"""Tiny CPU checks for S1 batch layouts and sample-separable DDP averaging.

The gradient comparison fixes inputs and disables random crops. Production
rank/microbatch seeds change with the layout, so this is mathematical averaging
equivalence within FP32 reduction tolerance, not bitwise training equivalence.
"""
import copy
from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

import pretrain
from losses.perceptual import adp_weight
from losses.reconstruction import adaptive_weight, reconstruction_loss
from utils.ema import EMA
from utils.s1_ablation import BATCH_LAYOUTS, loader_resources, validate_config, validate_runtime


ROOT = Path(__file__).resolve().parents[1]


class FakeLPIPS(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return (x - y).square().mean((1, 2, 3))


class FakeConvNext(nn.Module):
    def forward(self, x):
        # Spatial/channel features retain a separate row for every sample.
        return torch.cat((x.mean((2, 3)), x.square().mean((2, 3))), dim=1)


@contextmanager
def fake_encoders():
    encoders = ModuleType("models.encoders.perceptual")
    encoders.LPIPS, encoders.ConvNextFeatureExtractor = FakeLPIPS, FakeConvNext
    with patch.dict(sys.modules, {"models.encoders.perceptual": encoders}):
        yield


def fixture_config(accumulation=1):
    config, _ = pretrain.load_config(ROOT / "configs/pretrain/b80_perceptual_s100.yaml")
    config["train"]["gradient_accumulation_steps"] = accumulation
    config["loss"]["aux"]["random_crop"] = False
    return config


def fixture_inputs():
    x = torch.linspace(-.65, .85, 8 * 3 * 4 * 4).reshape(8, 3, 4, 4)
    y = torch.sin(x * 2.3) * .6
    return x, y


def fixture_auxiliary(config):
    with fake_encoders():
        raw = pretrain.build_auxiliary(config, "b", "unused.pth", device="cpu")
    return pretrain.AuxiliaryStream(raw, device="cpu")


class TinyReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([.55, -.2, .8]))
        self.bias = nn.Parameter(torch.tensor([-.05, .1, .03]))

    def forward(self, x, target, aux_loss_fn=None):
        prediction = x * self.scale.view(1, 3, 1, 1) + self.bias.view(1, 3, 1, 1)
        return reconstruction_loss(prediction, target, aux_loss_fn=aux_loss_fn,
                                   aux_target=target, times=torch.ones(len(x)),
                                   perceptual_threshold=1.)


def gradients(model):
    return torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters()])


def distributed_worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(seconds=40))
    try:
        x, y = fixture_inputs()
        results = {}
        for accumulation in (1, 2):
            config = fixture_config(accumulation)
            model = torch.nn.parallel.DistributedDataParallel(TinyReconstruction(),
                       broadcast_buffers=False, gradient_as_bucket_view=True)
            # Rank-strided rows cover the same eight global examples once.
            local_x, local_y = x[rank::4], y[rank::4]
            micro = 2 // accumulation
            batches = [(local_x[start:start + micro], local_y[start:start + micro])
                       for start in range(0, 2, micro)]
            pretrain.train_step(config, model, batches, fixture_auxiliary(config), 1,
                                family="b", rank=rank, device="cpu")
            results[str(accumulation)] = gradients(model).tolist()
        (directory / f"rank{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


class S1MultinodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_the_eight_batch_layouts_are_accepted(self):
        expected = {1: (64, 16), 2: (64, 8), 4: (64, 4), 8: (64, 2),
                    16: (64, 1), 32: (32, 1), 64: (16, 1), 128: (8, 1)}
        self.assertEqual(BATCH_LAYOUTS, expected)
        for world, (micro, accumulation) in expected.items():
            with self.subTest(world=world):
                config = fixture_config(accumulation)
                config["train"]["micro_batch_size"] = micro
                validate_config(config, "b")
                self.assertEqual(validate_runtime(config, world), 1024)
        for micro, accumulation in ((32, 2), (16, 4), (8, 2), (128, 1), (64, 3), (True, 1)):
            config = fixture_config(accumulation)
            config["train"]["micro_batch_size"] = micro
            with self.subTest(layout=(micro, accumulation)), self.assertRaises(ValueError):
                validate_config(config, "b")

    def test_runtime_rejections_atomically_record_layout_and_precision_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for world, micro, accumulation, dtype in ((32, 64, 1, "bf16"),
                    (16, 32, 2, "bf16"), (64, 16, 1, "fp32"), (256, 4, 1, "bf16")):
                config = fixture_config(accumulation)
                config["train"].update(micro_batch_size=micro, dtype=dtype)
                original_replace = os.replace
                replaced = []

                def replace(source, destination):
                    # The complete receipt must exist before atomic publication.
                    receipt = json.loads(Path(source).read_text())
                    self.assertEqual(receipt["global_batch"], world * micro * accumulation)
                    self.assertEqual(receipt["dtype"], dtype)
                    replaced.append(Path(destination))
                    original_replace(source, destination)

                with self.subTest(world=world, micro=micro, dtype=dtype):
                    with patch("utils.s1_ablation.os.replace", side_effect=replace):
                        with self.assertRaises(ValueError):
                            validate_runtime(config, world, output=output, rank=3)
                    receipt = json.loads((output / "s1_audit_rank3.json").read_text())
                    self.assertEqual(len(replaced), 1)
                    self.assertEqual(receipt["micro_batch_size"], micro)
                    self.assertEqual(receipt["accumulation"], accumulation)
                    self.assertEqual(receipt["required_global_batch"], 1024)
                    self.assertEqual(receipt["required_dtype"], "bf16")
                    self.assertEqual(receipt["expected_micro_batch_size"],
                                     BATCH_LAYOUTS.get(world, (None, None))[0])
                    self.assertEqual(receipt["expected_accumulation"],
                                     BATCH_LAYOUTS.get(world, (None, None))[1])
                    self.assertFalse(receipt["passed"])
                    self.assertFalse(list(output.glob("*.tmp")))

    def test_cpu_limits_reduce_workers_without_mutating_scientific_config(self):
        config = fixture_config()
        original = copy.deepcopy(config)
        self.assertEqual(loader_resources(10, environ={}),
                         dict(num_workers=10, threads=4, cpu_per_rank=None))
        for cpus, local, workers, threads, budget in ((128, 8, 10, 4, 16),
                (80, 8, 9, 1, 10), (32, 8, 3, 1, 4), (8, 8, 0, 1, 1)):
            actual = loader_resources(config["data"]["num_workers"], environ={
                "SLURM_CPUS_PER_TASK": str(cpus), "LOCAL_WORLD_SIZE": str(local),
                "WORLD_SIZE": "128"})
            self.assertEqual(actual, dict(num_workers=workers, threads=threads, cpu_per_rank=budget))
        self.assertEqual(config, original)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                loader_resources(10, environ={"SLURM_CPUS_PER_TASK": "0", "LOCAL_WORLD_SIZE": "8"},
                                 output=directory, rank=2)
            receipt = json.loads((Path(directory) / "s1_audit_rank2.json").read_text())
            self.assertEqual(receipt["gate"], "cpu_budget")
            self.assertEqual((receipt["cpus"], receipt["local_gpus"], receipt["requested_workers"]), (0, 8, 10))
            self.assertFalse(receipt["passed"])
            with self.assertRaisesRegex(ValueError, "at least one CPU"):
                loader_resources(10, environ={"SLURM_CPUS_PER_TASK": "4", "LOCAL_WORLD_SIZE": "8"},
                                 output=directory, rank=2)
            receipt = json.loads((Path(directory) / "s1_audit_rank2.json").read_text())
            self.assertEqual((receipt["cpus"], receipt["required_cpus"]), (4, 8))

    def test_adaptive_pixel_and_perceptual_losses_do_not_mix_samples(self):
        from losses.perceptual import PerceptualLoss
        values = torch.tensor([.1, .5, 3.], requires_grad=True)
        for fn in (adaptive_weight, adp_weight):
            whole = fn(values)
            singleton = torch.cat([fn(value.view(1)) for value in values])
            self.assertTrue(torch.equal(whole, singleton))
        x, y = fixture_inputs()
        with fake_encoders():
            perceptual = PerceptualLoss(random_crop=False).eval()
        x = x.requires_grad_()
        combined = perceptual(x, y, torch.ones(8, dtype=torch.bool))
        singles = [perceptual(x[i:i + 1], y[i:i + 1], torch.ones(1, dtype=torch.bool)) for i in range(8)]
        for index in range(3):
            self.assertTrue(torch.allclose(combined[index], torch.cat([row[index] for row in singles]),
                                           rtol=2e-6, atol=2e-6))
            self.assertTrue(bool((combined[index] > 0).all()))
        for random_crop in (False, True):
            perceptual.random_crop = random_crop
            torch.manual_seed(170)
            before = perceptual(x, y, torch.ones(8, dtype=torch.bool))
            changed = y.clone()
            changed[1:] += 2.
            torch.manual_seed(170)
            after = perceptual(x, changed, torch.ones(8, dtype=torch.bool))
            for first, second in zip(before, after):
                self.assertTrue(torch.equal(first[:1], second[:1]))
            jacobian_row = torch.autograd.grad(before[0][0], x)[0]
            self.assertTrue(torch.equal(jacobian_row[1:], torch.zeros_like(jacobian_row[1:])))
        # The actual reconstruction routine averages per-image adaptive terms.
        whole, _ = reconstruction_loss(x, y)
        singles = torch.stack([reconstruction_loss(x[i:i + 1], y[i:i + 1])[0] for i in range(8)])
        self.assertTrue(torch.allclose(whole, singles.mean(), rtol=1e-6, atol=1e-7))

    def test_gloo_four_rank_gradients_match_one_rank_fixed_global_batch(self):
        x, y = fixture_inputs()
        config, model = fixture_config(), TinyReconstruction()
        pretrain.train_step(config, model, [(x, y)], fixture_auxiliary(config), 1,
                            family="b", rank=0, device="cpu")
        expected = gradients(model).detach()
        with tempfile.TemporaryDirectory() as directory:
            # Fork shares the read-only Torch import; all children use one thread.
            mp.start_processes(distributed_worker, args=(directory,), nprocs=4,
                               join=True, start_method="fork")
            maximum_error = 0.
            for rank in range(4):
                results = json.loads((Path(directory) / f"rank{rank}.json").read_text())
                for accumulation, values in results.items():
                    actual = torch.tensor(values)
                    maximum_error = max(maximum_error, (actual - expected).abs().max().item())
                    with self.subTest(rank=rank, accumulation=accumulation):
                        self.assertTrue(torch.allclose(actual, expected, rtol=2e-5, atol=2e-6),
                                        f"max |gradient difference|={maximum_error}")
            print(f"Fixed-input Gloo W4/micro2 and W4/micro1/accum2 vs W1/micro8: "
                  f"max |gradient difference|={maximum_error:.3g}")
        ema = EMA(model, batch_size=1024)
        self.assertEqual(ema.state_dict()["batch_size"], 1024)


if __name__ == "__main__":
    unittest.main()
