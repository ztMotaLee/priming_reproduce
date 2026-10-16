"""Small CPU checks for the opt-in Stage-1 noise ablation and launchers.

No weights, data, CUDA, distributed process group, or network are needed.
PRIMING_RELEASE_ROOT can select a read-only official checkout. Otherwise the
official revision is read from Git; it is never checked out or modified.
"""
import copy
from contextlib import contextmanager, ExitStack
from functools import partial
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

import torch
from torch import nn
from omegaconf import OmegaConf

import pretrain
from data.imagenet import ResumeSampler
from models import priming
from utils.ema import EMA
from utils.optim import create_optimizer
from utils.pretrain_checkpoint import data_cursor, resume_pretraining, save_pretraining
from utils.random import mixed_seed, seed_all, snapshot_rng
from utils.state import exact_optimizer


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REVISION = "107cf0459e561a3e53640ef88750ed2aaff4957b"


def official_source(relative: str) -> str:
    reference = os.environ.get("PRIMING_RELEASE_ROOT")
    if reference:
        return (Path(reference) / relative).read_text()
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{OFFICIAL_REVISION}:{relative}"], text=True
    )


def new_directory() -> Path:
    # Use S1_TEST_TMPDIR to keep every test artifact in a chosen scratch folder.
    return Path(tempfile.mkdtemp(prefix="stage1-", dir=os.environ.get("S1_TEST_TMPDIR")))


class StubLPIPS(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, y):
        return (x - y).square().mean((1, 2, 3))


class StubConvNext(nn.Module):
    def forward(self, x):
        return x.mean((2, 3))


@contextmanager
def fake_perceptual():
    encoders = ModuleType("models.encoders.perceptual")
    encoders.LPIPS = StubLPIPS
    encoders.ConvNextFeatureExtractor = StubConvNext
    with patch.dict(sys.modules, {"models.encoders.perceptual": encoders}):
        yield


class S1AblationTests(unittest.TestCase):
    class StartupFinished(Exception):
        """Stop before config writes, DDP construction, or any training."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.official = ModuleType("_official_stage1_pretrain")
        cls.official.__file__ = "<official pretrain.py>"
        exec(compile(official_source("pretrain.py"), cls.official.__file__, "exec"), cls.official.__dict__)

    def config(self, name="b80_perceptual_s100"):
        return pretrain.load_config(ROOT / "configs" / "pretrain" / f"{name}.yaml")[0]

    def save_config(self, config):
        path = new_directory() / "recipe.yaml"
        OmegaConf.save(OmegaConf.create(config), path)
        return path

    def tiny_model(self, config):
        compact = copy.deepcopy(config)
        compact["data"].update(img_size=8, num_classes=3)
        compact["model"].update(patch_size=4, class_tokens=2)
        with patch.dict(priming.OST_models,
                        OST_B=partial(priming.OST, depth=1, hidden_size=32, num_heads=4)):
            model = pretrain.make_model(compact, device="cpu")
        return model

    def test_official_config_is_identical_in_subprocess(self):
        # A fresh interpreter imports the entire official pretrain module.
        child = """
import json, sys, types
payload = json.load(sys.stdin)
module = types.ModuleType('_official_pretrain_config')
module.__file__ = '<official pretrain.py>'
exec(compile(payload['source'], module.__file__, 'exec'), module.__dict__)
print(json.dumps([module.load_config(p) for p in payload['paths']], sort_keys=True))
"""
        names = ("b160_perceptual", "b160_no_perceptual", "l160_perceptual")
        paths = [str(ROOT / "configs" / "pretrain" / f"{name}.yaml") for name in names]
        completed = subprocess.run(
            [sys.executable, "-c", child], input=json.dumps(dict(source=official_source("pretrain.py"), paths=paths)),
            cwd=ROOT, text=True, capture_output=True, check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        current = [pretrain.load_config(path) for path in paths]
        self.assertEqual(json.loads(completed.stdout), json.loads(json.dumps(current)))

    def test_original_model_and_loss_sources_are_unchanged(self):
        for path in ("models/priming.py", "models/layers.py", "losses/reconstruction.py",
                     "losses/perceptual.py", "utils/random.py", "data/imagenet.py"):
            with self.subTest(path=path):
                self.assertEqual((ROOT / path).read_text(), official_source(path))

    def test_official_and_s080_losses_gradients_and_rng_are_bitwise_equal(self):
        official_config = self.config("b160_perceptual")
        control_config = self.config("b80_perceptual_s080")
        torch.manual_seed(17)
        template = self.tiny_model(official_config)
        with torch.no_grad():
            for parameter in template.parameters():
                parameter.copy_(torch.randn_like(parameter) * .1)
        initial = copy.deepcopy(template.state_dict())
        batches = [(torch.linspace(-1, 1, 384).reshape(2, 3, 8, 8),
                    torch.tensor([0, 2]), torch.zeros(2, 2, dtype=torch.long)) for _ in range(2)]
        results = []
        for runner, config in ((self.official, official_config), (pretrain, official_config),
                               (pretrain, control_config)):
            model = self.tiny_model(config)
            model.load_state_dict(initial)
            with fake_perceptual():
                auxiliary = runner.AuxiliaryStream(
                    runner.build_auxiliary(config, "b", "unused.pth", device="cpu"), device="cpu")
            torch.manual_seed(43)
            loss, metrics = runner.train_step(config, model, batches, auxiliary, 17,
                                              family="b", rank=3, device="cpu")
            results.append((loss.detach(), {k: v.detach() for k, v in metrics.items()},
                            {k: p.grad.clone() for k, p in model.named_parameters()}, torch.get_rng_state()))
        baseline = results[0]
        for result in results[1:]:
            self.assertTrue(torch.equal(baseline[0], result[0]))
            for key, value in baseline[1].items():
                self.assertTrue(torch.equal(value, result[1][key]), key)
            for key, value in baseline[2].items():
                self.assertTrue(torch.equal(value, result[2][key]), key)
            self.assertTrue(torch.equal(baseline[3], result[3]))
        self.assertEqual(results[2][1]["perceptual_active_fraction"].item(), 1.)
        self.assertAlmostEqual(results[2][1]["noise_level"].item(), .8)

    def test_pure_noise_input_is_exactly_the_sampled_noise(self):
        model = self.tiny_model(self.config())
        clean = torch.linspace(-1, 1, 384).reshape(2, 3, 8, 8)
        noise = torch.randn_like(clean)
        times = model.sample_t(2, "cpu", clean.dtype)
        self.assertTrue(torch.equal(times, torch.ones_like(times)))
        self.assertTrue(torch.equal(model.corrupt_inputs(clean, times, noise), noise))
        captured = []
        handle = model.net.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
        try:
            with patch("torch.randn_like", return_value=noise):
                model(clean, torch.tensor([0, 2]))
        finally:
            handle.remove()
        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(captured[0], noise))

    def test_perceptual_mask_and_both_terms_are_active_at_s100(self):
        config = self.config()
        model = self.tiny_model(config)
        with fake_perceptual():
            raw = pretrain.build_auxiliary(config, "b", "unused.pth", device="cpu")
        masks = []

        def observe(prediction, target, mask):
            masks.append(mask.detach().clone())
            return raw(prediction, target, mask)

        auxiliary = pretrain.AuxiliaryStream(observe, device="cpu")
        batch = (torch.ones(2, 3, 8, 8) * .5, torch.tensor([0, 2]))
        loss, metrics = pretrain.train_step(config, model, [batch, batch], auxiliary, 1,
                                            family="b", rank=0, device="cpu")
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(masks), 2)
        self.assertTrue(all(bool(mask.all()) for mask in masks))
        self.assertGreater(metrics["aux_loss_lpips"].item(), 0.)
        self.assertGreater(metrics["aux_loss_convnext"].item(), 0.)
        self.assertEqual(metrics["perceptual_active_fraction"].item(), 1.)
        self.assertEqual(metrics["noise_level"].item(), 1.)
        # Guard the actual runtime mask, even if a caller mutates a loaded recipe.
        model.perceptual_threshold = .8
        with self.assertRaises((ValueError, RuntimeError, AssertionError)):
            pretrain.train_step(config, model, [batch, batch], auxiliary, 2,
                                family="b", rank=0, device="cpu")

    def test_invalid_recipes_and_batches_are_rejected(self):
        from utils.s1_ablation import validate_runtime
        official = self.config("b160_perceptual")
        invalid = []
        changed_official = copy.deepcopy(official)
        changed_official["model"].update(t_min=1., t_max=1.)
        invalid.append(changed_official)
        for section, key, value in (("ablation", "noise_level", .3), ("model", "t_min", .9),
                                    ("loss", "perceptual_threshold", .8), ("train", "epochs", 81),
                                    ("train", "gradient_accumulation_steps", 3)):
            config = self.config()
            config[section][key] = value
            invalid.append(config)
        for config in invalid:
            with self.subTest(config=config), self.assertRaises((ValueError, TypeError)):
                pretrain.load_config(self.save_config(config))
        config = self.config()
        for world in (1, 2, 4, 8, 16):
            config["train"]["gradient_accumulation_steps"] = 16 // world
            validate_runtime(config, world)
        config["train"]["gradient_accumulation_steps"] = 2
        for world in (0, 3, 4, 32):
            with self.subTest(world=world), self.assertRaises(ValueError):
                validate_runtime(config, world)
        with self.assertRaises(ValueError):
            pretrain.train_step(config, self.tiny_model(config), [], None, 1,
                                family="b", rank=0, device="cpu")

    def test_documented_ablation_values_are_accepted(self):
        for noise in (.5, .6, .7, .8, .9, 1.):
            config = self.config()
            config["ablation"]["noise_level"] = noise
            config["model"].update(t_min=noise, t_max=noise)
            pretrain.load_config(self.save_config(config))
        for epochs in (80, 160):
            for accumulation in (1, 2, 4, 8, 16):
                config = self.config()
                config["train"].update(epochs=epochs, gradient_accumulation_steps=accumulation)
                pretrain.load_config(self.save_config(config))

    def test_failed_runtime_gates_write_their_inputs_before_raising(self):
        from utils.s1_ablation import validate_mask, validate_runtime
        config = self.config()
        output = new_directory()
        with self.assertRaises(ValueError):
            validate_runtime(config, 4, output=output, rank=2)
        receipt = json.loads((output / "s1_audit_rank2.json").read_text())
        self.assertEqual(receipt["gate"], "global_batch")
        self.assertEqual([receipt[k] for k in ("world_size", "micro_batch_size", "accumulation",
                                              "global_batch", "required_global_batch")], [4, 64, 2, 512, 1024])
        self.assertFalse(receipt["passed"])
        with self.assertRaises(ValueError):
            validate_mask(torch.tensor([True, False]), output=output, rank=2,
                          step=7, micro=3, noise_level=1., perceptual_threshold=.8)
        receipt = json.loads((output / "s1_audit_rank2.json").read_text())
        self.assertEqual(receipt["gate"], "perceptual_mask")
        self.assertEqual([receipt[k] for k in ("step", "micro", "active_samples", "total_samples",
                                              "perceptual_active_fraction", "noise_level",
                                              "perceptual_threshold")], [7, 3, 1, 2, .5, 1., .8])
        self.assertFalse(receipt["passed"])
        self.assertFalse(list(output.glob("*.tmp")))

    def test_rejected_launch_preserves_existing_output(self):
        parent = new_directory()
        output = parent / "prior-output"
        output.mkdir()
        sentinel = output / "sentinel.txt"
        sentinel.write_text("existing result\n")
        with patch.dict(os.environ, WORLD_SIZE="4", RANK="0"):
            with self.assertRaisesRegex(ValueError, "1024"):
                pretrain.train(self.config(), "b", SimpleNamespace(), output, None, None)
        self.assertEqual(sentinel.read_text(), "existing result\n")
        self.assertEqual(list(output.iterdir()), [sentinel])
        receipts = list(parent.glob("prior-output.s1_rejected_*/s1_audit_rank0.json"))
        self.assertEqual(len(receipts), 1)
        receipt = json.loads(receipts[0].read_text())
        self.assertEqual(receipt["global_batch"], 512)
        self.assertFalse(receipt["passed"])

    def output_bytes(self, output):
        return {path.relative_to(output): path.read_bytes()
                for path in output.rglob("*") if path.is_file()}

    @contextmanager
    def mocked_train_startup(self, *, valid_mask=True, resume_error=None, barrier_error=None):
        # Exercise train() and real audit I/O without creating data, CUDA, or DDP.
        parent = new_directory()
        output = parent / "prior-output"
        (output / "checkpoints").mkdir(parents=True)
        for name, content in {
            "s1_audit_rank0.json": b'{ "old": "rank-zero audit" }\n',
            "s1_audit_rank7.json": b'{ "old": "rank-seven audit" }\n',
            "config.json": b'{ "old": "configuration" }\n',
            "manifest.json": b'{ "old": "asset identity" }\n',
            "checkpoints/latest.pth": b"existing checkpoint\x00\xff\n",
            "sentinel.txt": b"existing result\n",
        }.items():
            (output / name).write_bytes(content)
        previous_rejected = parent / f"{output.name}.s1_rejected_{'0' * 32}"
        previous_rejected.mkdir()
        previous_receipt = previous_rejected / "s1_audit_rank0.json"
        previous_receipt.write_bytes(b'{ "old": "rejected launch" }\n')
        before, previous_bytes = self.output_bytes(output), previous_receipt.read_bytes()
        events, replacements = [], []
        original_replace, original_write = os.replace, Path.write_text
        checkpoint = output / "checkpoints/latest.pth"
        config = self.config()
        args = SimpleNamespace(resume=str(checkpoint), lpips_weights="unused.pth", smoke_steps=1)

        def replace(source, destination):
            destination = Path(destination)
            replacements.append(destination)
            original_replace(source, destination)
            events.append(("audit", destination))

        def write_text(path, *args, **kwargs):
            if path == output / "config.json":
                events.append(("config", path))
                raise self.StartupFinished()
            return original_write(path, *args, **kwargs)

        def resume(path, *args, **kwargs):
            events.append(("resume", path))
            if resume_error is not None:
                raise resume_error
            return 1, 0.

        def barrier(*args, **kwargs):
            events.append(("barrier", None))
            if barrier_error is not None and ("resume", checkpoint.resolve()) in events:
                raise barrier_error

        model = SimpleNamespace(perceptual_threshold=1. if valid_mask else .8,
                                sample_t=lambda count, device, dtype: torch.ones(count, dtype=dtype))
        with ExitStack() as stack:
            for target, kwargs in (
                ("utils.distributed.initialize", dict(return_value=(0, 8))),
                ("utils.distributed.broadcast_model", {}),
                ("torch.distributed.broadcast_object_list", {}),
                ("torch.distributed.barrier", dict(side_effect=barrier)),
                ("torch.distributed.destroy_process_group", {}),
                ("utils.assets.pretraining_manifest", dict(return_value={"fixture": "startup identity"})),
                ("data.imagenet.build_dataset", dict(return_value=range(1281167))),
                ("torch.utils.data.DataLoader", {}),
                ("utils.ema.EMA", {}),
                ("utils.optim.create_optimizer", {}),
                ("utils.random.seed_all", {}),
                ("utils.random.snapshot_rng", dict(return_value=None)),
                ("utils.pretrain_checkpoint.resume_pretraining", dict(side_effect=resume)),
                ("torch.set_num_threads", {}),
                ("torch.use_deterministic_algorithms", {}),
                ("logging.basicConfig", {}),
                ("torch.nn.parallel.DistributedDataParallel",
                 dict(side_effect=AssertionError("startup regression must not construct DDP"))),
                ("os.replace", dict(side_effect=replace)),
            ):
                stack.enter_context(patch(target, **kwargs))
            for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp", "enable_math_sdp"):
                stack.enter_context(patch.object(torch.backends.cuda, name))
            stack.enter_context(patch.object(torch.backends.cuda, "matmul", SimpleNamespace(allow_tf32=False)))
            stack.enter_context(patch.object(torch.backends, "cudnn",
                                             SimpleNamespace(allow_tf32=False, benchmark=False, deterministic=False)))
            stack.enter_context(patch.object(pretrain, "make_model", return_value=model))
            stack.enter_context(patch.object(pretrain, "build_auxiliary", return_value=None))
            stack.enter_context(patch.object(Path, "write_text", write_text))
            stack.enter_context(patch.dict(os.environ, WORLD_SIZE="8", RANK="0", LOCAL_RANK="0"))
            yield SimpleNamespace(output=output, config=config, args=args, checkpoint=checkpoint,
                                  before=before, previous_rejected=previous_rejected,
                                  previous_receipt=previous_receipt, previous_bytes=previous_bytes,
                                  events=events, replacements=replacements)

    def rejected_startup_receipt(self, fixture):
        self.assertEqual(self.output_bytes(fixture.output), fixture.before)
        self.assertFalse(any(path.is_relative_to(fixture.output) for path in fixture.replacements))
        self.assertEqual(fixture.previous_receipt.read_bytes(), fixture.previous_bytes)
        rejected = [path for path in fixture.output.parent.glob(f"{fixture.output.name}.s1_rejected_*")
                    if path != fixture.previous_rejected]
        self.assertEqual(len(rejected), 1)
        suffix = rejected[0].name.split(".s1_rejected_", 1)[1]
        self.assertEqual(UUID(suffix).hex, suffix)
        receipt_path = rejected[0] / "s1_audit_rank0.json"
        self.assertEqual(fixture.replacements, [receipt_path])
        self.assertEqual(list(rejected[0].iterdir()), [receipt_path])
        receipt = json.loads(receipt_path.read_text())
        self.assertFalse(receipt["passed"])
        return receipt

    def test_resume_identity_rejection_preserves_every_existing_output_byte(self):
        error = ValueError("Checkpoint asset identity changed: fixture identity mismatch")
        with self.mocked_train_startup(resume_error=error) as fixture:
            with self.assertRaisesRegex(ValueError, "identity changed") as caught:
                pretrain.train(fixture.config, "b", fixture.args, fixture.output, None, None)
            self.assertIs(caught.exception, error)
            self.assertIn(("resume", fixture.checkpoint.resolve()), fixture.events)
            receipt = self.rejected_startup_receipt(fixture)
            self.assertEqual(receipt["gate"], "resume")
            self.assertEqual(receipt["checkpoint"], str(fixture.checkpoint.resolve()))
            self.assertEqual(receipt["error"], repr(error))

    def test_startup_mask_rejection_preserves_every_existing_output_byte(self):
        with self.mocked_train_startup(valid_mask=False) as fixture:
            with self.assertRaisesRegex(ValueError, "perceptual mask"):
                pretrain.train(fixture.config, "b", fixture.args, fixture.output, None, None)
            self.assertNotIn(("resume", fixture.checkpoint.resolve()), fixture.events)
            receipt = self.rejected_startup_receipt(fixture)
            self.assertEqual(receipt["gate"], "perceptual_mask")
            self.assertEqual((receipt["active_samples"], receipt["total_samples"]), (0, 64))
            self.assertEqual((receipt["noise_level"], receipt["perceptual_threshold"]), (1., .8))

    def test_resume_success_records_runtime_audit_only_after_identity_acceptance(self):
        with self.mocked_train_startup() as fixture:
            with self.assertRaises(self.StartupFinished):
                pretrain.train(fixture.config, "b", fixture.args, fixture.output, None, None)
            audit_path = fixture.output / "s1_audit_rank0.json"
            self.assertEqual(fixture.events, [("barrier", None), ("resume", fixture.checkpoint.resolve()),
                                             ("barrier", None),
                                             ("audit", audit_path), ("config", fixture.output / "config.json")])
            self.assertEqual(fixture.replacements, [audit_path])
            receipt = json.loads(audit_path.read_text())
            self.assertEqual((receipt["gate"], receipt["global_batch"], receipt["passed"]),
                             ("global_batch", 1024, True))
            after = self.output_bytes(fixture.output)
            after.pop(Path("s1_audit_rank0.json"))
            before = dict(fixture.before)
            before.pop(Path("s1_audit_rank0.json"))
            self.assertEqual(after, before)
            self.assertEqual(fixture.previous_receipt.read_bytes(), fixture.previous_bytes)
            self.assertEqual(list(fixture.output.parent.glob(f"{fixture.output.name}.s1_rejected_*")),
                             [fixture.previous_rejected])

    def test_resume_barrier_failure_preserves_every_existing_output_byte(self):
        error = RuntimeError("fixture: another rank rejected startup")
        with self.mocked_train_startup(barrier_error=error) as fixture:
            with self.assertRaisesRegex(RuntimeError, "another rank rejected startup") as caught:
                pretrain.train(fixture.config, "b", fixture.args, fixture.output, None, None)
            self.assertIs(caught.exception, error)
            self.assertEqual(fixture.events, [("barrier", None), ("resume", fixture.checkpoint.resolve()),
                                             ("barrier", None)])
            self.assertEqual(self.output_bytes(fixture.output), fixture.before)
            self.assertEqual(fixture.replacements, [])
            self.assertEqual(fixture.previous_receipt.read_bytes(), fixture.previous_bytes)
            self.assertEqual(list(fixture.output.parent.glob(f"{fixture.output.name}.s1_rejected_*")),
                             [fixture.previous_rejected])

    def test_accumulation_four_syncs_only_the_final_microbatch(self):
        config = self.config()
        config["train"]["gradient_accumulation_steps"] = 4
        model = self.tiny_model(config)

        class Wrapped(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped, self.in_no_sync, self.observed = wrapped, False, []

            @contextmanager
            def no_sync(self):
                self.in_no_sync = True
                try:
                    yield
                finally:
                    self.in_no_sync = False

            def forward(self, *args, **kwargs):
                self.observed.append((self.in_no_sync, torch.initial_seed()))
                return self.wrapped(*args, **kwargs)

        wrapped = Wrapped(model)
        batch = (torch.ones(2, 3, 8, 8) * .5, torch.tensor([0, 2]))
        with fake_perceptual():
            auxiliary = pretrain.AuxiliaryStream(
                pretrain.build_auxiliary(config, "b", "unused.pth", device="cpu"), device="cpu")
        loss, metrics = pretrain.train_step(config, wrapped, [batch] * 4, auxiliary, 7,
                                            family="b", rank=2, device="cpu")
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual([v[0] for v in wrapped.observed], [True, True, True, False])
        self.assertEqual([v[1] for v in wrapped.observed],
                         [mixed_seed(config["train"]["seed"], 101, 2, 7, micro) for micro in range(4)])
        self.assertEqual(len(set(v[1] for v in wrapped.observed)), 4)
        self.assertEqual(metrics["perceptual_active_fraction"].item(), 1.)

    def test_sampler_coverage_and_resume_match_for_four_and_eight_gpus(self):
        dataset = range(5137)
        outputs = []
        for world, accumulation in ((4, 4), (8, 2)):
            config = self.config()
            config["train"]["gradient_accumulation_steps"] = accumulation
            local_update = 64 * accumulation
            streams = []
            resumed = []
            for rank in range(world):
                sampler = ResumeSampler(dataset, num_replicas=world, rank=rank, seed=1, drop_last=True)
                sampler.set_epoch(3)
                stream = list(sampler)
                self.assertTrue(all(epoch == 3 for epoch, _ in stream))
                streams.append([index for _, index in stream[:5 * local_update]])
                sampler.offset = 2 * local_update
                resumed.append([index for _, index in sampler][:3 * local_update])
                self.assertEqual(resumed[-1], streams[-1][2 * local_update:])
                self.assertEqual(data_cursor(2, config)["offset"], sampler.offset)
            updates = [set(index for stream in streams for index in
                           stream[step * local_update:(step + 1) * local_update]) for step in range(5)]
            self.assertTrue(all(len(update) == 1024 for update in updates))
            self.assertEqual(len(set.union(*updates)), 5120)
            self.assertEqual(sum(map(len, updates)), len(set.union(*updates)))
            self.assertEqual(set(index for stream in resumed for index in stream), set.union(*updates[2:]))
            outputs.append(updates)
        self.assertEqual(outputs[0], outputs[1])

    def test_checkpoint_retains_twenty_epoch_milestones_and_final_epoch(self):
        config = self.config()
        config["train"]["steps_per_epoch"] = 1
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        ema = EMA(model, batch_size=1024)
        folder = new_directory()
        transport = dict(get_rank=lambda: 0, get_world_size=lambda: 8,
                         broadcast_object_list=lambda *args, **kwargs: None, barrier=lambda: None)
        # Retention is shared by B and L. L avoids unrelated Muon-shard fixtures.
        with patch.multiple(torch.distributed, **transport):
            for step in (20, 21, 22, 23, 40, 41, 42, 43, 60, 61, 62, 63, 78, 79, 80):
                save_pretraining(folder, step, model, optimizer, ema, config, {}, family="l")
        retained = {int(path.stem.split("_")[-1]) for path in (folder / "checkpoints").glob("step_*.pth")}
        self.assertEqual(retained, {20, 40, 60, 78, 79, 80})
        self.assertEqual((folder / "checkpoints" / "latest.pth").resolve().name, "step_0000080.pth")

    def test_b_checkpoint_restores_the_next_update_bitwise(self):
        # World-one transport exercises the real B payload, Muon, EMA, and RNGs.
        # Tiny tensors replace production images; batch topology remains valid.
        config = self.config()
        config["train"]["gradient_accumulation_steps"] = 16
        seed_all(19)
        model = self.tiny_model(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * .1)
        ema = EMA(model, batch_size=1024)
        loader_rng = torch.Generator().manual_seed(101)
        with fake_perceptual():
            auxiliary = pretrain.AuxiliaryStream(
                pretrain.build_auxiliary(config, "b", "unused.pth", device="cpu"), device="cpu")
        batch = (torch.linspace(-1, 1, 384).reshape(2, 3, 8, 8), torch.tensor([0, 2]))
        folder, manifest = new_directory(), {"fixture": "tiny-cpu"}
        transport = dict(get_rank=lambda: 0, get_world_size=lambda: 1,
                         all_gather=lambda *args, **kwargs: None,
                         gather_object=lambda value, destinations, dst=0: destinations.__setitem__(0, value),
                         broadcast_object_list=lambda *args, **kwargs: None, barrier=lambda: None)
        with patch.multiple(torch.distributed, **transport):
            optimizer = create_optimizer(model, config["optimizer"])

            def update(step):
                optimizer.zero_grad(set_to_none=True)
                result = pretrain.train_step(config, model, [batch] * 16, auxiliary, step,
                                             family="b", rank=0, device="cpu")
                optimizer.step()
                ema.step(model)
                optimizer.zero_grad(set_to_none=True)
                return result

            update(1)
            torch.rand(3, generator=loader_rng)
            saved_loader_rng = loader_rng.get_state().clone()
            saved_rng = snapshot_rng()
            save_pretraining(folder, 1, model, optimizer, ema, config, manifest,
                             family="b", elapsed=3.5, loader_generator=loader_rng)
            checkpoint = folder / "checkpoints" / "latest.pth"
            saved_model = copy.deepcopy(model.state_dict())
            baseline_loss, baseline_metrics = update(2)
            uninterrupted = copy.deepcopy((model.state_dict(), optimizer.state_dict(), ema.state_dict(), snapshot_rng()))
            seed_all(999)
            loader_rng.manual_seed(999)
            step, elapsed = resume_pretraining(checkpoint, model, optimizer, ema, config, manifest,
                                               family="b", loader_generator=loader_rng)
            self.assertEqual((step, elapsed, ema.step_count), (1, 3.5, 1))
            exact_optimizer(model.state_dict(), saved_model)
            exact_optimizer(snapshot_rng(), saved_rng)
            self.assertTrue(torch.equal(loader_rng.get_state(), saved_loader_rng))
            resumed_loss, resumed_metrics = update(step + 1)
            self.assertTrue(torch.equal(baseline_loss, resumed_loss))
            exact_optimizer(baseline_metrics, resumed_metrics)
            exact_optimizer((model.state_dict(), optimizer.state_dict(), ema.state_dict(), snapshot_rng()), uninterrupted)
            with patch.object(torch.distributed, "get_world_size", return_value=2):
                with self.assertRaisesRegex(ValueError, "counters|topology"):
                    resume_pretraining(checkpoint, model, optimizer, ema, config, manifest,
                                       family="b", loader_generator=loader_rng)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                resume_pretraining(checkpoint, model, optimizer, ema, config, {"fixture": "changed"},
                                   family="b", loader_generator=loader_rng)


class S1LauncherTests(unittest.TestCase):
    def launch(self, storage, *, visible=None, fake_gpus=None, extra=None):
        environment = {**os.environ, "STORAGE": str(storage), "DRY_RUN": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "NOISE_LEVEL": "1.0", "EPOCHS": "80"}
        if visible is None:
            environment.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            environment["CUDA_VISIBLE_DEVICES"] = visible
        if fake_gpus is not None:
            binary = new_directory() / "nvidia-smi"
            binary.write_text("#!/usr/bin/env bash\nprintf '%s\\n' " + " ".join(shlex.quote(str(i)) for i in range(fake_gpus)) + "\n")
            binary.chmod(0o755)
            environment["PATH"] = str(binary.parent) + os.pathsep + environment["PATH"]
        environment.update(extra or {})
        return subprocess.run(["bash", str(ROOT / "train.sh")], cwd=new_directory(),
                              env=environment, capture_output=True, text=True)

    def test_visible_gpu_counts_and_accumulation(self):
        for count in (1, 2, 4, 8, 16):
            storage = new_directory() / "not-created"
            result = self.launch(storage, visible=",".join(map(str, range(count))))
            with self.subTest(count=count):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"GPU_COUNT={count}", result.stdout)
                self.assertIn(f"GRADIENT_ACCUMULATION_STEPS={16 // count}", result.stdout)
                self.assertIn(f"--nproc_per_node={count}", result.stdout)
                self.assertIn("--data-path", result.stdout)
                self.assertIn("--hf-home", result.stdout)
                self.assertIn("--torch-home", result.stdout)
                self.assertIn("--lpips-weights", result.stdout)
                self.assertIn("--output", result.stdout)
                self.assertFalse(storage.exists(), "DRY_RUN must not create storage")
        for visible in ("", "-1", "0,1,2"):
            with self.subTest(visible=visible):
                self.assertNotEqual(self.launch(new_directory(), visible=visible).returncode, 0)

    def test_nvidia_smi_fallback(self):
        result = self.launch(new_directory(), fake_gpus=4)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("GPU_COUNT=4", result.stdout)
        self.assertIn("GRADIENT_ACCUMULATION_STEPS=4", result.stdout)

    def test_resume_and_recipe_selection(self):
        storage = new_directory()
        output = storage / "outputs" / "pretrain" / "b80_perceptual_s100_4gpu"
        (output / "checkpoints").mkdir(parents=True)
        latest = output / "checkpoints" / "latest.pth"
        latest.touch()  # The shell checks identity metadata; it never reads tensors.
        (output / "config.json").write_text(json.dumps({"train": {"gradient_accumulation_steps": 4}}))
        result = self.launch(storage, visible="0,1,2,3", extra={"SMOKE_STEPS": "60"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--resume", result.stdout)
        self.assertIn(str(latest), result.stdout)
        self.assertIn("--smoke-steps 60", result.stdout)
        (output / "config.json").write_text(json.dumps({"train": {"gradient_accumulation_steps": 2}}))
        self.assertNotEqual(self.launch(storage, visible="0,1,2,3").returncode, 0)
        for noise, epochs, expected in (("0.8", "80", "b80_perceptual_s080"),
                                        ("1.0", "160", "b160_perceptual_s100")):
            result = self.launch(new_directory(), visible="0", extra={"NOISE_LEVEL": noise, "EPOCHS": epochs})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(expected, result.stdout)
        for extra in ({"NOISE_LEVEL": ".7"}, {"EPOCHS": "81"}, {"SMOKE_STEPS": "0"}):
            self.assertNotEqual(self.launch(new_directory(), visible="0", extra=extra).returncode, 0)

    def test_slurm_signal_uses_absolute_storage_after_changing_directory(self):
        repository, launch_directory = new_directory(), new_directory()
        child_script = repository / "train.sh"
        child_script.write_text(
            '#!/usr/bin/env bash\nset -eu\ncd "$(dirname "$0")"\n'
            'echo READY\nfor ((i=0;i<100;i++)); do\n'
            '[[ ! -f "$PREEMPT_FLAG_FILE" ]] || exit 75\nsleep 0.02\ndone\nexit 1\n')
        environment = {**os.environ, "STORAGE": "relative-storage", "SLURM_NNODES": "1",
                       "SLURM_JOB_ID": "fixture", "SLURM_SUBMIT_DIR": str(repository)}
        child = subprocess.Popen(["bash", str(ROOT / "train.slurm")], cwd=launch_directory,
                                 env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "READY")
            child.send_signal(signal.SIGUSR1)
            _, stderr = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 75, stderr)
            self.assertTrue((launch_directory / "relative-storage/jobs/preempt-fixture").is_file())
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()

    def test_shell_syntax_and_shellcheck_when_installed(self):
        scripts = [ROOT / "setup.sh", ROOT / "train.sh", ROOT / "train.slurm"]
        for script in scripts:
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True, text=True)
        shellcheck = shutil.which("shellcheck")
        if shellcheck:
            result = subprocess.run([shellcheck, *map(str, scripts)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shell_argument_guards_and_slurm_wrapper_without_slurm(self):
        environment = {**os.environ, "STORAGE": str(new_directory()), "DRY_RUN": "1",
                       "CUDA_VISIBLE_DEVICES": "0,1,2,3", "SLURM_SUBMIT_DIR": str(ROOT),
                       "SLURM_NNODES": "1", "NOISE_LEVEL": "1.0", "EPOCHS": "80"}
        for name in ("setup.sh", "train.sh", "train.slurm"):
            result = subprocess.run(["bash", str(ROOT / name), "unexpected"], env=environment,
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0, name)
        accepted = subprocess.run(["bash", str(ROOT / "train.slurm")], env=environment,
                                  cwd=new_directory(), text=True, capture_output=True)
        self.assertEqual(accepted.returncode, 0, accepted.stdout + accepted.stderr)
        self.assertIn("GPU_COUNT=4", accepted.stdout)
        rejected = subprocess.run(["bash", str(ROOT / "train.slurm")],
                                  env={**environment, "SLURM_NNODES": "2"}, text=True, capture_output=True)
        self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
