"""Small CPU checkpoint tests; no GPU, dataset, or production checkpoints."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from utils.ema import EMA
from utils.optim import create_optimizer
from utils import pretrain_checkpoint as checkpoint
from utils.random import seed_all, snapshot_rng
from utils.state import exact_optimizer, optimizer_layout, validate_optimizer_shard


class TinyModel(nn.Module):
    def __init__(self, matrices=7):
        super().__init__()
        self.matrices = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for _ in range(matrices)])
        self.bias = nn.Parameter(torch.randn(4))


def config():
    return dict(ablation=dict(name="stage1_noise_level", noise_level=1.),
                model=dict(model="tiny"), data={}, loss={},
                train=dict(seed=19, micro_batch_size=2, gradient_accumulation_steps=1,
                           steps_per_epoch=3, milestone_interval=20, keep_n_ckpts=3),
                optimizer=dict(optimizer="muon", lr=.001, muon_momentum=.95,
                               muon_weight_decay=0., beta1=.9, beta2=.95))


def bundle():
    recipe = config()
    model = TinyModel()
    optimizer = create_optimizer(model, recipe["optimizer"])
    ema = EMA(model, batch_size=8)
    loader = torch.Generator().manual_seed(101)
    return recipe, model, optimizer, ema, loader


def _four_rank_worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/rendezvous", rank=rank,
                            world_size=4, timeout=timedelta(seconds=45))
    try:
        seed_all(17)
        recipe, model, optimizer, ema, loader = bundle()
        seed_all(41 + rank)
        loader.manual_seed(101 + rank)
        original_backend = dist.get_backend

        def backend(group=None):
            # Exercise the real CPU subgroup branch without requiring NCCL/GPU.
            return "nccl" if group is None else original_backend(group)

        def save(step):
            with patch.object(dist, "get_backend", side_effect=backend):
                checkpoint.save_pretraining(folder, step, model, optimizer, ema, recipe,
                                             {"fixture": "four-rank"}, family="b", elapsed=3.5,
                                             loader_generator=loader)

        def update(step):
            random.random()
            np.random.rand()
            torch.rand(2)
            torch.rand(2, generator=loader)
            for index, parameter in enumerate(model.parameters()):
                parameter.grad = torch.full_like(parameter, .01 * (index + 1) * step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.step(model)

        save(0)
        group = checkpoint._cpu_checkpoint_group
        assert original_backend(group) == "gloo"
        update(1)
        save(1)
        assert checkpoint._cpu_checkpoint_group is group
        path = Path(folder) / "checkpoints/latest.pth"
        saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        assert len(saved["rank_states"]) == saved["world_size"] == 4
        local = saved["rank_states"][rank]
        assert local["shard"] == dict(muon_tensors=len(range(rank, 7, 4)), adam_tensors=1)
        update(2)
        uninterrupted = copy.deepcopy((model.state_dict(), optimizer.state_dict(), ema.state_dict(),
                                       snapshot_rng(), loader.get_state()))
        seed_all(999)
        loader.manual_seed(999)
        restored = checkpoint.resume_pretraining(path, model, optimizer, ema, recipe,
                                                 {"fixture": "four-rank"}, family="b",
                                                 loader_generator=loader)
        assert restored == (1, 3.5)
        update(2)
        exact_optimizer((model.state_dict(), optimizer.state_dict(), ema.state_dict(),
                         snapshot_rng(), loader.get_state()), uninterrupted)
        (Path(folder) / f"rank-{rank}.json").write_text(json.dumps(dict(rank=rank, passed=True)))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Requires the Gloo CPU backend")
class S1MultinodeCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = TemporaryDirectory(prefix="s1-checkpoint-", dir=os.environ.get("TMPDIR", "/tmp"))
        cls.folder = Path(cls.temporary.name)
        # Fork shares imported CPU libraries; each rank uses one thread and tiny tensors.
        mp.start_processes(_four_rank_worker, args=(str(cls.folder),), nprocs=4,
                           join=True, start_method="fork")
        cls.source = torch.load(cls.folder / "checkpoints/latest.pth", map_location="cpu",
                                mmap=True, weights_only=False)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def resume(self, state, world=4):
        path = self.folder / "candidate.pth"
        torch.save(state, path)
        recipe, model, optimizer, ema, loader = bundle()
        with patch.object(dist, "get_rank", return_value=0), patch.object(dist, "get_world_size", return_value=world):
            result = checkpoint.resume_pretraining(path, model, optimizer, ema, recipe,
                                                   {"fixture": "four-rank"}, family="b",
                                                   loader_generator=loader)
        return result, optimizer

    def test_four_real_gloo_ranks_restore_the_next_update_bitwise(self):
        for rank in range(4):
            self.assertEqual(json.loads((self.folder / f"rank-{rank}.json").read_text()),
                             dict(rank=rank, passed=True))
        _, model, optimizer, _, _ = bundle()
        self.assertEqual(self.source["optimizer_layout"], optimizer_layout(model, optimizer))
        self.assertEqual(self.source["contract"]["world_size"], 4)

    def test_changed_world_size_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "original world size"):
            self.resume(self.source, world=8)
        invalid = copy.deepcopy(self.source)
        invalid["rank_states"].pop()
        with self.assertRaisesRegex(ValueError, "rank topology"):
            self.resume(invalid)

    def test_integer_counter_dtypes_are_preserved(self):
        for dtype in (torch.int32, torch.int64):
            state = copy.deepcopy(self.source)
            state["step"] = torch.tensor(1, dtype=dtype)
            auxiliary = next(group for group in state["rank_states"][0]["optimizer"]["param_groups"]
                             if not group["use_muon"])
            identifier = auxiliary["params"][0]
            state["rank_states"][0]["optimizer"]["state"][identifier]["step"] = torch.tensor(1, dtype=dtype)
            (step, _), optimizer = self.resume(state)
            self.assertEqual(step, 1)
            restored = next(group for group in optimizer.param_groups if not group["use_muon"])
            self.assertEqual(optimizer.state[restored["params"][0]]["step"].dtype, dtype)

    def test_fractional_boolean_and_nonscalar_counters_are_rejected(self):
        for name in ("step", "current_step", "samples_seen"):
            for value in (1.5, True, torch.tensor(1.), torch.tensor(True), torch.tensor([1])):
                with self.subTest(name=name, value=value):
                    state = copy.deepcopy(self.source)
                    state[name] = value
                    with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                        self.resume(state)
        state = copy.deepcopy(self.source)
        states = state["rank_states"][0]["optimizer"]["state"]
        next(value for value in states.values() if "step" in value)["step"] = 1.5
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            self.resume(state)

    def test_rank_stream_cursor_layout_and_optimizer_shards_remain_strict(self):
        mutations = (
            lambda state: state["rank_states"][0].update(rank=1),
            lambda state: state["rank_states"][0]["streams"].update(seed=999),
            lambda state: state["rank_states"][0]["next_data"].update(offset=999),
            lambda state: state["optimizer_layout"][0]["parameters"].reverse(),
            lambda state: next(value for value in state["rank_states"][0]["optimizer"]["state"].values()
                               if "momentum_buffer" in value).clear(),
        )
        for mutate in mutations:
            state = copy.deepcopy(self.source)
            mutate(state)
            with self.assertRaises(ValueError):
                self.resume(state)

    def test_128_rank_ownership_metadata_without_128_processes(self):
        matrices = [nn.Parameter(torch.ones(2, 2)) for _ in range(113)]
        auxiliary = nn.Parameter(torch.ones(3))
        counts = []
        for rank in range(128):
            state = {parameter: dict(momentum_buffer=torch.ones_like(parameter))
                     for index, parameter in enumerate(matrices) if index % 128 == rank}
            state[auxiliary] = dict(step=50, exp_avg=torch.ones_like(auxiliary),
                                    exp_avg_sq=torch.ones_like(auxiliary))
            optimizer = SimpleNamespace(param_groups=[dict(params=matrices, use_muon=True),
                                                      dict(params=[auxiliary], use_muon=False)], state=state)
            shard = validate_optimizer_shard(optimizer, 50, rank, 128)
            self.assertEqual(shard["adam_tensors"], 1)
            self.assertEqual(shard["muon_tensors"], int(rank < 113))
            counts.append(shard["muon_tensors"])
        self.assertEqual(sum(counts), 113)

    def test_official_and_cpu_transport_keep_the_existing_collective(self):
        with patch.object(dist, "is_initialized", return_value=True), \
             patch.object(dist, "get_backend", return_value="nccl"), \
             patch.object(dist, "new_group") as new_group:
            self.assertIsNone(checkpoint._checkpoint_group({}))
            new_group.assert_not_called()
        with patch.object(dist, "is_initialized", return_value=True), \
             patch.object(dist, "get_backend", return_value="gloo"), \
             patch.object(dist, "new_group") as new_group:
            self.assertIsNone(checkpoint._checkpoint_group(config()))
            new_group.assert_not_called()


if __name__ == "__main__":
    unittest.main()
