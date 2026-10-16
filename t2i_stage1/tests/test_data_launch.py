"""Small CPU-only data/launcher regressions; no model or network downloads."""
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from t2i.data import WebDatasetStream, expand_shards, shard_identity
from t2i.prepare import PairWriter, atomic_json


CODE = Path(__file__).resolve().parents[1]


def write_shard(path, keys):
    with tarfile.open(path, "w") as archive:
        for key in keys:
            image = io.BytesIO()
            Image.new("RGB", (11, 9), (int(key) % 256, 20, 30)).save(image, format="PNG")
            for suffix, value in (("png", image.getvalue()), ("txt", f"caption {key}".encode()),
                                  ("json", b'{"metadata": true}')):
                member = tarfile.TarInfo(f"{key}.{suffix}")
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))


class DataTests(unittest.TestCase):
    def test_forward_normalization_and_seek_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_shard(root / "00000.tar", range(9))
            stream = WebDatasetStream(root, image_size=8)
            first = stream.next_batch(3)
            self.assertEqual(first["images"].shape, (3, 8, 8, 3))
            self.assertEqual(first["images"].dtype, np.float32)
            self.assertTrue(np.all(np.abs(first["images"]) <= 1))
            state = json.loads(json.dumps(stream.state_dict()))
            self.assertGreater(state["streams"][0]["byte_offset"], 0)
            self.assertEqual(len(stream.streams[0]._tar.members), 0)
            expected = stream.next_batch(12)
            resumed = WebDatasetStream(root, image_size=8, state=state)
            actual = resumed.next_batch(12)
            self.assertEqual(expected["keys"], actual["keys"])
            self.assertEqual(expected["captions"], actual["captions"])
            np.testing.assert_array_equal(expected["images"], actual["images"])
            stream.load_state_dict(state)
            self.assertEqual(stream.next_batch(12)["keys"], expected["keys"])
            stream.close(); resumed.close()

    def test_disjoint_shards_and_few_shards(self):
        for shards, hosts in ((4, 2), (1, 4), (2, 4)):
            with self.subTest(shards=shards, hosts=hosts), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for index in range(shards):
                    write_shard(root / f"{index:05d}.tar", range(index * 8, (index + 1) * 8))
                streams = [WebDatasetStream(root, process_index=rank, process_count=hosts, image_size=4)
                           for rank in range(hosts)]
                keys = [set(s.next_batch(shards * 8 // hosts)["keys"]) for s in streams]
                self.assertEqual(sum(len(v) for v in keys), len(set.union(*keys)))
                self.assertEqual(len(set.union(*keys)), shards * 8)
                for stream in streams: stream.close()

    def test_mixture_restart_and_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                source = root / f"source{index}"
                source.mkdir()
                write_shard(source / "part.tar", range(index * 10, index * 10 + 6))
            spec = {"sources": [{"root": f"source{i}", "weight": w} for i, w in enumerate((.06,.016,.04))]}
            (root / "sources.json").write_text(json.dumps(spec))
            stream = WebDatasetStream(root, image_size=4)
            stream.next_batch(7)
            state = stream.state_dict()
            expected = stream.next_batch(20)["keys"]
            resumed = WebDatasetStream(root, image_size=4, state=state)
            self.assertEqual(resumed.next_batch(20)["keys"], expected)
            with self.assertRaisesRegex(ValueError, "topology"):
                WebDatasetStream(root, process_count=2, image_size=4, state=state)
            spec["sources"][0]["weight"] = 1
            (root / "sources.json").write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "weights"):
                WebDatasetStream(root, image_size=4, state=state)
            stream.close(); resumed.close()

    def test_changed_shard_and_device_independent_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "00000.tar"
            write_shard(path, range(4))
            stream = WebDatasetStream(root, image_size=4)
            state = stream.state_dict()
            stat = path.stat()
            values = dict(st_ino=stat.st_ino, st_uid=stat.st_uid, st_size=stat.st_size,
                          st_mtime_ns=stat.st_mtime_ns, st_dev=54)
            with mock.patch("t2i.data.Path.stat", return_value=type("Stat", (), values)()):
                before = shard_identity(path)
            values["st_dev"] = 987
            with mock.patch("t2i.data.Path.stat", return_value=type("Stat", (), values)()):
                self.assertEqual(shard_identity(path), before)
            self.assertNotIn("device", before)
            with path.open("ab") as handle: handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "shards"):
                WebDatasetStream(root, image_size=4, state=state)
            stream.close()

    def test_brace_expansion_and_bad_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2): write_shard(root / f"{index:05d}.tar", range(2))
            self.assertEqual(len(expand_shards(str(root / "{00000..00001}.tar"))), 2)
            bad = root / "bad.tar"
            with tarfile.open(bad, "w") as archive:
                entry = tarfile.TarInfo("0.txt"); entry.size = 5
                archive.addfile(entry, io.BytesIO(b"hello"))
            stream = WebDatasetStream(bad, image_size=4)
            with self.assertRaisesRegex(ValueError, "Unpaired"):
                stream.next_batch(1)
            stream.close()

    def test_preparation_create_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "receipt.json", {"value": 1}, create_once=True)
            atomic_json(root / "receipt.json", {"value": 1}, create_once=True)
            with self.assertRaises(FileExistsError):
                atomic_json(root / "receipt.json", {"value": 2}, create_once=True)
            writer = PairWriter(root / "pairs", shard_size=1)
            image = io.BytesIO(); Image.new("RGB", (4,4)).save(image, format="JPEG")
            writer.add("0", image.getvalue(), "example")
            writer.finish({"test": True})
            self.assertEqual(json.loads((root / "pairs/preparation.json").read_text())["samples"], 1)
            duplicate = PairWriter(root / "pairs", shard_size=1)
            with self.assertRaises(FileExistsError): duplicate.add("0", image.getvalue(), "example")


class LaunchTests(unittest.TestCase):
    def run_script(self, script, args=(), env=None):
        run_env = dict(os.environ, DRY_RUN="1", NODES="16", GPUS_PER_NODE="8",
                       JAX_COORDINATOR_ADDRESS="node0:12355")
        for key in ("SLURM_NNODES", "SLURM_SUBMIT_DIR", "STAGE1_CODE_DIR"):
            run_env.pop(key, None)
        run_env.update(env or {})
        return subprocess.run(["bash", str(CODE / script), *args], cwd=CODE,
                              env=run_env, capture_output=True, text=True)

    def test_shell_syntax(self):
        for script in ("setup.sh", "train.sh", "train.slurm"):
            result = subprocess.run(["bash", "-n", str(CODE / script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_multinode_dry_run(self):
        result = self.run_script("train.slurm", ["--microbatch", "2", "--steps", "30", "--smoke"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nodes=16 GPUs_per_node=8 processes_per_node=1", result.stdout)
        self.assertIn("effective_per_GPU=8", result.stdout)
        self.assertIn("--export=ALL", result.stdout)
        self.assertIn("--ntasks=16", result.stdout)
        self.assertIn("--gpus-per-task=8", result.stdout)

    def test_argument_guard_rejects(self):
        for args, env in ((["--microbatch", "0"], {}), (["--noise", "bad"], {}),
                          (["--checkpoint-every", "0"], {}),
                          (["--steps"], {}), (["--unknown"], {}), ([], {"NODES": "3"}),
                          ([], {"GPUS_PER_NODE": "bad"})):
            with self.subTest(args=args, env=env):
                self.assertNotEqual(self.run_script("train.slurm", args, env).returncode, 0)

    def test_failed_numeric_gate_has_atomic_receipt(self):
        result = self.run_script("train.slurm", ["--microbatch", "0", "--steps", "30", "--smoke"])
        self.assertNotEqual(result.returncode, 0)
        line = next(line for line in result.stderr.splitlines() if line.startswith("Launch preflight receipt: "))
        receipt = json.loads(Path(line.split(": ", 1)[1]).read_text())
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["values"], {"nodes": 16, "gpus": 8, "microbatch": 0,
                                              "steps": 30, "smoke": 1, "checkpoint_every": 5000})
        self.assertEqual(receipt["global_batch"], 1024)
        self.assertEqual(receipt["maximum_steps"], 290000)

    def test_public_entrypoints_dry_run(self):
        setup = self.run_script("setup.sh")
        self.assertEqual(setup.returncode, 0, setup.stderr)
        self.assertIn("JAX=0.4.31", setup.stdout)
        train = self.run_script("train.sh", env={"JAX_PROCESS_COUNT":"16", "JAX_PROCESS_INDEX":"7",
                                                  "CC12M_ROOT":"/data/cc12m/*.tar", "RESUME":"/runs/step_20"})
        self.assertEqual(train.returncode, 0, train.stderr)
        self.assertIn("process_count=16 process_index=7 global_batch=1024", train.stdout)
        self.assertIn("-m t2i.train", train.stdout)
        self.assertIn("--resume /runs/step_20", train.stdout)


if __name__ == "__main__":
    unittest.main()
