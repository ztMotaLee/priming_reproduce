"""Small CPU regressions for DataLoader's filesystem-backed AF_UNIX sockets.

Use the installed training interpreter. No weights, dataset, CUDA, or scheduler
are needed. S1_TEST_TMPDIR can keep all test files in a chosen temporary folder.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CACHE_KEYS = (
    "TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE",
    "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_COMPILATION_CACHE",
)


class S1TmpdirTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="s1-", dir=os.environ.get("S1_TEST_TMPDIR"))
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.storage = self.folder / "storage"
        self.storage.mkdir()
        self.short = self.folder / "short"
        self.short.mkdir()
        self.long = self.folder / ("long-" + "x" * 120)
        self.long.mkdir()
        self.environment = {
            **os.environ, "STORAGE": str(self.storage), "DRY_RUN": "1",
            "NOISE_LEVEL": "1.0", "EPOCHS": "80", "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "S1_TMPDIR": str(self.short), "PYTHONDONTWRITEBYTECODE": "1",
            "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        for key in ("OUTPUT_DIR", "SMOKE_STEPS", "SLURM_JOB_ID", "SLURM_SUBMIT_DIR"):
            self.environment.pop(key, None)

    def launch(self, **overrides):
        return subprocess.run(
            ["bash", str(ROOT / "train.sh")], cwd=self.folder,
            env={**self.environment, **overrides}, capture_output=True, text=True, timeout=30)

    def legacy_environment(self):
        lines = [f"export TMPDIR={shlex.quote(str(self.long))}"]
        lines.extend(f"export {name}=legacy-cache" for name in CACHE_KEYS)
        (self.storage / "s1_env.sh").write_text("\n".join(lines) + "\n")

    def binary(self, name, content):
        target = self.storage / "env/bin" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        target.chmod(0o755)
        return target

    def test_long_tmpdir_reproduces_multiprocessing_af_unix_failure(self):
        # A fresh interpreter avoids tempfile/multiprocessing's cached directory.
        child = """
from multiprocessing.connection import Listener
try:
    with Listener(family='AF_UNIX'):
        raise SystemExit('expected the overlong AF_UNIX address to fail')
except OSError as error:
    assert 'AF_UNIX path too long' in str(error), str(error)
    print(str(error))
"""
        result = subprocess.run(
            [sys.executable, "-c", child], env={**self.environment, "TMPDIR": str(self.long)},
            text=True, capture_output=True, timeout=15)
        self.assertGreater(len(os.fsencode(self.long)), 108)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("AF_UNIX path too long", result.stdout)

    def test_dry_run_ignores_legacy_tmpdir_and_checks_a_real_socket(self):
        self.legacy_environment()
        before = sorted(self.short.iterdir())
        result = self.launch(TMPDIR=str(self.long))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"TMPDIR={self.short}\n", result.stdout)
        self.assertIn("COMMAND:", result.stdout)
        self.assertEqual(sorted(self.short.iterdir()), before, "probe must remove its socket")
        self.assertFalse((self.storage / "configs").exists())
        self.assertFalse((self.storage / "outputs").exists())

    def test_default_is_tmp_even_with_a_legacy_environment(self):
        # Check shell selection without creating test files outside S1_TEST_TMPDIR.
        # Other tests exercise the real bind and DataLoader in a short test folder.
        self.legacy_environment()
        self.binary("python", '#!/usr/bin/env bash\nprintf "PROBE_TMPDIR=%s\\n" "$TMPDIR"\n')
        environment = dict(self.environment)
        environment.pop("S1_TMPDIR")
        result = subprocess.run(
            ["bash", str(ROOT / "train.sh")], env=environment, cwd=self.folder,
            text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PROBE_TMPDIR=/tmp\n", result.stdout)
        self.assertIn("TMPDIR=/tmp\n", result.stdout)

    def test_bad_override_fails_before_launch_with_a_clear_remedy(self):
        self.legacy_environment()
        for directory in (self.long, self.folder / "missing"):
            with self.subTest(directory=directory):
                result = self.launch(S1_TMPDIR=str(directory))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("AF_UNIX", result.stderr)
                self.assertIn("S1_TMPDIR", result.stderr)
                self.assertIn("short, writable directory", result.stderr)
                self.assertNotIn("COMMAND:", result.stdout)
                self.assertFalse((self.storage / "configs").exists())
                self.assertFalse((self.storage / "outputs").exists())

    def test_relative_override_remains_absolute_after_changing_directory(self):
        result = self.launch(S1_TMPDIR="short")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"TMPDIR={self.short}\n", result.stdout)

    def test_real_two_worker_cpu_loader_receives_the_launchers_tmpdir(self):
        self.legacy_environment()
        # Skip only the GPU environment check. Execute the actual inline socket
        # probe and runtime-config generation with the production interpreter.
        self.binary("python", (
            '#!/usr/bin/env bash\nset -eu\n'
            '[[ "${1:-}" != scripts/check_environment.py ]] || exit 0\n'
            f'exec {shlex.quote(sys.executable)} "$@"\n'))
        self.binary("torchrun", f"#!{sys.executable}\n" + """
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)
assert all(name not in os.environ for name in json.loads(os.environ['TEST_CACHE_KEYS']))
sys.path.insert(0, os.getcwd())
import pretrain

def cpu_train(*args):
    assert os.environ['TMPDIR'] == os.environ['TEST_EXPECTED_TMPDIR'], os.environ['TMPDIR']
    # Exercise the actual main/cache setup, even with a long compiler-cache root.
    assert os.environ['TORCHINDUCTOR_CACHE_DIR'].startswith(os.environ['TEST_CACHE_BASE'])
    assert torch.multiprocessing.get_sharing_strategy() == 'file_descriptor'
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2, num_workers=2,
                        timeout=10, multiprocessing_context='fork')
    batches = [batch[0].tolist() for batch in loader]
    assert batches == [[0, 1], [2, 3], [4, 5], [6, 7]], batches
    print('CPU_DATALOADER=' + json.dumps(dict(tmpdir=os.environ['TMPDIR'], batches=batches,
                                            workers=2, sharing='file_descriptor')))
    return 0

pretrain.train = cpu_train
os.environ['PRIMING_LOCAL_TMP'] = os.environ['TEST_CACHE_BASE']
script = next(i for i, argument in enumerate(sys.argv) if argument.endswith('/pretrain.py'))
sys.argv = sys.argv[script:]
raise SystemExit(pretrain.main())
""")
        result = self.launch(
            DRY_RUN="0", TMPDIR=str(self.long), TEST_CACHE_KEYS=json.dumps(CACHE_KEYS),
            TEST_EXPECTED_TMPDIR=str(self.short), TEST_CACHE_BASE=str(self.long))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith("CPU_DATALOADER="))
        receipt = json.loads(line.split("=", 1)[1])
        self.assertEqual(receipt["tmpdir"], str(self.short))
        self.assertEqual(receipt["workers"], 2)
        self.assertEqual(receipt["batches"], [[0, 1], [2, 3], [4, 5], [6, 7]])
        self.assertTrue((self.storage / "configs/b80_perceptual_s100_4gpu.yaml").is_file())
        self.assertFalse((self.storage / "outputs").exists(), "no real training was launched")

    def test_setup_does_not_save_its_install_tmpdir(self):
        # Environment/download checks are stubs; this tests the actual setup
        # environment writer without installing or fetching anything.
        self.binary("python", '#!/usr/bin/env bash\nexit 0\n')
        result = subprocess.run(
            ["bash", str(ROOT / "setup.sh")], cwd=self.folder,
            env={**self.environment, "S1_SKIP_DATASET": "1"},
            capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        generated = (self.storage / "s1_env.sh").read_text()
        self.assertNotIn("export TMPDIR=", generated)
        self.assertIn("export STORAGE=", generated)
        self.assertTrue((self.storage / "tmp").is_dir())

    def test_non_ablation_main_keeps_the_official_cache_tmpdir(self):
        import pretrain

        def configure(*args, **kwargs):
            os.environ["TMPDIR"] = str(self.long)
            tempfile.tempdir = str(self.long)
            return None, None

        def train(*args):
            self.assertEqual(os.environ["TMPDIR"], str(self.long))
            self.assertEqual(tempfile.tempdir, str(self.long))
            return 0

        arguments = ["pretrain.py", "--config", str(ROOT / "configs/pretrain/b160_perceptual.yaml"),
                     "--output", str(self.folder / "not-created")]
        with patch.dict(os.environ, TMPDIR=str(self.short)), patch.object(tempfile, "tempdir", None), \
                patch.object(sys, "argv", arguments), \
                patch.object(pretrain, "configure_environment", side_effect=configure), \
                patch.object(pretrain, "train", side_effect=train):
            self.assertEqual(pretrain.main(), 0)


if __name__ == "__main__":
    unittest.main()
