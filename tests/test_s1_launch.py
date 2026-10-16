"""CPU-only launcher checks; scheduler commands are never executed."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from unittest.mock import call

from utils import s1_launch as launch


ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="s1launch-", dir=os.environ.get("S1_TEST_TMPDIR"))
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.storage = self.folder / "storage"
        self.short = self.folder / "tmp"
        self.short.mkdir()
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith("SLURM_") and key not in (
                                "OUTPUT_DIR", "SMOKE_STEPS", "S1_CONFIG_DIR", "S1_REPO_ROOT", "PREEMPT_FLAG_FILE")}
        self.environment.update(STORAGE=str(self.storage), DRY_RUN="1", NOISE_LEVEL="1.0", EPOCHS="80",
                                SLURM_NNODES="16", SLURM_GPUS_ON_NODE="8", SLURM_JOB_ID="123456",
                                SLURM_JOB_NODELIST="worker[01-16]", SLURM_SUBMIT_DIR=str(ROOT),
                                S1_REPO_ROOT=str(ROOT), S1_TMPDIR=str(self.short), TMPDIR=str(self.short),
                                CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")

    def run_wrapper(self, **overrides):
        return subprocess.run(["bash", str(ROOT / "train.slurm")], cwd=self.folder,
                              env={**self.environment, **overrides}, text=True, capture_output=True, timeout=20)

    def test_all_batch_shapes_preserve_1024(self):
        expected = ((64, 16), (64, 8), (64, 4), (64, 2), (64, 1), (32, 1), (16, 1), (8, 1))
        for world, shape in zip(launch.WORLD_SIZES, expected):
            with self.subTest(world=world):
                self.assertEqual(launch.batch_shape(world), shape)
                self.assertEqual(world * shape[0] * shape[1], 1024)
        for world in (0, 3, 24, 256):
            with self.assertRaisesRegex(ValueError, "World size"):
                launch.batch_shape(world)

    def test_single_node_entrypoint_preserves_all_supported_batch_shapes(self):
        for world in launch.WORLD_SIZES:
            environment = {**self.environment, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(world)))}
            result = subprocess.run(["bash", str(ROOT / "train.sh")], cwd=self.folder, env=environment,
                                    text=True, capture_output=True, timeout=15)
            with self.subTest(world=world):
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                micro, accumulation = launch.batch_shape(world)
                self.assertIn(f"MICRO_BATCH_SIZE={micro}\nGRADIENT_ACCUMULATION_STEPS={accumulation}", result.stdout)
                self.assertIn(f"--standalone --nproc_per_node={world}", result.stdout)
                self.assertFalse(self.storage.exists())

    def test_node_list_and_gpu_metadata_variants(self):
        self.assertEqual(launch.expand_nodelist("rack[1-2]node[01-02],spare"),
                         ["rack1node01", "rack1node02", "rack2node01", "rack2node02", "spare"])
        for name, value in (("SLURM_GPUS_ON_NODE", "8"), ("SLURM_GPUS_ON_NODE", "gpu:device:8"),
                            ("SLURM_GPUS_PER_NODE", "8(x16)"), ("SLURM_JOB_GPUS", "0-7"),
                            ("SLURM_JOB_GPUS", "0,1,2,3,4,5,6,7")):
            with self.subTest(name=name, value=value):
                environment = {key: item for key, item in self.environment.items()
                               if key not in ("SLURM_GPUS_ON_NODE", "CUDA_VISIBLE_DEVICES")}
                environment[name] = value
                topology = launch.allocation(environment)
                self.assertEqual((topology["nodes"], topology["gpus"], topology["world"]), (16, 8, 128))
                self.assertEqual(topology["master_addr"], "worker01")
                self.assertEqual(topology["master_port"], 15000 + 123456 % 45000)
        with self.assertRaisesRegex(ValueError, "same GPU count"):
            launch.slurm_gpu_count({"SLURM_GPUS_PER_NODE": "8(x8),4(x8)"}, 16)
        with self.assertRaisesRegex(ValueError, "SLURM_JOB_NODELIST"):
            launch.allocation({"SLURM_NNODES": "2", "SLURM_GPUS_ON_NODE": "8"})

    def test_visible_gpu_validation(self):
        self.assertEqual(launch.visible_gpu_count("0, 1,GPU-abcd,MIG-abcd/1/2"), 4)
        for value in ("", "-1"):
            self.assertEqual(launch.visible_gpu_count(value), 0)
        for value in ("0,0", ",0", "0,", "0,,1", "0-7"):
            with self.assertRaises(ValueError):
                launch.visible_gpu_count(value)

    def test_16_node_dry_run_prints_final_commands_without_writes(self):
        result = self.run_wrapper(S1_CONFIG_DIR="runtime-configs")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("GPU_COUNT=128\nMICRO_BATCH_SIZE=8\nGRADIENT_ACCUMULATION_STEPS=1", result.stdout)
        commands = [line for line in result.stdout.splitlines() if line.startswith("NODE_COMMAND[")]
        self.assertEqual(len(commands), 16)
        for rank, command in enumerate(commands):
            self.assertIn(f"host=worker{rank + 1:02}", command)
            self.assertIn(f"--node_rank={rank}", command)
            self.assertIn("-m torch.distributed.run --nnodes=16 --nproc_per_node=8", command)
            self.assertIn("--master_addr=worker01", command)
            self.assertIn(str(self.folder / "runtime-configs"), command)
        self.assertFalse(self.storage.exists(), "DRY_RUN must not create STORAGE, configs, outputs, or flags")
        self.assertEqual(list(self.short.iterdir()), [])

    def test_resume_world_and_config_micro_are_checked_before_launch(self):
        output = self.folder / "run"
        (output / "checkpoints").mkdir(parents=True)
        latest = output / "checkpoints/latest.pth"
        latest.touch()
        config = output / "config.json"
        config.write_text(json.dumps({"train": {"micro_batch_size": 8, "gradient_accumulation_steps": 1}}))
        result = self.run_wrapper(OUTPUT_DIR="run")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"--resume {latest}", result.stdout)
        config.write_text(json.dumps({"train": {"micro_batch_size": 16, "gradient_accumulation_steps": 1}}))
        result = self.run_wrapper(OUTPUT_DIR="run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint used 64, requested 128", result.stderr)
        self.assertNotIn("NODE_COMMAND", result.stdout)

    def test_long_socket_directory_is_rejected_before_launch(self):
        directory = self.folder / ("x" * 120)
        directory.mkdir()
        result = self.run_wrapper(S1_TMPDIR=str(directory))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AF_UNIX", result.stderr)
        self.assertIn("S1_TMPDIR", result.stderr)
        self.assertFalse(self.storage.exists())

    def test_real_launch_orchestration_is_mocked_and_config_written_once(self):
        environment = {**self.environment, "DRY_RUN": "0", "S1_CONFIG_DIR": str(self.folder / "configs")}
        calls = []

        def run(command, **options):
            calls.append(command)
            if "--node-launch" in command:
                output = self.storage / "outputs/pretrain/b80_perceptual_s100_128gpu"
                checkpoint = output / "checkpoints/step_0000030.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
                (output / "safe_stop.json").write_text(json.dumps(dict(
                    state="SAFE_STOPPED", launch_id=os.environ["S1_LAUNCH_ID"], completed=30,
                    checkpoint=str(checkpoint))))
                return SimpleNamespace(returncode=1)
            return SimpleNamespace(returncode=0)

        with patch.dict(os.environ, environment, clear=True), patch.object(launch.subprocess, "run", side_effect=run), \
                patch.object(launch, "write_config") as write, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.run_slurm(), 75)
            write.assert_called_once()
            self.assertEqual(write.call_args.args[2:], (8, 1, 80))
            self.assertEqual(write.call_args.args[1].parent, self.folder / "configs")
        scheduler_calls = [command for command in calls if command[0] == "srun"]
        self.assertEqual(len(scheduler_calls), 2)
        self.assertIn("--preflight", scheduler_calls[0])
        self.assertIn("--node-launch", scheduler_calls[1])
        for command in scheduler_calls:
            self.assertIn("--ntasks-per-node=1", command)
            self.assertIn("--ntasks=16", command)
            self.assertIn("--kill-on-bad-exit=1", command)

    def test_node_preflight_rejects_different_gpu_visibility_without_cuda(self):
        with patch.dict(os.environ, TMPDIR=str(self.short), CUDA_VISIBLE_DEVICES="0,1"):
            with self.assertRaisesRegex(ValueError, "expected 8 GPUs"):
                launch.preflight(8)

    def test_node_preflight_computes_on_every_gpu_with_mocked_cuda(self):
        tensor = SimpleNamespace(item=lambda: 1.0)
        output = io.StringIO()
        with patch.dict(os.environ, TMPDIR=str(self.short), CUDA_VISIBLE_DEVICES="0,1,2,3"), \
                patch("torch.cuda.device_count", return_value=4), \
                patch("torch.cuda.set_device") as bind, \
                patch("torch.cuda.get_device_name", side_effect=lambda index: f"Example GPU {index}"), \
                patch("torch.ones", return_value=tensor) as ones, contextlib.redirect_stdout(output):
            launch.preflight(4)
        self.assertEqual(bind.call_args_list, [call(index) for index in range(4)])
        self.assertEqual(ones.call_args_list, [call(1, device=f"cuda:{index}") for index in range(4)])
        self.assertEqual(output.getvalue().count("PREFLIGHT_GPU "), 4)
        self.assertIn("name='Example GPU 3' probe_value=1.0", output.getvalue())
        self.assertIn("PREFLIGHT_OK", output.getvalue())

    def test_node_preflight_rejects_a_failed_gpu_probe_with_mocked_cuda(self):
        output = io.StringIO()
        with patch.dict(os.environ, TMPDIR=str(self.short), CUDA_VISIBLE_DEVICES="0,1"), \
                patch("torch.cuda.device_count", return_value=2), patch("torch.cuda.set_device"), \
                patch("torch.cuda.get_device_name", return_value="Example GPU"), \
                patch("torch.ones", side_effect=[SimpleNamespace(item=lambda: 1.0), SimpleNamespace(item=lambda: 0.0)]), \
                contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(ValueError, "GPU 1 probe returned 0.0"):
                launch.preflight(2)
        self.assertIn("gpus=2 expected_gpus=2", output.getvalue())
        self.assertIn("device=1 name='Example GPU' probe_value=0.0", output.getvalue())
        self.assertNotIn("PREFLIGHT_OK", output.getvalue())

    def test_config_generation_changes_only_batch_shape_and_epoch(self):
        from utils.config import load_yaml

        source = ROOT / "configs/pretrain/b80_perceptual_s100.yaml"
        original = load_yaml(source)
        for world in launch.WORLD_SIZES:
            target = self.folder / f"configs/{world}.yaml"
            micro, accumulation = launch.batch_shape(world)
            launch.write_config(source, target, micro, accumulation, 80)
            actual = load_yaml(target)
            expected = load_yaml(source)
            expected["train"].update(micro_batch_size=micro, gradient_accumulation_steps=accumulation)
            self.assertEqual(actual, expected)
            self.assertEqual(original["optimizer"], actual["optimizer"])

    def test_wrapper_cleans_legacy_caches_before_its_first_python(self):
        binary = self.storage / "env/bin/python"
        binary.parent.mkdir(parents=True)
        checks = "\n".join(f'[[ ! ${{{name}+present}} ]] || exit 91' for name in launch.CACHE_VARIABLES)
        binary.write_text(f"#!/usr/bin/env bash\nset -eu\n{checks}\nexec {shlex.quote(sys.executable)} \"$@\"\n")
        binary.chmod(0o755)
        (self.storage / "s1_env.sh").write_text("\n".join(
            f"export {name}=legacy" for name in launch.CACHE_VARIABLES) + "\nexport TMPDIR=/does/not/exist\n")
        result = self.run_wrapper(**{name: "inherited" for name in launch.CACHE_VARIABLES})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"TMPDIR={self.short}", result.stdout)

    def test_batch_usr1_touches_shared_flag_and_returns_child_status(self):
        binary = self.storage / "env/bin/python"
        binary.parent.mkdir(parents=True)
        binary.write_text(
            '#!/usr/bin/env bash\nset -eu\necho READY\n'
            'for ((i=0;i<200;i++)); do\n'
            '[[ ! -f "$PREEMPT_FLAG_FILE" ]] || exit 75\nsleep 0.01\ndone\nexit 1\n')
        binary.chmod(0o755)
        (self.storage / "s1_env.sh").touch()
        environment = {**self.environment, "DRY_RUN": "0", "PREEMPT_FLAG_FILE": "shared/stop"}
        child = subprocess.Popen(["bash", str(ROOT / "train.slurm")], cwd=self.folder, env=environment,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertEqual(child.stdout.readline().strip(), "READY")
            child.send_signal(signal.SIGUSR1)
            _, stderr = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 75, stderr)
            self.assertTrue((self.folder / "shared/stop").is_file())
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()


if __name__ == "__main__":
    unittest.main()
