"""Exercise the CC12M-only public boundary without downloads or training."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from t2i import prepare


CODE = Path(__file__).resolve().parents[1]


class Cc12mPrepareTests(unittest.TestCase):
    def main(self, argv):
        output, error = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["prepare", *argv]), \
             mock.patch("t2i.recaption.ensure_caption_receipt", return_value={"caption_source": "caption_llava"}), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            prepare.main()
        return output.getvalue(), error.getvalue()

    def test_check_only_never_discovers_curated(self):
        with tempfile.TemporaryDirectory(prefix="cc12m-check-") as directory:
            root = Path(directory)
            shard = root / "part-00000.tar"
            with tarfile.open(shard, "w") as archive:
                member = tarfile.TarInfo("0.txt")
                member.size = 4
                archive.addfile(member, io.BytesIO(b"tiny"))
            for curated in (None, "", str(root / "does-not-exist")):
                with self.subTest(curated=curated):
                    argv = ["check", "--cc12m-only", "--cc12m-root", str(root)]
                    if curated is not None:
                        argv += ["--curated-root", curated]
                    with mock.patch.object(prepare, "expand_shards", wraps=prepare.expand_shards) as expand:
                        output, error = self.main(argv)
                    expand.assert_called_once_with(str(root))
                    self.assertEqual(json.loads(output)["shards"], 1)
                    self.assertEqual(bool(error), bool(curated))

    def test_check_only_requires_cc12m(self):
        with self.assertRaisesRegex(ValueError, "CC12M_ROOT must be set"):
            self.main(["check", "--cc12m-only"])

    def test_datasets_only_downloads_missing_cc12m(self):
        for existing in (False, True):
            with self.subTest(existing=existing), \
                 mock.patch.object(prepare, "prepare_cc12m") as cc12m, \
                 mock.patch.object(prepare, "prepare_existing_cc12m") as existing_cc12m, \
                 mock.patch.object(prepare, "prepare_curated", side_effect=AssertionError("curated accessed")):
                argv = ["datasets", "--download", "--cc12m-only", "--curated-root", "/absent/curated"]
                if existing:
                    argv += ["--cc12m-root", "/data/cc12m"]
                self.main(argv)
                self.assertEqual(cc12m.call_count, 0 if existing else 1)
                self.assertEqual(existing_cc12m.call_count, 1 if existing else 0)

    def test_datasets_only_still_requires_download_opt_in(self):
        with self.assertRaises(SystemExit) as caught:
            self.main(["datasets", "--cc12m-only"])
        self.assertNotEqual(caught.exception.code, 0)

    def test_paper_datasets_still_prepares_both(self):
        with mock.patch.object(prepare, "prepare_cc12m") as cc12m, \
             mock.patch.object(prepare, "prepare_curated") as curated:
            self.main(["datasets", "--download"])
        cc12m.assert_called_once()
        curated.assert_called_once()


class Cc12mShellTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cc12m-shell-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("setup.sh", "train.sh", "train.slurm"):
            shutil.copy2(CODE / name, self.root / name)
        (self.root / "t2i").mkdir()
        (self.root / "t2i" / "train.py").write_text("")
        self.env = dict(os.environ, DRY_RUN="1", NODES="16", GPUS_PER_NODE="8",
                        JAX_COORDINATOR_ADDRESS="node0:12355", AUDIT_PYTHON=sys.executable,
                        TMPDIR=str(self.root), STAGE1_AUDIT_DIR=str(self.root))
        for name in ("CC12M_ONLY", "CC12M_ROOT", "CURATED_ROOT", "SLURM_JOB_ID", "SLURM_NNODES",
                     "SLURM_SUBMIT_DIR", "STAGE1_CODE_DIR", "JAX_PROCESS_COUNT", "JAX_PROCESS_INDEX"):
            self.env.pop(name, None)

    def run_script(self, script, args=(), env=None):
        settings = dict(self.env)
        settings.update(env or {})
        return subprocess.run(["bash", str(self.root / script), *args], cwd=self.root,
                              env=settings, capture_output=True, text=True)

    def receipt(self, result):
        prefix = "Launch preflight receipt: "
        line = next(line for line in result.stderr.splitlines() if line.startswith(prefix))
        return json.loads(Path(line[len(prefix):]).read_text())

    def test_defaults_and_both_plans_show_actual_commands(self):
        for script in ("setup.sh", "train.sh", "train.slurm"):
            for plan in (None, "0", "1"):
                with self.subTest(script=script, plan=plan):
                    extra = {} if plan is None else {"CC12M_ONLY": plan}
                    result = self.run_script(script, env=extra)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("CC12M_ONLY=" + (plan or "0"), result.stdout)
                    self.assertIn("command:", result.stdout)
                    self.assertEqual("--cc12m-only" in result.stdout, plan == "1")
                    self.assertEqual("--curated-root" in result.stdout, plan != "1")

    def test_invalid_values_fail_before_python(self):
        for script in ("setup.sh", "train.sh", "train.slurm"):
            for value in ("", "2", "yes", "-1"):
                with self.subTest(script=script, value=value):
                    result = self.run_script(script, env={"CC12M_ONLY": value,
                                                         "AUDIT_PYTHON": "/must-not-run"})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("CC12M_ONLY must be 0 or 1", result.stderr)

    def test_wrappers_reject_conflicting_python_plan_switch(self):
        for script in ("train.sh", "train.slurm"):
            result = self.run_script(script, ["--cc12m-only"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Use CC12M_ONLY=1", result.stderr)

    def test_launch_dose_gate_uses_selected_plan_and_records_rejections(self):
        for only, maximum in (("0", 290000), ("1", 250000)):
            cases = (([], True), (["--steps", str(maximum)], True),
                     (["--steps", "1", "--smoke"], True),
                     (["--steps", str(maximum), "--smoke"], True),
                     (["--steps", str(maximum - 1)], False),
                     (["--steps", str(maximum + 1), "--smoke"], False),
                     (["--steps", "0", "--smoke"], False))
            for args, accepted in cases:
                with self.subTest(only=only, args=args):
                    result = self.run_script("train.slurm", args, {"CC12M_ONLY": only})
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)
                    receipt = self.receipt(result)
                    self.assertEqual(receipt["accepted"], accepted)
                    self.assertEqual(receipt["maximum_steps"], maximum)
                    self.assertEqual(receipt["production_steps"], maximum)
                    self.assertEqual(receipt["data_plan"], "cc12m-only" if only == "1" else "paper-two-stage")
                    self.assertEqual(set(receipt["values"]), {"nodes", "gpus", "microbatch", "steps",
                                                               "smoke", "checkpoint_every"})

    def test_only_ignores_set_curated_root(self):
        for script in ("setup.sh", "train.sh", "train.slurm"):
            result = self.run_script(script, env={"CC12M_ONLY": "1", "CURATED_ROOT": "/absent/curated"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ignoring CURATED_ROOT", result.stdout)
            self.assertNotIn("--curated-root", result.stdout)
            self.assertNotIn("/absent/curated", result.stdout)

    def test_download_plan_omits_curated_and_skips_provided_cc12m(self):
        for existing in (False, True):
            extra = {"CC12M_ONLY": "1", "DOWNLOAD_DATA": "1"}
            if existing:
                extra["CC12M_ROOT"] = "/A/B/C"
            result = self.run_script("setup.sh", env=extra)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("curated raw downloads 0 GB", result.stdout)
            self.assertIn("CC12M URL metadata 3.58 GB", result.stdout)
            self.assertEqual("--cc12m-root /A/B/C" in result.stdout.split("command:")[1], existing)
        full = self.run_script("setup.sh", env={"CC12M_ONLY": "0", "DOWNLOAD_DATA": "1"})
        self.assertIn("curated raw downloads 231.45 GB", full.stdout)

    def fake_setup_environment(self):
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        hostname = fake_bin / "hostname"
        hostname.write_text("#!/usr/bin/env bash\nprintf '%s\\n' compute-fixture\n")
        hostname.chmod(0o755)
        venv = self.root / "venv"
        (venv / "bin").mkdir(parents=True)
        python = venv / "bin" / "python"
        python.write_text("#!/usr/bin/env bash\nprintf '%q ' \"$@\" >> \"$SETUP_CALL_LOG\"\nprintf '\\n' >> \"$SETUP_CALL_LOG\"\n"
                          "if [[ ${3:-} == resolve ]]; then printf '%s\\n' /A/B/C; fi\n")
        python.chmod(0o755)
        assets = self.root / "data" / "assets"
        for group in ("t5", "perceptual"):
            (assets / group).mkdir(parents=True)
            (assets / group / "assets.sha256.json").write_text("{}")
        return {"DRY_RUN": "0", "VENV_DIR": str(venv), "PYTHON": str(python),
                "PATH": str(fake_bin) + os.pathsep + self.env["PATH"], "DATA_DIR": str(self.root / "data"),
                "SETUP_CALL_LOG": str(self.root / "calls.log"), "CC12M_ROOT": "/A/B/C",
                "DOWNLOAD_DATA": "1"}

    def test_setup_writes_plan_and_explicit_environment_wins(self):
        extra = self.fake_setup_environment()
        for plan in ("1", "0"):
            result = self.run_script("setup.sh", env=dict(extra, CC12M_ONLY=plan))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("if [[ -z ${CC12M_ONLY+x} ]]; then CC12M_ONLY=" + plan + "; fi",
                          (self.root / ".stage1.env").read_text())
            saved = self.run_script("train.sh")
            self.assertEqual(saved.returncode, 0, saved.stderr)
            self.assertIn("CC12M_ONLY=" + plan, saved.stdout)
            opposite = "0" if plan == "1" else "1"
            for script in ("train.sh", "train.slurm"):
                override = self.run_script(script, env={"CC12M_ONLY": opposite})
                self.assertEqual(override.returncode, 0, override.stderr)
                self.assertIn("CC12M_ONLY=" + opposite, override.stdout)
        lines = (self.root / "calls.log").read_text().splitlines()
        only = next(line for line in lines if "t2i.prepare datasets" in line)
        self.assertIn("--cc12m-only", only)
        self.assertIn("--cc12m-root /A/B/C", only)
        self.assertNotIn("--curated-root", only)


    def test_stage_b_setup_after_cc12m_only_downloads_curated_once(self):
        extra = self.fake_setup_environment()
        first = self.run_script("setup.sh", env=dict(extra, CC12M_ONLY="1"))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertNotIn("CURATED_ROOT=", (self.root / ".stage1.env").read_text())
        # New shell: paths come only from .stage1.env, as in the README.
        shell = {k: v for k, v in extra.items() if k not in ("CC12M_ROOT", "DATA_DIR")}
        shell.update(DOWNLOAD_DATA="1")
        settings = dict(self.env)
        settings.update(shell)
        result = subprocess.run(["bash", "-c", "source .stage1.env; export CC12M_ONLY=0 "
                                 "OUTPUT_DIR=$DATA_DIR/run-b; bash setup.sh"],
                                cwd=self.root, env=settings, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        prepared = [line for line in (self.root / "calls.log").read_text().splitlines()
                    if "t2i.prepare datasets" in line][-1]
        self.assertIn("--cc12m-root /A/B/C", prepared)
        self.assertNotIn("--cc12m-only", prepared)
        self.assertNotIn("--curated-root", prepared)
        saved = (self.root / ".stage1.env").read_text()
        self.assertIn("CC12M_ONLY=0", saved)
        self.assertIn("CURATED_ROOT=", saved)


if __name__ == "__main__":
    unittest.main()
