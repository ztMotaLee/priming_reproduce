import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from launcher import duration, plan, submit


class LauncherTests(unittest.TestCase):
    def environment(self, root):
        return dict(NODES="2", GPUS_PER_NODE="4", PARTITION="example", ACCOUNT="example",
                    CPUS_PER_TASK="8", MEM_PER_NODE="128G", TIME="12:00:00",
                    DATA_ROOT=str(root / "data"), OUTPUT_DIR=str(root / "run"))

    def test_plan_is_read_only_and_keeps_global_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = self.environment(root)
            for nodes, gpus in ((1, 4), (2, 4), (4, 8), (8, 8)):
                chain = plan(dict(env, NODES=str(nodes), GPUS_PER_NODE=str(gpus)))
                self.assertEqual(chain["global_caption_groups"], 64)
                self.assertEqual(len(chain["jobs"]), 30)
                self.assertIsNone(chain["jobs"][0]["dependency"])
                self.assertIn("--export=ALL", chain["jobs"][0]["command"])
                self.assertIn("--no-requeue", chain["jobs"][0]["command"])
                self.assertTrue(any(x.startswith("--chdir=") for x in chain["jobs"][0]["command"]))
            self.assertEqual(list(root.iterdir()), [])

    def test_parameter_guards(self):
        env = self.environment(Path("/example"))
        for change in ({"NODES": "3"}, {"PARTITION": ""}, {"ACCOUNT": "YOUR_ACCOUNT"},
                       {"TIME": "00:00:00"}, {"TIME": "1:90:00"},
                       {"CPUS_PER_TASK": "0"}, {"MEM_PER_NODE": "0"},
                       {"END_STEP": "30001"}, {"STEPS_PER_SEGMENT": "0"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan(dict(env, **change))
        self.assertEqual(duration("1-00:00:00"), 86400)

    def test_readout_boundaries(self):
        chain = plan(dict(self.environment(Path("/example")), STEPS_PER_SEGMENT="1750"))
        ends = {job["end_step"] for job in chain["jobs"]}
        self.assertTrue(set(chain["readout_steps"]).issubset(ends))

    def test_mock_submission_dependencies_and_failure_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            env = dict(self.environment(root), END_STEP="3000")
            chain = plan(env)
            calls = []
            def fake(command, **kwargs):
                calls.append((command, kwargs))
                return type("Result", (), {"stdout": str(100 + len(calls))})()
            with patch("launcher.subprocess.run", side_effect=fake), contextlib.redirect_stdout(io.StringIO()):
                submit(chain, env)
            self.assertFalse(any(x.startswith("--dependency") for x in calls[0][0]))
            self.assertIn("--dependency=afterok:101", calls[1][0])
            self.assertIn("--dependency=afterok:102", calls[2][0])
            self.assertEqual(calls[1][1]["env"]["START_STEP"], "1000")
            self.assertEqual(calls[1][1]["env"]["BIGFIELD_CODE_ROOT"], str(Path(__file__).resolve().parents[1]))
            self.assertFalse((root / "run").exists())
            with patch("launcher.subprocess.run", side_effect=RuntimeError("stop")) as run:
                with self.assertRaises(RuntimeError):
                    submit(chain, env)
                self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
