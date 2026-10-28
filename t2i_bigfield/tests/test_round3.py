"""QOS discovery and readable plans using PATH-injected scheduler fixtures."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest import mock

import test_workflow as legacy


QOS_KEYS = ("CPU_QOS", "GPU_QOS_SHORT", "GPU_QOS_LONG")
NODE_NAMES = tuple("qa_node_" + str(index) for index in range(4)) + ("qa_cpu_node",)
INVENTORY = "".join("gpu_x|" + node + "|gpu:h100:4|32|131072|1-00:00:00\n"
                    for node in NODE_NAMES[:-1]) + "cpu_x|qa_cpu_node|N/A|32|131072|2-00:00:00\n"

# Preserve the existing fake job/readback implementation. Only read-only QOS
# discovery and --test-only acceptance differ from the shared fixture.
QOS_SLURM = legacy.FAKE_SLURM.replace(
    "if tool=='sacctmgr':\n print('demo|')",
    """if tool=='sacctmgr':
 if os.environ.get('FAKE_ASSOC_UNAVAILABLE')=='1':
  print('association query unavailable',file=sys.stderr);sys.exit(1)
 if 'qos' in args:print(os.environ.get('FAKE_QOS_ROWS',''),end='')
 else:print(os.environ.get('FAKE_ASSOC_ROWS','acct_x||'),end='')""").replace(
    """ if '--test-only' in args:
  if os.environ.get('FAKE_REJECT_TEST_ONLY')=='1':
   print('job-origin submission disabled',file=sys.stderr);sys.exit(1)
  sys.exit(0)""",
    """ if '--test-only' in args:
  if os.environ.get('FAKE_TEST_UNAVAILABLE')=='1':
   print('unsupported option --test-only',file=sys.stderr);sys.exit(1)
  opts=options(args)
  qos=opts.get('qos','none')
  rejected=json.loads(os.environ.get('FAKE_QOS_REJECT','[]'))
  if qos in rejected or '*' in rejected:
   print('resource and QOS combination rejected',file=sys.stderr);sys.exit(1)
  shapes=json.loads(os.environ.get('FAKE_QOS_SHAPE_REJECT','[]'))
  if any(all(opts.get(key,'none' if key=='qos' else '')==value
             for key,value in shape.items()) for shape in shapes):
   print('resource shape and QOS combination rejected',file=sys.stderr);sys.exit(1)
  print(os.environ.get('FAKE_PROBE_STDOUT',''),end='')
  sys.exit(0)""").replace("QOS=row.get('qos','normal')", "QOS=row.get('qos','qa_default')")


def option(arguments, name):
    for index, value in enumerate(arguments):
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
    return None


class Round3Tests(unittest.TestCase):
    def setUp(self):
        # Composition reuses setup without duplicating the predecessor tests.
        self.fixture = legacy.WorkflowTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        for tool in ("sbatch", "squeue", "sacct", "sinfo", "sacctmgr", "scontrol"):
            path = self.fixture.bin / tool
            path.write_text("#!" + sys.executable + "\n" + QOS_SLURM)
        temporary = self.fixture.root / "tmp"
        temporary.mkdir()
        self.fixture.env.update(ACCOUNT="acct_x", CPU_PARTITION="cpu_x", FAKE_SINFO=INVENTORY,
                                FAKE_ASSOC_ROWS="acct_x||\n", TMPDIR=str(temporary))
        for key in QOS_KEYS:
            self.fixture.env.pop(key, None)

    def run_plan(self, *, json_output=True, extra=None, args=()):
        arguments = ("--yes", *args)
        if json_output:
            return self.fixture.run_workflow(*arguments, extra=extra)
        return subprocess.run(["bash", str(self.fixture.code / "run.sh"),
                               "--output-root", str(self.fixture.output),
                               "--data-root", str(self.fixture.data), *arguments],
                              cwd=legacy.ROOT, env=dict(self.fixture.env, **(extra or {})),
                              text=True, capture_output=True, timeout=30)

    def plan(self, **kwargs):
        result = self.run_plan(**kwargs)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def probes(self):
        return [row for row in self.fixture.calls("sbatch") if "--test-only" in row["args"]]

    def assert_no_jobs_or_run(self):
        self.assertFalse(self.fixture.output.exists())
        self.assertEqual(self.fixture.job_rows(), [])
        self.assertTrue(all("--test-only" in row["args"] for row in self.fixture.calls("sbatch")))

    def qos_site(self, qos="qa_short,qa_medium,qa_long",
                 rows="qa_long|1-00:00:00|\nqa_short|01:00:00|\nqa_medium|02:00:00|\n"):
        return {"FAKE_ASSOC_ROWS": "acct_x||" + qos + "|\n", "FAKE_QOS_ROWS": rows}

    def test_no_qos_association_uses_default_none_and_omits_qos_flag(self):
        plan = self.plan()
        self.assertEqual({plan["site"][key] for key in QOS_KEYS}, {"none"})
        self.assertTrue(all(plan["site"]["_sources"][key] == "default none (no association QOS)" for key in QOS_KEYS))
        self.assertTrue(all(not any(arg.startswith("--qos") for arg in row["command_preview"])
                            for row in plan["jobs"]))
        self.assertTrue(self.probes(), "The site default is also checked with --test-only")
        self.assertTrue(all(option(row["args"], "--qos") is None for row in self.probes()))
        self.assert_no_jobs_or_run()

    def test_multiple_qos_choose_smallest_sufficient_maxwall_per_resource_class(self):
        plan = self.plan(extra=self.qos_site())
        self.assertEqual([plan["site"][key] for key in QOS_KEYS],
                         ["qa_medium", "qa_short", "qa_long"])
        self.assertTrue(all(plan["site"]["_sources"][key] == "detected+test-only verified" for key in QOS_KEYS))
        probes = self.probes()
        cpu = next(row for row in probes if option(row["args"], "--partition") == "cpu_x")
        self.assertEqual(option(cpu["args"], "--time"), "01:05:00")
        self.assertEqual(option(cpu["args"], "--qos"), "qa_medium")
        self.assertTrue(any(option(row["args"], "--qos") == "qa_short"
                            and option(row["args"], "--partition") == "gpu_x" for row in probes))
        self.assertTrue(all("--no-requeue" in row["args"] and "--export=ALL" in row["args"]
                            for row in probes))
        self.assert_no_jobs_or_run()

    def test_test_only_rejects_first_candidate_then_accepts_next(self):
        plan = self.plan(extra=dict(self.qos_site(), FAKE_QOS_REJECT='["qa_short"]'))
        self.assertEqual(plan["site"]["GPU_QOS_SHORT"], "qa_medium")
        gpu_probes = [option(row["args"], "--qos") for row in self.probes()
                      if option(row["args"], "--partition") == "gpu_x"
                      and option(row["args"], "--time") == "01:00:00"]
        self.assertEqual(gpu_probes[:2], ["qa_short", "qa_medium"])
        self.assert_no_jobs_or_run()

    def test_long_qos_rejected_for_single_gpu_selects_next_valid_for_every_shape(self):
        extra = self.qos_site("qa_short,qa_medium,qa_long,qa_backup",
                              "qa_short|01:00:00|\nqa_medium|02:00:00|\n"
                              "qa_long|1-00:00:00|\nqa_backup|2-00:00:00|\n")
        extra["FAKE_QOS_SHAPE_REJECT"] = json.dumps([{"qos": "qa_long", "gpus-per-node": "h100:1"}])
        plan = self.plan(extra=extra)
        self.assertEqual(plan["site"]["GPU_QOS_LONG"], "qa_backup")
        for name in ("qa_long", "qa_backup"):
            probes = [row["args"] for row in self.probes() if option(row["args"], "--qos") == name
                      and option(row["args"], "--partition") == "gpu_x"]
            self.assertTrue(any(option(row, "--gpus-per-node") == "h100:4" for row in probes))
            self.assertTrue(any(option(row, "--gpus-per-node") == "h100:1" for row in probes))
        self.assert_no_jobs_or_run()

    def test_named_and_default_qos_rejected_for_single_gpu_stop_before_reservation(self):
        extra = self.qos_site("qa_short,qa_medium,qa_long,qa_backup",
                              "qa_short|01:00:00|\nqa_medium|02:00:00|\n"
                              "qa_long|1-00:00:00|\nqa_backup|2-00:00:00|\n")
        extra["FAKE_QOS_SHAPE_REJECT"] = json.dumps([
            {"qos": name, "gpus-per-node": "h100:1"} for name in ("qa_long", "qa_backup", "none")])
        result = self.run_plan(extra=extra)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STOP:", result.stderr)
        self.assertIn("GPU_QOS_LONG", result.stderr)
        self.assertIn("root 待核", result.stderr)
        self.assertTrue(any(option(row["args"], "--qos") is None
                            and option(row["args"], "--gpus-per-node") == "h100:1"
                            for row in self.probes()))
        self.assert_no_jobs_or_run()

    def test_cpu_qos_must_accept_light_advance_and_summary_resources(self):
        extra = self.qos_site()
        extra["FAKE_QOS_SHAPE_REJECT"] = json.dumps([
            {"qos": "qa_medium", "partition": "cpu_x", "cpus-per-task": "1"}])
        plan = self.plan(extra=extra)
        self.assertEqual(plan["site"]["CPU_QOS"], "qa_long")
        cpu_probes = [row["args"] for row in self.probes()
                      if option(row["args"], "--partition") == "cpu_x"]
        self.assertTrue(any(option(row, "--qos") == "qa_medium"
                            and option(row, "--cpus-per-task") == "4" for row in cpu_probes))
        self.assertTrue(any(option(row, "--qos") == "qa_medium"
                            and option(row, "--cpus-per-task") == "1" for row in cpu_probes))
        self.assertTrue(any(option(row, "--qos") == "qa_long"
                            and option(row, "--cpus-per-task") == "1" for row in cpu_probes))
        self.assert_no_jobs_or_run()

    def test_every_qos_and_default_rejected_stops_without_reservation(self):
        result = self.run_plan(extra=dict(self.qos_site(), FAKE_QOS_REJECT='["*"]'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STOP:", result.stderr)
        self.assertIn("QOS discovery incomplete", result.stderr)
        self.assertIn("root 待核", result.stderr)
        self.assertTrue(any(option(row["args"], "--qos") is None for row in self.probes()))
        self.assert_no_jobs_or_run()

    def test_test_only_unavailable_stops_for_confirmation_after_default_probe(self):
        result = self.run_plan(extra=dict(self.qos_site(), FAKE_TEST_UNAVAILABLE="1"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("root 待核", result.stderr)
        self.assertTrue(any(option(row["args"], "--qos") is None for row in self.probes()))
        self.assert_no_jobs_or_run()

    def test_only_shorter_long_qos_caps_partition_walltime(self):
        plan = self.plan(extra=self.qos_site("qa_limited", "qa_limited|12:00:00|\n"))
        self.assertEqual(plan["site"]["GPU_QOS_LONG"], "qa_limited")
        self.assertEqual(self.fixture.workflow.duration(plan["site"]["MAX_WALLTIME"]), 43200)
        self.assertTrue(any("MAX_WALLTIME" in note and "12:00:00" in note for note in plan["notes"]))
        self.assertTrue(all(row["resources"]["walltime_seconds"] <= 43200 for row in plan["jobs"]))
        self.assert_no_jobs_or_run()

    def test_short_qos_does_not_hide_sole_usable_long_qos_walltime_cap(self):
        extra = self.qos_site("qa_short,qa_limited",
                              "qa_short|01:00:00|\nqa_limited|12:00:00|\n")
        extra["FAKE_QOS_SHAPE_REJECT"] = json.dumps([
            {"qos": "none", "partition": "gpu_x", "time": "24:00:00"}])
        plan = self.plan(extra=extra)
        self.assertEqual(plan["site"]["GPU_QOS_SHORT"], "qa_short")
        self.assertEqual(plan["site"]["GPU_QOS_LONG"], "qa_limited")
        self.assertEqual(self.fixture.workflow.duration(plan["site"]["MAX_WALLTIME"]), 43200)
        evidence = plan["site"]["_qos_discovery"]["GPU_QOS_LONG"]
        self.assertEqual(evidence["original_gpu_max_walltime_seconds"], 86400)
        self.assertEqual(evidence["effective_gpu_max_walltime_seconds"], 43200)
        probes = [row["args"] for row in self.probes()
                  if option(row["args"], "--partition") == "gpu_x"
                  and option(row["args"], "--qos") == "qa_limited"]
        for gpus, seconds in (("h100:4", 43200), ("h100:1", 28800)):
            self.assertTrue(any(option(row, "--gpus-per-node") == gpus
                                and self.fixture.workflow.duration(option(row, "--time")) == seconds
                                for row in probes), (gpus, seconds, probes))
        self.assertTrue(any("MAX_WALLTIME" in note and "12:00:00" in note
                            for note in plan["notes"]))
        self.assert_no_jobs_or_run()

    def test_capped_long_qos_rejected_for_either_shape_stops_before_reservation(self):
        for rejected_gpu_count in ("h100:4", "h100:1"):
            with self.subTest(rejected_gpu_count=rejected_gpu_count):
                extra = self.qos_site("qa_short,qa_limited",
                                      "qa_short|01:00:00|\nqa_limited|12:00:00|\n")
                extra["FAKE_QOS_SHAPE_REJECT"] = json.dumps([
                    {"qos": "qa_limited", "partition": "gpu_x",
                     "gpus-per-node": rejected_gpu_count},
                    {"qos": "none", "partition": "gpu_x", "time": "24:00:00"}])
                before = len(self.probes())
                result = self.run_plan(extra=extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("STOP:", result.stderr)
                self.assertIn("GPU_QOS_LONG", result.stderr)
                probes = [row["args"] for row in self.probes()[before:]]
                self.assertTrue(any(option(row, "--qos") == "qa_limited"
                                    and option(row, "--gpus-per-node") == rejected_gpu_count
                                    for row in probes))
                self.assertTrue(any(option(row, "--qos") is None
                                    and option(row, "--partition") == "gpu_x"
                                    and self.fixture.workflow.duration(option(row, "--time")) == 86400
                                    for row in probes))
                self.assert_no_jobs_or_run()

    def test_rejected_capped_long_qos_default_fallback_keeps_original_partition_cap(self):
        extra = self.qos_site("qa_short,qa_limited",
                              "qa_short|01:00:00|\nqa_limited|12:00:00|\n")
        extra["FAKE_QOS_SHAPE_REJECT"] = json.dumps([
            {"qos": "qa_limited", "partition": "gpu_x", "gpus-per-node": "h100:1"}])
        plan = self.plan(extra=extra)
        self.assertEqual(plan["site"]["GPU_QOS_LONG"], "none")
        self.assertEqual(self.fixture.workflow.duration(plan["site"]["MAX_WALLTIME"]), 86400)
        self.assertFalse(any("MAX_WALLTIME reduced" in note for note in plan["notes"]))
        self.assertTrue(any(option(row["args"], "--qos") is None
                            and option(row["args"], "--partition") == "gpu_x"
                            and self.fixture.workflow.duration(option(row["args"], "--time")) == 86400
                            for row in self.probes()))
        self.assert_no_jobs_or_run()

    def test_short_qos_alone_does_not_cap_long_jobs_to_one_hour(self):
        plan = self.plan(extra=self.qos_site("qa_short", "qa_short|01:00:00|\n"))
        self.assertEqual(plan["site"]["GPU_QOS_SHORT"], "qa_short")
        self.assertEqual(plan["site"]["GPU_QOS_LONG"], "none")
        self.assertEqual(self.fixture.workflow.duration(plan["site"]["MAX_WALLTIME"]), 86400)
        self.assertTrue(any(option(row["args"], "--qos") is None
                            and option(row["args"], "--partition") == "gpu_x"
                            and self.fixture.workflow.duration(option(row["args"], "--time")) == 86400
                            for row in self.probes()))
        self.assert_no_jobs_or_run()

    def test_none_and_explicit_empty_environment_override_autodiscovery(self):
        for value in ("NoNe", ""):
            with self.subTest(value=value):
                overrides = dict(self.qos_site(), **{key: value for key in QOS_KEYS})
                plan = self.plan(extra=overrides)
                self.assertEqual({plan["site"][key] for key in QOS_KEYS}, {"none"})
                self.assertTrue(all(plan["site"]["_sources"][key] == "user override" for key in QOS_KEYS))
                self.assertTrue(all(not any(arg.startswith("--qos") for arg in row["command_preview"])
                                    for row in plan["jobs"]))
        self.assert_no_jobs_or_run()

    def test_site_env_none_overrides_discovery(self):
        supplied = self.fixture.root / "explicit.env"
        supplied.write_text("CPU_QOS=none\nGPU_QOS_SHORT=NONE\nGPU_QOS_LONG=''\n")
        plan = self.plan(extra=self.qos_site(), args=("--site-env", str(supplied)))
        self.assertEqual({plan["site"][key] for key in QOS_KEYS}, {"none"})
        self.assertTrue(all(plan["site"]["_sources"][key] == "user override" for key in QOS_KEYS))
        self.assert_no_jobs_or_run()

    def test_sacctmgr_unavailable_falls_back_to_default_with_explanation(self):
        plan = self.plan(extra={"FAKE_ASSOC_UNAVAILABLE": "1"})
        self.assertEqual({plan["site"][key] for key in QOS_KEYS}, {"none"})
        self.assertTrue(any("unavailable" in note.lower() for note in plan["notes"]))
        self.assert_no_jobs_or_run()

    def test_legacy_site_qos_binding_is_unchanged_and_rejects_none_override(self):
        values = dict(zip(QOS_KEYS, ("qa_cpu_old", "qa_gpu_old", "qa_long_old")))
        first = self.run_plan(args=("--submit",), extra=values)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        saved = self.fixture.output / "state/site.json"
        old = json.loads(saved.read_text())
        old.pop("_sources", None)
        old.pop("_qos_discovery", None)
        saved.write_text(json.dumps(old))
        before = saved.read_bytes()
        env_path = self.fixture.output / "state/site.env"
        original_env = env_path.read_bytes()
        count = len(self.fixture.job_rows())
        preview = self.plan(extra=self.qos_site())
        self.assertEqual({key: preview["site"][key] for key in QOS_KEYS}, values)
        conflict = self.run_plan(extra={"CPU_QOS": "none"})
        self.assertNotEqual(conflict.returncode, 0)
        self.assertIn("bind", conflict.stderr)
        supplied = self.fixture.root / "conflict.env"
        supplied.write_text("CPU_QOS=none\n")
        conflict = self.run_plan(args=("--site-env", str(supplied)))
        self.assertNotEqual(conflict.returncode, 0)
        self.assertEqual(saved.read_bytes(), before)
        self.assertEqual(env_path.read_bytes(), original_env)
        self.assertEqual(len(self.fixture.job_rows()), count)

    def test_default_summary_is_one_screen_and_has_required_fields_without_nodes(self):
        result = self.run_plan(json_output=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertLessEqual(len(result.stdout.splitlines()), 40)
        for value in ("PLAN_ONLY", "ACCOUNT", "PARTITION", "CPU_QOS", "GPU_QOS_SHORT", "GPU_QOS_LONG",
                      "H100", "data", "bank", "smoke", "advance", "bash", "--submit"):
            self.assertIn(value, result.stdout)
        for value in ("storage", "inode", "queue", "default none"):
            self.assertIn(value, result.stdout.lower())
        for name in NODE_NAMES:
            self.assertNotIn(name, result.stdout)
        self.assert_no_jobs_or_run()

    def test_summary_next_command_retains_yes_for_noninteractive_use(self):
        result = self.run_plan(json_output=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        next_line = next(line for line in result.stdout.splitlines() if line.startswith("Next: "))
        command = shlex.split(next_line[len("Next: "):])
        self.assertEqual(command[:2], ["bash", "run.sh"])
        self.assertIn("--submit", command)
        self.assertIn("--yes", command)
        self.assertEqual(command[command.index("--output-root") + 1], str(self.fixture.output))
        self.assert_no_jobs_or_run()

    def test_json_plan_keeps_structure_with_inventory_counts_and_no_node_names(self):
        result = self.run_plan()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        plan = json.loads(result.stdout)
        for key in ("mode", "site", "jobs", "checks", "notes", "run_prefix"):
            self.assertIn(key, plan)
        inventory = plan["checks"]["resource_inventory"]
        self.assertEqual(inventory["compatible_gpu_node_count"], 4)
        self.assertEqual(inventory["compatible_cpu_node_count"], 1)
        self.assertTrue(inventory["physical_nodes"])
        self.assertTrue(all("node_count" in row and "partition" in row
                            and "node" not in row for row in inventory["physical_nodes"]))
        for name in NODE_NAMES:
            self.assertNotIn(name, result.stdout)
        self.assert_no_jobs_or_run()

    def test_successful_test_only_node_diagnostics_do_not_leak_into_json(self):
        result = self.run_plan(extra={"FAKE_PROBE_STDOUT":
                                     "Job 0 would start on nodes " + ",".join(NODE_NAMES) + "\n"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        plan = json.loads(result.stdout)
        self.assertTrue(all(row["accepted"] for cls in plan["site"]["_qos_discovery"].values()
                            for row in cls["probes"]))
        for name in NODE_NAMES:
            self.assertNotIn(name, result.stdout)
            self.assertNotIn(name, result.stderr)
        self.assert_no_jobs_or_run()

    def test_default_submit_prints_compact_job_table_with_held_and_released_states(self):
        result = self.run_plan(json_output=False, args=("--submit",),
                               extra={key: "none" for key in QOS_KEYS})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertLessEqual(len(result.stdout.splitlines()), 40)
        self.assertTrue(result.stdout.startswith("SUBMIT\n"))
        self.assertIn("Submission | job ID | held/released", result.stdout)
        self.assertFalse(any(line.startswith("{") for line in result.stdout.splitlines()))
        for key in ("data", "bank", "smoke", "advance"):
            saved = self.fixture.output / "state/jobs" / (key + ".json")
            row = json.loads(saved.read_text())
            self.assertIn(key + " | " + str(row["job_id"]) + " | ", result.stdout)
        self.assertIn("data | 100 | held", result.stdout)
        self.assertIn("data | 100 | released", result.stdout)
        for name in NODE_NAMES:
            self.assertNotIn(name, result.stdout)

    def test_plan_only_leaves_existing_fixture_files_unchanged_and_creates_no_run(self):
        def files():
            return {str(path.relative_to(self.fixture.root)): path.read_bytes()
                    for path in self.fixture.root.rglob("*")
                    if path.is_file() and self.fixture.mock not in path.parents}
        before = files()
        self.plan(extra=self.qos_site())
        self.assertEqual(files(), before)
        self.assert_no_jobs_or_run()

    def test_failing_plan_only_storage_gate_reports_numbers_without_writing_files(self):
        before = {str(path.relative_to(self.fixture.root)) for path in self.fixture.root.rglob("*")}
        result = self.run_plan(extra={"BIGFIELD_MIN_FREE_BYTES": str(1 << 90)})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GATE_REPORT", result.stderr)
        self.assertIn("free_bytes", result.stderr)
        self.assertIn("free_inodes", result.stderr)
        self.assertIn(str(1 << 90), result.stderr)
        after = {str(path.relative_to(self.fixture.root)) for path in self.fixture.root.rglob("*")}
        self.assertEqual(after - before, {"mock/calls.jsonl"})
        self.assert_no_jobs_or_run()

    def test_none_readback_records_actual_default_qos(self):
        result = self.run_plan(args=("--submit",), extra={key: "none" for key in QOS_KEYS})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.fixture.job_rows())
        self.assertTrue(all("qos" not in row for row in self.fixture.job_rows()))
        readbacks = self.fixture.output / "state/readbacks"
        self.assertTrue(all(json.loads(path.read_text())["parsed"]["QOS"] == "qa_default"
                            for path in readbacks.glob("*.json")))

    def test_interactive_confirmation_displays_qos_values_and_sources(self):
        args = argparse.Namespace(site_env=None, output_root=str(self.fixture.output),
                                  data_root=str(self.fixture.data), yes=False)
        stream = io.StringIO()
        with mock.patch.dict(os.environ, self.fixture.env, clear=True), \
                mock.patch.object(sys.stdin, "isatty", return_value=True), \
                mock.patch("builtins.input", return_value=""), \
                contextlib.redirect_stderr(stream):
            site, _ = self.fixture.workflow.site_configuration(args)
        text = stream.getvalue()
        for key in QOS_KEYS:
            self.assertEqual(site[key], "none")
            self.assertIn(key, text)
        self.assertIn("default none", text)
        self.assert_no_jobs_or_run()

    def test_interactive_json_uses_real_input_and_keeps_all_prompts_on_stderr(self):
        # Do not mock builtins.input: its real prompt behavior exposed the bug.
        stdin = io.StringIO("\n" * 64)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, self.fixture.env, clear=True), \
                mock.patch.object(sys, "stdin", stdin), \
                mock.patch.object(stdin, "isatty", return_value=True), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.fixture.workflow.main(["--json", "--output-root", str(self.fixture.output),
                                        "--data-root", str(self.fixture.data)])
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["mode"], "PLAN_ONLY")
        self.assertTrue(stdout.getvalue().lstrip().startswith("{"))
        for key in ("ACCOUNT", *QOS_KEYS):
            self.assertIn(key + " [", stderr.getvalue())
            self.assertNotIn(key + " [", stdout.getvalue())
        self.assertIn("default none", stderr.getvalue())
        for name in NODE_NAMES:
            self.assertNotIn(name, stdout.getvalue())
        self.assert_no_jobs_or_run()


if __name__ == "__main__":
    unittest.main()
