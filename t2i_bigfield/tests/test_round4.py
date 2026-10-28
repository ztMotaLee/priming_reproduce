"""QOS selection when scheduler limits are incomplete; fake Slurm only."""
from __future__ import annotations

import json
import shlex
import unittest

import test_round3 as previous


QOS_KEYS = previous.QOS_KEYS
DEFAULT_SOURCE = "site default (QOS limits not published)"
AMBIGUOUS_SOURCE = "ambiguous-needs-confirmation"
VERIFIED_SOURCE = "detected+test-only verified"


class Round4Tests(unittest.TestCase):
    def setUp(self):
        # Composition adds only the new cases to unittest discovery.
        self.previous = previous.Round3Tests(methodName="runTest")
        self.previous.setUp()
        self.addCleanup(self.previous.doCleanups)
        self.fixture = self.previous.fixture

    def plan(self, **kwargs):
        return self.previous.plan(**kwargs)

    def run_plan(self, **kwargs):
        return self.previous.run_plan(**kwargs)

    def site(self, rows="qa_low||1|\nqa_high||80|\n", names="qa_low,qa_high"):
        return self.previous.qos_site(names, rows)

    def assert_no_reservation(self):
        self.previous.assert_no_jobs_or_run()
        self.assertEqual(self.fixture.calls("scontrol"), [])

    def probes(self, key=None):
        rows = self.previous.probes()
        if key is None:
            return rows
        def belongs(row):
            args = row["args"]
            partition = previous.option(args, "--partition")
            seconds = self.fixture.workflow.duration(previous.option(args, "--time"))
            if key == "CPU_QOS":
                return partition == "cpu_x"
            if partition != "gpu_x":
                return False
            return (seconds <= 3600) == (key == "GPU_QOS_SHORT")
        return [row for row in rows if belongs(row)]

    def candidate_order(self, key):
        order = []
        for row in self.probes(key):
            name = previous.option(row["args"], "--qos") or "none"
            if not order or order[-1] != name:
                order.append(name)
        return order

    def assert_default_none(self, plan):
        self.assertEqual([plan["site"][key] for key in QOS_KEYS], ["none"] * 3)
        self.assertEqual([plan["site"]["_sources"][key] for key in QOS_KEYS],
                         [DEFAULT_SOURCE] * 3)
        self.assertEqual(self.fixture.workflow.duration(plan["site"]["MAX_WALLTIME"]), 86400)
        self.assertTrue(self.probes())
        self.assertTrue(all(previous.option(row["args"], "--qos") is None
                            for row in self.probes()))
        self.assertTrue(all(not any(arg.startswith("--qos") for arg in row["command_preview"])
                            for row in plan["jobs"]))
        for key in QOS_KEYS:
            self.assertEqual(self.candidate_order(key), ["none"])
        self.assert_no_reservation()

    def test_all_blank_maxwall_prefers_none_for_all_three_resource_classes(self):
        plan = self.plan(extra=self.site())
        self.assert_default_none(plan)
        queries = self.fixture.calls("sacctmgr")
        self.assertTrue(any("format=Name,MaxWall,Priority" in row["args"] for row in queries))

    def test_unlimited_and_infinite_limits_prefer_none(self):
        plan = self.plan(extra=self.site("qa_low|UNLIMITED|1|\nqa_high|infinite|80|\n"))
        self.assert_default_none(plan)

    def test_missing_qos_rows_prefer_none(self):
        plan = self.plan(extra=self.site(rows=""))
        self.assert_default_none(plan)

    def test_missing_associated_row_and_blank_other_row_prefer_none(self):
        plan = self.plan(extra=self.site("qa_high||80|\n"))
        self.assert_default_none(plan)

    def test_unassociated_finite_limit_does_not_disambiguate_associated_qos(self):
        plan = self.plan(extra=self.site("qa_low||1|\nqa_high||80|\nqa_other|24:00:00|900|\n"))
        self.assert_default_none(plan)

    def test_default_none_is_checked_for_light_cpu_and_single_gpu_requests(self):
        plan = self.plan(extra=self.site())
        cpu = [row["args"] for row in self.probes("CPU_QOS")]
        self.assertEqual({previous.option(args, "--cpus-per-task") for args in cpu}, {"1", "4"})
        long_gpu = [row["args"] for row in self.probes("GPU_QOS_LONG")]
        self.assertEqual({previous.option(args, "--gpus-per-node") for args in long_gpu},
                         {"h100:1", "h100:4"})
        self.assertTrue(all("--test-only" in row["args"] and "--export=ALL" in row["args"]
                            and "--no-requeue" in row["args"] for row in self.probes()))
        self.assert_default_none(plan)

    def test_none_rejection_tries_high_priority_then_name_and_marks_ambiguity(self):
        extra = self.site("qa_low||1|\nqa_high_z||80|\nqa_high_a||80|\n",
                          "qa_low,qa_high_z,qa_high_a")
        extra["FAKE_QOS_REJECT"] = '["none", "qa_high_a"]'
        plan = self.plan(extra=extra)
        for key in QOS_KEYS:
            self.assertEqual(plan["site"][key], "qa_high_z")
            self.assertEqual(plan["site"]["_sources"][key], AMBIGUOUS_SOURCE)
            self.assertEqual(self.candidate_order(key), ["none", "qa_high_a", "qa_high_z"])
        self.assertTrue(any("ambiguous" in note.lower() and "mentor must confirm" in note.lower()
                            for note in plan["notes"]))
        self.assert_no_reservation()

    def test_named_fallback_must_accept_applicable_single_gpu_shape(self):
        extra = self.site("qa_low||1|\nqa_high||80|\n")
        extra["FAKE_QOS_SHAPE_REJECT"] = json.dumps([
            {"qos": "none", "partition": "gpu_x", "gpus-per-node": "h100:1"},
            {"qos": "qa_high", "partition": "gpu_x", "gpus-per-node": "h100:1"}])
        plan = self.plan(extra=extra)
        self.assertEqual(plan["site"]["GPU_QOS_LONG"], "qa_low")
        self.assertEqual(plan["site"]["_sources"]["GPU_QOS_LONG"], AMBIGUOUS_SOURCE)
        self.assertEqual(self.candidate_order("GPU_QOS_LONG"), ["none", "qa_high", "qa_low"])
        for name in (None, "qa_high", "qa_low"):
            requests = [row["args"] for row in self.probes("GPU_QOS_LONG")
                        if previous.option(row["args"], "--qos") == name]
            self.assertEqual({previous.option(args, "--gpus-per-node") for args in requests},
                             {"h100:1", "h100:4"})
        self.assertEqual(plan["site"]["CPU_QOS"], "none")
        self.assertEqual(plan["site"]["GPU_QOS_SHORT"], "none")
        self.assert_no_reservation()

    def test_light_cpu_rejection_changes_only_cpu_qos(self):
        extra = self.site()
        extra["FAKE_QOS_SHAPE_REJECT"] = json.dumps([
            {"qos": "none", "partition": "cpu_x", "cpus-per-task": "1"}])
        plan = self.plan(extra=extra)
        self.assertEqual(plan["site"]["CPU_QOS"], "qa_high")
        self.assertEqual(plan["site"]["_sources"]["CPU_QOS"], AMBIGUOUS_SOURCE)
        self.assertEqual(self.candidate_order("CPU_QOS"), ["none", "qa_high"])
        for key in ("GPU_QOS_SHORT", "GPU_QOS_LONG"):
            self.assertEqual(plan["site"][key], "none")
            self.assertEqual(plan["site"]["_sources"][key], DEFAULT_SOURCE)
        self.assert_no_reservation()

    def test_all_ambiguous_candidates_rejected_stops_without_reserving_run(self):
        extra = dict(self.site(), FAKE_QOS_REJECT='["*"]')
        result = self.run_plan(extra=extra)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("STOP:", result.stderr)
        self.assertIn("root 待核", result.stderr)
        for key in QOS_KEYS:
            self.assertIn(key, result.stderr)
            self.assertEqual(self.candidate_order(key), ["none", "qa_high", "qa_low"])
        self.assert_no_reservation()

    def test_finite_maxwall_ties_choose_highest_priority_per_resource_class(self):
        rows = ("qa_short_low|01:00:00|2|\nqa_short_high|01:00:00|20|\n"
                "qa_cpu_low|02:00:00|3|\nqa_cpu_high|02:00:00|30|\n"
                "qa_long_low|24:00:00|4|\nqa_long_high|24:00:00|40|\n")
        names = ",".join(line.split("|")[0] for line in rows.splitlines())
        plan = self.plan(extra=self.site(rows, names))
        self.assertEqual([plan["site"][key] for key in QOS_KEYS],
                         ["qa_cpu_high", "qa_short_high", "qa_long_high"])
        self.assertEqual([plan["site"]["_sources"][key] for key in QOS_KEYS],
                         [VERIFIED_SOURCE] * 3)
        self.assertTrue(all(self.candidate_order(key)[0] != "none" for key in QOS_KEYS))
        self.assert_no_reservation()

    def test_finite_maxwall_and_priority_ties_choose_name(self):
        plan = self.plan(extra=self.site("qa_z|24:00:00|20|\nqa_a|24:00:00|20|\n", "qa_z,qa_a"))
        self.assertEqual([plan["site"][key] for key in QOS_KEYS], ["qa_a"] * 3)
        self.assertTrue(all(self.candidate_order(key) == ["qa_a"] for key in QOS_KEYS))
        self.assert_no_reservation()

    def test_finite_shortest_sufficient_maxwall_precedes_higher_priority(self):
        extra = self.site("qa_short|01:00:00|1|\nqa_medium|02:00:00|2|\nqa_long|24:00:00|999|\n",
                          "qa_short,qa_medium,qa_long")
        plan = self.plan(extra=extra)
        self.assertEqual([plan["site"][key] for key in QOS_KEYS],
                         ["qa_medium", "qa_short", "qa_long"])
        self.assert_no_reservation()

    def test_user_overrides_remain_authoritative_when_limits_are_missing(self):
        for value in ("qa_manual", "NoNe", ""):
            with self.subTest(value=value):
                before = len(self.probes())
                extra = dict(self.site(), **{key: value for key in QOS_KEYS})
                plan = self.plan(extra=extra)
                expected = "qa_manual" if value == "qa_manual" else "none"
                self.assertEqual([plan["site"][key] for key in QOS_KEYS], [expected] * 3)
                self.assertEqual([plan["site"]["_sources"][key] for key in QOS_KEYS],
                                 ["user override"] * 3)
                self.assertEqual(len(self.probes()), before)
        self.assert_no_reservation()

    def test_legacy_binding_is_not_rediscovered_or_silently_changed(self):
        values = dict(zip(QOS_KEYS, ("qa_old_cpu", "qa_old_short", "qa_old_long")))
        site = self.plan(extra=values)["site"]
        site.pop("_sources", None)
        site.pop("_qos_discovery", None)
        state = self.fixture.output / "state"
        state.mkdir(parents=True)
        saved = state / "site.json"
        saved.write_text(json.dumps(site))
        site_env = state / "site.env"
        site_env.write_text("".join(key + "=" + shlex.quote(value) + "\n"
                                    for key, value in values.items()))
        before = {path.name: path.read_bytes() for path in state.iterdir()}
        query_count = len(self.fixture.calls("sacctmgr"))
        plan = self.plan(extra=self.site(), args=("--site-env", str(site_env)))
        self.assertEqual({key: plan["site"][key] for key in QOS_KEYS}, values)
        self.assertEqual(len(self.fixture.calls("sacctmgr")), query_count)
        self.assertEqual(self.probes(), [])
        conflict = self.run_plan(extra=dict(self.site(), CPU_QOS="none"))
        self.assertNotEqual(conflict.returncode, 0)
        self.assertIn("bind", conflict.stderr)
        self.assertEqual({path.name: path.read_bytes() for path in state.iterdir()}, before)
        self.assertEqual(self.fixture.job_rows(), [])
        self.assertEqual(self.fixture.calls("scontrol"), [])

    def test_summary_shows_site_default_source_without_names_or_large_json(self):
        result = self.run_plan(json_output=False, extra=self.site())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertLessEqual(len(result.stdout.splitlines()), 40)
        self.assertIn(DEFAULT_SOURCE, result.stdout)
        for key in QOS_KEYS:
            self.assertIn(key + " | none | " + DEFAULT_SOURCE, result.stdout)
        for node in previous.NODE_NAMES:
            self.assertNotIn(node, result.stdout)
        self.assert_no_reservation()

    def test_summary_ambiguity_warning_is_visible_and_still_one_screen(self):
        result = self.run_plan(json_output=False, extra=dict(self.site(), FAKE_QOS_REJECT='["none"]'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertLessEqual(len(result.stdout.splitlines()), 40)
        warning = [line for line in result.stdout.splitlines() if line.startswith("WARNING:")]
        self.assertTrue(warning, result.stdout)
        self.assertTrue(any("ambiguous" in line.lower() and "mentor must confirm" in line.lower()
                            for line in warning))
        for key in QOS_KEYS:
            self.assertIn(key + " | qa_high | " + AMBIGUOUS_SOURCE, result.stdout)
        self.assert_no_reservation()


if __name__ == "__main__":
    unittest.main()
