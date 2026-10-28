"""Small workflow checks; every scheduler executable comes from a local fixture."""
from __future__ import annotations

import json
import hashlib
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]

# The mock records commands and allocation attributes. It never starts workers.
FAKE_SLURM = r'''import json,os,pathlib,sys
root=pathlib.Path(os.environ['FAKE_SLURM_ROOT'])
tool=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:]
with (root/'calls.jsonl').open('a') as f:
 f.write(json.dumps(dict(tool=tool,args=args))+'\n')
def options(argv):
 result={}
 index=0
 while index<len(argv):
  arg=argv[index]
  if arg.startswith('--'):
   if '=' in arg:
    key,value=arg[2:].split('=',1)
   elif index+1<len(argv) and not argv[index+1].startswith('-'):
    key,value=arg[2:],argv[index+1]; index+=1
   else:key,value=arg[2:],'yes'
   result[key]=value
  index+=1
 return result
if tool=='sacctmgr':
 print('demo|')
elif tool=='sinfo':
 if 'FAKE_SINFO' in os.environ:print(os.environ['FAKE_SINFO'],end='')
 else:
  for index in range(int(os.environ.get('FAKE_GPU_NODES','4'))):
   print('compute*|gpu'+str(index)+'|'+os.environ.get('FAKE_GRES','gpu:h100:4')+'|32|131072|1-00:00:00')
  print('cpu|cpu0|N/A|32|131072|2-00:00:00')
elif tool=='squeue':
 if os.environ.get('FAKE_QUEUE_BLOCKED')=='1':
  print('queue query unavailable',file=sys.stderr);sys.exit(1)
 if 'FAKE_QUEUE' in os.environ:print(os.environ['FAKE_QUEUE'],end='')
 else:
  for p in sorted(root.glob('job_*.json')):
   row=json.loads(p.read_text())
   print(p.stem.split('_')[1]+'|'+row.get('job-name','')+'|PENDING')
elif tool=='sacct':
 print(os.environ.get('FAKE_ACCOUNTING',''),end='')
elif tool=='sbatch':
 if '--test-only' in args:
  if os.environ.get('FAKE_REJECT_TEST_ONLY')=='1':
   print('job-origin submission disabled',file=sys.stderr);sys.exit(1)
  sys.exit(0)
 if os.environ.get('FAKE_REJECT_ADVANCE')=='1' and any('advance' in arg for arg in args):
  print('job-origin submission disabled',file=sys.stderr); sys.exit(1)
 if os.environ.get('FAKE_REJECT_NESTED')=='1' and os.environ.get('SLURM_JOB_ID'):
  print('sbatch: error: Batch job submission failed: Job submit from compute nodes is disabled',file=sys.stderr);sys.exit(1)
 if os.environ.get('FAKE_AMBIGUOUS_SUBMIT')=='1':
  print('sbatch: error: Batch job submission failed: Socket timed out on send/recv operation',file=sys.stderr);sys.exit(1)
 count=len(list(root.glob('job_*.json')))
 job=str(100+count)
 row=options(args)
 row['argv']=args
 (root/('job_'+job+'.json')).write_text(json.dumps(row))
 print(job)
elif tool=='scontrol':
 if args[0]=='release':
  job=args[-1]
  if os.environ.get('FAKE_RELEASE_FAIL')=='1' or os.environ.get('FAKE_RELEASE_FAIL_ID')==job:
   print('release temporarily unavailable',file=sys.stderr);sys.exit(1)
  path=root/('job_'+job+'.json')
  row=json.loads(path.read_text());row['released']=True;path.write_text(json.dumps(row))
  if os.environ.get('FAKE_RELEASE_LOST_RESPONSE')=='1':
   print('release response unavailable',file=sys.stderr);sys.exit(1)
  sys.exit(0)
 job=args[-1]
 row=json.loads((root/('job_'+job+'.json')).read_text())
 nodes=int(row.get('nodes','1'))
 tasks=int(row.get('ntasks',str(nodes)))
 cpus=tasks*int(row.get('cpus-per-task','1'))
 gpu=row.get('gpus-per-node','')
 tres='gres/gpu:'+gpu if gpu else '(null)'
 if row.get('gres'):tres='gres/'+row['gres']
 dependency=row.get('dependency','(null)')
 if dependency!='(null)':
  if os.environ.get('FAKE_COMPACT_DEPENDENCY')=='1':dependency+='(unfulfilled)'
  else:dependency=','.join('afterok:'+job+'(unfulfilled)' for job in dependency.split(':')[1:])
 limit=row.get('time','01:00:00')
 hour,minute,second=map(int,limit.split(':'))
 total=((hour*3600+minute*60+second+59)//60)*60
 limit=f'{total//3600:02d}:{total//60%60:02d}:00'
 values=dict(JobId=job,JobName=row.get('job-name',''),Account=row.get('account',''),
  Partition=row.get('partition',''),QOS=row.get('qos','normal'),NumNodes=str(nodes),
  NumCPUs=str(cpus),NumTasks=os.environ.get('FAKE_NUMTASKS',str(tasks)),
  MinMemoryNode=os.environ.get('FAKE_MEM',row.get('mem','1G')),
  TimeLimit=limit,Dependency=dependency,
  Requeue='0',TresPerNode=tres,JobState=row.get('job_state','PENDING'),
  Reason=row.get('reason','JobHeldUser' if 'hold' in row and not row.get('released') else 'Dependency'),AllocNodeSid='unused')
 values['NumNodes']=os.environ.get('FAKE_NUM_NODES',values['NumNodes'])
 values['CPUs/Task']=os.environ.get('FAKE_CPU',row.get('cpus-per-task','1'))
 if os.environ.get('FAKE_BAD_READBACK')=='1':values['Account']='different'
 print(' '.join(key+'='+value for key,value in values.items())+' AllocNode:Sid=host:1 ReqB:S:C:T=0:0:*:*')
else:sys.exit('Unexpected fake command: '+tool)
'''


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workflow-")
        self.root = Path(self.temporary.name)
        self.mock = self.root / "mock"
        self.mock.mkdir()
        self.data = self.root / "data"
        self.data.mkdir()
        self.output = self.root / "output"
        # Exercise unmodified controller scripts against their real source
        # checker in a tiny frozen fixture; no production worker is executed.
        self.code = self.root / "code"
        self.code.mkdir()
        for name in ("workflow.py", "run.sh", "freeze.py", "worker.slurm", "worker.py", "status.sh"):
            shutil.copyfile(ROOT / name, self.code / name)
        (self.code / "configs").mkdir()
        shutil.copyfile(ROOT / "configs/bigfield_tf.yaml", self.code / "configs/bigfield_tf.yaml")
        frozen = subprocess.run([sys.executable, "-B", "freeze.py"], cwd=self.code,
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(frozen.returncode, 0, frozen.stdout + frozen.stderr)
        specification = importlib.util.spec_from_file_location("workflow_fixture", self.code / "workflow.py")
        self.workflow = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(self.workflow)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for tool in ("sbatch", "squeue", "sacct", "sinfo", "sacctmgr", "scontrol"):
            path = self.bin / tool
            path.write_text("#!" + sys.executable + "\n" + FAKE_SLURM)
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        FAKE_SLURM_ROOT=str(self.mock), DATA_ROOT=str(self.data),
                        OUTPUT_ROOT=str(self.output), PYTHON=sys.executable,
                        CPU_PARTITION="cpu", CPU_TIME="01:05:00", CPU_QOS="-",
                        GPU_QOS_SHORT="-", GPU_QOS_LONG="-",
                        BIGFIELD_MIN_FREE_BYTES="0", BIGFIELD_MIN_FREE_INODES="0",
                        JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for key in ("ACCOUNT", "PARTITION", "NODES", "GPUS_PER_NODE", "MAX_WALLTIME",
                    "CPUS_PER_TASK", "MEM_PER_NODE", "GPU_TYPE", "SITE_ENV", "SLURM_JOB_ID", "SLURM_NNODES"):
            self.env.pop(key, None)

    def tearDown(self):
        self.temporary.cleanup()

    def run_workflow(self, *args, extra=None, stdin=None):
        return subprocess.run(["bash", str(self.code / "run.sh"), "--output-root", str(self.output),
                               "--data-root", str(self.data), *args],
                              cwd=ROOT, env=dict(self.env, **(extra or {})), text=True,
                              input=stdin, capture_output=True, timeout=30)

    def calls(self, tool=None):
        path = self.mock / "calls.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [row for row in rows if tool is None or row["tool"] == tool]

    def job_rows(self):
        return [json.loads(path.read_text()) for path in sorted(self.mock.glob("job_*.json"))]

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def submit_initial(self, **extra):
        result = self.run_workflow("--yes", "--submit", extra=extra)
        self.assert_ok(result)
        return result

    def test_plan_only_discovers_layout_without_reserving_run_root(self):
        result = self.run_workflow("--yes")
        self.assert_ok(result)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["mode"], "PLAN_ONLY")
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls("sbatch"))
        self.assertTrue(self.calls("sinfo"))
        self.assertTrue(self.calls("sacctmgr"))

    def test_first_noninteractive_run_requires_explicit_site_confirmation(self):
        result = self.run_workflow()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("confirm", result.stderr.lower())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls("sbatch"))

    def test_invalid_topology_and_unknown_option_fail_before_submission(self):
        for args, extra in ((("--yes", "--submit"), {"NODES": "3"}),
                            (("--unknown",), {})):
            with self.subTest(args=args, extra=extra):
                result = self.run_workflow(*args, extra=extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.calls("sbatch"))

    def test_unavailable_queue_is_root_review_without_run_reservation(self):
        result = self.run_workflow("--yes", "--submit", extra={"FAKE_QUEUE_BLOCKED": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("root", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls("sbatch"))

    def test_source_check_failure_precedes_output_and_journal_reservation(self):
        wrapper = self.code / "worker.slurm"
        wrapper.write_text(wrapper.read_text() + "\n# Deliberately changed frozen fixture.\n")
        result = self.run_workflow("--yes", "--submit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source manifest", result.stderr.lower())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls("sbatch"))

    def test_storage_failure_records_numbers_without_reserving_root(self):
        impossible = str(1 << 90)
        result = self.run_workflow("--yes", "--submit", extra={"BIGFIELD_MIN_FREE_BYTES": impossible})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls("sbatch"))
        evidence = Path(result.stderr.strip().rsplit("Gate evidence: ", 1)[1])
        gate = json.loads(evidence.read_text())
        self.assertEqual(gate["min_free_bytes"], int(impossible))
        self.assertTrue(all("free_bytes" in row and "free_inodes" in row for row in gate["storage"]))

    def test_noninteractive_submission_persists_site_and_is_idempotent(self):
        self.submit_initial()
        self.assertTrue((self.output / "state/site.env").is_file())
        initial = len(self.calls("sbatch"))
        self.assertGreaterEqual(initial, 3)
        result = self.run_workflow("--yes", "--submit")
        self.assert_ok(result)
        self.assertEqual(len(self.calls("sbatch")), initial)
        self.assertTrue(all("--no-requeue" in row["args"] and "--export=ALL" in row["args"]
                            for row in self.calls("sbatch")))

    def interrupt_registered_wave(self, after="bank"):
        publish = self.workflow.atomic_json
        def interrupted(path, value, **kwargs):
            publish(path, value, **kwargs)
            if Path(path) == self.output / "state/jobs" / (after + ".json") and value.get("readback_verified") is True:
                raise KeyboardInterrupt("simulated controller interruption after registered readback")
        with mock.patch.dict(os.environ, self.env, clear=True), \
                mock.patch.object(self.workflow, "atomic_json", side_effect=interrupted), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            self.workflow.main(["--yes", "--submit"])

    def release_calls(self):
        return [row for row in self.calls("scontrol") if row["args"][0] == "release"]

    def test_partial_wave_resume_keeps_verified_root_held_until_missing_jobs_are_checked(self):
        self.interrupt_registered_wave()
        state = self.output / "state"
        root = json.loads((state / "jobs/data.json").read_text())
        self.assertEqual(root["status"], "held")
        self.assertTrue(root["readback_verified"])
        self.assertTrue(root["release_pending"])
        journal = state / "journals/data.json"
        original = journal.read_bytes()
        self.assertEqual(json.loads(original)["wave_keys"], ["data", "bank", "smoke", "advance"])
        self.assertEqual(len(self.calls("sbatch")), 2)
        self.assertFalse(self.release_calls())
        preview = self.run_workflow("--yes")
        self.assert_ok(preview)
        self.assertEqual(json.loads(preview.stdout)["existing_jobs"]["data"]["status"], "held")
        self.assertEqual([row["key"] for row in json.loads(preview.stdout)["jobs"]], ["smoke", "advance"])
        self.assertFalse(self.release_calls())
        self.assert_ok(self.run_workflow("--yes", "--submit"))
        self.assertEqual(len(self.calls("sbatch")), 4)
        self.assertEqual(len(self.release_calls()), 1)
        calls = self.calls()
        release_index = next(index for index, row in enumerate(calls) if row["tool"] == "scontrol" and row["args"][0] == "release")
        for identifier in ("100", "101", "102", "103"):
            self.assertTrue(any(row["tool"] == "scontrol" and row["args"] == ["show", "job", "-o", identifier]
                                for row in calls[:release_index]))
        self.assertEqual(journal.read_bytes(), original)
        self.assertFalse(json.loads((state / "jobs/data.json").read_text())["release_pending"])
        self.assert_ok(self.run_workflow("--yes", "--submit"))
        self.assertEqual(len(self.calls("sbatch")), 4)
        self.assertEqual(len(self.release_calls()), 1)

    def test_failed_release_empty_plan_rechecks_whole_wave_and_resumes_without_resubmission(self):
        first = self.run_workflow("--yes", "--submit", extra={"FAKE_RELEASE_FAIL": "1"})
        self.assertNotEqual(first.returncode, 0)
        self.assertIn("bash run.sh --yes --submit", first.stderr)
        self.assertEqual(len(self.calls("sbatch")), 4)
        before = len(self.calls())
        preview = self.run_workflow("--yes")
        self.assert_ok(preview)
        self.assertEqual(json.loads(preview.stdout)["jobs"], [])
        self.assertEqual(json.loads(preview.stdout)["existing_jobs"]["data"]["status"], "held")
        self.assertEqual(len(self.release_calls()), 1)
        self.assert_ok(self.run_workflow("--yes", "--submit"))
        self.assertEqual(len(self.calls("sbatch")), 4)
        self.assertEqual(len(self.release_calls()), 2)
        shows = [row["args"][-1] for row in self.calls()[before:] if row["tool"] == "scontrol" and row["args"][0] == "show"]
        self.assertEqual(set(shows), {"100", "101", "102", "103"})
        root = json.loads((self.output / "state/jobs/data.json").read_text())
        self.assertEqual(root["release_state"], "released")
        self.assertFalse(root["release_pending"])

    def test_lost_release_response_is_reconciled_without_repeating_release(self):
        first = self.run_workflow("--yes", "--submit", extra={"FAKE_RELEASE_LOST_RESPONSE": "1"})
        self.assertNotEqual(first.returncode, 0)
        root = json.loads((self.output / "state/jobs/data.json").read_text())
        self.assertEqual(root["release_state"], "requested")
        self.assertTrue(root["release_pending"])
        fixture = self.mock / "job_100.json"
        scheduled = json.loads(fixture.read_text())
        scheduled.update(job_state="RUNNING", reason="None")
        fixture.write_text(json.dumps(scheduled))
        self.assert_ok(self.run_workflow("--yes", "--submit"))
        self.assertEqual(len(self.calls("sbatch")), 4)
        self.assertEqual(len(self.release_calls()), 1)
        root = json.loads((self.output / "state/jobs/data.json").read_text())
        self.assertEqual(root["release_state"], "released")
        self.assertEqual(root["status"], "running")
        self.assertFalse(root["release_pending"])

    def test_partial_wave_bad_existing_readback_stops_before_new_journal_or_release(self):
        self.interrupt_registered_wave()
        journals = self.output / "state/journals"
        before = {path.name: path.read_bytes() for path in journals.iterdir()}
        failed = self.run_workflow("--yes", "--submit", extra={"FAKE_BAD_READBACK": "1"})
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("Scheduler readback differs", failed.stderr)
        self.assertEqual(len(self.calls("sbatch")), 2)
        self.assertFalse(self.release_calls())
        self.assertEqual({path.name: path.read_bytes() for path in journals.iterdir()}, before)

    def test_empty_plan_bad_child_readback_keeps_old_root_held(self):
        self.assertNotEqual(self.run_workflow("--yes", "--submit", extra={"FAKE_RELEASE_FAIL": "1"}).returncode, 0)
        child = self.mock / "job_103.json"
        row = json.loads(child.read_text())
        row["account"] = "wrong-account"
        child.write_text(json.dumps(row))
        before = len(self.release_calls())
        result = self.run_workflow("--yes", "--submit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Scheduler readback differs", result.stderr)
        self.assertEqual(len(self.calls("sbatch")), 4)
        self.assertEqual(len(self.release_calls()), before)
        self.assertTrue(json.loads((self.output / "state/jobs/data.json").read_text())["release_pending"])

    def test_unknown_held_root_disposition_stops_before_new_submission_or_release(self):
        self.interrupt_registered_wave()
        fixture = self.mock / "job_100.json"
        row = json.loads(fixture.read_text())
        row["reason"] = "JobHeldAdmin"
        fixture.write_text(json.dumps(row))
        result = self.run_workflow("--yes", "--submit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexplained scheduler disposition", result.stderr)
        self.assertIn("bash run.sh --yes --submit", result.stderr)
        self.assertEqual(len(self.calls("sbatch")), 2)
        self.assertFalse(self.release_calls())

    def test_unknown_queue_status_keeps_partial_wave_stopped(self):
        self.interrupt_registered_wave()
        rows = [(path.stem.split("_")[1], json.loads(path.read_text())) for path in sorted(self.mock.glob("job_*.json"))]
        queue = "".join(identifier + "|" + row["job-name"] + "|UNKNOWN\n" for identifier, row in rows)
        result = self.run_workflow("--yes", "--submit", extra={"FAKE_QUEUE": queue})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown scheduler state", result.stderr)
        self.assertIn("bash run.sh --yes --submit", result.stderr)
        self.assertEqual(len(self.calls("sbatch")), 2)
        self.assertFalse(self.release_calls())

    def test_unbound_held_wave_fails_closed_without_submitting_missing_jobs(self):
        self.interrupt_registered_wave()
        # An unrecognized old journal cannot authorize guessing the original
        # wave. Preserve it and stop for an explicit provenance inspection.
        journal = self.output / "state/journals/data.json"
        row = json.loads(journal.read_text())
        row.pop("wave_keys")
        journal.write_text(json.dumps(row))
        result = self.run_workflow("--yes", "--submit")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("wave binding", result.stderr)
        self.assertEqual(len(self.calls("sbatch")), 2)
        self.assertFalse(self.release_calls())

    def test_legacy_held_empty_plan_is_preserved_and_stops_for_missing_wave_binding(self):
        self.assertNotEqual(self.run_workflow("--yes", "--submit", extra={"FAKE_RELEASE_FAIL": "1"}).returncode, 0)
        job_path = self.output / "state/jobs/data.json"
        job = json.loads(job_path.read_text())
        for key in ("release_pending", "release_state", "readback_verified"):
            job.pop(key, None)
        job_path.write_text(json.dumps(job))
        journal = self.output / "state/journals/data.json"
        row = json.loads(journal.read_text())
        row.pop("wave_keys")
        journal.write_text(json.dumps(row))
        before = len(self.release_calls())
        result = self.run_workflow("--yes", "--submit")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["jobs"], [])
        self.assertEqual(json.loads(result.stdout)["existing_jobs"]["data"]["status"], "held")
        self.assertIn("wave binding", result.stderr)
        self.assertEqual(len(self.calls("sbatch")), 4)
        self.assertEqual(len(self.release_calls()), before)

    def test_lost_release_response_accepts_only_receipt_proven_dependency_clearing(self):
        self.assertNotEqual(self.run_workflow("--yes", "--submit", extra={"FAKE_RELEASE_LOST_RESPONSE": "1"}).returncode, 0)
        root = json.loads((self.output / "state/jobs/data.json").read_text())
        receipts = self.output / "state/receipts"
        receipts.mkdir()
        (receipts / "data.json").write_text(json.dumps(dict(status="complete", source_manifest_sha256=root["spec"]["source_manifest_sha256"])))
        bank_path = self.mock / "job_101.json"
        bank = json.loads(bank_path.read_text())
        bank["job_state"] = "RUNNING"
        bank["dependency"] = "(null)"
        bank_path.write_text(json.dumps(bank))
        self.assert_ok(self.run_workflow("--yes", "--submit"))
        self.assertEqual(len(self.calls("sbatch")), 4)
        self.assertEqual(len(self.release_calls()), 1)
        root = json.loads((self.output / "state/jobs/data.json").read_text())
        self.assertEqual(root["status"], "complete")
        self.assertEqual(root["release_state"], "released")

    def test_unproven_dependency_clearing_or_new_edge_stops_lost_release_recovery(self):
        self.assertNotEqual(self.run_workflow("--yes", "--submit", extra={"FAKE_RELEASE_LOST_RESPONSE": "1"}).returncode, 0)
        bank_path = self.mock / "job_101.json"
        bank = json.loads(bank_path.read_text())
        for dependency in ("(null)", "afterok:100:999"):
            with self.subTest(dependency=dependency):
                bank["dependency"] = dependency
                bank_path.write_text(json.dumps(bank))
                failed = self.run_workflow("--yes", "--submit")
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn("dependency binding differs", failed.stderr)
                self.assertEqual(len(self.calls("sbatch")), 4)
                self.assertEqual(len(self.release_calls()), 1)

    def test_multiple_roots_resume_after_one_released_and_the_next_release_failed(self):
        preview = self.run_workflow("--yes")
        self.assert_ok(preview)
        plan = json.loads(preview.stdout)
        first = plan["jobs"][0]
        second = dict(first, key="independent", job_name=plan["run_prefix"] + "independent")
        plan["jobs"] = [first, second]
        with mock.patch.dict(os.environ, dict(self.env, FAKE_RELEASE_FAIL_ID="101"), clear=True), \
                mock.patch.object(self.workflow, "stage_plan", return_value=plan), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Release stopped"):
            self.workflow.main(["--yes", "--submit"])
        self.assertEqual(len(self.calls("sbatch")), 2)
        root0 = json.loads((self.output / "state/jobs/data.json").read_text())
        root1 = json.loads((self.output / "state/jobs/independent.json").read_text())
        self.assertEqual(root0["release_state"], "released")
        self.assertEqual(root1["release_state"], "requested")
        empty = dict(plan, jobs=[])
        with mock.patch.dict(os.environ, self.env, clear=True), \
                mock.patch.object(self.workflow, "stage_plan", return_value=empty), contextlib.redirect_stdout(io.StringIO()):
            self.workflow.main(["--yes", "--submit"])
        self.assertEqual(len(self.calls("sbatch")), 2)
        self.assertEqual([row["args"][-1] for row in self.release_calls()], ["100", "101", "101"])
        for key in ("data", "independent"):
            job = json.loads((self.output / "state/jobs" / (key + ".json")).read_text())
            self.assertEqual(job["release_state"], "released")
            self.assertFalse(job["release_pending"])

    def test_bundle_does_not_release_a_waiting_training_wave(self):
        self.assertNotEqual(self.run_workflow("--yes", "--submit", extra={"FAKE_RELEASE_FAIL": "1"}).returncode, 0)
        before = len(self.release_calls())
        self.assert_ok(self.run_workflow("--yes", "--submit", "--bundle"))
        self.assertEqual(len(self.calls("sbatch")), 5)
        self.assertEqual([row["args"][-1] for row in self.release_calls()[before:]], ["104"])
        root = json.loads((self.output / "state/jobs/data.json").read_text())
        self.assertEqual(root["status"], "held")
        self.assertTrue(root["release_pending"])

    def test_completed_bundle_reconciles_lost_release_response_from_bound_receipt(self):
        self.submit_initial()
        first = self.run_workflow("--yes", "--submit", "--bundle", extra={"FAKE_RELEASE_LOST_RESPONSE": "1"})
        self.assertNotEqual(first.returncode, 0)
        state = self.output / "state"
        jobs = self.workflow.existing_jobs(state)
        key = next(key for key in jobs if key.startswith("bundle_"))
        job = jobs[key]
        self.assertTrue(job["release_pending"])
        self.assertEqual(job["release_state"], "requested")
        self.workflow.atomic_json(state / "receipts" / (key + ".json"),
                                  dict(status="complete", source_manifest_sha256=job["spec"]["source_manifest_sha256"]))
        fixture = self.mock / ("job_" + job["job_id"] + ".json")
        row = json.loads(fixture.read_text())
        row["job_state"] = "COMPLETED"
        fixture.write_text(json.dumps(row))
        submits, releases = len(self.calls("sbatch")), len(self.release_calls())
        self.assert_ok(self.run_workflow("--yes", "--submit", "--bundle"))
        self.assertEqual(len(self.calls("sbatch")), submits)
        self.assertEqual(len(self.release_calls()), releases)
        final = json.loads((state / "jobs" / (key + ".json")).read_text())
        self.assertEqual(final["status"], "complete")
        self.assertFalse(final["release_pending"])

    def test_explicit_site_overrides_are_respected(self):
        self.submit_initial(ACCOUNT="chosen", PARTITION="compute", NODES="2", GPUS_PER_NODE="4",
                            CPUS_PER_TASK="4", MEM_PER_NODE="64G", MAX_WALLTIME="12:00:00")
        site = (self.output / "state/site.env").read_text()
        self.assertIn("chosen", site)
        self.assertIn("compute", site)
        rows = self.job_rows()
        self.assertTrue(all(row["account"] == "chosen" for row in rows))

    def test_manual_32_gpu_layout_preserves_16_gpu_automatic_default(self):
        automatic = self.run_workflow("--yes")
        self.assert_ok(automatic)
        site = json.loads(automatic.stdout)["site"]
        self.assertEqual(int(site["NODES"]) * int(site["GPUS_PER_NODE"]), 16)
        manual = self.run_workflow("--yes", extra={"NODES": "8", "GPUS_PER_NODE": "4", "FAKE_GPU_NODES": "8"})
        self.assert_ok(manual)
        plan = json.loads(manual.stdout)
        self.assertEqual(plan["mode"], "PLAN_ONLY")
        self.assertEqual(int(plan["site"]["NODES"]) * int(plan["site"]["GPUS_PER_NODE"]), 32)
        smoke = next(job for job in plan["jobs"] if job["stage"] == "smoke")
        self.assertEqual(smoke["resources"]["nodes"], 8)
        self.assertEqual(smoke["resources"]["gpus_per_node"], 4)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls("sbatch"))

    def test_default_layout_counts_unique_physical_gpu_nodes(self):
        result = self.run_workflow("--yes", extra={"FAKE_GPU_NODES": "1"})
        self.assert_ok(result)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["site"]["NODES"], "1")
        self.assertEqual(plan["site"]["GPUS_PER_NODE"], "4")
        self.assertEqual(plan["checks"]["resource_inventory"]["compatible_gpu_nodes"], ["gpu0"])
        inventory = "compute|only_gpu|gpu:h100:4|32|131072|1-00:00:00\n" * 4 + "cpu|cpu0|N/A|32|131072|2-00:00:00\n"
        repeated = self.run_workflow("--yes", extra={"FAKE_SINFO": inventory})
        self.assert_ok(repeated)
        self.assertEqual(json.loads(repeated.stdout)["site"]["NODES"], "1")
        self.assertFalse(self.output.exists())

    def test_cpu_default_skips_partition_shorter_than_3900_seconds(self):
        self.env.pop("CPU_PARTITION")
        inventory = "compute|gpu0|gpu:h100:4|32|131072|1-00:00:00\na_short|cpu0|N/A|32|131072|00:30:00\nz_long|cpu1|N/A|32|131072|2-00:00:00\n"
        result = self.run_workflow("--yes", extra={"FAKE_SINFO": inventory})
        self.assert_ok(result)
        self.assertEqual(json.loads(result.stdout)["site"]["CPU_PARTITION"], "z_long")
        self.assertFalse(self.output.exists())

    def test_inventory_preserves_unlimited_and_finite_partition_limits(self):
        inventory = ("compute|gpu0|gpu:h100:4|32|131072|infinite\n"
                     "compute|gpu1|gpu:h100:4|32|131072|UNLIMITED\n"
                     "finite|gpu2|gpu:h100:4|32|131072|1-00:00:00\n"
                     "cpu|cpu0|N/A|32|131072|2-00:00:00\n")
        with mock.patch.dict(os.environ, dict(self.env, FAKE_SINFO=inventory), clear=True):
            rows = {row["node"]: row for row in self.workflow.partition_inventory()}
        self.assertEqual(set(rows), {"gpu0", "gpu1", "gpu2", "cpu0"})
        for node in ("gpu0", "gpu1"):
            self.assertIsNone(rows[node]["limit_seconds"])
            self.assertIs(rows[node]["limit_unlimited"], True)
        self.assertEqual(rows["gpu2"]["limit_seconds"], 86400)
        self.assertIs(rows["gpu2"]["limit_unlimited"], False)
        self.assertFalse(self.calls("sbatch"))

    def test_unlimited_gpu_and_cpu_discovery_pass_preflight_with_finite_requests(self):
        self.env.pop("CPU_PARTITION")
        for limit in ("infinite", "UNLIMITED", "InFiNiTe"):
            with self.subTest(limit=limit):
                inventory = ("compute|gpu0|gpu:h100:4|32|131072|" + limit + "\n"
                             "cpu|cpu0|N/A|32|131072|" + limit + "\n")
                result = self.run_workflow("--yes", extra={"FAKE_SINFO": inventory})
                self.assert_ok(result)
                plan = json.loads(result.stdout)
                self.assertEqual(plan["mode"], "PLAN_ONLY")
                self.assertEqual(plan["site"]["NODES"], "1")
                self.assertEqual(plan["site"]["GPUS_PER_NODE"], "4")
                self.assertEqual(self.workflow.duration(plan["site"]["MAX_WALLTIME"]), 86400)
                self.assertEqual(plan["site"]["CPU_PARTITION"], "cpu")
                checked = plan["checks"]["resource_inventory"]
                self.assertEqual(checked["compatible_gpu_nodes"], ["gpu0"])
                self.assertEqual(checked["compatible_cpu_nodes"], ["cpu0"])
                self.assertTrue(all(job["resources"]["walltime_seconds"] <= 86400
                                    for job in plan["jobs"]))
                self.assertFalse(self.output.exists())
                self.assertFalse(self.calls("sbatch"))

    def test_unlimited_partition_accepts_explicit_finite_walltime_override(self):
        for limit in ("infinite", "UNLIMITED"):
            with self.subTest(limit=limit):
                inventory = ("compute|gpu0|gpu:h100:4|32|131072|" + limit + "\n"
                             "cpu|cpu0|N/A|32|131072|2-00:00:00\n")
                extra = dict(FAKE_SINFO=inventory, ACCOUNT="chosen", PARTITION="compute",
                             NODES="1", GPUS_PER_NODE="4", MAX_WALLTIME="24:00:00")
                result = self.run_workflow("--yes", extra=extra)
                self.assert_ok(result)
                plan = json.loads(result.stdout)
                self.assertEqual(plan["site"]["MAX_WALLTIME"], "24:00:00")
                self.assertEqual(plan["checks"]["resource_inventory"]["compatible_gpu_nodes"], ["gpu0"])
                self.assertEqual(plan["checks"]["requested_capacity"]["gpu_walltime_seconds"], 86400)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.calls("sbatch"))

    def test_explicit_overrides_exceeding_physical_capacity_fail_before_reservation(self):
        inventory = "compute|gpu0|gpu:h100:4|32|131072|1-00:00:00\ncpu|cpu0|N/A|32|131072|2-00:00:00\nshort|cpu1|N/A|32|131072|00:30:00\n"
        cases = ({"NODES": "4"}, {"GPUS_PER_NODE": "8", "NODES": "1"},
                 {"CPUS_PER_TASK": "64"}, {"MEM_PER_NODE": "256G"},
                 {"MAX_WALLTIME": "2-00:00:00"}, {"CPU_PARTITION": "short"})
        for extra in cases:
            with self.subTest(extra=extra):
                result = self.run_workflow("--yes", "--submit", extra=dict(FAKE_SINFO=inventory, **extra))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())
                self.assertFalse(self.calls("sbatch"))
                evidence = Path(result.stderr.strip().rsplit("Gate evidence: ", 1)[1])
                gate = json.loads(evidence.read_text())
                self.assertIn("requested_capacity", gate)
                self.assertIn("physical_nodes", gate["resource_inventory"])

    def test_queue_collision_fails_before_reserving_root(self):
        # A duplicate identity must be rejected before site/journal reservation.
        first = self.run_workflow("--yes")
        self.assert_ok(first)
        plan = json.loads(first.stdout)
        prefix = plan["run_prefix"]
        self.assertTrue(prefix, "PLAN_ONLY must expose its unique job-name prefix")
        result = self.run_workflow("--yes", "--submit", extra={"FAKE_QUEUE": "100|" + prefix + "-data|RUNNING\n"})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls("sbatch"))

    def test_failed_accounting_stage_stops_without_retry(self):
        self.submit_initial()
        jobs = list((self.output / "state/jobs").glob("*.json"))
        row = json.loads(jobs[0].read_text())
        before = len(self.calls("sbatch"))
        accounting = "".join(str(json.loads(path.read_text())["job_id"]) + "|FAILED|\n" for path in jobs)
        result = self.run_workflow("--yes", "--submit", extra={"FAKE_QUEUE": "", "FAKE_ACCOUNTING": accounting})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.calls("sbatch")), before)
        self.assertIn("fail", (result.stdout + result.stderr).lower())

    def test_queue_requires_both_job_id_and_expected_job_name(self):
        self.submit_initial()
        job = json.loads((self.output / "state/jobs/data.json").read_text())
        before = len(self.calls("sbatch"))
        result = self.run_workflow("--yes", "--submit", extra={
            "FAKE_QUEUE": str(job["job_id"]) + "|unrelated_name|RUNNING\n"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("JobName binding differs", result.stderr)
        self.assertEqual(len(self.calls("sbatch")), before)

    def test_root_is_held_and_bad_readback_never_released(self):
        result = self.run_workflow("--yes", "--submit", extra={"FAKE_BAD_READBACK": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.calls("sbatch")), 1)
        self.assertIn("--hold", self.calls("sbatch")[0]["args"])
        self.assertFalse(any(row["args"][0] == "release" for row in self.calls("scontrol")))
        repeated = self.run_workflow("--yes", "--submit")
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(len(self.calls("sbatch")), 1)

    def test_node_ranges_and_colon_slash_readback_fields_are_accepted(self):
        self.submit_initial(NODES="1", FAKE_NUM_NODES="1-1")
        self.assertEqual(len(self.calls("sbatch")), 4)
        self.assertTrue(any(row["args"][0] == "release" for row in self.calls("scontrol")))

    def test_generic_gpu_requires_explicit_confirmation_and_uses_untyped_requests(self):
        # Untyped scheduling information cannot itself certify H100 hardware.
        unknown = self.run_workflow("--yes", extra={"FAKE_GRES": "gpu:8"})
        self.assertNotEqual(unknown.returncode, 0)
        self.assertFalse(self.output.exists())
        for index, spelling in enumerate(("generic", "")):
            with self.subTest(gpu_type=spelling):
                self.output = self.root / ("generic_output_" + str(index))
                self.env["OUTPUT_ROOT"] = str(self.output)
                self.submit_initial(FAKE_GRES="gpu:8", GPU_TYPE=spelling)
                site = json.loads((self.output / "state/site.json").read_text())
                self.assertEqual(site["GPU_TYPE"], "generic")
                self.assertEqual(site["NODES"], "2")
                rows = [row for row in self.job_rows() if row["job-name"].endswith("smoke")]
                self.assertEqual(rows[-1]["gpus-per-node"], "8")
                self.assertNotIn("--gpus-per-node=:8", rows[-1]["argv"])
                before = len(self.calls("sbatch"))
                repeated = self.run_workflow("--yes", "--submit", extra={"FAKE_GRES": "gpu:8", "GPU_TYPE": spelling})
                self.assert_ok(repeated)
                self.assertEqual(len(self.calls("sbatch")), before)

    def test_afterok_readback_accepts_both_forms_and_rejects_or_and_extra_dependencies(self):
        self.submit_initial()
        job = json.loads((self.output / "state/jobs/bank.json").read_text())
        stored = json.loads((self.output / "state/readbacks/bank.json").read_text())
        original = stored["text"]
        for dependency in ("afterok:101:102", "afterok:102(unfulfilled),afterok:101(unfulfilled)",
                           "afterok:101:102(unfulfilled)"):
            with self.subTest(dependency=dependency):
                text = original.replace("Dependency=" + self.workflow.parse_scontrol(original)["Dependency"],
                                        "Dependency=" + dependency)
                self.workflow.validate_scontrol(text, job["spec"], job["job_id"], ["101", "102"])
        for dependency in ("afterok:101?afterok:102", "afterok:101,afterany:102",
                           "afterok:101:102:103", "afterok:101(unfulfilled)",
                           "afterok:101:102,afterok:999", "afterok:101:102(unknown)"):
            with self.subTest(dependency=dependency):
                text = original.replace("Dependency=" + self.workflow.parse_scontrol(original)["Dependency"],
                                        "Dependency=" + dependency)
                with self.assertRaisesRegex(RuntimeError, "Dependency"):
                    self.workflow.validate_scontrol(text, job["spec"], job["job_id"], ["101", "102"])

    def test_readback_task_cpu_memory_mismatch_stops_held_wave(self):
        for index, extra in enumerate(({"FAKE_NUMTASKS": "2"}, {"FAKE_CPU": "1"}, {"FAKE_MEM": "1M"})):
            with self.subTest(extra=extra):
                self.output = self.root / ("bad_resources_" + str(index))
                self.env["OUTPUT_ROOT"] = str(self.output)
                before = len(self.calls("sbatch"))
                result = self.run_workflow("--yes", "--submit", extra=extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(self.calls("sbatch")), before + 1)
        self.assertFalse(any(row["args"][0] == "release" for row in self.calls("scontrol")))

    def test_status_empty_run_is_read_only(self):
        result = subprocess.run(["bash", str(self.code / "status.sh"), "--output-root", str(self.output)],
                                cwd=ROOT, env=self.env, text=True, capture_output=True, timeout=30)
        self.assert_ok(result)
        self.assertFalse(self.output.exists())
        self.assertIn("progress", result.stdout.lower())

    def complete_preparation(self, *, seconds_per_step=2.0):
        directory = self.output / "state/receipts"
        directory.mkdir(parents=True, exist_ok=True)
        for key in ("data", "bank", "smoke"):
            job = json.loads((self.output / "state/jobs" / (key + ".json")).read_text())
            receipt = dict(status="complete", source_manifest_sha256=job["spec"]["source_manifest_sha256"])
            if key == "smoke":
                receipt.update(validated=True, seconds_per_step=seconds_per_step, startup_seconds=5.0,
                               checkpoint_seconds=7.0, peak_memory_bytes=1024)
            (directory / (key + ".json")).write_text(json.dumps(receipt))

    def test_measured_smoke_builds_bounded_chain_and_exact_readouts(self):
        self.submit_initial(FAKE_REJECT_ADVANCE="1", MAX_WALLTIME="01:00:00")
        self.complete_preparation(seconds_per_step=5.0)
        result = self.run_workflow("--yes", extra={"MAX_WALLTIME": "01:00:00"})
        self.assert_ok(result)
        plan = json.loads(result.stdout)
        measured = plan["measured_segmentation"]
        self.assertEqual(measured["steps_per_segment"], int((3600 - 12 - 240) // (5 * 1.15)))
        self.assertEqual(measured["signal_budget_seconds"], 240)
        trains = [job for job in plan["jobs"] if job["stage"] == "train"]
        self.assertEqual(trains[0]["start_step"], 0)
        self.assertEqual(trains[-1]["end_step"], 30000)
        for previous, following in zip(trains, trains[1:]):
            self.assertEqual(previous["end_step"], following["start_step"])
        self.assertTrue(all(job["resources"]["walltime_seconds"] <= 3600 for job in trains))
        self.assertEqual([job["readout_step"] for job in plan["jobs"] if job["stage"] == "eval"],
                         [1000, 2000, 4000, 8000, 12000, 16000, 24000, 30000])
        self.assertGreater(plan["production_h100_hours"], 0)
        self.assertTrue(all(job["resources"]["walltime_seconds"] % 60 == 0 for job in trains))
        expected_hours = sum(job["resources"]["walltime_seconds"] for job in trains) * 16 / 3600
        self.assertAlmostEqual(plan["production_h100_hours"], expected_hours)

    def test_walltime_rounding_binds_budget_qos_and_partition_limit(self):
        self.assertEqual(self.workflow.walltime(2312), "00:39:00")
        self.assertEqual(self.workflow.requested_seconds(2312), 2340)
        site = {"CPU_TIME": "01:05:00", "MAX_WALLTIME": "02:00:00", "CPU_QOS": "cpu",
                "GPU_QOS_SHORT": "short", "GPU_QOS_LONG": "long", "CPUS_PER_TASK": "8",
                "CPU_CPUS_PER_TASK": "4", "MEM_PER_NODE": "128G", "CPU_MEM_PER_NODE": "32G",
                "NODES": "4", "GPUS_PER_NODE": "4", "PARTITION": "compute", "CPU_PARTITION": "cpu",
                "ACCOUNT": "demo", "GPU_TYPE": "h100"}
        requested = self.workflow.resources(site, "train", 3601)
        self.assertEqual(requested["walltime_seconds"], 3660)
        self.assertEqual(requested["qos"], "long")
        self.assertEqual(self.workflow.resources(site, "train", 3599)["qos"], "short")
        self.assertEqual(self.workflow.resources(site, "train", 3600)["qos"], "short")
        smoke = dict(validated=True, seconds_per_step=1, startup_seconds=0, checkpoint_seconds=0)
        plan = self.workflow.segment_plan(smoke, 301)
        self.assertEqual(plan["schedulable_seconds"], 300)
        self.assertTrue(all(job["walltime_seconds"] == 300 for job in plan["jobs"]))
        self.assertEqual(plan["jobs"][0]["end_step"], 52)
        with self.assertRaises(ValueError):
            self.workflow.segment_plan(smoke, 299)
        with self.assertRaisesRegex(ValueError, "Rounded GPU walltime"):
            self.workflow.resources(dict(site, MAX_WALLTIME="01:00:01"), "train", 3601)

    def test_signal_budget_protects_short_tail_before_earliest_usr1(self):
        smoke = dict(validated=True, seconds_per_step=10, startup_seconds=600,
                     checkpoint_seconds=7)
        plan = self.workflow.segment_plan(smoke, 3600, start=29975)
        self.assertEqual(plan["signal_budget_seconds"], 240)
        self.assertEqual(len(plan["jobs"]), 1)
        tail = plan["jobs"][0]
        self.assertEqual((tail["start_step"], tail["end_step"]), (29975, 30000))
        self.assertEqual(tail["walltime_seconds"], 1140)
        nominal_seconds = 600 + 25 * 10 + 7
        earliest_signal_seconds = tail["walltime_seconds"] - 180 - 60
        self.assertGreaterEqual(earliest_signal_seconds, nominal_seconds)

    def test_signal_budget_limits_capacity_and_every_segment_after_rounding(self):
        smoke = dict(validated=True, seconds_per_step=10, startup_seconds=600,
                     checkpoint_seconds=7)
        plan = self.workflow.segment_plan(smoke, 3601)
        self.assertEqual(plan["steps_per_segment"], int((3600 - 600 - 7 - 240) // 11.5))
        self.assertEqual(plan["schedulable_seconds"], 3600)
        for job in plan["jobs"]:
            with self.subTest(start=job["start_step"], end=job["end_step"]):
                updates = job["end_step"] - job["start_step"]
                measured_with_margin = updates * 10 * 1.15 + 600 + job["checkpoint_saves"] * 7
                self.assertGreaterEqual(job["walltime_seconds"], measured_with_margin + 240)
                self.assertEqual(job["walltime_seconds"] % 60, 0)
                self.assertLessEqual(job["walltime_seconds"], 3600)
        self.assertEqual(plan["jobs"][-1]["end_step"], 30000)

    def test_signal_budget_rejects_partition_that_cannot_fit_one_update(self):
        smoke = dict(validated=True, seconds_per_step=10, startup_seconds=600,
                     checkpoint_seconds=7)
        for maximum in (240, 600, 899):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                self.workflow.segment_plan(smoke, maximum, start=29999)
        plan = self.workflow.segment_plan(smoke, 900, start=29999)
        self.assertEqual(plan["jobs"][0]["walltime_seconds"], 900)
        self.assertEqual(plan["jobs"][0]["end_step"], 30000)

    def test_unusable_partition_time_persists_gate_numbers_before_reserving_root(self):
        result = self.run_workflow("--yes", "--submit", extra={"MAX_WALLTIME": "00:00:59"})
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls("sbatch"))
        evidence = Path(result.stderr.strip().rsplit("Gate evidence: ", 1)[1])
        gate = json.loads(evidence.read_text())
        self.assertEqual(gate["walltime_limits_seconds"]["MAX_WALLTIME"], 59)
        self.assertEqual(gate["gpu_maximum_seconds"], 59)
        self.assertEqual(gate["schedulable_gpu_seconds"], 0)
        self.assertEqual(gate["minute_seconds"], 60)

    def test_rounded_requests_pass_minute_normalizing_scheduler_and_multi_parent_summary(self):
        self.submit_initial(FAKE_REJECT_ADVANCE="1")
        self.complete_preparation(seconds_per_step=2.0)
        self.assert_ok(self.run_workflow("--yes", "--submit"))
        rows = self.job_rows()
        trains = [row for row in rows if "_train_" in row["job-name"]]
        self.assertEqual(trains[0]["time"], "00:43:00")
        summary = next(row for row in rows if row["job-name"].endswith("summary"))
        self.assertEqual(len(summary["dependency"].split(":")) - 1, 8)
        self.assertTrue(any(row["args"][0] == "release" for row in self.calls("scontrol")))

    def test_nested_submission_unavailable_keeps_production_journals_unreserved(self):
        self.submit_initial(FAKE_REJECT_ADVANCE="1")
        self.complete_preparation()
        before = {path.name for path in (self.output / "state/journals").iterdir()}
        result = self.run_workflow("--yes", "--submit", "--advance",
                                   extra={"SLURM_JOB_ID": "999", "FAKE_REJECT_TEST_ONLY": "1"})
        self.assertIn(result.returncode, (0, 69), result.stdout + result.stderr)
        fallback = json.loads((self.output / "state/advance_fallback.json").read_text())
        self.assertEqual(fallback["auto_advance"], "manual_required")
        self.assertIn("bash run.sh", fallback["resume_command"])
        self.assertEqual({path.name for path in (self.output / "state/journals").iterdir()}, before)
        manual = self.run_workflow("--yes", "--submit")
        self.assert_ok(manual)
        self.assertGreater(len(list((self.output / "state/journals").iterdir())), len(before))
        # The first production root uses verified smoke evidence, without
        # afterok on the already completed smoke scheduler job.
        first = next(row for row in self.job_rows() if row["job-name"].endswith("train_000000_001000"))
        self.assertNotIn("dependency", first)
        self.assertIn("--hold", first["argv"])

    def test_missing_or_timed_out_nested_probe_uses_manual_path_without_journal(self):
        self.submit_initial(FAKE_REJECT_ADVANCE="1")
        self.complete_preparation()
        before = {path.name for path in (self.output / "state/journals").iterdir()}
        original = self.workflow.command
        for failure in (FileNotFoundError("sbatch unavailable"), subprocess.TimeoutExpired("sbatch --test-only", 45)):
            def probe(command, **kwargs):
                if command[0] == "sbatch" and "--test-only" in command:
                    raise failure
                return original(command, **kwargs)
            with self.subTest(failure=type(failure).__name__), mock.patch.dict(os.environ, dict(self.env, SLURM_JOB_ID="999"), clear=True), \
                    mock.patch.object(self.workflow, "command", side_effect=probe), contextlib.redirect_stdout(io.StringIO()):
                result = self.workflow.main(["--yes", "--submit", "--advance"])
                self.assertEqual(result, 69)
            fallback = json.loads((self.output / "state/advance_fallback.json").read_text())
            self.assertEqual(fallback["auto_advance"], "manual_required")
            self.assertEqual({path.name for path in (self.output / "state/journals").iterdir()}, before)
        self.assert_ok(self.run_workflow("--yes", "--submit"))

    def test_actual_nested_policy_rejection_retains_journal_and_allows_one_manual_continuation(self):
        self.submit_initial(FAKE_REJECT_ADVANCE="1")
        self.complete_preparation()
        nested = dict(SLURM_JOB_ID="999", FAKE_REJECT_NESTED="1")
        result = self.run_workflow("--yes", "--submit", "--advance", extra=nested)
        self.assertEqual(result.returncode, 69, result.stdout + result.stderr)
        key = "train_000000_001000"
        record = json.loads((self.output / "state/jobs" / (key + ".json")).read_text())
        self.assertEqual(record["status"], "manual_required")
        self.assertTrue(record["no_job_created"])
        self.assertIsNone(record["job_id"])
        journal = self.output / "state/journals" / (key + ".json")
        digest = hashlib.sha256(journal.read_bytes()).hexdigest()
        count = len(self.calls("sbatch"))
        self.assertEqual(self.run_workflow("--yes", "--submit", "--advance", extra=nested).returncode, 69)
        self.assertEqual(len(self.calls("sbatch")), count)
        manual = self.run_workflow("--yes", "--submit")
        self.assert_ok(manual)
        continuation = json.loads((self.output / "state/jobs" / (key + "_c001.json")).read_text())
        self.assertEqual(continuation["spec"]["continuation_of"], key)
        self.assertEqual(hashlib.sha256(journal.read_bytes()).hexdigest(), digest)
        self.assertTrue((self.output / "state/journals" / (key + "_c001.json")).is_file())
        after = len(self.calls("sbatch"))
        self.assert_ok(self.run_workflow("--yes", "--submit"))
        self.assertEqual(len(self.calls("sbatch")), after)
        evaluation = json.loads((self.output / "state/jobs/eval_001000.json").read_text())
        self.assertEqual(evaluation["spec"]["parent_key"], key + "_c001")

    def test_nested_nonzero_socket_timeout_stays_ambiguous_and_blocks_manual_retry(self):
        self.submit_initial(FAKE_REJECT_ADVANCE="1")
        self.complete_preparation()
        result = self.run_workflow("--yes", "--submit", "--advance", extra={
            "SLURM_JOB_ID": "999", "FAKE_AMBIGUOUS_SUBMIT": "1"})
        self.assertNotEqual(result.returncode, 69)
        self.assertNotEqual(result.returncode, 0)
        key = "train_000000_001000"
        record = json.loads((self.output / "state/jobs" / (key + ".json")).read_text())
        self.assertEqual(record["status"], "submission_failed")
        self.assertIsNone(record["job_id"])
        before = len(self.calls("sbatch"))
        self.assertNotEqual(self.run_workflow("--yes", "--submit").returncode, 0)
        self.assertEqual(len(self.calls("sbatch")), before)
        self.assertFalse((self.output / "state/journals" / (key + "_c001.json")).exists())

    def test_real_submission_timeout_is_ambiguous_and_never_retried(self):
        self.submit_initial(FAKE_REJECT_ADVANCE="1")
        self.complete_preparation()
        original = self.workflow.command
        def timeout(command, **kwargs):
            if command[0] == "sbatch":
                raise subprocess.TimeoutExpired("sbatch", 45)
            return original(command, **kwargs)
        with mock.patch.dict(os.environ, self.env, clear=True), mock.patch.object(self.workflow, "command", side_effect=timeout), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Submission stopped"):
            self.workflow.main(["--yes", "--submit"])
        record = json.loads((self.output / "state/jobs/train_000000_001000.json").read_text())
        self.assertEqual(record["status"], "submission_failed")
        self.assertIsNone(record["job_id"])
        self.assertTrue((self.output / "state/journals/train_000000_001000.json").is_file())
        before = len(self.calls("sbatch"))
        repeated = self.run_workflow("--yes", "--submit")
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(len(self.calls("sbatch")), before)

    def test_status_table_and_bundle_include_scores_and_pending_params(self):
        self.submit_initial()
        self.complete_preparation()
        def write(name, value):
            path = self.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        write("train/progress.json", dict(step=1000, seconds=2.0))
        write("readouts/step_001000/geneval/scores/geneval.summary.json", dict(overall_category_balanced=.42))
        write("state/receipts/eval_001000.json", dict(status="complete", benchmarks={"dpg": {"status": "scoring pending"}}))
        write("train/checkpoints/step_001000/checkpoint.json", dict(step=1000))
        params = self.output / "train/checkpoints/step_001000/params.msgpack"
        params.write_bytes(b"toy params")
        sample = self.output / "readouts/step_001000/samples.png"
        sample.write_bytes(b"unused image fixture")
        result = subprocess.run(["bash", str(self.code / "status.sh"), "--output-root", str(self.output), "--bundle"],
                                cwd=self.code, env=self.env, text=True, capture_output=True, timeout=30)
        self.assert_ok(result)
        self.assertIn("1000/30000", result.stdout)
        self.assertIn("| 1000 | 0.4200 | scoring pending |", result.stdout)
        self.assertIn("D5:", result.stdout)
        self.assertTrue((self.output / "results/results.md").is_file())
        with tarfile.open(self.output / "results/return_bundle.tar") as archive:
            names = set(archive.getnames())
        self.assertIn("results/results.md", names)
        self.assertIn("results/return_manifest.json", names)
        self.assertIn("train/checkpoints/step_001000/params.msgpack", names)
        self.assertFalse(any(name.endswith(".png") for name in names))

    def test_explicit_recovery_pins_latest_manifest_and_preserves_old_journals(self):
        self.submit_initial(FAKE_REJECT_ADVANCE="1")
        self.complete_preparation()
        self.assert_ok(self.run_workflow("--yes", "--submit"))
        key = "train_000000_001000"
        old = json.loads((self.output / "state/jobs" / (key + ".json")).read_text())
        journals = self.output / "state/journals"
        old_digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in journals.iterdir()}
        checkpoint = self.output / "train/checkpoints/step_000500"
        checkpoint.mkdir(parents=True)
        manifest = checkpoint / "checkpoint.json"
        manifest.write_text(json.dumps(dict(schema="bigfield-checkpoint-v1", step=500, data_cursor=32000)))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        (self.output / "train/latest.json").write_text(json.dumps(dict(step=500, checkpoint=checkpoint.name, checkpoint_sha256=digest)))
        queue = "".join(path.stem.split("_")[1] + "|" + json.loads(path.read_text())["job-name"] + "|PENDING\n"
                        for path in sorted(self.mock.glob("job_*.json"))
                        if path.stem.split("_")[1] != str(old["job_id"]))
        extra = dict(FAKE_QUEUE=queue, FAKE_ACCOUNTING=str(old["job_id"]) + "|FAILED|\n")
        preview = self.run_workflow("--yes", "--recover", key, extra=extra)
        self.assert_ok(preview)
        plan = json.loads(preview.stdout)
        first = next(job for job in plan["jobs"] if job["stage"] == "train")
        self.assertEqual(first["start_step"], 500)
        self.assertEqual(first["parent_checkpoint"]["manifest_sha256"], digest)
        self.assertTrue(first["key"].endswith("_r001"))
        self.assertFalse((self.output / "state/recovery.json").exists())
        executed = self.run_workflow("--yes", "--submit", "--recover", key, extra=extra)
        self.assert_ok(executed)
        self.assertTrue((self.output / "state/recoveries/r001.json").is_file())
        for name, expected in old_digests.items():
            self.assertEqual(hashlib.sha256((journals / name).read_bytes()).hexdigest(), expected)
        new_root = next(row for row in self.job_rows() if row["job-name"].endswith("train_000500_001000_r001"))
        self.assertNotIn("dependency", new_root)

    def readout_recovery_fixture(self):
        self.submit_initial(FAKE_REJECT_ADVANCE="1")
        self.complete_preparation()
        self.assert_ok(self.run_workflow("--yes", "--submit"))
        state = self.output / "state"
        jobs = self.workflow.existing_jobs(state)
        site = json.loads((state / "site.json").read_text())
        source = jobs["eval_001000"]["spec"]["source_manifest_sha256"]
        checkpoint = dict(path=str(self.output / "train/checkpoints/step_001000"),
                          manifest_sha256="a" * 64, step=1000)
        producer = dict(status="complete", source_manifest_sha256=source, checkpoint=checkpoint)
        (state / "receipts/train_000000_001000.json").write_text(json.dumps(producer))
        jobs = {key: row for key, row in jobs.items()
                if key in ("data", "bank", "smoke", "train_000000_001000", "eval_001000")}
        recovery = dict(generation="r001", checkpoint=checkpoint)
        return site, state, jobs, source, recovery

    def test_recovery_reuses_pending_and_running_readout_with_original_summary_dependency(self):
        site, state, jobs, source, recovery = self.readout_recovery_fixture()
        for status in ("pending", "running"):
            with self.subTest(status=status):
                jobs["eval_001000"]["status"] = status
                plan = self.workflow.stage_plan(site, state, jobs, source, recovery=recovery)
                self.assertFalse(any(spec["stage"] == "eval" and spec["end_step"] == 1000
                                     for spec in plan["jobs"]))
                summary = next(spec for spec in plan["jobs"] if spec["stage"] == "summary")
                self.assertIn("eval_001000", summary["dependencies"])
                self.assertNotIn("eval_001000_r001", summary["dependencies"])
                bound = self.workflow.bind_dependencies(dict(summary, dependencies=["eval_001000"]), state, jobs)
                self.assertEqual(bound["dependency_receipts"]["eval_001000"]["producer"], "eval_001000")

    def test_recovery_readout_binding_mismatch_or_uncertain_state_stops(self):
        site, state, jobs, source, recovery = self.readout_recovery_fixture()
        evaluation = jobs["eval_001000"]
        evaluation["status"] = "running"
        for name, wrong in (("source_manifest_sha256", "b" * 64),
                            ("data_root", "/different/input"), ("output_root", "/different/output")):
            with self.subTest(binding=name):
                original = evaluation["spec"][name]
                evaluation["spec"][name] = wrong
                with self.assertRaisesRegex(RuntimeError, "binding differs"):
                    self.workflow.stage_plan(site, state, jobs, source, recovery=recovery)
                evaluation["spec"][name] = original
        for status in ("failed", "submission_failed", "uncertain"):
            evaluation["status"] = status
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, "failed or uncertain"):
                self.workflow.stage_plan(site, state, jobs, source, recovery=recovery)
        evaluation["status"] = "running"
        changed = dict(recovery, checkpoint=dict(recovery["checkpoint"], manifest_sha256="b" * 64))
        with self.assertRaisesRegex(RuntimeError, "checkpoint binding differs"):
            self.workflow.stage_plan(site, state, jobs, source, recovery=changed)

    def test_bundle_schedules_one_small_cpu_job_without_advancing_training(self):
        self.submit_initial()
        count = len(self.calls("sbatch"))
        first = self.run_workflow("--yes", "--bundle", "--submit")
        self.assert_ok(first)
        self.assertEqual(len(self.calls("sbatch")), count + 1)
        row = self.job_rows()[-1]
        self.assertTrue(row["job-name"].split("_")[-2] == "bundle")
        self.assertNotIn("gpus-per-node", row)
        self.assertNotIn("dependency", row)
        self.assertLessEqual(int(row["cpus-per-task"]), 1)
        repeated = self.run_workflow("--yes", "--bundle", "--submit")
        self.assert_ok(repeated)
        self.assertEqual(len(self.calls("sbatch")), count + 1)


if __name__ == "__main__":
    unittest.main()
