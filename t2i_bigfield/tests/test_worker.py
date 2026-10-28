"""Small filesystem tests of runtime bindings, failure evidence and return data."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import tarfile
import tempfile
import unittest
import subprocess
import sys
import types
from unittest.mock import patch

import worker


class WorkerTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get('TMPDIR', tempfile.gettempdir()))
        self.root = Path(tempfile.mkdtemp(prefix='bigfield_worker_', dir=base))
        self.state = self.root / 'state'
        self.state.mkdir()
        stat = self.root.stat()
        worker.atomic(self.state / 'binding.json', dict(
            source_manifest_sha256=worker.sha(worker.HERE / 'sources.sha256'),
            output_identity=dict(path=str(self.root), inode=stat.st_ino, uid=stat.st_uid)))

    def test_storage_failure_records_all_numbers_before_raise(self):
        with self.assertRaisesRegex(RuntimeError, 'Storage gate'):
            worker.storage_gate(self.state, 'toy', [self.root], min_bytes=1 << 100)
        gate = worker.read(self.state / 'gates/toy.storage.json')
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['observations'][0]['required_bytes'], 1 << 100)
        self.assertIn('available_inodes', gate['observations'][0])

    def test_nonfinite_gate_evidence_is_published(self):
        worker.atomic(self.state / 'numeric.json', dict(value=float('nan'), nested=[float('inf')]))
        self.assertEqual(worker.read(self.state / 'numeric.json'), dict(value='nan', nested=['inf']))

    def test_create_once_receipt_retains_first_content(self):
        path = self.state / 'receipt.json'
        worker.atomic(path, dict(value=1), once=True)
        with self.assertRaises(FileExistsError):
            worker.atomic(path, dict(value=2), once=True)
        self.assertEqual(worker.read(path), dict(value=1))

    def test_parent_metadata_sha_is_checked_before_payload_use(self):
        checkpoint = self.root / 'checkpoint'
        checkpoint.mkdir()
        worker.atomic(checkpoint / 'checkpoint.json', dict(schema='bigfield-checkpoint-v1', step=2))
        spec = dict(key='next', parent_checkpoint=dict(path=str(checkpoint), manifest_sha256='0' * 64))
        with self.assertRaisesRegex(ValueError, 'manifest SHA'):
            worker.parent_checkpoint(self.state, spec, {})
        self.assertFalse((self.state / 'bindings/next.parent.json').exists())

    def test_receipt_pinned_by_sha_and_source(self):
        worker.atomic(self.state / 'receipts/first.json', dict(status='complete', source_manifest_sha256='source'))
        worker.atomic(self.state / 'journals/first.json', dict(key='first'))
        spec = dict(key='next', dependencies=['first'], source_manifest_sha256='source',
                    dependency_receipts={'first': {'sha256': '0' * 64}})
        with self.assertRaisesRegex(ValueError, 'receipt SHA'):
            worker.dependency_bindings(self.state, spec)
        spec['dependency_receipts']['first']['sha256'] = worker.sha(self.state / 'receipts/first.json')
        result = worker.dependency_bindings(self.state, spec)
        self.assertEqual(result['first']['journal_sha256'], worker.sha(self.state / 'journals/first.json'))

    def test_smoke_rank_mismatch_publishes_failed_numeric_gate(self):
        spec = dict(key='smoke', smoke_output=str(self.root / 'smoke'),
                    resources={'nodes': 2, 'gpus_per_node': 4})
        def launch(state, spec, start, end, output, **kwargs):
            for rank in range(2):
                value = dict(complete=True, nodes=2, local_devices=4,
                             final=dict(step=end, digest='different' if rank else 'one'),
                             update_seconds=[10., 2.] if start == 0 else [3.],
                             startup_seconds=20., checkpoint_seconds=1., memory=[{'peak_bytes_in_use': 10}])
                worker.atomic(self.state / 'ranks' / ('smoke.%d-%d.rank%d.json' % (start, end, rank)), value)
        with patch.object(worker, 'launch_ranks', side_effect=launch), patch.object(worker, 'checkpoint_record', return_value={'path': 'toy'}):
            with self.assertRaisesRegex(ValueError, 'Smoke continuity'):
                worker.smoke_stage(self.state, spec)
        self.assertFalse(worker.read(self.state / 'gates/smoke.numeric.json')['validated'])

    def test_status_has_all_fixed_reads_and_no_invented_score(self):
        text = worker.status_text(self.root, refresh=False)
        for step in worker.READOUTS:
            self.assertIn('| %d |' % step, text)
        self.assertIn('pending smoke', text)
        self.assertNotIn('0.7951', text)

    def test_status_metric_names_and_dose_are_read_from_results(self):
        panel = self.root / 'readouts/step_001000'
        worker.atomic(panel / 'geneval/scores/geneval.summary.json', dict(overall_category_balanced=.75))
        worker.atomic(panel / 'dpg/scores/dpg.json', dict(fullrow_corrected_overall=81.5))
        worker.atomic(self.root / 'train/readouts/step_001000.dose.json', dict(status='PENDING'))
        text = worker.status_text(self.root, refresh=False)
        self.assertIn('| 1000 | 0.7500 | 81.5000 |', text)
        self.assertIn('D5:', text)

    def test_bundle_includes_pending_params_but_no_full_state_or_images(self):
        checkpoint = self.root / 'train/checkpoints/step_001000'
        checkpoint.mkdir(parents=True)
        worker.atomic(checkpoint / 'checkpoint.json', dict(step=1000))
        (checkpoint / 'params.msgpack').write_bytes(b'toy params')
        (checkpoint / 'state.msgpack').write_bytes(b'toy optimizer')
        images = self.root / 'readouts/step_001000/geneval/samples'
        images.mkdir(parents=True)
        (images / 'sample.png').write_bytes(b'toy image')
        worker.write_status(self.root, bundle=True)
        with tarfile.open(self.root / 'results/return_bundle.tar') as archive:
            names = archive.getnames()
        self.assertIn('train/checkpoints/step_001000/params.msgpack', names)
        self.assertIn('results/return_manifest.json', names)
        self.assertFalse(any(name.endswith(('state.msgpack', '.png')) for name in names))

    def test_heavy_stage_rejects_login_context(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'allocated'):
                worker.allocated()

    def copied_wrapper(self):
        spool = self.root / 'spool'
        spool.mkdir()
        wrapper = spool / 'slurm_script'
        wrapper.write_bytes((worker.HERE / 'worker.slurm').read_bytes())
        return wrapper

    def wrapper_environment(self):
        return dict(os.environ, SLURM_JOB_ID='fixture', SLURMD_NODENAME='fixture',
                    BIGFIELD_PYTHON='/bin/echo')

    def test_relocated_batch_wrapper_uses_bound_source_working_directory(self):
        wrapper = self.copied_wrapper()
        args = ['--stage', 'data', '--state', str(self.state), '--key', 'data']
        result = subprocess.run(['bash', str(wrapper)] + args, cwd=worker.HERE,
                                env=self.wrapper_environment(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(shlex.split(result.stdout), [str(worker.HERE / 'worker.py')] + args)
        self.assertNotIn(str(wrapper.parent / 'worker.py'), result.stdout)

    def test_relocated_wrapper_rejects_wrong_or_incomplete_source_directory(self):
        wrapper = self.copied_wrapper()
        for missing in ('worker.py', 'freeze.py'):
            with self.subTest(missing=missing):
                source = self.root / missing.replace('.', '_')
                source.mkdir()
                present = 'freeze.py' if missing == 'worker.py' else 'worker.py'
                (source / present).write_text('# Fixture only; never executed.\n')
                result = subprocess.run(['bash', str(wrapper)], cwd=source,
                                        env=self.wrapper_environment(), capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('Source working directory', result.stderr)
                self.assertIn(missing, result.stderr)
                self.assertEqual(result.stdout, '')

    def test_wrapper_clears_legacy_caches_before_first_interpreter(self):
        wrapper = self.copied_wrapper()
        interpreter = self.root / 'fake_python'
        interpreter.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' +
                               'for name in ' + ' '.join(worker.CACHES) + '; do\n' +
                               '  [[ ! -v $name ]] || { echo "uncleared cache: $name" >&2; exit 44; }\n' +
                               'done\nprintf "%s\\n" "$1"\n')
        interpreter.chmod(0o755)
        env = self.wrapper_environment()
        env.update({name: 'inherited-cache-fixture' for name in worker.CACHES})
        env['BIGFIELD_PYTHON'] = str(interpreter)
        result = subprocess.run(['bash', str(wrapper)], cwd=worker.HERE, env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(worker.HERE / 'worker.py'))

    def test_wrapper_requires_allocation_before_first_interpreter(self):
        wrapper = self.copied_wrapper()
        for missing in ('SLURM_JOB_ID', 'SLURMD_NODENAME'):
            with self.subTest(missing=missing):
                env = self.wrapper_environment()
                env.pop(missing)
                result = subprocess.run(['bash', str(wrapper)], cwd=worker.HERE, env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn('allocated compute node', result.stderr)
                self.assertEqual(result.stdout, '')

    def test_status_refuses_unrelated_existing_output_before_writing(self):
        unrelated = self.root / 'unrelated'
        unrelated.mkdir()
        sentinel = unrelated / 'old.txt'
        sentinel.write_text('old output')
        with self.assertRaisesRegex(ValueError, 'unbound'):
            worker.write_status(unrelated, bundle=True)
        self.assertEqual(list(unrelated.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(), 'old output')

    def test_compute_path_keeps_virtual_environment_entrypoint(self):
        path = self.root / 'venv/bin/python'
        path.parent.mkdir(parents=True)
        path.symlink_to(sys.executable)
        self.assertEqual(worker.compute_path(path), path)
        self.assertNotEqual(worker.compute_path(path), path.resolve())

    def test_recovered_eval_uses_new_panel_and_unavailable_scorer_stays_pending(self):
        checkpoint = self.root / 'checkpoint'
        worker.atomic(checkpoint / 'checkpoint.json', dict(step=1000))
        data = self.root / 'data'
        setup = dict(eval_assets=str(data / 'evaluation'),
                     evaluation_assets={b: {'status': 'PASS'} for b in ('geneval', 'dpg')},
                     scoring={b: {'status': 'PASS', 'python': sys.executable} for b in ('geneval', 'dpg')})
        worker.atomic(data / 'setup/status.json', setup)
        spec = dict(key='eval_001000_r001', end_step=1000, output_root=str(self.root),
                    data_root=str(data), python=sys.executable)
        old = self.root / 'readouts/step_001000/old.txt'
        old.parent.mkdir(parents=True)
        old.write_text('preserve')
        with patch.object(worker, 'parent_checkpoint', return_value=checkpoint), patch.object(worker, 'storage_gate'), \
             patch.object(worker, 'command') as commands, patch.object(worker.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'unavailable CUDA')):
            result = worker.evaluation_stage(self.state, spec, {})
        self.assertTrue(result['panel_path'].endswith('step_001000_r001'))
        self.assertEqual(old.read_text(), 'preserve')
        self.assertEqual(commands.call_count, 2)  # Both panels sampled; no score invoked.
        self.assertTrue(all(r['status'] == 'scoring pending' for r in result['benchmarks'].values()))
        for benchmark in ('geneval', 'dpg'):
            gate = worker.read(self.state / 'gates' / ('eval_001000_r001.' + benchmark + '.environment.json'))
            self.assertFalse(gate['ready'])

    def test_bound_step_zero_checkpoint_restores_into_existing_training_directory(self):
        checkpoint = self.root / 'train/checkpoints/step_000000'
        worker.atomic(checkpoint / 'checkpoint.json', dict(schema='bigfield-checkpoint-v1', step=0))
        spec = dict(key='train_zero', stage='train', start_step=0, end_step=1000,
                    source_manifest_sha256='source', data_root=str(self.root / 'data'),
                    output_root=str(self.root), code_root=str(worker.HERE), python=sys.executable,
                    train_output=str(self.root / 'train'),
                    parent_checkpoint=dict(path=str(checkpoint), manifest_sha256=worker.sha(checkpoint / 'checkpoint.json')))
        worker.atomic(self.state / 'jobs/train_zero.json', dict(spec=spec))
        args = types.SimpleNamespace(state=self.state, key='train_zero', stage='train')
        with patch.object(worker, 'allocated'), patch('freeze.check', return_value='source'), \
             patch.object(worker, 'dependency_bindings', return_value={}), patch.object(worker, 'storage_gate'), \
             patch.object(worker, 'launch_ranks') as launch, patch.object(worker, 'checkpoint_record', return_value={'step': 1000}):
            result = worker.execute_stage(args)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(launch.call_args.kwargs['resume'], checkpoint)
        self.assertEqual(worker.read(self.state / 'bindings/train_zero.parent.json')['step'], 0)

    def test_cold_start_still_refuses_nonempty_training_directory(self):
        output = self.root / 'train'
        output.mkdir()
        sentinel = output / 'old.txt'
        sentinel.write_text('preserve')
        spec = dict(key='train_cold', stage='train', start_step=0, end_step=1000,
                    source_manifest_sha256='source', data_root=str(self.root / 'data'),
                    output_root=str(self.root), code_root=str(worker.HERE), python=sys.executable,
                    train_output=str(output))
        worker.atomic(self.state / 'jobs/train_cold.json', dict(spec=spec))
        args = types.SimpleNamespace(state=self.state, key='train_cold', stage='train')
        with patch.object(worker, 'allocated'), patch('freeze.check', return_value='source'), \
             patch.object(worker, 'dependency_bindings', return_value={}), patch.object(worker, 'storage_gate'), \
             patch.object(worker, 'launch_ranks') as launch:
            with self.assertRaisesRegex(FileExistsError, 'Cold-start'):
                worker.execute_stage(args)
        launch.assert_not_called()
        self.assertEqual(sentinel.read_text(), 'preserve')
        self.assertEqual(worker.read(self.state / 'receipts/train_cold.json')['status'], 'failed')

    def test_recovered_scored_panel_omits_unneeded_params(self):
        checkpoint = self.root / 'train/checkpoints/step_001000'
        worker.atomic(checkpoint / 'checkpoint.json', dict(step=1000))
        (checkpoint / 'params.msgpack').write_bytes(b'toy parameters')
        panel = self.root / 'readouts/step_001000_r001'
        worker.atomic(panel / 'geneval/scores/geneval.summary.json', dict(overall_category_balanced=.8))
        worker.atomic(panel / 'dpg/scores/dpg.json', dict(fullrow_corrected_overall=82.))
        worker.atomic(self.state / 'receipts/eval_001000_r001.json',
                      dict(stage='eval', step=1000, status='complete', panel_path=str(panel), finished_at=10))
        worker.write_status(self.root, bundle=True)
        with tarfile.open(self.root / 'results/return_bundle.tar') as archive:
            self.assertNotIn('train/checkpoints/step_001000/params.msgpack', archive.getnames())
        self.assertIn('| 1000 | 0.8000 | 82.0000 |', (self.root / 'results/results.md').read_text())

    def test_late_scores_invalidate_bundle_without_receipt_change(self):
        checkpoint = self.root / 'train/checkpoints/step_001000'
        worker.atomic(checkpoint / 'checkpoint.json', dict(step=1000))
        (checkpoint / 'params.msgpack').write_bytes(b'toy parameters')
        panel = self.root / 'readouts/step_001000_r001'
        worker.atomic(self.state / 'receipts/eval_001000_r001.json',
                      dict(stage='eval', step=1000, status='complete', panel_path=str(panel), finished_at=10))
        worker.write_status(self.root, bundle=True)
        token = worker.read(self.root / 'results/return_bundle.receipt.json')['state_token']
        worker.atomic(panel / 'geneval/scores/geneval.summary.json', dict(overall_category_balanced=.9))
        worker.atomic(panel / 'dpg/scores/dpg.json', dict(fullrow_corrected_overall=83.))
        worker.write_status(self.root, bundle=True)
        self.assertNotEqual(token, worker.read(self.root / 'results/return_bundle.receipt.json')['state_token'])
        with tarfile.open(self.root / 'results/return_bundle.tar') as archive:
            self.assertNotIn('train/checkpoints/step_001000/params.msgpack', archive.getnames())
            self.assertEqual(archive.extractfile('results/results.md').read(),
                             (self.root / 'results/results.md').read_bytes())
        worker.write_status(self.root, bundle=True)
        self.assertEqual(worker.read(self.root / 'results/bundle_gate.json')['action'], 'reuse')

    def test_nontraining_failure_does_not_suggest_training_recovery(self):
        worker.atomic(self.state / 'jobs/bank.json', dict(spec=dict(key='bank', stage='bank'), status='failed'))
        text = worker.status_text(self.root, refresh=False)
        self.assertIn('failed bank stage logs', text)
        self.assertNotIn('--recover bank', text)


if __name__ == '__main__':
    unittest.main()
