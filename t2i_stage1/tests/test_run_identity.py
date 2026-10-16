"""Tiny CPU regressions for output ownership and exact run provenance."""
from copy import deepcopy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from t2i.checkpoint import (claim_run, read_run_owner, restore_checkpoint,
                            save_checkpoint, sha256, verify_checkpoint)
from t2i.config import Recipe, batch_layout
from t2i.run_identity import (directory_identity, new_run_identity,
                              validate_run_identity)
from t2i.train import create_state


def binding():
    return dict(source_sha256='c' * 64, recipe=Recipe().as_dict(),
                noise='logit-normal', layout=batch_layout(1024, 1, 1, 1),
                assets=dict(t5='d' * 64, perceptual='e' * 64),
                data=dict(A='a' * 64, B='b' * 64))


def snapshot(directory):
    """Capture both old contents and modification identity before a rejection."""
    return {str(path.relative_to(directory)):
            (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
            for path in sorted(Path(directory).rglob('*')) if path.is_file()}


def checkpoint_metadata(run, cursor=1):
    return {**run['binding'], 'run_identity': run,
            'data_states': [{'A': {'cursor': cursor}, 'B': {'cursor': 0}}]}


def rewrite_metadata(path, metadata):
    """Retain payload hashes so provenance tests reach the identity check."""
    (path / 'metadata.json').write_text(json.dumps(metadata, sort_keys=True))
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['files']['metadata.json'] = sha256(path / 'metadata.json')
    (path / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True))
    receipt = json.loads((path / 'receipt.json').read_text())
    receipt['manifest_sha256'] = sha256(path / 'manifest.json')
    (path / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True))


class RunIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only two scalar parameters and Adam/EMA state; no model/asset loads.
        cls.initial = create_state({'w': jnp.asarray([1., 2.])}, Recipe())
        cls.one = cls.initial.apply_gradients(grads={'w': jnp.asarray([.1, .2])})
        cls.two = cls.one.apply_gradients(grads={'w': jnp.asarray([.3, .4])})

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='run-identity-',
                                                     dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def new_output(self, name='run', config=None):
        output = self.root / name
        output.mkdir()
        run, parent = claim_run(output, binding() if config is None else config)
        self.assertIsNone(parent)
        return output, run

    def save(self, output, run, state=None, cursor=1):
        return save_checkpoint(output, self.one if state is None else state,
                               checkpoint_metadata(run, cursor))

    def assert_unchanged_rejection(self, output, operation, pattern=None):
        before = snapshot(output)
        if pattern is None:
            with self.assertRaises(ValueError):
                operation()
        else:
            with self.assertRaisesRegex(ValueError, pattern):
                operation()
        self.assertEqual(snapshot(output), before)

    def test_fresh_runs_get_distinct_launch_identity(self):
        _, first = self.new_output('first')
        _, second = self.new_output('second')
        self.assertEqual(first['binding'], second['binding'])
        self.assertNotEqual(first['launch_id'], second['launch_id'])
        self.assertNotEqual(first['sha256'], second['sha256'])
        self.assertIsNone(first['starting_checkpoint'])
        self.assertEqual(validate_run_identity(first), first)

    def test_same_run_resume_restores_exact_state_and_next_update(self):
        output, run = self.new_output()
        path = self.save(output, run)
        before = snapshot(output)
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': 'new-job-after-USR1'}):
            current, parent = claim_run(output, binding(), path)
        self.assertEqual((current, parent), (run, run))
        self.assertEqual(snapshot(output), before)
        restored, _ = restore_checkpoint(path, self.initial,
            {**binding(), 'run_identity': parent})
        for actual, expected in zip(jax.tree_util.tree_leaves(restored),
                                    jax.tree_util.tree_leaves(self.one)):
            np.testing.assert_array_equal(actual, expected)
        resumed = restored.apply_gradients(grads={'w': jnp.asarray([.3, .4])})
        for actual, expected in zip(jax.tree_util.tree_leaves(resumed),
                                    jax.tree_util.tree_leaves(self.two)):
            np.testing.assert_array_equal(actual, expected)
        next_path = self.save(output, current, resumed, cursor=2)
        self.assertEqual(verify_checkpoint(next_path)['step'], 2)
        self.assertEqual(read_run_owner(output), run)

    def test_usr1_same_step_identical_save_is_read_only(self):
        output, run = self.new_output()
        path = self.save(output, run)
        before = snapshot(output)
        self.assertEqual(self.save(output, run), path)
        self.assertEqual(snapshot(output), before)

    def test_internal_20_to_30_resume_keeps_original_run_identity(self):
        output, run = self.new_output()
        state = self.initial
        grads = {'w': jnp.asarray([.1, .2])}
        for _ in range(20):
            state = state.apply_gradients(grads=grads)
        parent = self.save(output, run, state, cursor=20)
        resumed_run, parent_run = claim_run(output, binding(), parent)
        resumed, _ = restore_checkpoint(parent, self.initial,
            {**binding(), 'run_identity': parent_run})
        for _ in range(10):
            resumed = resumed.apply_gradients(grads=grads)
        final = self.save(output, resumed_run, resumed, cursor=30)
        self.assertEqual(verify_checkpoint(final)['step'], 30)
        self.assertEqual(read_run_owner(output), run)
        self.assertEqual(json.loads((final / 'metadata.json').read_text())['run_identity'], run)

    def test_interleaved_runs_cannot_write_same_output(self):
        first, first_run = self.new_output('first')
        second, second_run = self.new_output('second')
        self.save(first, first_run, self.two, cursor=2)
        other_path = self.save(second, second_run)
        self.assert_unchanged_rejection(first,
            lambda: claim_run(first, binding(), other_path), 'new output directory')
        self.assert_unchanged_rejection(first,
            lambda: self.save(first, second_run, self.two, cursor=2), 'new output directory')
        self.assert_unchanged_rejection(first,
            lambda: self.save(first, second_run, self.one), 'new output directory')

    def test_same_identity_different_state_cannot_reuse_step(self):
        output, run = self.new_output()
        self.save(output, run)
        changed = [self.one.replace(params={'w': jnp.asarray([9., 2.])}),
                   self.one.replace(ema_params={'w': jnp.asarray([3., 4.])}),
                   self.one.replace(rng=jax.random.PRNGKey(123)),
                   self.one.replace(opt_state=self.two.opt_state)]
        for state in changed:
            with self.subTest(state=str(state.params)):
                self.assert_unchanged_rejection(output,
                    lambda: self.save(output, run, state), 'new output directory')

    def test_same_identity_different_cursor_metadata_cannot_reuse_step(self):
        output, run = self.new_output()
        self.save(output, run)
        self.assert_unchanged_rejection(output,
            lambda: self.save(output, run, cursor=2), 'new output directory')

    def test_binding_changes_rejected_without_output_writes(self):
        output, run = self.new_output()
        path = self.save(output, run)
        variants = {}
        for name in ('noise', 'recipe', 'source', 'data', 'assets', 'layout'):
            changed = binding()
            if name == 'noise':
                changed['noise'] = 'fixed-0.8'
            elif name == 'recipe':
                changed['recipe']['ema_decay'] = .99
            elif name == 'source':
                changed['source_sha256'] = 'f' * 64
            elif name == 'data':
                changed['data']['A'] = 'f' * 64
            elif name == 'assets':
                changed['assets']['t5'] = 'f' * 64
            else:
                changed['layout']['microbatch'] = 2
            variants[name] = changed
        for name, changed in variants.items():
            with self.subTest(binding=name):
                self.assert_unchanged_rejection(output,
                    lambda: claim_run(output, changed, path))

    def test_owner_identity_tampering_rejected(self):
        output, run = self.new_output()
        path = self.save(output, run)
        owner_path = output / 'owner.json'
        original = json.loads(owner_path.read_text())
        for field in ('launch_id', 'binding_sha256', 'sha256'):
            owner = deepcopy(original)
            owner['run_identity'][field] = '0' * len(owner['run_identity'][field])
            owner_path.write_text(json.dumps(owner))
            with self.subTest(field=field):
                self.assert_unchanged_rejection(output,
                    lambda: claim_run(output, binding(), path), 'new output directory')

    def test_checkpoint_identity_tampering_rejected_after_payload_rehash(self):
        output, run = self.new_output()
        path = self.save(output, run)
        metadata = json.loads((path / 'metadata.json').read_text())
        metadata['run_identity']['binding']['noise'] = 'fixed-0.8'
        rewrite_metadata(path, metadata)
        # Hash integrity alone is valid; immutable run provenance is not.
        verify_checkpoint(path)
        self.assert_unchanged_rejection(output,
            lambda: claim_run(output, binding(), path), 'new output directory')
        self.assert_unchanged_rejection(output,
            lambda: restore_checkpoint(path, self.initial,
                {**binding(), 'run_identity': run}), 'new output directory')

    def test_self_consistent_other_run_checkpoint_is_rejected(self):
        output, run = self.new_output()
        path = self.save(output, run)
        metadata = json.loads((path / 'metadata.json').read_text())
        metadata['run_identity'] = new_run_identity(binding())
        rewrite_metadata(path, metadata)
        verify_checkpoint(path)
        self.assert_unchanged_rejection(output,
            lambda: claim_run(output, binding(), path), 'new output directory')
        self.assert_unchanged_rejection(output,
            lambda: restore_checkpoint(path, self.initial,
                {**binding(), 'run_identity': run}), 'run_identity')

    def test_output_without_owner_or_missing_resume_is_rejected(self):
        unowned = self.root / 'unowned'
        unowned.mkdir()
        (unowned / 'old-progress.json').write_text('old output')
        self.assert_unchanged_rejection(unowned, lambda: claim_run(unowned, binding()))
        output, run = self.new_output()
        self.save(output, run)
        self.assert_unchanged_rejection(output, lambda: claim_run(output, binding()))

    def test_stale_same_run_checkpoint_cannot_rewind_existing_output(self):
        output, run = self.new_output()
        stale = self.save(output, run)
        latest = self.save(output, run, self.two, cursor=2)
        self.assert_unchanged_rejection(output,
            lambda: claim_run(output, binding(), stale), 'latest checkpoint')
        self.assertEqual(claim_run(output, binding(), latest), (run, run))

    def test_new_output_resume_has_fresh_identity_and_exact_parent_pin(self):
        source, parent_run = self.new_output('source')
        parent = self.save(source, parent_run)
        before = snapshot(source)
        output = self.root / 'continuation'
        output.mkdir()
        run, restored_run = claim_run(output, binding(), parent)
        self.assertEqual(restored_run, parent_run)
        self.assertNotEqual(run['launch_id'], parent_run['launch_id'])
        self.assertNotEqual(run['sha256'], parent_run['sha256'])
        self.assertEqual(run['starting_checkpoint'], dict(path=str(parent.resolve()),
            step=1, receipt_sha256=sha256(parent / 'receipt.json'),
            manifest_sha256=sha256(parent / 'manifest.json'),
            run_identity_sha256=parent_run['sha256']))
        restored, _ = restore_checkpoint(parent, self.initial,
            {**binding(), 'run_identity': restored_run})
        resumed = restored.apply_gradients(grads={'w': jnp.asarray([.3, .4])})
        latest = self.save(output, run, resumed, cursor=2)
        self.assertEqual(claim_run(output, binding(), latest), (run, run))
        self.assertEqual(snapshot(source), before)

    def test_save_requires_owner_and_consistent_metadata(self):
        output, run = self.new_output()
        inconsistent = checkpoint_metadata(run)
        inconsistent['noise'] = 'fixed-1.0'
        self.assert_unchanged_rejection(output,
            lambda: save_checkpoint(output, self.one, inconsistent), 'new output directory')
        missing = checkpoint_metadata(run)
        missing.pop('run_identity')
        self.assert_unchanged_rejection(output,
            lambda: save_checkpoint(output, self.one, missing), 'new output directory')
        unowned = self.root / 'no-owner'
        unowned.mkdir()
        self.assert_unchanged_rejection(unowned,
            lambda: save_checkpoint(unowned, self.one, checkpoint_metadata(run)))

    def test_run_identity_ignores_shared_filesystem_device_number(self):
        output, _ = self.new_output()
        actual = output.stat()
        def stat_with_device(device):
            return SimpleNamespace(st_ino=actual.st_ino, st_uid=actual.st_uid,
                                   st_dev=device)
        with mock.patch('t2i.run_identity.Path.stat', return_value=stat_with_device(54)):
            login = directory_identity(output)
        with mock.patch('t2i.run_identity.Path.stat', return_value=stat_with_device(999)):
            compute = directory_identity(output)
        self.assertEqual(login, compute)
        self.assertEqual(set(login), {'path', 'inode', 'uid'})

    def test_main_rejects_cross_run_before_first_existing_output_write(self):
        from t2i import train
        from t2i import data, model, perceptual, text
        first, first_run = self.new_output('first')
        second, second_run = self.new_output('second')
        self.save(first, first_run, self.two, cursor=2)
        other = self.save(second, second_run)
        before = snapshot(first)
        fake_streams = [SimpleNamespace(manifest_sha256=binding()['data'][stage])
                        for stage in ('A', 'B')]
        # Stop at the admission gate. No encoder, perceptual net or forward runs.
        with ExitStack() as stack:
            replacements = [(train, 'initialize_distributed', {}),
                (train, 'runtime_topology_gate', {'passed': True}),
                (train, 'source_digest', binding()['source_sha256']),
                (train.jax, 'process_index', 0), (train.jax, 'process_count', 1),
                (train.jax, 'local_device_count', 1), (train.jax, 'device_count', 1),
                (train.jax, 'local_devices', [])]
            for target, name, value in replacements:
                stack.enter_context(mock.patch.object(target, name, return_value=value))
            stack.enter_context(mock.patch.object(train.multihost_utils, 'sync_global_devices'))
            stack.enter_context(mock.patch.object(train, 'validate_assets',
                side_effect=lambda path, kind: binding()['assets'][kind]))
            stack.enter_context(mock.patch.object(data, 'WebDatasetStream', side_effect=fake_streams))
            writers = stack.enter_context(mock.patch.object(train, 'atomic_json'))
            build = stack.enter_context(mock.patch.object(model, 'build_model'))
            encoder = stack.enter_context(mock.patch.object(text, 'load_text_encoder'))
            features = stack.enter_context(mock.patch.object(perceptual, 'load_perceptual'))
            with self.assertRaisesRegex(ValueError, 'new output directory'):
                train.main(['--output-dir', str(first), '--resume', str(other),
                            '--cc12m-root', 'fake-A', '--curated-root', 'fake-B',
                            '--smoke', '--steps', '30'])
            writers.assert_not_called()
            build.assert_not_called()
            encoder.assert_not_called()
            features.assert_not_called()
        self.assertEqual(snapshot(first), before)


if __name__ == '__main__':
    unittest.main()
