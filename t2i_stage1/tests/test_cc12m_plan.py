"""Tiny CPU checks for Stage A-only plans and exact Stage B continuation."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from PIL import Image

from t2i import train
from t2i.checkpoint import (claim_run, restore_checkpoint, save_checkpoint,
                            sha256, verify_checkpoint)
from t2i.config import Recipe, batch_layout
from t2i.data import WebDatasetStream
from t2i.data_plan import (CC12M_ONLY, PAPER, plan_name, plan_steps,
                           stage_b_admission, valid_steps)


def binding(only=True):
    return dict(source_sha256='c' * 64, recipe=Recipe().as_dict(),
                noise='logit-normal', layout=batch_layout(1024, 1, 1, 1),
                assets=dict(t5='d' * 64, perceptual='e' * 64),
                data_plan=plan_name(only),
                data=dict(A='a' * 64) if only else dict(A='a' * 64, B='b' * 64))


def snapshot(directory):
    return {str(path.relative_to(directory)):
            (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
            for path in sorted(Path(directory).rglob('*')) if path.is_file()}


def state_at(step):
    state = train.create_state({'w': jnp.asarray([1., 2.])}, Recipe())
    components = [component._replace(count=jnp.int32(step))
                  if 'count' in getattr(component, '_fields', ()) else component
                  for component in state.opt_state]
    return state.replace(step=jnp.int32(step), opt_state=tuple(components),
                         ema_params={'w': jnp.asarray([.9, 1.9])},
                         rng=jax.random.PRNGKey(123))


def tiny_update(state):
    # Real AdamW/schedule and the production EMA equation; no model forward.
    next_state = state.apply_gradients(grads={'w': jnp.asarray([.1, .2])})
    ema = jax.tree_util.tree_map(lambda e, p: Recipe().ema_decay * e +
                                (1 - Recipe().ema_decay) * p,
                                state.ema_params, next_state.params)
    return next_state.replace(ema_params=ema)


def write_tar(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = io.BytesIO()
    Image.new('RGB', (2, 2), color='red').save(image, format='PNG')
    with tarfile.open(path, 'w') as archive:
        for name, value in [('0.png', image.getvalue()), ('0.txt', b'red image')]:
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    return str(path)


def assert_state_equal(test, actual, expected):
    # TrainState's static optimizer function objects are separately constructed.
    actual = serialization.to_state_dict(actual)
    expected = serialization.to_state_dict(expected)
    test.assertEqual(jax.tree_util.tree_structure(actual),
                     jax.tree_util.tree_structure(expected))
    for left, right in zip(jax.tree_util.tree_leaves(actual),
                           jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(left, right)


class CC12MPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='cc12m-plan-',
                                                     dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def new_output(self, name):
        output = self.root / name
        output.mkdir()
        return output

    def parent(self, name='parent', *, config=None, step=250000, cursors=None):
        output = self.new_output(name)
        config = binding() if config is None else config
        run, _ = claim_run(output, config)
        if cursors is None:
            cursors = {stage: {'cursor': 7 if stage == 'A' else 0}
                       for stage in config['data']}
        metadata = {**config, 'run_identity': run, 'data_states': [cursors]}
        path = save_checkpoint(output, state_at(step), metadata)
        return output, run, path

    def test_plan_limits_and_default_cli_keep_recipe_constants(self):
        recipe = Recipe()
        self.assertEqual((recipe.stage_a_steps, recipe.stage_b_steps,
                          recipe.global_batch, recipe.lr_a, recipe.lr_b,
                          recipe.seed, recipe.image_size),
                         (250000, 40000, 1024, 4e-4, 4e-5, 42, 512))
        for only, maximum in ((False, 290000), (True, 250000)):
            self.assertEqual(plan_steps(recipe, only), maximum)
            self.assertTrue(valid_steps(maximum, False, recipe, only))
            for steps in (0, -1, 20, maximum - 1, maximum + 1):
                self.assertFalse(valid_steps(steps, False, recipe, only))
            for steps in (1, 20, maximum):
                self.assertTrue(valid_steps(steps, True, recipe, only))
            for steps in (0, -1, maximum + 1):
                self.assertFalse(valid_steps(steps, True, recipe, only))
            args = train.parse_args(['--cc12m-only'] if only else [])
            self.assertEqual((args.cc12m_only, args.steps), (only, maximum))
            self.assertEqual((args.noise, args.microbatch, args.checkpoint_every),
                             ('logit-normal', 1, 5000))
        self.assertEqual(Recipe().as_dict(), recipe.as_dict())

    def test_rejected_steps_are_persisted_before_raise_without_old_writes(self):
        output, _, _ = self.parent()
        before = snapshot(output)
        for only, steps, smoke in ((True, 250001, True), (True, 249999, False),
                                   (False, 250000, False), (False, 290001, True)):
            with self.subTest(only=only, steps=steps, smoke=smoke):
                stderr = io.StringIO()
                argv = ['--output-dir', str(output), '--steps', str(steps)]
                if only:
                    argv.append('--cc12m-only')
                if smoke:
                    argv.append('--smoke')
                with mock.patch.object(train, 'initialize_distributed', return_value={}), \
                     mock.patch.object(train, 'runtime_topology_gate', return_value={'passed': True}), \
                     mock.patch.object(train, 'source_digest', return_value='c' * 64), \
                     mock.patch.object(train.multihost_utils, 'sync_global_devices'), \
                     redirect_stderr(stderr):
                    with self.assertRaisesRegex(ValueError, 'preflight receipt'):
                        train.main(argv)
                line = next(row for row in stderr.getvalue().splitlines()
                            if row.startswith('Rejected configuration receipt: '))
                audit = Path(line.split(': ', 1)[1])
                record = json.loads(audit.read_text())
                self.assertFalse(record['passed'])
                self.assertEqual(record['steps'], steps)
                self.assertEqual(record['max_steps'], 250000 if only else 290000)
                self.assertEqual(record['data_plan'], plan_name(only))
                self.assertEqual(record['global_batch'], 1024)
                self.assertEqual(record['smoke'], smoke)
                self.assertIn('layout', record)
                self.assertEqual(snapshot(output), before)

    def test_cc12m_only_never_constructs_curated_and_accepts_missing_root(self):
        a = write_tar(self.root / 'A' / 'one.tar')
        for curated in ('', str(self.root / 'absent-curated')):
            args = SimpleNamespace(cc12m_root=a, curated_root=curated, cc12m_only=True)
            with redirect_stdout(io.StringIO()) as stdout, \
                 mock.patch('t2i.data.WebDatasetStream', wraps=WebDatasetStream) as factory:
                streams = train.data_streams(args, dict(image_size=2))
            self.addCleanup(streams['A'].close)
            self.assertEqual(set(streams), {'A'})
            factory.assert_called_once_with(a, image_size=2)
            self.assertEqual(streams['A'].next_batch(1)['captions'], ['red image'])
            self.assertEqual(bool(stdout.getvalue()), bool(curated))

    def test_binding_and_checkpoint_metadata_only_bind_a(self):
        _, run, path = self.parent()
        metadata = json.loads((path / 'metadata.json').read_text())
        self.assertEqual(run['binding']['data_plan'], CC12M_ONLY)
        self.assertEqual(metadata['data_plan'], CC12M_ONLY)
        self.assertEqual(metadata['data'], {'A': 'a' * 64})
        self.assertEqual(set(metadata['data_states'][0]), {'A'})
        self.assertEqual(verify_checkpoint(path)['step'], 250000)
        self.assertEqual(path.name, 'checkpoint_00250000')

    def test_same_directory_plan_changes_explain_environment_and_preserve_output(self):
        for only in (True, False):
            output, _, parent = self.parent('run-' + str(only), config=binding(only))
            before = snapshot(output)
            expected = 'CC12M_ONLY=' + ('1' if only else '0')
            for resume in ('', str(parent)):
                with self.subTest(only=only, resume=resume):
                    with self.assertRaisesRegex(ValueError, expected) as error:
                        claim_run(output, binding(not only), resume)
                    self.assertIn('set ' + expected + ' again', str(error.exception))
                    self.assertIn('new output directory', str(error.exception))
                    self.assertEqual(snapshot(output), before)

    def test_main_checks_plan_before_assets_or_streams_or_existing_receipts(self):
        output, _, parent = self.parent()
        before = snapshot(output)
        with mock.patch.object(train, 'initialize_distributed', return_value={}), \
             mock.patch.object(train, 'runtime_topology_gate', return_value={'passed': True}), \
             mock.patch.object(train, 'source_digest', return_value='c' * 64), \
             mock.patch.object(train.multihost_utils, 'sync_global_devices'), \
             mock.patch.object(train, 'validate_assets') as assets, \
             mock.patch.object(train, 'data_streams') as streams, \
             mock.patch.object(train, 'atomic_json') as write:
            with self.assertRaisesRegex(ValueError, 'set CC12M_ONLY=1 again'):
                train.main(['--output-dir', str(output), '--resume', str(parent)])
            assets.assert_not_called()
            streams.assert_not_called()
            write.assert_not_called()
        self.assertEqual(snapshot(output), before)

    def test_stage_b_gate_covers_every_binding_rejection(self):
        previous = binding()
        destination = binding(False)
        parent = {'step': 250000, 'run_identity': {'binding': previous}}
        self.assertTrue(stage_b_admission(parent, destination, 250000)['passed'])
        cases = []
        changed = deepcopy(parent); changed['run_identity']['binding']['data_plan'] = PAPER
        cases.append(('parent_plan', changed, destination, 250000))
        changed = deepcopy(destination); changed['data_plan'] = CC12M_ONLY
        cases.append(('destination_plan', parent, changed, 250000))
        for step in (249999, 250001):
            cases.append(('parent_step', parent, destination, step))
        changed = deepcopy(parent); changed['step'] = 249999
        cases.append(('parent_step', changed, destination, 250000))
        changed = deepcopy(parent); changed['run_identity']['binding']['data']['B'] = 'b' * 64
        cases.append(('parent_data', changed, destination, 250000))
        for data in ({'A': 'a' * 64}, {'B': 'b' * 64},
                     {'A': 'a' * 64, 'B': 'b' * 64, 'C': 'f' * 64}):
            changed = deepcopy(destination); changed['data'] = data
            cases.append(('destination_data', parent, changed, 250000))
        changed = deepcopy(destination); changed['data']['A'] = 'f' * 64
        cases.append(('a_manifest', parent, changed, 250000))
        for key in ('source_sha256', 'recipe', 'noise', 'layout', 'assets'):
            changed = deepcopy(destination)
            if key == 'recipe': changed[key]['seed'] += 1
            elif key == 'layout': changed[key]['microbatch'] = 2
            elif key == 'assets': changed[key]['t5'] = 'f' * 64
            else: changed[key] = 'different'
            cases.append(('unchanged_binding', parent, changed, 250000))
        for alter in ('missing', 'extra'):
            changed = deepcopy(destination)
            if alter == 'missing': changed.pop('noise')
            else: changed['extra'] = True
            cases.append(('unchanged_binding', parent, changed, 250000))
        for check, metadata, current, step in cases:
            with self.subTest(check=check, step=step, current=current):
                gate = stage_b_admission(metadata, current, step)
                self.assertFalse(gate['passed'])
                self.assertFalse(gate['checks'][check])
                self.assertEqual(gate['parent_binding'], metadata['run_identity']['binding'])
                self.assertEqual(gate['destination_binding'], current)
                self.assertEqual(gate['parent_step'], step)
                self.assertEqual(gate['metadata_step'], metadata['step'])

    def test_stage_b_new_output_pins_parent_and_inherits_exact_state(self):
        source, parent_run, parent = self.parent()
        old = snapshot(source)
        output = self.new_output('stage-b')
        run, restored_run = claim_run(output, binding(False), parent)
        self.assertEqual(restored_run, parent_run)
        self.assertNotEqual(run['sha256'], parent_run['sha256'])
        self.assertEqual(run['starting_checkpoint'], dict(path=str(parent.resolve()),
            step=250000, receipt_sha256=sha256(parent / 'receipt.json'),
            manifest_sha256=sha256(parent / 'manifest.json'),
            run_identity_sha256=parent_run['sha256']))
        restored, metadata = restore_checkpoint(parent, state_at(0),
            {**parent_run['binding'], 'run_identity': restored_run})
        assert_state_equal(self, restored, state_at(250000))
        self.assertEqual(metadata['data_states'], [{'A': {'cursor': 7}}])
        a, b = mock.Mock(), mock.Mock()
        audit = self.root / 'data-restore-success.json'
        train.restore_data_streams({'A': a, 'B': b}, metadata, 0, audit_path=audit)
        a.load_state_dict.assert_called_once_with({'cursor': 7})
        b.load_state_dict.assert_not_called()
        self.assertTrue(json.loads(audit.read_text())['passed'])
        assert_state_equal(self, tiny_update(restored), tiny_update(state_at(250000)))
        self.assertEqual(snapshot(source), old)

    def test_stage_b_rejections_leave_parent_and_destination_unchanged_and_audit(self):
        source, _, parent = self.parent()
        old = snapshot(source)
        cases = []
        for key in ('source_sha256', 'recipe', 'noise', 'layout', 'assets', 'data'):
            changed = binding(False)
            if key == 'recipe': changed[key]['ema_decay'] = .99
            elif key == 'layout': changed[key]['microbatch'] = 2
            elif key == 'assets': changed[key]['perceptual'] = 'f' * 64
            elif key == 'data': changed[key]['A'] = 'f' * 64
            else: changed[key] = 'different'
            cases.append((key, changed, parent))
        for step in (249999, 250001):
            _, _, invalid = self.parent('parent-' + str(step), step=step)
            cases.append(('step-' + str(step), binding(False), invalid))
        _, _, paper = self.parent('paper-parent', config=binding(False))
        changed = binding(False); changed['data']['B'] = 'f' * 64
        cases.append(('paper-parent', changed, paper))
        changed = binding(False); changed['data'].pop('B')
        cases.append(('missing-B', changed, parent))
        for name, changed, resume in cases:
            with self.subTest(name=name):
                output = self.new_output('bad-' + name)
                stdout = io.StringIO()
                with redirect_stdout(stdout), self.assertRaisesRegex(ValueError, 'failed:'):
                    claim_run(output, changed, resume)
                record_path = Path(stdout.getvalue().strip().split(': ', 1)[1])
                record = json.loads(record_path.read_text())
                self.assertFalse(record['passed'])
                self.assertEqual(record['destination_binding'], changed)
                self.assertEqual(record['required_step'], 250000)
                self.assertEqual(snapshot(output), {})
                self.assertEqual(snapshot(source), old)
        output = self.new_output('nonempty')
        (output / 'old.txt').write_text('preserve me')
        before = snapshot(output)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'new_output'):
            claim_run(output, binding(False), parent)
        self.assertEqual(snapshot(output), before)

    def test_stage_b_rejects_metadata_step_after_valid_payload_rehash(self):
        source, _, parent = self.parent()
        metadata = json.loads((parent / 'metadata.json').read_text())
        metadata['step'] = 249999
        (parent / 'metadata.json').write_text(json.dumps(metadata))
        manifest = json.loads((parent / 'manifest.json').read_text())
        manifest['files']['metadata.json'] = sha256(parent / 'metadata.json')
        (parent / 'manifest.json').write_text(json.dumps(manifest))
        receipt = json.loads((parent / 'receipt.json').read_text())
        receipt['manifest_sha256'] = sha256(parent / 'manifest.json')
        (parent / 'receipt.json').write_text(json.dumps(receipt))
        before = snapshot(source)
        self.assertEqual(verify_checkpoint(parent)['step'], 250000)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'parent_step'):
            claim_run(self.new_output('bad-metadata'), binding(False), parent)
        self.assertEqual(snapshot(source), before)

    def test_parent_receipt_and_manifest_step_rejections_are_audited_before_claim(self):
        for document, step in (('receipt', 250001), ('manifest', 250001),
                               ('receipt', 250000.5), ('manifest', True)):
            with self.subTest(document=document, step=step):
                source, _, parent = self.parent(document + '-' + str(step))
                path = parent / (document + '.json')
                value = json.loads(path.read_text())
                value['step'] = step
                path.write_text(json.dumps(value))
                if document == 'manifest':
                    receipt = json.loads((parent / 'receipt.json').read_text())
                    receipt['manifest_sha256'] = sha256(path)
                    (parent / 'receipt.json').write_text(json.dumps(receipt))
                before = snapshot(source)
                output = self.new_output('destination-' + document + '-' + str(step))
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(ValueError):
                    claim_run(output, binding(False), parent)
                line = next(row for row in stderr.getvalue().splitlines()
                            if row.startswith('Rejected parent receipt: '))
                record = json.loads(Path(line.split(': ', 1)[1]).read_text())
                self.assertFalse(record['passed'])
                self.assertEqual(record['phase'], 'receipt_manifest_step')
                expected_receipt = step if document == 'receipt' else 250000
                expected_manifest = step if document == 'manifest' else 250000
                self.assertEqual(record['values']['receipt_step'], expected_receipt)
                self.assertEqual(record['values']['manifest_step'], expected_manifest)
                self.assertEqual(snapshot(source), before)
                self.assertEqual(snapshot(output), {})

    def test_restoring_wrong_cursor_plan_or_topology_fails_closed(self):
        valid = dict(data={'A': 'a' * 64}, layout={'processes': 1},
                     data_states=[{'A': {'cursor': 7}}])
        for number, states in enumerate(([], [{'B': {}}], [{'A': {}, 'B': {}}], [{}, {}], None)):
            streams = {'A': mock.Mock(), 'B': mock.Mock()}
            audit = self.root / ('bad-data-restore-' + str(number) + '.json')
            with self.subTest(states=states):
                with self.assertRaisesRegex(ValueError, 'data cursors'):
                    train.restore_data_streams(streams, {**valid, 'data_states': states},
                                              0, audit_path=audit)
                record = json.loads(audit.read_text())
                self.assertFalse(record['passed'])
                self.assertFalse(record['cursors_match_plan'])
                self.assertEqual(record['gate'], 'restore_data_plan')
                values = record['values']
                self.assertEqual(values['state_count'], len(states) if states is not None else None)
                self.assertEqual(values['expected_processes'], 1)
                self.assertEqual(values['expected_stages'], ['A'])
                self.assertEqual(values['stream_stages'], ['A', 'B'])
                self.assertEqual(values['rank'], 0)
                self.assertEqual(values['data_state'], states[0] if states else None)
                for stream in streams.values():
                    stream.load_state_dict.assert_not_called()

    def run_tiny_main(self, output, parent, *, only, steps):
        """Run the real train loop with 2px tars and a two-parameter update.

        A batch consumes one real record then broadcasts a zero-stride image
        view. The production stage selection/save/restore path stays real;
        model, feature loading and gradient evaluation are mocked.
        """
        from t2i import model, perceptual, text
        original_streams = train.data_streams
        original_batch = WebDatasetStream.next_batch
        streams_seen = {}

        def streams(args, common):
            result = original_streams(args, {**common, 'image_size': 2})
            streams_seen.update(result)
            for stream in result.values(): self.addCleanup(stream.close)
            return result

        def next_batch(stream, count):
            one = original_batch(stream, 1)
            return {**one, 'images': np.broadcast_to(np.zeros((1, 1, 1, 3), np.float32),
                                                     (count, 512, 512, 3)),
                    'captions': one['captions'] * count}

        def tokenize(captions):
            tokens = np.broadcast_to(np.ones((1, 256), np.int32), (len(captions), 256))
            return {'input_ids': tokens, 'attention_mask': tokens}

        def update(state, *unused):
            state = train.jax_utils.replicate(tiny_update(train.jax_utils.unreplicate(state)))
            return state, {'loss': np.asarray([1.], np.float32)}

        args = ['--output-dir', str(output), '--resume', str(parent), '--smoke',
                '--steps', str(steps), '--cc12m-root', self.a_root,
                '--data-dir', str(self.root / 'caption-cache')]
        if only: args.append('--cc12m-only')
        else: args += ['--curated-root', self.b_root]
        with ExitStack() as stack:
            patches = [(train, 'initialize_distributed', {}),
                       (train, 'runtime_topology_gate', {'passed': True}),
                       (train, 'source_digest', 'c' * 64),
                       (model, 'build_model', SimpleNamespace()),
                       (perceptual, 'load_perceptual', SimpleNamespace(params=jnp.float32(1))) ]
            for target, name, value in patches:
                stack.enter_context(mock.patch.object(target, name, return_value=value))
            stack.enter_context(mock.patch.object(train, 'validate_assets',
                side_effect=lambda path, kind: binding()['assets'][kind]))
            stack.enter_context(mock.patch.object(train, 'data_streams', side_effect=streams))
            stack.enter_context(mock.patch.object(WebDatasetStream, 'next_batch', next_batch))
            stack.enter_context(mock.patch.object(train.jax, 'jit',
                side_effect=lambda function: lambda key: {'w': jnp.asarray([1., 2.])}))
            stack.enter_context(mock.patch.object(text, 'load_text_encoder',
                return_value=(None, jnp.float32(1), tokenize)))
            stack.enter_context(mock.patch.object(train, 'make_update', return_value=update))
            stack.enter_context(mock.patch.object(train, 'gather_json', side_effect=lambda value: [value]))
            stack.enter_context(mock.patch.object(train.multihost_utils, 'sync_global_devices'))
            stack.enter_context(mock.patch.object(train.multihost_utils, 'process_allgather',
                                                  side_effect=lambda value: value))
            stack.enter_context(mock.patch.object(train.signal, 'signal'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            train.main(args)
        return streams_seen

    def test_real_loop_boundary_b_is_fresh_and_stage_b_matches_uninterrupted(self):
        self.a_root = write_tar(self.root / 'data-A' / 'one.tar')
        self.b_root = write_tar(self.root / 'data-B' / 'one.tar')
        from t2i.prepare import REMOTE_LOCK, atomic_json
        from t2i.recaption import ensure_caption_receipt
        a_dir = Path(self.a_root).parent
        atomic_json(a_dir / 'preparation.json', dict(caption_source='caption_llava',
            source=REMOTE_LOCK['cc12m'], files={'one.tar': sha256(self.a_root)}), create_once=True)
        self.a_root = str(a_dir)
        caption = ensure_caption_receipt(self.a_root, self.root / 'caption-cache')
        initial = {}
        for stage, root in (('A', self.a_root), ('B', self.b_root)):
            stream = WebDatasetStream(root, image_size=2)
            self.addCleanup(stream.close)
            initial[stage] = stream.state_dict()
        self.assertEqual(Recipe().stage(249999), 'A')
        self.assertEqual(Recipe().stage(250000), 'B')
        configs = {}
        for only in (False, True):
            config = binding(only)
            config['data_captions'] = {'A': caption}
            for stage in config['data']:
                config['data'][stage] = initial[stage]['manifest_sha256']
            configs[only] = config
        full_out, full_run, full_parent = self.parent('full', config=configs[False], step=249999,
                                                     cursors=deepcopy(initial))
        only_out, only_run, only_parent = self.parent('only', config=configs[True], step=249999,
            cursors={'A': deepcopy(initial['A'])})
        self.run_tiny_main(full_out, full_parent, only=False, steps=250000)
        self.run_tiny_main(only_out, only_parent, only=True, steps=250000)
        full_boundary = full_out / 'checkpoint_00250000'
        only_boundary = only_out / 'checkpoint_00250000'
        full_meta = json.loads((full_boundary / 'metadata.json').read_text())
        only_meta = json.loads((only_boundary / 'metadata.json').read_text())
        fresh_b = WebDatasetStream(self.b_root, image_size=2)
        self.addCleanup(fresh_b.close)
        self.assertEqual(full_meta['data_states'][0]['B'], fresh_b.state_dict())
        self.assertEqual(full_meta['data_states'][0]['A'], only_meta['data_states'][0]['A'])
        self.assertEqual(full_meta['data_states'][0]['A']['mix_counter'], 1)
        self.assertEqual(set(only_meta['data_states'][0]), {'A'})
        self.assertEqual(only_meta['data_plan'], CC12M_ONLY)
        full_state, _ = restore_checkpoint(full_boundary, state_at(0),
            {**configs[False], 'run_identity': full_run})
        only_state, _ = restore_checkpoint(only_boundary, state_at(0),
            {**configs[True], 'run_identity': only_run})
        assert_state_equal(self, full_state, only_state)
        only_before = snapshot(only_out)
        continuation = self.new_output('continue-b')
        self.run_tiny_main(full_out, full_boundary, only=False, steps=250001)
        self.run_tiny_main(continuation, only_boundary, only=False, steps=250001)
        left, left_meta = restore_checkpoint(full_out / 'checkpoint_00250001', state_at(0),
                                             {**configs[False], 'run_identity': full_run})
        right, right_meta = restore_checkpoint(continuation / 'checkpoint_00250001', state_at(0),
                                              configs[False])
        assert_state_equal(self, left, right)
        self.assertEqual(left_meta['data_states'], right_meta['data_states'])
        self.assertEqual(right_meta['data_states'][0]['B']['mix_counter'], 1)
        self.assertEqual(snapshot(only_out), only_before)


if __name__ == '__main__':
    unittest.main()
