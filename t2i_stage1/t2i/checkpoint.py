"""Atomic exact-state checkpoints, verified receipts, and MiniT2I params export."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import jax
import numpy as np
from flax import serialization

from .run_identity import (canonical_bytes, directory_identity, new_run_identity,
                           validate_run_identity)
from .data_plan import plan_mismatch_message, stage_b_admission


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value, *, create=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        if create:
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size >= 8 << 20:
        raise ValueError('Missing/unsafe run metadata: ' + str(path))
    return json.loads(path.read_text())


def _metadata_identity(metadata):
    run = validate_run_identity(metadata.get('run_identity'))
    if any(metadata.get(key) != value for key, value in run['binding'].items()):
        raise ValueError('Checkpoint metadata differs from run identity; use a new output directory')
    return run


def read_run_owner(output):
    owner = _read_json(Path(output) / 'owner.json')
    run = validate_run_identity(owner.get('run_identity'))
    if (owner.get('schema') != 'priming-stage1-owner-v1'
            or owner.get('identity') != directory_identity(output)):
        raise ValueError('Output directory ownership differs; use a new output directory')
    return run


def checkpoint_provenance(path):
    """Read and verify the exact parent before claiming or writing an output."""
    path = Path(path).resolve()
    directory = tempfile.mkdtemp(prefix='stage1-parent-', dir=os.environ.get('TMPDIR', '/tmp'))
    audit = Path(directory) / 'preflight.json'
    try:
        receipt = verify_checkpoint(path, audit_path=audit)
    except (ValueError, TypeError):
        if audit.exists():
            print('Rejected parent receipt: ' + str(audit), file=sys.stderr, flush=True)
        raise
    metadata = _read_json(path / 'metadata.json')
    run = _metadata_identity(metadata)
    pin = dict(path=str(path), step=int(receipt['step']),
               receipt_sha256=sha256(path / 'receipt.json'),
               manifest_sha256=receipt['manifest_sha256'], run_identity_sha256=run['sha256'])
    return metadata, pin


def claim_run(output, binding, resume='', *, create=True):
    """Admit a launch before any existing output writes; rank zero creates once.

    Same-directory resumes keep the run identity and must use its latest exact
    checkpoint. A continuation into a new empty directory starts a new identity
    whose origin pins the parent receipt/manifest and parent run identity.
    """
    output = Path(output).resolve()
    parent, pin = checkpoint_provenance(resume) if resume else (None, None)
    owner = output / 'owner.json'
    if os.path.lexists(owner):
        run = read_run_owner(output)
        if run['binding'].get('data_plan') != binding.get('data_plan'):
            raise ValueError(plan_mismatch_message(run['binding']))
    if parent is not None and parent['run_identity']['binding'] != binding:
        gate = stage_b_admission(parent, binding, pin['step'])
        gate['new_output'] = not os.path.lexists(owner) and not any(output.iterdir())
        gate['passed'] = gate['passed'] and gate['new_output']
        directory = tempfile.mkdtemp(prefix='stage1-continuation-',
                                     dir=os.environ.get('TMPDIR', '/tmp'))
        audit = Path(directory) / 'preflight.json'
        atomic_json(audit, gate)
        print('Stage B admission receipt: ' + str(audit), flush=True)
        if not gate['passed']:
            failed = [key for key, passed in gate['checks'].items() if not passed]
            if not gate['new_output']:
                failed.append('new_output')
            raise ValueError('Resume configuration/data/assets differ; Stage B requires '
                             'a new output directory and an exact CC12M-only '
                             'step-250000 parent; failed: ' + ', '.join(failed))
    if os.path.lexists(owner):
        if (parent is None or run['binding'] != binding
                or parent['run_identity'] != run):
            raise ValueError('Resume run identity differs from output owner; use a new output directory')
        latest = _read_json(output / 'latest.json')
        expected = output / f"checkpoint_{pin['step']:08d}"
        if (Path(resume).resolve() != expected or latest.get('checkpoint') != expected.name
                or latest.get('step') != pin['step']
                or latest.get('receipt_sha256') != pin['receipt_sha256']):
            raise ValueError('Resume must use the latest checkpoint owned by this run; use a new output directory')
    else:
        if not create or any(output.iterdir()):
            raise ValueError('Training requires an owned run or an empty new output directory')
        run = new_run_identity(binding, pin)
        atomic_json(owner, dict(schema='priming-stage1-owner-v1',
                    identity=directory_identity(output), run_identity=run), create=True)
    return run, None if parent is None else parent['run_identity']


def _resume_audit(audit_path, values, passed, phase):
    if audit_path is None:
        return
    def json_value(value):
        if isinstance(value, np.ndarray):
            return json_value(value.tolist())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        if isinstance(value, dict):
            return {key: json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(item) for item in value]
        return value
    atomic_json(audit_path, dict(gate='strict_resume_step', phase=phase,
                                values=json_value(values), passed=passed))


def _integer_scalar(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return (array.shape == () and array.dtype.kind in 'iuf'
            and bool(np.isfinite(array)) and bool(array == np.floor(array)))


def _value_shape(value):
    try:
        return list(np.shape(value))
    except (TypeError, ValueError):
        return None


def _state_audit_values(receipt, metadata, step, counters):
    values = dict(receipt_step=receipt['step'], restored_step=step,
                  metadata_step=metadata.get('step'), expected_step=receipt['step'],
                  actual_step=step, optimizer_count_count=len(counters),
                  expected_optimizer_count_count=2, **counters)
    values.update({key + '_shape': _value_shape(value) for key, value in
                   dict(receipt_step=receipt['step'], restored_step=step,
                        metadata_step=metadata.get('step'), **counters).items()})
    return values


def _serialized_structure(value):
    """Keep structural evidence without copying large optimizer tensors to JSON."""
    if isinstance(value, dict):
        return {key: _serialized_structure(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialized_structure(item) for item in value]
    dtype = getattr(value, 'dtype', None)
    return {'shape': _value_shape(value),
            'dtype': str(dtype if dtype is not None else np.asarray(value).dtype)}


def assert_tree_compatible(actual, target):
    # serialization.from_bytes alone can silently accept wrong leaf shapes.
    aa, at = jax.tree_util.tree_flatten_with_path(actual)
    bb, bt = jax.tree_util.tree_flatten_with_path(target)
    if at != bt or len(aa) != len(bb):
        raise ValueError('Checkpoint state structure differs')
    for (path, a), (_, b) in zip(aa, bb):
        if np.shape(a) != np.shape(b):
            raise ValueError('Checkpoint state leaf shape differs')
        # Step dtype may change; shape, numerical value and optimizer count govern it.
        is_step = len(path) == 1 and getattr(path[0], 'name', None) == 'step'
        kinds = (np.asarray(a).dtype.kind, np.asarray(b).dtype.kind)
        if kinds[0] != kinds[1] and not (is_step and all(k in 'iuf' for k in kinds)):
            raise ValueError('Checkpoint state leaf dtype family differs')


def save_checkpoint(output, state, metadata):
    """Only rank zero calls this; caller synchronizes every rank afterwards."""
    step = int(np.asarray(state.step))
    output = Path(output)
    run = _metadata_identity(metadata)
    if read_run_owner(output) != run:
        raise ValueError('Checkpoint run identity differs from output owner; use a new output directory')
    metadata = {**metadata, 'step': step}
    target = output / f'checkpoint_{step:08d}'
    if os.path.lexists(target):
        # A USR1 at an already saved step must not rewrite the checkpoint.
        verify_checkpoint(target)
        if (canonical_bytes(_read_json(target / 'metadata.json')) != canonical_bytes(metadata)
                or sha256(target / 'state.msgpack') !=
                   hashlib.sha256(serialization.to_bytes(state)).hexdigest()):
            raise ValueError('Existing checkpoint state/metadata or run identity differs; use a new output directory')
        return target
    pending = Path(tempfile.mkdtemp(prefix=f'.checkpoint_{step:08d}-', dir=output))
    (pending / 'state.msgpack').write_bytes(serialization.to_bytes(state))
    # A standard legacy Flax checkpoint file readable by MiniT2I utils.ckpt_util.
    export = pending / 'minit2i'
    export.mkdir()
    payload = {'params': {'net': state.params}}
    (export / f'checkpoint_{step}').write_bytes(serialization.msgpack_serialize(payload))
    # Separate params-only exports make online versus EMA explicit even for
    # legacy readers that select ema_params regardless of prefer_ema=False.
    ema_export = pending / 'minit2i_ema'
    ema_export.mkdir()
    ema_payload = {'params': {'net': state.ema_params}}
    (ema_export / f'checkpoint_{step}').write_bytes(serialization.msgpack_serialize(ema_payload))
    atomic_json(pending / 'metadata.json', metadata)
    files = {str(p.relative_to(pending)): sha256(p)
             for p in sorted(pending.rglob('*')) if p.is_file()}
    atomic_json(pending / 'manifest.json', {'schema': 'priming-stage1-checkpoint-v1',
                                         'step': step, 'files': files})
    atomic_json(pending / 'receipt.json', {'step': step, 'status': 'COMPLETE',
                                        'manifest_sha256': sha256(pending / 'manifest.json')})
    os.rename(pending, target)
    atomic_json(output / 'latest.json', {'checkpoint': target.name, 'step': step,
                                       'receipt_sha256': sha256(target / 'receipt.json')})
    return target


def verify_checkpoint(path, audit_path=None):
    path = Path(path)
    receipt = json.loads((path / 'receipt.json').read_text())
    manifest = json.loads((path / 'manifest.json').read_text())
    if receipt['status'] != 'COMPLETE' or sha256(path / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Checkpoint receipt/manifest binding differs')
    values = dict(receipt_step=receipt.get('step'), manifest_step=manifest.get('step'),
                  expected_step=receipt.get('step'), actual_step=manifest.get('step'),
                  receipt_step_shape=_value_shape(receipt.get('step')),
                  manifest_step_shape=_value_shape(manifest.get('step')))
    if not all(_integer_scalar(value) for value in (manifest.get('step'), receipt.get('step'))):
        _resume_audit(audit_path, values, False, 'receipt_manifest_step')
        raise ValueError('Checkpoint receipt/manifest steps must be finite integer scalars')
    steps_match = int(manifest['step']) == int(receipt['step'])
    if not steps_match:
        _resume_audit(audit_path, values, False, 'receipt_manifest_step')
        raise ValueError('Checkpoint receipt step differs')
    expected = set(manifest['files']) | {'manifest.json', 'receipt.json'}
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Checkpoint inventory differs')
    for name, digest in manifest['files'].items():
        p = path / name
        if p.is_symlink() or not p.resolve().is_relative_to(path.resolve()) or sha256(p) != digest:
            raise ValueError('Checkpoint payload differs: ' + name)
    return receipt


def restore_checkpoint(path, target, expected_metadata, audit_path=None):
    path = Path(path)
    receipt = verify_checkpoint(path, audit_path=audit_path)
    metadata = json.loads((path / 'metadata.json').read_text())
    _metadata_identity(metadata)
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError('Resume binding differs: ' + key)
    # Flax restores against the target structure before returning a State. Audit
    # the uncoerced msgpack values first, including malformed optimizer trees
    # that from_state_dict rejects before the existing restored-state gate.
    raw = serialization.msgpack_restore((path / 'state.msgpack').read_bytes())
    raw_step = raw.get('step') if isinstance(raw, dict) else None
    raw_optimizer = raw.get('opt_state') if isinstance(raw, dict) else None
    raw_counters = {f'optimizer_count_{key}': component['count']
                    for key, component in (raw_optimizer.items()
                        if isinstance(raw_optimizer, dict) else ())
                    if isinstance(component, dict) and 'count' in component}
    raw_values = _state_audit_values(receipt, metadata, raw_step, raw_counters)
    raw_values.update(raw_step=raw_step,
        raw_optimizer_structure=_serialized_structure(raw_optimizer),
        target_optimizer_structure=_serialized_structure(serialization.to_state_dict(target.opt_state)))
    _resume_audit(audit_path, raw_values, False, 'raw_state')
    state = serialization.from_state_dict(target, raw)
    restored_step = np.asarray(state.step)
    counters = {f'optimizer_count_{i}': np.asarray(component.count)
                for i, component in enumerate(state.opt_state)
                if 'count' in getattr(component, '_fields', ())}
    values = _state_audit_values(receipt, metadata, restored_step, counters)
    # Retain the deciding raw values even if shape/dtype compatibility rejects
    # the checkpoint before the scalar numerical gate can run.
    _resume_audit(audit_path, values, False, 'restored_state')
    assert_tree_compatible(state, target)
    if len(counters) != 2:
        raise ValueError('Expected Adam and learning-rate schedule counters')
    steps = dict(receipt_step=receipt['step'], restored_step=restored_step,
                 metadata_step=metadata.get('step'), **counters)
    passed = all(_integer_scalar(value) for value in steps.values())
    passed = passed and len({int(value) for value in steps.values()}) == 1
    _resume_audit(audit_path, values, passed, 'restored_state')
    if not passed:
        raise ValueError('Restored step or Adam count differs from receipt')
    # Normalize the scalar step dtype only after asserting its exact value.
    state = state.replace(step=np.asarray(restored_step, dtype=np.int32))
    return state, metadata
