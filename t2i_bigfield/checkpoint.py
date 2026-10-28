"""Atomic full-state checkpoints; exact AdamW moment restoration."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
import numpy as np
def scalar_step(value):
    a = np.asarray(value)
    if a.shape != () or a.dtype.kind not in 'iuf' or (not np.isfinite(a)):
        raise ValueError('Checkpoint step must be a finite integral scalar')
    result = int(a)
    if result < 0 or float(a) != result:
        raise ValueError('Checkpoint step must be nonnegative integral')
    return result

def restore_adamw_state(tx, params, saved_state, expected_updates):
    """Restore exact moments, allowing Orbax's empty-NamedTuple encoding only."""
    from flax import serialization
    target = tx.init(params)
    template = serialization.to_state_dict(target)
    raw = saved_state if isinstance(saved_state, dict) else serialization.to_state_dict(saved_state)

    def reject(path, reason):
        raise ValueError('Saved AdamW state at ' + path + ': ' + reason)

    def normalize(schema, saved, target_value, path='opt_state'):
        if saved is None:
            if isinstance(schema, dict) and (not schema) and isinstance(target_value, tuple) and (getattr(target_value, '_fields', None) == ()):
                return {}
            reject(path, 'unexpected None')
        if isinstance(schema, dict):
            if not isinstance(saved, dict):
                reject(path, 'expected state dictionary, got ' + type(saved).__name__)
            missing = set(schema) - set(saved)
            extra = set(saved) - set(schema)
            if missing:
                reject(path + '/' + str(sorted(missing, key=str)[0]), 'missing key')
            if extra:
                reject(path + '/' + str(sorted(extra, key=str)[0]), 'extra key')
            result = {}
            for (key, value) in schema.items():
                if hasattr(target_value, '_fields'):
                    child = getattr(target_value, key)
                elif isinstance(target_value, (tuple, list)):
                    child = target_value[int(key)]
                else:
                    child = target_value[key]
                result[key] = normalize(value, saved[key], child, path + '/' + str(key))
            return result
        if isinstance(saved, dict):
            reject(path, 'expected array, got state dictionary')
        expected_shape = tuple(schema.shape)
        expected_dtype = np.dtype(schema.dtype)
        try:
            actual = np.asarray(saved)
        except (TypeError, ValueError) as error:
            reject(path, 'invalid array: ' + str(error))
        if actual.shape != expected_shape:
            reject(path, 'shape ' + str(actual.shape) + ' != ' + str(expected_shape))
        if actual.dtype != expected_dtype:
            reject(path, 'dtype ' + str(actual.dtype) + ' != ' + str(expected_dtype))
        return saved
    normalized = normalize(template, raw, target)
    state = serialization.from_state_dict(target, normalized)
    counts = optimizer_counts(state)
    if not counts or any((value != expected_updates for (_, value) in counts)):
        bad = [path for (path, value) in counts if value != expected_updates]
        raise ValueError('Saved AdamW counters do not equal posttrain updates at ' + ','.join(bad or ['opt_state']))
    verify_adamw_state_bytes(state, raw)
    return state

def verify_adamw_state_bytes(state, raw_saved_state):
    """Independently compare every count/mu/nu leaf with raw checkpoint arrays.

    This comparison deliberately consumes the unnormalized target=None tree;
    empty transform states contribute no AdamW leaves. It neither initializes
    an optimizer nor relies on tree digests or the schema-normalization pass.
    """
    from flax import serialization

    def leaves(value, path=(), moment=None):
        if isinstance(value, dict):
            for (key, child) in value.items():
                role = key if moment is None and key in ('count', 'mu', 'nu') else moment
                yield from leaves(child, path + (str(key),), role)
        elif moment is not None:
            yield (path, value)
    restored = dict(leaves(serialization.to_state_dict(state)))
    saved = raw_saved_state if isinstance(raw_saved_state, dict) else serialization.to_state_dict(raw_saved_state)
    raw = dict(leaves(saved))
    if set(restored) != set(raw):
        path = sorted(set(restored) ^ set(raw))[0]
        raise ValueError('AdamW byte verification leaf mismatch at opt_state/' + '/'.join(path))
    if not restored:
        raise ValueError('AdamW byte verification has no count/mu/nu leaves at opt_state')
    for (path, value) in restored.items():
        expected = raw[path]
        location = 'opt_state/' + '/'.join(path)
        if value is None or expected is None:
            raise ValueError('AdamW byte verification found None at ' + location)
        actual_array = np.asarray(value)
        raw_array = np.asarray(expected)
        if actual_array.shape != raw_array.shape or actual_array.dtype != raw_array.dtype or actual_array.tobytes() != raw_array.tobytes():
            raise ValueError('AdamW count/mu/nu dtype/shape/bytes differ at ' + location)
    return len(restored)

def optimizer_counts(state):

    def visit(value, path=''):
        if hasattr(value, '_fields'):
            for name in value._fields:
                child = getattr(value, name)
                if name == 'count':
                    yield [path + '.count', scalar_step(child)]
                else:
                    yield from visit(child, path + '.' + name)
        elif isinstance(value, dict):
            for name in sorted(value):
                if name == 'count':
                    yield [path + '/count', scalar_step(value[name])]
                else:
                    yield from visit(value[name], path + '/' + str(name))
        elif isinstance(value, (tuple, list)):
            for (i, child) in enumerate(value):
                yield from visit(child, path + '/' + str(i))
    return list(visit(state))
def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda : stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()

def tree_digest(tree):
    """Digest serialized state paths and exact array dtype/shape/bytes."""
    from flax import serialization
    h = hashlib.sha256()

    def walk(value, path=()):
        if isinstance(value, dict):
            for k in sorted(value, key=str):
                walk(value[k], path + (str(k),))
        elif value is None:
            h.update(json.dumps([path, 'None']).encode())
        else:
            a = np.asarray(value)
            h.update(json.dumps([path, str(a.dtype), a.shape]).encode())
            h.update(a.tobytes())
    walk(serialization.to_state_dict(tree))
    return h.hexdigest()

def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def numeric(value):
    """Strict JSON data, retaining non-finite gate evidence as explicit strings."""
    if isinstance(value, dict):
        return {str(k): numeric(v) for (k, v) in value.items()}
    if isinstance(value, (tuple, list)):
        return [numeric(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return numeric(value.tolist())
    if hasattr(value, 'shape'):
        return numeric(np.asarray(value))
    if isinstance(value, float) and (not math.isfinite(value)):
        return str(value)
    return value

def atomic_json(path, value, *, create_once=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(numeric(value), f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    if create_once:
        os.link(tmp, path)
        tmp.unlink()
    else:
        os.replace(tmp, path)
    fsync_directory(path.parent)

PERMANENT_STEPS = (1000,2000,4000,8000,12000,16000,24000,30000)

def segment_report_path(output, start_step, end_step):
    """Keep each invocation's status without preventing manual continuation."""
    return Path(output)/('segment_%06d_%06d_%s.json' % (start_step,end_step,uuid.uuid4().hex))

def _rng_for_step(seed, step):
    import jax
    key = jax.random.PRNGKey(seed)
    return np.asarray(key if step == 0 else jax.random.fold_in(key, step),np.uint32)


def save_checkpoint(directory, *, params, opt_state, step, rng, training_seed, binding):
    from flax import serialization
    directory=Path(directory)
    if directory.exists(): raise FileExistsError(directory)
    step=scalar_step(step)
    counts=optimizer_counts(opt_state)
    if not counts or any(n!=step for _,n in counts): raise ValueError('AdamW clocks differ')
    if set(params)!={'pt_net'} or not np.array_equal(rng,_rng_for_step(training_seed,step)):
        raise ValueError('Generator branch or RNG clock differs')
    payload=dict(params=params,opt_state=serialization.to_state_dict(opt_state),
                 step=np.int32(step),rng=np.asarray(rng,np.uint32),data_cursor=np.int64(64*step),
                 training_seed=np.int64(training_seed),binding=binding)
    raw=serialization.msgpack_serialize(serialization.to_state_dict(payload),in_place=False)
    directory.parent.mkdir(parents=True,exist_ok=True)
    temporary=directory.with_name('.'+directory.name+'.'+uuid.uuid4().hex);temporary.mkdir()
    with (temporary/'state.msgpack').open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    # A params-only payload is the optional return artifact for independent evaluation.
    params_raw=serialization.msgpack_serialize(serialization.to_state_dict(params),in_place=False)
    with (temporary/'params.msgpack').open('xb') as f:
        f.write(params_raw);f.flush();os.fsync(f.fileno())
    manifest=dict(schema='bigfield-checkpoint-v1',step=step,data_cursor=64*step,
                  training_seed=training_seed,rng=np.asarray(rng,np.uint32).tolist(),binding=binding,
                  parameter_sha256=tree_digest(params),optimizer_sha256=tree_digest(payload['opt_state']),
                  files_sha256={name:sha256_file(temporary/name) for name in ('state.msgpack','params.msgpack')},
                  permanent=step in PERMANENT_STEPS)
    atomic_json(temporary/'checkpoint.json',manifest,create_once=True)
    fsync_directory(temporary);os.rename(temporary,directory);fsync_directory(directory.parent)
    return manifest


def restore_checkpoint(directory, *, tx, binding, expected_step=None):
    from flax import serialization
    directory=Path(directory);manifest=json.loads((directory/'checkpoint.json').read_text())
    if manifest.get('schema')!='bigfield-checkpoint-v1' or manifest.get('binding')!=binding:
        raise ValueError('Checkpoint scientific configuration/data/weights binding differs')
    if expected_step is not None and manifest['step']!=expected_step:
        raise ValueError('Checkpoint step differs')
    for name,sha in manifest['files_sha256'].items():
        if sha256_file(directory/name)!=sha: raise ValueError('Checkpoint payload SHA differs: '+name)
    payload=serialization.msgpack_restore((directory/'state.msgpack').read_bytes())
    step=scalar_step(payload['step']);seed=scalar_step(payload['training_seed'])
    if (step!=manifest['step'] or scalar_step(payload['data_cursor'])!=step*64
        or seed!=manifest['training_seed'] or payload['binding']!=binding
        or np.asarray(payload['rng']).shape!=(2,) or np.asarray(payload['rng']).dtype!=np.dtype('uint32')
        or not np.array_equal(payload['rng'],_rng_for_step(seed,step))
        or np.asarray(payload['rng']).tolist()!=manifest['rng']
        or tree_digest(payload['params'])!=manifest['parameter_sha256']
        or tree_digest(payload['opt_state'])!=manifest['optimizer_sha256']):
        raise ValueError('Checkpoint state, clock, RNG, cursor or digest differs')
    payload['opt_state']=restore_adamw_state(tx,payload['params'],payload['opt_state'],step)
    return payload,manifest


def load_params(directory):
    """Load only evaluation params after checking the payload SHA and tree digest."""
    from flax import serialization
    directory=Path(directory);meta=json.loads((directory/'checkpoint.json').read_text())
    path=directory/'params.msgpack'
    if sha256_file(path)!=meta['files_sha256']['params.msgpack']:
        raise ValueError('Params payload SHA differs')
    params=serialization.msgpack_restore(path.read_bytes())
    if set(params)!={'pt_net'} or tree_digest(params)!=meta['parameter_sha256']:
        raise ValueError('Params branch or tree SHA differs')
    return params
