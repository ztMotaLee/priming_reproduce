"""Registered D5 dose diagnostics; observations never retune or gate training."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np


def flatten(tree, prefix=()):
    for key, value in tree.items():
        if hasattr(value, 'items'):
            yield from flatten(value, prefix + (str(key),))
        else:
            yield prefix + (str(key),), value


def role_for_path(path):
    """Preserve the registered generator parameter-role precedence."""
    p = '/'.join(path)
    tokens = set(path)
    if 't_embedder' in tokens or 'pooled_embedder' in tokens:
        return 'inactive'
    if any(s in p for s in ('txt_', 'aux_pooled_embedder', 'mask_token', 'context_tokens')):
        return 'text_interface'
    if any(s in p for s in ('omega_', 'h_embedder', 't_min_', 't_max_', 'time_tokens')):
        return 'controls'
    return 'synthesis'


def gradient_metric_function(jax, jnp):
    """Original pre-clip global and role metrics, evaluated on reduced gradients."""
    def metrics(native, field, weight):
        result = {}
        a, b = dict(flatten(native)), dict(flatten(field))
        for group in ('all', 'synthesis', 'text_interface', 'controls', 'inactive'):
            selected = [path for path in a if group == 'all' or role_for_path(path) == group]
            if not selected:
                continue
            ns = sum(jnp.sum(jnp.square(a[p].astype(jnp.float32))) for p in selected)
            fs = sum(jnp.sum(jnp.square(b[p].astype(jnp.float32))) for p in selected)
            joint = sum(jnp.sum(jnp.square(a[p]+weight*b[p])) for p in selected)
            dot = sum(jnp.sum(a[p]*b[p]) for p in selected)
            nn, fn, cn = jnp.sqrt(ns), jnp.sqrt(fs), jnp.sqrt(joint)
            result[group] = dict(native_norm=nn, field_norm=fn, combined_norm=cn,
                                 rho=jnp.where(nn > 0, weight*fn/nn, jnp.nan),
                                 gradient_cosine=jnp.where(nn*fn > 0, dot/(nn*fn), jnp.nan))
        result['clip_factor'] = jnp.minimum(1., 1./result['all']['combined_norm'])
        result['clipped'] = result['all']['combined_norm'] > 1.
        return result
    return jax.pmap(metrics, in_axes=(0, 0, None))


def update_diagnostics(jax, a, b):
    """Original float64 host comparison of actual clip+AdamW output updates."""
    aa = jax.tree_util.tree_leaves(a)
    bb = jax.tree_util.tree_leaves(b)
    if len(aa) != len(bb) or not aa:
        raise ValueError('Update trees differ')
    base_sq = combined_sq = diff_sq = dot = 0.0
    for x, y in zip(aa, bb):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        base_sq += float(np.sum(x * x))
        combined_sq += float(np.sum(y * y))
        diff_sq += float(np.sum((y - x) ** 2))
        dot += float(np.sum(x * y))
    if not (base_sq > 0 and combined_sq > 0 and math.isfinite(diff_sq)):
        raise ValueError('Invalid actual optimizer update')
    return dict(relative_update_difference=math.sqrt(diff_sq / base_sq),
                combined_update_norm_ratio=math.sqrt(combined_sq / base_sq),
                update_cosine=dot / math.sqrt(base_sq * combined_sq))


def load_dose_history(directory, through_step):
    """Recover applied records only; a checkpoint rollback supersedes later logs.

    A resumed segment may repeat an unsaved tail. Its last applied record wins;
    absent or invalid observations remain visibly incomplete diagnostics.
    """
    result = {}
    path = Path(directory)/'training.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
                if not isinstance(row,dict):continue
                step = int(row['step'])
            except (json.JSONDecodeError,KeyError,TypeError,ValueError):
                # An interrupted append is missing diagnostic evidence, not a
                # reason to reject a separately verified full-state checkpoint.
                continue
            if row.get('status') == 'APPLIED' and 1 <= step <= through_step:
                result[step] = row.get('rho_t')
    return result


def append_training_record(path, record):
    """Append without altering old bytes; isolate an interrupted final record."""
    with Path(path).open('ab+') as stream:
        stream.seek(0,2)
        if stream.tell():
            stream.seek(-1,2)
            if stream.read(1)!=b'\n':stream.write(b'\n')
        stream.write((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode())


def dose_status(by_step, through_step, *, rho0, calibration_receipt_sha256):
    """Report inclusive D5 windows with unchanged strict thresholds.

    A future/missing window is explicit rather than a training approval gate.
    A flagged completed window cannot establish field ineffectiveness.
    """
    def valid(value):
        return value is not None and math.isfinite(float(value)) and float(value) >= 0
    initial = float(rho0)
    windows = []
    for start, end in ((901, 1000), (1901, 2000)):
        observed = [float(by_step[i]) for i in range(start, end+1)
                    if i <= through_step and valid(by_step.get(i))]
        complete = len(observed) == end-start+1
        median = float(np.median(observed)) if complete else None
        mark = ('剂量塌缩' if median < initial/10 else '剂量主导' if median > 10*initial else None) if complete else None
        windows.append(dict(start=start, end=end, observed_steps=len(observed),
                            expected_steps=end-start+1,
                            status='COMPLETE' if complete else 'PENDING' if through_step < end else 'INCOMPLETE',
                            median_rho=median, rho0=initial, lower=initial/10, upper=10*initial, flag=mark))
    flagged = any(x['flag'] is not None for x in windows)
    complete = all(x['status'] == 'COMPLETE' for x in windows) and valid(by_step.get(1))
    status = ('UNMAINTAINED' if flagged else 'MAINTAINED' if complete
              else 'INCOMPLETE' if through_step >= 2000 or any(x['status'] == 'INCOMPLETE' for x in windows)
              else 'PENDING')
    return dict(schema='bigfield-dose-diagnostics-v1', through_step=int(through_step),
                rho0=initial, calibration_receipt_sha256=calibration_receipt_sha256,
                initial_step_present=valid(by_step.get(1)), windows=windows, status=status,
                dose_unmaintained=True if flagged else False if complete else None,
                diagnostic_only=True)
