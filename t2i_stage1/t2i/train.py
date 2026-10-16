"""Two-stage Priming reconstruction pretraining. Never submits scheduler jobs."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils
from flax.training import train_state
from jax.experimental import multihost_utils

from .checkpoint import (atomic_json, claim_run, read_run_owner, restore_checkpoint,
                         save_checkpoint, sha256)
from .config import Recipe, batch_layout
from .run_identity import directory_identity


class State(train_state.TrainState):
    ema_params: object
    rng: object


def create_state(params, recipe):
    tx = optax.adamw(recipe.learning_rate, b1=recipe.beta1, b2=recipe.beta2,
                     weight_decay=recipe.weight_decay)
    return State.create(apply_fn=None, params=params, tx=tx, ema_params=params,
                        rng=jax.random.PRNGKey(recipe.seed))


def make_update(model, encode, perceptual, recipe, noise_mode, microbatch):
    from .noise import sample_noise_levels, corrupt_inputs
    from .losses import reconstruction_loss

    def update(state, images, ids, masks, text_params, perceptual_params):
        # Local images: [accumulation, microbatch, H, W, C].
        n = images.shape[0]
        base_rng = jax.random.fold_in(state.rng, state.step)
        base_rng = jax.random.fold_in(base_rng, jax.lax.axis_index('data'))
        def body(carry, batch):
            grads_sum, loss_sum, mse_sum = carry
            image, token, mask, index = batch
            noise_key, s_key, crop_key = jax.random.split(jax.random.fold_in(base_rng, index), 3)
            fixed = .8 if noise_mode != 'fixed-1.0' else 1.
            mode = 'logit_normal' if noise_mode == 'logit-normal' else 'fixed'
            s = sample_noise_levels(s_key, microbatch, mode=mode, fixed_s=fixed)
            noisy = corrupt_inputs(noise_key, image, s, noise_std=recipe.noise_std)
            context = jax.lax.stop_gradient(encode(text_params, token, mask))
            def loss(params):
                pred = jax.checkpoint(model.apply)({'params': params}, noisy, context, mask)
                pf = partial(perceptual.apply, perceptual_params)
                return reconstruction_loss(pred, image, s, perceptual_fn=pf,
                                           perceptual_threshold=recipe.perceptual_threshold,
                                           perceptual_rng=crop_key)
            (value, metrics), grads = jax.value_and_grad(loss, has_aux=True)(state.params)
            return (jax.tree_util.tree_map(jnp.add, grads_sum, grads), loss_sum + value,
                    mse_sum + metrics['loss_mse']), None
        carry = (jax.tree_util.tree_map(jnp.zeros_like, state.params), jnp.float32(0), jnp.float32(0))
        (grads, loss, mse), _ = jax.lax.scan(body, carry, (images, ids, masks, jnp.arange(n)))
        grads = jax.tree_util.tree_map(lambda g: jax.lax.pmean(g / n, 'data'), grads)
        new_state = state.apply_gradients(grads=grads)
        ema = jax.tree_util.tree_map(lambda e, p: recipe.ema_decay * e + (1-recipe.ema_decay) * p,
                                    state.ema_params, new_state.params)
        new_state = new_state.replace(ema_params=ema)
        return new_state, {'loss': jax.lax.pmean(loss / n, 'data'),
                           'loss_mse': jax.lax.pmean(mse / n, 'data'),
                           'grad_norm': optax.global_norm(grads),
                           'lr': recipe.learning_rate(state.step)}
    # EMA initially shares the parameter buffers; donation would alias on step 1.
    return jax.pmap(update, axis_name='data')


def source_digest():
    root = Path(__file__).resolve().parents[1]
    files = list((root / 't2i').glob('*.py')) + list((root / 'models').glob('*.py'))
    files += list((root / 'utils').glob('*.py'))
    values = {str(p.relative_to(root)): sha256(p) for p in sorted(files)}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def visible_gpu_ids(environ):
    """Enumerate CUDA visibility without initializing any JAX backend."""
    if 'CUDA_VISIBLE_DEVICES' in environ:
        value = environ['CUDA_VISIBLE_DEVICES'].strip()
        return [] if value in ('', '-1') else [v.strip() for v in value.split(',')]
    # This branch is runtime-only on the GPU machine; CPU tests mock the probe.
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=15)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def distributed_configuration(environ=None):
    """Bind one process per node to every visible GPU before JAX discovery."""
    env = os.environ if environ is None else environ
    names = ('JAX_PROCESS_COUNT', 'JAX_PROCESS_INDEX', 'JAX_COORDINATOR_ADDRESS',
             'SLURM_JOB_ID', 'SLURM_NNODES', 'SLURM_NTASKS', 'SLURM_PROCID',
             'SLURM_NODEID', 'SLURM_LOCALID', 'STAGE1_GPUS_PER_NODE', 'GPUS_PER_NODE',
             'SLURM_GPUS_PER_TASK', 'SLURM_GPUS_PER_NODE', 'SLURM_GPUS_ON_NODE',
             'CUDA_VISIBLE_DEVICES')
    raw = {name: env.get(name) for name in names}
    errors = []
    def number(name, value, minimum=1):
        try:
            parsed = int(value)
            if parsed < minimum:
                raise ValueError
            return parsed
        except (ValueError, TypeError):
            errors.append('Invalid ' + name)
            return None
    size = number('process count', env.get('JAX_PROCESS_COUNT', env.get('SLURM_NNODES', '1')))
    rank = number('process index', env.get('JAX_PROCESS_INDEX', env.get('SLURM_PROCID',
                  env.get('SLURM_NODEID', '0'))), minimum=0)
    requests = {}
    for name in ('STAGE1_GPUS_PER_NODE', 'GPUS_PER_NODE', 'SLURM_GPUS_PER_TASK',
                 'SLURM_GPUS_PER_NODE', 'SLURM_GPUS_ON_NODE'):
        if name in env:
            requests[name] = number(name, env[name].rsplit(':', 1)[-1])
    try:
        visible = visible_gpu_ids(env)
    except (OSError, subprocess.SubprocessError) as exc:
        visible = []
        errors.append('GPU visibility probe failed: ' + str(exc))
    expected_local = next(iter(requests.values()), None)
    if not requests:
        if 'SLURM_JOB_ID' in env or size != 1:
            errors.append('Explicit GPUs-per-node request required for Slurm/multi-node')
        else:
            expected_local = len(visible)
    if expected_local is None or expected_local <= 0:
        errors.append('Expected GPU count must be positive')
    if any(value != expected_local for value in requests.values()):
        errors.append('GPU requests disagree')
    if len(visible) != expected_local or len(set(visible)) != len(visible):
        errors.append('Visible GPU count differs from request or contains duplicates')
    if size is not None and rank is not None and rank >= size:
        errors.append('Process index outside process count')
    for name in ('SLURM_NNODES', 'SLURM_NTASKS'):
        if name in env and number(name, env[name]) != size:
            errors.append(name + ' differs from one-process-per-node request')
    for name in ('SLURM_PROCID', 'SLURM_NODEID'):
        if name in env and number(name, env[name], minimum=0) != rank:
            errors.append(name + ' differs from process index')
    if 'SLURM_LOCALID' in env and number('SLURM_LOCALID', env['SLURM_LOCALID'], minimum=0) != 0:
        errors.append('Launcher requires one process per node')
    address = env.get('JAX_COORDINATOR_ADDRESS')
    if size is not None and size > 1 and not address:
        errors.append('Missing JAX_COORDINATOR_ADDRESS')
    return dict(raw=raw, expected_processes=size, expected_rank=rank,
                requested_gpu_counts=requests, expected_local_devices=expected_local,
                expected_global_devices=(size * expected_local
                    if size is not None and expected_local is not None else None),
                visible_gpu_ids=visible, visible_gpu_count=len(visible),
                local_device_ids=list(range(len(visible))),
                coordinator_address=address, errors=errors, passed=not errors)


def initialize_distributed(audit_path=None):
    configuration = distributed_configuration()
    if audit_path is None:
        # Output ownership is established only after distributed initialization.
        # Use a new per-process directory so early failures cannot alter a run.
        directory = tempfile.mkdtemp(prefix='stage1-distributed-', dir=os.environ.get('TMPDIR', '/tmp'))
        audit_path = Path(directory) / 'preflight.json'
    configuration['audit_path'] = str(audit_path)
    configuration['phase'] = 'before_initialize'
    atomic_json(audit_path, configuration)
    print('Distributed preflight receipt: ' + str(audit_path), file=sys.stderr, flush=True)
    if not configuration['passed']:
        raise ValueError('Invalid GPU request/visibility; inspect distributed preflight receipt')
    try:
        if configuration['expected_processes'] > 1:
            # JAX 0.4.31 otherwise infers [SLURM_LOCALID], exposing only one GPU.
            jax.distributed.initialize(coordinator_address=configuration['coordinator_address'],
                num_processes=configuration['expected_processes'],
                process_id=configuration['expected_rank'],
                local_device_ids=configuration['local_device_ids'])
    except Exception as exc:
        configuration.update(phase='initialize_failed', passed=False,
                             initialization_error=repr(exc))
        atomic_json(audit_path, configuration)
        raise
    configuration['phase'] = 'initialized'
    atomic_json(audit_path, configuration)
    return configuration


def runtime_topology_gate(expected):
    """Return every actual/expected value; the caller persists before raising."""
    local = jax.local_devices()
    devices = jax.devices()
    actual = dict(processes=jax.process_count(), rank=jax.process_index(),
                  local_devices=jax.local_device_count(), global_devices=jax.device_count(),
                  local_inventory=[dict(id=d.id, process_index=d.process_index,
                      platform=d.platform, kind=getattr(d, 'device_kind', '')) for d in local],
                  global_inventory=[dict(id=d.id, process_index=d.process_index,
                      platform=d.platform, kind=getattr(d, 'device_kind', '')) for d in devices])
    expected_per_process = {rank: expected['expected_local_devices']
                           for rank in range(expected['expected_processes'])}
    actual_per_process = {rank: sum(d.process_index == rank for d in devices)
                          for rank in expected_per_process}
    passed = (expected['passed'] and actual['processes'] == expected['expected_processes']
              and actual['rank'] == expected['expected_rank']
              and actual['local_devices'] == len(local) == expected['expected_local_devices']
              and actual['global_devices'] == len(devices) == expected['expected_global_devices']
              and actual_per_process == expected_per_process
              and all(d.platform == 'gpu' for d in devices + local))
    return dict(expected=expected, actual=actual, actual_devices_per_process=actual_per_process,
                expected_devices_per_process=expected_per_process, passed=passed)


def gather_json(value):
    encoded = json.dumps(value, sort_keys=True).encode()
    if len(encoded) > 65532:
        raise ValueError('Data cursor exceeds the 64 KiB metadata bound')
    buf = np.zeros(65536, np.uint8)
    buf[:4] = np.frombuffer(len(encoded).to_bytes(4, 'little'), np.uint8)
    buf[4:4+len(encoded)] = np.frombuffer(encoded, np.uint8)
    gathered = np.asarray(multihost_utils.process_allgather(buf, tiled=False)).reshape(-1, 65536)
    return [json.loads(bytes(row[4:4+int.from_bytes(bytes(row[:4]), 'little')])) for row in gathered]


def validate_assets(path, kind):
    """Validate actual local payloads before model initialization; never download."""
    if kind == 't5':
        from .assets import validate_t5_assets
        return validate_t5_assets(path, verify_payloads=True)['manifest_sha256']
    if kind != 'perceptual':
        raise ValueError('Unknown asset kind: ' + kind)
    from .perceptual_assets import validate_perceptual_assets
    return validate_perceptual_assets(path, verify_payloads=True)['manifest_sha256']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cc12m-root', default=os.environ.get('CC12M_ROOT', ''))
    p.add_argument('--curated-root', default=os.environ.get('CURATED_ROOT', ''))
    p.add_argument('--output-dir', default=os.environ.get('OUTPUT_DIR', 'runs/stage1'))
    p.add_argument('--resume', default=os.environ.get('RESUME', ''))
    p.add_argument('--t5-model', default=os.environ.get('T5_MODEL', 'data/assets/t5'))
    p.add_argument('--perceptual-dir', default=os.environ.get('PERCEPTUAL_DIR', 'data/assets/perceptual'))
    p.add_argument('--steps', type=int, default=290000)
    p.add_argument('--checkpoint-every', type=int, default=5000)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--noise', choices=['logit-normal', 'fixed-0.8', 'fixed-1.0'], default='logit-normal')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    distributed = initialize_distributed()  # Must precede JAX backend discovery.
    rank = jax.process_index()
    out = Path(args.output_dir).resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    multihost_utils.sync_global_devices('output-directory')
    recipe = Recipe()
    source = source_digest()
    # Persist every numerical gate input before raising, even on failure.
    topology = runtime_topology_gate(distributed)
    gate = dict(steps=args.steps, checkpoint_every=args.checkpoint_every,
                microbatch=args.microbatch, max_steps=290000,
                processes=jax.process_count(), local_devices=jax.local_device_count(),
                global_devices=jax.device_count(), global_batch=recipe.global_batch,
                smoke=args.smoke, devices=[str(d) for d in jax.local_devices()],
                topology=topology)
    try:
        layout = batch_layout(recipe.global_batch, jax.process_count(),
                              jax.local_device_count(), args.microbatch)
        valid = (0 < args.steps <= 290000 and args.checkpoint_every > 0
                 and (args.smoke or args.steps == 290000)
                 and topology['passed'])
    except ValueError:
        layout = None; valid = False
    gate.update(layout=layout, passed=valid)
    if not valid:
        # Rejected launches must not modify an existing run's receipts.
        directory = tempfile.mkdtemp(prefix='stage1-rejected-', dir=os.environ.get('TMPDIR', '/tmp'))
        audit = Path(directory) / f'preflight-rank{rank}.json'
        atomic_json(audit, gate)
        print('Rejected configuration receipt: ' + str(audit), file=sys.stderr, flush=True)
        raise ValueError('Invalid production configuration; inspect preflight receipt')
    assets = dict(t5=validate_assets(args.t5_model, 't5'),
                  perceptual=validate_assets(args.perceptual_dir, 'perceptual'))
    from .data import WebDatasetStream
    from .model import build_model
    from .perceptual import load_perceptual
    from .text import load_text_encoder
    common = dict(process_index=rank, process_count=jax.process_count(), seed=recipe.seed,
                  image_size=recipe.image_size)
    streams = {'A': WebDatasetStream(args.cc12m_root, **common),
               'B': WebDatasetStream(args.curated_root, **common)}
    binding = dict(source_sha256=source, recipe=recipe.as_dict(), noise=args.noise,
                   layout=layout, assets=assets,
                   data={stage: stream.manifest_sha256 for stage, stream in streams.items()})
    # No existing output file is written until source/configuration/data/assets
    # and the exact resume provenance have passed the run admission gate.
    admission = {}
    if rank == 0:
        run, parent_run = claim_run(out, binding, args.resume)
        admission = dict(run=run, parent_run=parent_run)
    multihost_utils.sync_global_devices('output-owner')
    admission = gather_json(admission)[0]
    run, parent_run = admission['run'], admission['parent_run']
    if run != read_run_owner(out) or run['binding'] != binding:
        raise ValueError('Run configuration differs between processes')
    binding['run_identity'] = run
    gate['run_identity_sha256'] = run['sha256']
    atomic_json(out / f'preflight-rank{rank}.json', gate)
    model = build_model(image_size=512)
    # Real model initialization belongs on the allocated GPU node only.
    params = jax.jit(lambda k: model.init(k, jnp.zeros((1,512,512,3)),
                    jnp.zeros((1,256,1024)), jnp.ones((1,256)))['params'])(jax.random.PRNGKey(recipe.seed))
    state = create_state(params, recipe)
    if args.resume:
        state, metadata = restore_checkpoint(args.resume, state, {**binding, 'run_identity': parent_run},
                                             audit_path=out / f'restore-gate-rank{rank}.json')
        for stage in ('A', 'B'):
            streams[stage].load_state_dict(metadata['data_states'][rank][stage])
    start = int(np.asarray(state.step))
    atomic_json(out / f'resume-rank{rank}.json', dict(start_step=start, target_step=args.steps,
                                                   passed=start < args.steps))
    if start >= args.steps:
        raise ValueError('Resume target must be later than restored step')
    encode, text_params, tokenize = load_text_encoder(args.t5_model)
    perceptual = load_perceptual(args.perceptual_dir)
    state = jax_utils.replicate(state)
    text_params = jax_utils.replicate(text_params)
    pp = jax_utils.replicate(perceptual.params)
    update = make_update(model, encode, perceptual, recipe, args.noise, args.microbatch)
    stop = {'requested': False}
    def request_stop(signum, frame):
        stop['requested'] = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    wall_start = time.monotonic()
    for step in range(start, args.steps):
        stage = recipe.stage(step)
        batch = streams[stage].next_batch(layout['per_process'])
        text = tokenize(batch['captions'])
        prefix = (layout['local_devices'], layout['accumulation_steps'], args.microbatch)
        images = batch['images'].reshape(*prefix,512,512,3)
        ids = text['input_ids'].reshape(*prefix,256)
        masks = text['attention_mask'].reshape(*prefix,256)
        state, metrics = update(state, images, ids, masks, text_params, pp)
        values = {k: float(np.asarray(v)[0]) for k, v in jax.device_get(metrics).items()}
        finite = all(np.isfinite(v) for v in values.values())
        audited = {k: v if np.isfinite(v) else str(v) for k,v in values.items()}
        atomic_json(out / f'progress-rank{rank}.json', dict(step=step+1, stage=stage,
                    metrics=audited, finite=finite, elapsed_seconds=time.monotonic()-wall_start,
                    target_step=args.steps))
        if not finite:
            raise FloatingPointError('Nonfinite update; all gate metrics persisted to progress receipt')
        requested = bool(np.any(multihost_utils.process_allgather(np.int32(stop['requested']))))
        save = ((step+1) % args.checkpoint_every == 0 or step+1 in (250000,args.steps) or requested)
        if save:
            data_states = gather_json({k:v.state_dict() for k,v in streams.items()})
            if rank == 0:
                cpu_state = jax.device_get(jax_utils.unreplicate(state))
                save_checkpoint(out, cpu_state, {**binding, 'data_states': data_states})
            multihost_utils.sync_global_devices(f'checkpoint-{step+1}')
        if rank == 0 and ((step+1) % 10 == 0 or save):
            print(json.dumps(dict(step=step+1,stage=stage,**audited)), flush=True)
        if requested:
            break
    if jax.process_count() > 1:
        jax.distributed.shutdown()


if __name__ == '__main__':
    main()
