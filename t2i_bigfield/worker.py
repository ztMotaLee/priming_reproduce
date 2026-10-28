"""Allocated workflow stages and a lightweight progress/return-bundle reader.

Scientific operations delegate to the frozen preparation, Runtime and evaluator.
This module imports only the standard library until an allocated stage starts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import time
import uuid

HERE = Path(__file__).resolve().parent
READOUTS = (1000, 2000, 4000, 8000, 12000, 16000, 24000, 30000)
CACHES = ('TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE',
          'PYTORCH_PRETRAINED_BERT_CACHE', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE',
          'JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_COMPILATION_CACHE')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def atomic(path, value, *, once=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        def numeric(item):
            if isinstance(item, dict):
                return {str(k): numeric(v) for k, v in item.items()}
            if isinstance(item, (list, tuple)):
                return [numeric(v) for v in item]
            return str(item) if isinstance(item, float) and not math.isfinite(item) else item
        json.dump(numeric(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    if once:
        os.link(temporary, path); temporary.unlink()
    else:
        os.replace(temporary, path)


def allocated():
    node = os.environ.get('SLURMD_NODENAME', '')
    if (not os.environ.get('SLURM_JOB_ID') or not node or
            socket.gethostname().split('.')[0] != node.split('.')[0]):
        raise RuntimeError('Run heavy stages in an allocated compute-node task')


def compute_path(path):
    # Preserve the virtual environment entrypoint; resolving its executable
    # symlink before exec would select the base interpreter and lose packages.
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    if path.resolve().parts[1:2] == ('projects',):
        raise ValueError('Runtime paths must use compute-visible shared storage')
    return path


def command(args, log, *, cpu=False):
    env = dict(os.environ)
    for key in CACHES:
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if cpu:
        env['JAX_PLATFORMS'] = 'cpu'
    else:
        env.pop('JAX_PLATFORMS', None)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('x') as stream:
        subprocess.run([str(x) for x in args], cwd=HERE, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def storage_gate(state, key, roots, *, min_bytes=8 << 30, min_inodes=10000):
    observations = []
    for root in roots:
        root = Path(root)
        ancestor = root
        while not ancestor.exists():
            ancestor = ancestor.parent
        stat = os.statvfs(ancestor)
        observations.append(dict(path=str(root), available_bytes=stat.f_bavail * stat.f_frsize,
                                 available_inodes=stat.f_favail, required_bytes=min_bytes,
                                 required_inodes=min_inodes))
    passed = all(x['available_bytes'] >= min_bytes and x['available_inodes'] >= min_inodes for x in observations)
    atomic(Path(state) / 'gates' / (key + '.storage.json'), dict(passed=passed, observations=observations))
    if not passed:
        raise RuntimeError('Storage gate failed; inspect the recorded byte/inode observations')


def dependency_bindings(state, spec):
    """Resolve producer-bound receipts before opening their checkpoint payloads."""
    state = Path(state)
    records = {}
    for key in spec.get('dependencies', []):
        path = state / 'receipts' / (key + '.json')
        receipt = read(path)
        if not receipt or receipt.get('status') != 'complete':
            raise ValueError('Dependency has no complete receipt: ' + key)
        if receipt.get('source_manifest_sha256') != spec['source_manifest_sha256']:
            raise ValueError('Dependency source binding differs: ' + key)
        pin = spec.get('dependency_receipts', {}).get(key, {})
        if pin.get('sha256') and sha(path) != pin['sha256']:
            raise ValueError('Dependency receipt SHA differs: ' + key)
        journal = state / 'journals' / (key + '.json')
        if not journal.is_file():
            raise ValueError('Dependency submission journal is missing: ' + key)
        if pin.get('journal_sha256') and sha(journal) != pin['journal_sha256']:
            raise ValueError('Dependency journal SHA differs: ' + key)
        records[key] = dict(path=str(path), receipt_sha256=sha(path),
                            journal_sha256=sha(journal), receipt=receipt)
    atomic(state / 'bindings' / (spec['key'] + '.json'), records, once=True)
    return records


def parent_checkpoint(state, spec, bindings):
    parent = spec.get('parent_checkpoint')
    if not parent and spec.get('parent_key'):
        parent = bindings.get(spec['parent_key'], {}).get('receipt', {}).get('checkpoint')
    if not parent:
        raise ValueError('A verified parent checkpoint is required')
    path = Path(parent['path'])
    actual = sha(path / 'checkpoint.json')
    expected = parent.get('manifest_sha256', parent.get('checkpoint_sha256'))
    if actual != expected:
        raise ValueError('Parent checkpoint manifest SHA differs')
    meta = read(path / 'checkpoint.json')
    if meta.get('schema') != 'bigfield-checkpoint-v1':
        raise ValueError('Unknown parent checkpoint schema')
    atomic(Path(state) / 'bindings' / (spec['key'] + '.parent.json'),
           dict(path=str(path), manifest_sha256=actual, step=meta['step']), once=True)
    return path


def checkpoint_record(output, step):
    path = Path(output) / 'checkpoints' / ('step_%06d' % step)
    meta = read(path / 'checkpoint.json')
    if not meta or meta.get('step') != step or meta.get('data_cursor') != step * 64:
        raise ValueError('Checkpoint endpoint/cursor differs')
    for name, digest in meta['files_sha256'].items():
        if sha(path / name) != digest:
            raise ValueError('Checkpoint payload SHA differs: ' + name)
    return dict(path=str(path), step=step, manifest_sha256=sha(path / 'checkpoint.json'))


def launch_ranks(state, spec, start, end, output, *, resume=None, smoke=False):
    resources = spec['resources']
    args = ['srun', '--kill-on-bad-exit=1', '--ntasks=' + str(resources['nodes']),
            '--ntasks-per-node=1', spec['python'], HERE / 'worker.py', '--rank',
            '--state', state, '--key', spec['key'], '--start-step', str(start),
            '--end-step', str(end), '--output', output]
    if resume:
        args += ['--resume', resume]
    if smoke:
        args += ['--smoke-rank']
    command(args, Path(spec['output_root']) / 'logs' / ('%s.%d-%d.log' % (spec['key'], start, end)))


def verify_single_gpu(spec):
    code = ("import jax; d=jax.local_devices(); "
            "assert len(d)==1 and d[0].platform=='gpu' and 'H100' in d[0].device_kind, "
            "'Expected one allocated visible H100'; print(d)")
    command([spec['python'], '-c', code],
            Path(spec['output_root']) / 'logs' / (spec['key'] + '.devices.log'))


def rank_main(args, spec):
    allocated()
    for key in CACHES + ('JAX_DEFAULT_MATMUL_PRECISION',):
        os.environ.pop(key, None)
    os.environ.pop('JAX_PLATFORMS', None)
    os.environ.update(XLA_PYTHON_CLIENT_ALLOCATOR='bfc', XLA_PYTHON_CLIENT_PREALLOCATE='true',
                      XLA_PYTHON_CLIENT_MEM_FRACTION='0.90')
    os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true')
    from freeze import check
    if check() != spec['source_manifest_sha256']:
        raise ValueError('Rank source manifest SHA differs')
    import numpy as np
    import jax
    from distributed import initialize, barrier, world_any
    from training import Runtime, load_config
    from checkpoint import tree_digest, optimizer_counts, _rng_for_step
    from data import sample_caption_indices
    rank = int(os.environ['SLURM_PROCID'])
    resources = spec['resources']
    rendezvous = args.state / ('rendezvous.%s.%d-%d.json' % (args.key, args.start_step, args.end_step))
    if rank == 0:
        atomic(rendezvous, dict(address=socket.gethostname() + ':' + str(20000 + int(os.environ['SLURM_JOB_ID']) % 30000)), once=True)
    deadline = time.monotonic() + 120
    while not rendezvous.is_file():
        if time.monotonic() > deadline:
            raise RuntimeError('Coordinator address publication timed out')
        time.sleep(0.2)
    geometry = initialize(resources['nodes'], resources['gpus_per_node'], read(rendezvous)['address'])
    paused = [False]
    def pause(signum, frame):
        paused[0] = True
    signal.signal(signal.SIGUSR1, pause)
    signal.signal(signal.SIGTERM, pause)
    beginning = time.monotonic()
    rt = Runtime(load_config(), spec['data_root'], Path(spec['data_root']) / 'weights',
                 args.output, geometry, resume=args.resume)
    startup = time.monotonic() - beginning
    if rt.step != args.start_step:
        raise ValueError('Requested segment start differs from restored clock')
    def snapshot():
        params, optimizer = rt.host(rt.params), rt.host(rt.opt_state)
        finite = all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((params, optimizer)))
        counts = optimizer_counts(optimizer)
        value = dict(step=rt.step, data_cursor=rt.step * 64, rng=rt.rng.tolist(), counters=counts,
                     parameter_sha256=tree_digest(params), optimizer_sha256=tree_digest(optimizer),
                     binding_sha256=hashlib.sha256(json.dumps(rt.binding, sort_keys=True).encode()).hexdigest(), finite=bool(finite))
        passed = finite and counts and all(n == rt.step for _, n in counts) and np.array_equal(rt.rng, _rng_for_step(rt.seed, rt.step))
        atomic(args.state / 'ranks' / ('%s.%d-%d.rank%d.snapshot%d.json' % (args.key, args.start_step, args.end_step, rank, rt.step)),
               dict(value, passed=bool(passed)), once=True)
        if world_any(not passed):
            raise ValueError('Rank state is nonfinite or its clocks/RNG differ')
        return value
    initial = snapshot() if args.smoke_rank else dict(step=rt.step, rng=rt.rng.tolist(), data_cursor=rt.step * 64)
    if args.smoke_rank and args.start_step == 2:
        prior = read(args.state / 'ranks' / ('%s.0-2.rank%d.json' % (args.key, rank)))
        if initial != prior['final']:
            raise ValueError('Full restored state differs from the pre-restart rank state')
    updates, saves = [], []
    while rt.step < args.end_step and not world_any(paused[0]):
        record = rt.update()
        expected = sample_caption_indices(rt.rows, rt.step - 1, rt.seed)
        passed = record.get('gradient_finite') and record.get('status') == 'APPLIED' and record['batch_ids_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
        if args.smoke_rank:
            atomic(args.state / 'ranks' / ('%s.rank%d.update%06d.json' % (args.key, rank, rt.step)),
                   dict(step=rt.step, finite=bool(record.get('gradient_finite')), batch_ids_sha256=record['batch_ids_sha256'],
                        seconds=record['seconds'], passed=bool(passed)), once=True)
        if world_any(not passed):
            raise ValueError('Rank update is nonfinite or used a different caption stream')
        updates.append(float(record['seconds']))
        if rt.step % rt.config['checkpoint_interval'] == 0 or rt.step == args.end_step:
            started = time.monotonic(); rt.save(); saves.append(time.monotonic() - started)
    if rt.step != args.end_step:
        started = time.monotonic(); rt.save(); saves.append(time.monotonic() - started)
    final = snapshot() if args.smoke_rank else dict(step=rt.step, rng=rt.rng.tolist(), data_cursor=rt.step * 64)
    memories = [d.memory_stats() or {} for d in jax.local_devices()]
    result = dict(rank=rank, nodes=geometry.nodes, local_devices=geometry.local_devices,
                  initial=initial, final=final, update_seconds=updates, startup_seconds=startup,
                  checkpoint_seconds=max(saves, default=0), memory=memories,
                  complete=rt.step == args.end_step, dose=rt.dose_status)
    atomic(args.state / 'ranks' / ('%s.%d-%d.rank%d.json' % (args.key, args.start_step, args.end_step, rank)), result, once=True)
    barrier('worker_segment_complete')
    if rt.step != args.end_step:
        raise RuntimeError('Segment paused; inspect the saved checkpoint before explicit recovery')


def smoke_stage(state, spec):
    output = Path(spec['smoke_output'])
    if output.exists():
        raise FileExistsError('Smoke output already exists; inspect failed stage without automatic retry')
    launch_ranks(state, spec, 0, 2, output, smoke=True)
    saved = checkpoint_record(output, 2)
    # Pin the exact full-state checkpoint before the fresh distributed process.
    atomic(Path(state) / 'bindings' / 'smoke.restart.json', saved, once=True)
    launch_ranks(state, spec, 2, 3, output, resume=saved['path'], smoke=True)
    rows = [read(Path(state) / 'ranks' / ('%s.%d-%d.rank%d.json' % (spec['key'], start, end, rank)))
            for start, end in ((0, 2), (2, 3)) for rank in range(spec['resources']['nodes'])]
    passed = all(r and r['complete'] and r['nodes'] == spec['resources']['nodes'] and
                 r['local_devices'] == spec['resources']['gpus_per_node'] for r in rows)
    finals = [r['final'] for r in rows] if passed else []
    for start in (0, spec['resources']['nodes']):
        group = finals[start:start + spec['resources']['nodes']]
        passed = passed and bool(group) and all(x == group[0] for x in group)
    # Exclude the compilation-heavy first update; retain every timing as evidence.
    steady = [r['update_seconds'][1] for r in rows[:spec['resources']['nodes']] if len(r['update_seconds']) == 2]
    timing = dict(validated=bool(passed), rank_reports=rows,
                  seconds_per_step=max(steady, default=float('inf')),
                  startup_seconds=max((r['startup_seconds'] + r['update_seconds'][0] for r in rows), default=0),
                  checkpoint_seconds=max((r['checkpoint_seconds'] for r in rows), default=0),
                  peak_memory_bytes=max((m.get('peak_bytes_in_use', m.get('bytes_in_use', 0))
                                         for r in rows for m in r['memory']), default=0),
                  topology=spec['resources'], timing_basis='Second update of 0->2; slowest rank. Startup includes first-update compilation.')
    atomic(Path(state) / 'gates' / 'smoke.numeric.json', timing)
    if not passed or not all(math.isfinite(timing[k]) and timing[k] > 0 for k in ('seconds_per_step', 'startup_seconds', 'checkpoint_seconds')):
        raise ValueError('Smoke continuity/topology/timing validation failed')
    timing['checkpoint'] = checkpoint_record(output, 3)
    return timing


def evaluation_stage(state, spec, bindings):
    checkpoint = parent_checkpoint(state, spec, bindings)
    step = spec.get('end_step', read(checkpoint / 'checkpoint.json')['step'])
    # Explicit recovery gets a new panel, preserving partial earlier output.
    generation = spec['key'].split('_r', 1)[1] if '_r' in spec['key'] else None
    panel_name = 'step_%06d' % step + ('_r' + generation if generation else '')
    output = Path(spec['output_root']) / 'readouts' / panel_name
    setup = read(Path(spec['data_root']) / 'setup/status.json', {})
    assets = compute_path(setup.get('eval_assets', str(Path(spec['data_root']) / 'evaluation')))
    result = dict(step=step, panel_path=str(output), checkpoint=dict(path=str(checkpoint), manifest_sha256=sha(checkpoint / 'checkpoint.json'), step=step), benchmarks={})
    for benchmark, metric in (('geneval', 'overall_category_balanced'), ('dpg', 'fullrow_corrected_overall')):
        storage_gate(state, spec['key'] + '.' + benchmark, [spec['output_root']], min_bytes=8 << 30, min_inodes=10000)
        images = output / benchmark / 'samples'
        score_output = output / benchmark / 'scores'
        command([spec['python'], HERE / 'eval_sample.py', '--benchmark', benchmark, '--checkpoint', checkpoint,
                 '--assets', assets, '--weights-root', Path(spec['data_root']) / 'weights',
                 '--native-root', HERE / 'bigfield_native', '--output', images],
                Path(spec['output_root']) / 'logs' / (spec['key'] + '.' + benchmark + '.sample.log'))
        scoring = setup.get('scoring', {}).get(benchmark, {})
        available = scoring.get('status') in ('ready', 'complete', 'PASS', 'READY') and Path(scoring.get('python', '')).is_file()
        assets_ready = setup.get('evaluation_assets', {}).get(benchmark, {}).get('status') in ('ready', 'complete', 'PASS', 'READY')
        if available and assets_ready:
            # Wheels can install successfully yet lack a working CUDA runtime.
            # Only readiness falls back; scientific scoring failures stay fatal.
            imports = 'import mmcv,mmdet,open_clip' if benchmark == 'geneval' else 'import modelscope,transformers'
            probe = compute_path(scoring['python'])
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
            for key in CACHES:
                env.pop(key, None)
            try:
                run = subprocess.run([str(probe), '-c', 'import torch; ' + imports +
                                      '; assert torch.cuda.is_available() and torch.cuda.device_count()==1'],
                                     cwd=HERE, env=env, text=True, capture_output=True, timeout=120)
                available = run.returncode == 0
                reason = (run.stdout + run.stderr)[-4000:]
            except (OSError, subprocess.TimeoutExpired) as error:
                available = False
                reason = str(error)
            atomic(Path(state) / 'gates' / (spec['key'] + '.' + benchmark + '.environment.json'),
                   dict(ready=available, python=str(probe), reason=reason))
            if not available:
                scoring = dict(scoring, error='Allocated scorer environment unavailable: ' + reason)
        if available and assets_ready:
            scoring_python = compute_path(scoring['python'])
            command([scoring_python, HERE / 'score.py', '--benchmark', benchmark, '--images', images,
                     '--assets', assets, '--output', score_output],
                    Path(spec['output_root']) / 'logs' / (spec['key'] + '.' + benchmark + '.score.log'))
            summary = read(score_output / ('geneval.summary.json' if benchmark == 'geneval' else 'dpg.json'))
            value = float(summary[metric])
            atomic(Path(state) / 'gates' / (spec['key'] + '.' + benchmark + '.score.json'),
                   dict(value=value if math.isfinite(value) else str(value), metric=metric, finite=math.isfinite(value)))
            if not math.isfinite(value):
                raise ValueError('Nonfinite score')
            result['benchmarks'][benchmark] = dict(status='complete', metric=metric, value=value)
        else:
            result['benchmarks'][benchmark] = dict(status='scoring pending', metric=metric,
                                                   reason=scoring.get('error', 'Scoring environment/assets unavailable'),
                                                   params=str(checkpoint / 'params.msgpack'))
    return result


def execute_stage(args):
    allocated()
    job = read(args.state / 'jobs' / (args.key + '.json'))
    spec = job['spec']
    receipt_path = args.state / 'receipts' / (args.key + '.json')
    if receipt_path.exists():
        raise FileExistsError('Stage already has a receipt; automatic retries are disabled')
    begun = time.time()
    base = dict(key=args.key, stage=args.stage, source_manifest_sha256=spec['source_manifest_sha256'], started_at=begun)
    try:
        from freeze import check
        if check() != spec['source_manifest_sha256']:
            raise ValueError('Allocated source manifest SHA differs')
        if spec['stage'] != args.stage or spec['key'] != args.key:
            raise ValueError('Worker stage/key differs from its submission binding')
        identity = read(args.state / 'binding.json', {}).get('output_identity')
        if identity:
            stat = Path(spec['output_root']).stat()
            if identity != dict(path=str(Path(spec['output_root'])), inode=stat.st_ino, uid=stat.st_uid):
                raise ValueError('Run output path/inode/uid differs')
        for path in (spec['data_root'], spec['output_root'], spec['code_root'], spec['python']):
            compute_path(path)
        bindings = dependency_bindings(args.state, spec)
        storage_gate(args.state, args.key, [spec['data_root'], spec['output_root']])
        python = spec['python']; data = Path(spec['data_root'])
        if args.stage == 'data':
            action = ['check', '--kind', 'prepared'] if (data / 'curated').exists() else ['rebuild']
            command([python, HERE / 'prepare_data.py', '--root', data, *action],
                    Path(spec['output_root']) / 'logs' / 'data.prepare.log', cpu=True)
            result = dict(data_manifest_sha256=sha(data / 'curated/manifest.json'))
        elif args.stage == 'bank':
            verify_single_gpu(spec)
            action = ['check', '--kind', 'bank'] if (data / 'bank').exists() else ['dino']
            command([python, HERE / 'prepare_data.py', '--root', data, *action],
                    Path(spec['output_root']) / 'logs' / 'bank.prepare.log')
            result = dict(bank_receipt_sha256=sha(data / 'bank/receipt.json'))
        elif args.stage == 'smoke':
            result = smoke_stage(args.state, spec)
        elif args.stage == 'train':
            has_parent = (spec.get('parent_checkpoint') is not None or
                          spec.get('parent_key') is not None or spec['start_step'] != 0)
            parent = parent_checkpoint(args.state, spec, bindings) if has_parent else None
            output = Path(spec['train_output'])
            if parent is None and output.exists() and any(output.iterdir()):
                raise FileExistsError('Cold-start training output already contains files')
            launch_ranks(args.state, spec, spec['start_step'], spec['end_step'], output, resume=parent)
            result = dict(checkpoint=checkpoint_record(output, spec['end_step']))
        elif args.stage == 'eval':
            verify_single_gpu(spec)
            result = evaluation_stage(args.state, spec, bindings)
        elif args.stage == 'summary':
            result = dict(results=str(write_status(Path(spec['output_root']), bundle=True)))
        elif args.stage == 'advance':
            run = subprocess.run([python, HERE / 'workflow.py', '--advance', '--submit', '--yes',
                                  '--output-root', spec['output_root']], cwd=HERE, text=True, capture_output=True)
            # Only the orchestrator's explicit unavailability result enables this
            # fallback. Other errors retain a failed receipt and halt progression.
            if run.returncode == 69:
                result = dict(auto_advance='manual_required', reason=run.stderr,
                              resume_command='bash run.sh --submit --yes --output-root ' + spec['output_root'])
            elif run.returncode:
                raise RuntimeError('Automatic advance failed: ' + run.stdout + run.stderr)
            else:
                result = dict(auto_advance='submitted', output=run.stdout)
        else:
            raise ValueError('Unknown stage: ' + args.stage)
        receipt = dict(base, **result, status='complete', finished_at=time.time())
        atomic(receipt_path, receipt, once=True)
        return receipt
    except BaseException as error:
        atomic(receipt_path, dict(base, status='failed', error=type(error).__name__ + ': ' + str(error),
                                  finished_at=time.time(), recovery='Inspect logs and latest verified checkpoint; no automatic retry.'), once=True)
        raise


def evaluation_receipt(root, readout):
    state = Path(root) / 'state'
    receipt = read(state / 'receipts' / ('eval_%06d.json' % readout), {})
    candidates = [read(p, {}) for p in (state / 'receipts').glob('*.json')]
    candidates = [r for r in candidates if r.get('stage') == 'eval' and r.get('step') == readout]
    if candidates:
        receipt = max(candidates, key=lambda r: (r.get('status') == 'complete', r.get('finished_at', 0)))
    return receipt


def evaluation_panel(root, readout):
    receipt = evaluation_receipt(root, readout)
    return Path(receipt.get('panel_path', Path(root) / 'readouts' / ('step_%06d' % readout)))


def status_text(root, *, refresh=True):
    root = Path(root)
    state = root / 'state'
    jobs = [read(p) for p in sorted((state / 'jobs').glob('*.json'))]
    scheduler_note = None
    if refresh and jobs:
        from workflow import queue_snapshot, refresh_jobs
        mapped = {job['spec']['key']: job for job in jobs}
        try:
            refresh_jobs(state, mapped, queue_snapshot(), submit=False)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            scheduler_note = str(error)
        jobs = list(mapped.values())
    stages = {}
    failures = []
    for job in jobs:
        spec = job['spec']; key = spec.get('key', job.get('key'))
        receipt = read(state / 'receipts' / (key + '.json'), {})
        status = 'superseded' if job.get('superseded_by') else receipt.get('status', job.get('status', 'unknown'))
        stages.setdefault(spec['stage'], []).append(status)
        if status.lower() in ('failed', 'submission_failed', 'cancelled', 'timeout', 'node_fail', 'out_of_memory', 'uncertain'):
            failures.append((key, spec['stage'], receipt.get('error', job.get('error', status))))
    progress = read(root / 'train/progress.json', {})
    latest = read(root / 'train/latest.json', {})
    step = max(int(progress.get('step', 0)), int(latest.get('step', 0)))
    speed = progress.get('seconds')
    smoke = read(state / 'receipts/smoke.json', {})
    speed = speed or smoke.get('seconds_per_step')
    eta = ((30000 - step) * float(speed) / 3600) if speed else None
    lines = ['Big-field progress', 'Output: ' + str(root),
             'Stages: ' + ('; '.join(k + '=' + ','.join(sorted(set(v))) for k, v in stages.items()) or 'not submitted'),
             'Training: %d/30000 | s/update: %s | compute ETA: %s (excludes queue/evaluation)' %
             (step, ('%.3f' % speed) if speed else 'pending smoke', ('%.1f h' % eta) if eta is not None else 'pending smoke'),
             '', '| Step | GenEval category-balanced | DPG corrected | Checkpoint SHA | D5 | Status |',
             '|---:|---:|---:|---|---|---|']
    for readout in READOUTS:
        receipt = evaluation_receipt(root, readout)
        panel = evaluation_panel(root, readout)
        values = []
        for benchmark, metric, file in (('geneval', 'overall_category_balanced', 'geneval.summary.json'),
                                         ('dpg', 'fullrow_corrected_overall', 'dpg.json')):
            summary = read(panel / benchmark / 'scores' / file, {})
            result = receipt.get('benchmarks', {}).get(benchmark, {})
            value = summary.get(metric, result.get('value'))
            values.append(('%.4f' % value) if value is not None else result.get('status', 'pending'))
        checkpoint = root / 'train/checkpoints' / ('step_%06d' % readout) / 'checkpoint.json'
        digest = sha(checkpoint) if checkpoint.is_file() else 'pending'
        dose = read(root / 'train/readouts' / ('step_%06d.dose.json' % readout), {})
        lines.append('| %d | %s | %s | %s | %s | %s |' %
                     (readout, *values, digest, dose.get('status', 'PENDING'), receipt.get('status', 'pending')))
    dose = read(root / 'train/dose_summary.json', {})
    lines += ['', 'D5: ' + dose.get('status', 'PENDING') + ' (original two diagnostic windows).']
    if scheduler_note:
        lines += ['Scheduler: ' + scheduler_note + ' (root 待核 where unavailable).']
    for key, stage, error in failures:
        lines += ['STOP ' + key + ': ' + str(error), 'Logs: ' + str(root / 'logs')]
        if stage == 'train':
            lines += ['Recovery: inspect the failed stage/checkpoint, then preview bash run.sh --recover ' + key + '; add --submit only after review.']
        else:
            lines += ['Recovery: inspect the failed ' + stage + ' stage logs and report the failure; automatic retries are disabled.']
    advance = read(state / 'receipts/advance.json', {})
    if advance.get('auto_advance') == 'manual_required':
        lines += ['Automatic submission unavailable: ' + advance.get('resume_command', 'bash run.sh --submit')]
    return '\n'.join(lines) + '\n'


def write_status(root, *, bundle=False, include_params=False):
    root = Path(root)
    if root.is_dir():
        binding = read(root / 'state/binding.json', {})
        stat = root.stat()
        actual = dict(path=str(root), inode=stat.st_ino, uid=stat.st_uid)
        if (binding.get('output_identity') != actual or
                binding.get('source_manifest_sha256') != sha(HERE / 'sources.sha256')):
            raise ValueError('Refusing to write status into an unbound or different run directory')
    text = status_text(root)
    results = root / 'results'
    if root.is_dir():
        results.mkdir(exist_ok=True)
        target = results / 'results.md'
        temp = results / ('.results.' + uuid.uuid4().hex)
        temp.write_text(text); os.replace(temp, target)
    if bundle:
        if not root.is_dir():
            raise FileNotFoundError('Run output has not been reserved')
        limit = 20_000_000
        destination = results / 'return_bundle.tar'
        manifest_path = results / 'return_manifest.json'
        receipt_path = results / 'return_bundle.receipt.json'
        gate_path = results / 'bundle_gate.json'
        host = socket.gethostname().split('.')[0]
        node = os.environ.get('SLURMD_NODENAME', '').split('.')[0]
        on_compute = bool(os.environ.get('SLURM_JOB_ID') and node and node == host)

        def identity(path):
            stat = path.stat()
            return dict(path=str(path), inode=stat.st_ino, uid=stat.st_uid,
                        bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)

        # A summary writes its own receipt after returning from this function.
        # Exclude summary receipts so a final allocated bundle remains reusable.
        markers = [root / 'train' / name for name in ('latest.json', 'progress.json', 'dose_summary.json')]
        markers += [root / 'state/binding.json', HERE / 'sources.sha256']
        for readout in READOUTS:
            panel = evaluation_panel(root, readout)
            markers += [panel / b / 'scores' / name for b, name in
                        (('geneval', 'geneval.summary.json'), ('dpg', 'dpg.json'))]
            markers.append(root / 'train/readouts' / ('step_%06d.dose.json' % readout))
        for path in sorted((root / 'state/receipts').glob('*.json')):
            if path.stat().st_size <= 65536 and read(path, {}).get('stage') == 'summary':
                continue
            markers.append(path)
        state_rows = [identity(path) for path in markers if path.is_file()]
        state_token = hashlib.sha256(json.dumps(state_rows, sort_keys=True).encode()).hexdigest()
        prior = read(receipt_path, {}) if receipt_path.is_file() and receipt_path.stat().st_size <= 65536 else {}
        reusable = (destination.is_file() and manifest_path.is_file() and manifest_path.stat().st_size <= limit
                    and prior.get('state_token') == state_token
                    and prior.get('archive_identity') == identity(destination)
                    and (prior.get('created_on_compute') or destination.stat().st_size <= limit)
                    and (not include_params or prior.get('include_params_requested')))
        if reusable and sha(manifest_path) == prior.get('manifest_sha256'):
            atomic(gate_path, dict(action='reuse', passed=True, on_compute=on_compute,
                                   login_archive_limit_bytes=limit, archive_bytes=destination.stat().st_size,
                                   parameter_bytes_read=0, archive_bytes_read=0, state_token=state_token))
            print(text, end='')
            print('Reusing the verified bundle snapshot; no parameter or archive payload was read.')
            print('Return: ' + str(results / 'results.md') + ' + ' + str(destination))
            return results / 'results.md'
        candidates = set()
        for directory in ('state', 'logs', 'train/readouts'):
            candidates.update(p for p in (root / directory).rglob('*') if p.is_file())
        panels = root / 'readouts'
        for panel in panels.iterdir() if panels.is_dir() else ():
            if not panel.is_dir():
                continue
            metadata_roots = [panel]
            for benchmark in ('geneval', 'dpg'):
                metadata_roots += [panel / benchmark, panel / benchmark / 'samples', panel / benchmark / 'scores']
            for directory in metadata_roots:
                if directory.is_dir():
                    candidates.update(p for p in directory.iterdir() if p.suffix in ('.json', '.jsonl', '.txt') and p.is_file())
        candidates.update(p for p in (root / 'train').glob('*.json*') if p.is_file())
        candidates.add(results / 'results.md')
        for readout in READOUTS:
            path = root / 'train/checkpoints' / ('step_%06d' % readout)
            if (path / 'checkpoint.json').is_file():
                candidates.add(path / 'checkpoint.json')
                summaries = [evaluation_panel(root, readout) / b / 'scores' / f
                             for b, f in (('geneval', 'geneval.summary.json'), ('dpg', 'dpg.json'))]
                if include_params or not all(p.is_file() for p in summaries):
                    if (path / 'params.msgpack').is_file():
                        candidates.add(path / 'params.msgpack')
        setup = read(root / 'state/site.json', {})
        data_root = Path(setup.get('DATA_ROOT', root.parent / 'bigfield_data'))
        external = [data_root / 'setup/status.json', data_root / 'curated/manifest.json', data_root / 'bank/receipt.json',
                    HERE / 'configs/bigfield_tf.yaml', HERE / 'sources.sha256']
        external = [path for path in external if path.is_file()]
        selected = [(str(path.relative_to(root)), path) for path in sorted(candidates)]
        selected += [('inputs/' + ('setup.json' if path.name == 'status.json' else path.name), path) for path in external]
        observations = [dict(archive_path=name, **identity(path)) for name, path in selected]
        parameter_bytes = sum(row['bytes'] for row in observations if Path(row['path']).name == 'params.msgpack')
        total_bytes = sum(row['bytes'] for row in observations)
        # Bound padded file bodies, PAX/path headers, the manifest and tar trailer
        # using stat only. No parameter/log body is opened before this gate.
        manifest_estimate = len(json.dumps(dict(files_sha256={name: '0' * 64 for name, _ in selected}),
                                           sort_keys=True, indent=2).encode()) + 1
        archive_upper_bytes = 10240 + sum(((row['bytes'] + 511) // 512) * 512 + 8192 +
                                          ((len(row['archive_path'].encode()) + 511) // 512) * 512
                                          for row in observations)
        archive_upper_bytes += ((manifest_estimate + 511) // 512) * 512 + 8192
        available = os.statvfs(results)
        free_bytes, free_inodes = available.f_bavail * available.f_frsize, available.f_favail
        required_bytes = archive_upper_bytes + manifest_estimate + 65536
        gate = dict(action='build', on_compute=on_compute, hostname=host, allocated_node=node,
                    login_archive_limit_bytes=limit, candidate_files=len(observations),
                    candidate_bytes=total_bytes, parameter_bytes=parameter_bytes,
                    metadata_bytes=total_bytes - parameter_bytes,
                    archive_upper_bytes=archive_upper_bytes, manifest_estimate_bytes=manifest_estimate,
                    free_bytes=free_bytes, required_bytes=required_bytes, free_inodes=free_inodes,
                    required_inodes=8, state_token=state_token, observations=observations)
        gate['allocation_passed'] = on_compute or archive_upper_bytes <= limit
        gate['storage_passed'] = free_bytes >= required_bytes and free_inodes >= gate['required_inodes']
        gate['passed'] = gate['allocation_passed'] and gate['storage_passed']
        atomic(gate_path, gate)
        if not gate['allocation_passed']:
            import shlex
            preview = 'bash run.sh --yes --bundle --output-root ' + shlex.quote(str(root))
            print(text, end='')
            raise RuntimeError('Bundle needs allocated compute (' + str(archive_upper_bytes) +
                               ' bytes upper bound; login limit ' + str(limit) +
                               '). Preview ' + preview + '; after confirmation add --submit. Numeric gate: ' + str(gate_path))
        if not gate['storage_passed']:
            raise RuntimeError('Bundle storage gate failed; inspect ' + str(gate_path))
        records = {}
        temporary = results / ('.bundle.' + uuid.uuid4().hex)
        with tarfile.open(temporary, 'w') as archive:
            expected_sizes = {row['archive_path']: row['bytes'] for row in observations}
            for name, path in selected:
                with path.open('rb') as stream:
                    info = archive.gettarinfo(fileobj=stream, arcname=name)
                    archive_upper_bytes += ((info.size + 511) // 512 - (expected_sizes[name] + 511) // 512) * 512
                    if not on_compute and archive_upper_bytes > limit:
                        gate.update(passed=False, allocation_passed=False, archive_upper_bytes=archive_upper_bytes,
                                    changed_file=name, changed_file_bytes=info.size)
                        atomic(gate_path, gate)
                        raise RuntimeError('Bundle grew beyond the login-node size limit; preview bash run.sh --yes --bundle, then add --submit after confirmation. Numeric gate: ' + str(gate_path))
                    digest = hashlib.sha256()
                    class SnapshotReader:
                        def read(self, count=-1):
                            value = stream.read(count)
                            digest.update(value)
                            return value
                    archive.addfile(info, SnapshotReader())
                    records[name] = digest.hexdigest()
            # Hash the exact bytes streamed into tar, including a stable prefix
            # of any log being appended by an independent active stage.
            atomic(manifest_path, dict(files_sha256=records))
            archive.add(manifest_path, arcname='results/return_manifest.json')
        os.replace(temporary, destination)
        atomic(receipt_path, dict(schema='bigfield-return-bundle-v2', created_on_compute=on_compute,
                                  include_params_requested=include_params, state_token=state_token,
                                  archive_identity=identity(destination), archive_sha256=sha(destination),
                                  manifest_sha256=sha(manifest_path)))
    print(text, end='')
    if bundle:
        print('Return: ' + str(results / 'results.md') + ' + ' + str(results / 'return_bundle.tar'))
    return results / 'results.md'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--bundle', action='store_true')
    parser.add_argument('--include-params', action='store_true')
    parser.add_argument('--output-root', type=Path, default=Path(os.environ.get('OUTPUT_ROOT', HERE.parent.parent / 'bigfield_run')))
    parser.add_argument('--state', type=Path)
    parser.add_argument('--key')
    parser.add_argument('--stage', choices=('data', 'bank', 'smoke', 'train', 'eval', 'advance', 'summary'))
    parser.add_argument('--rank', action='store_true')
    parser.add_argument('--smoke-rank', action='store_true')
    parser.add_argument('--start-step', type=int)
    parser.add_argument('--end-step', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(argv)
    if args.status:
        write_status(args.output_root.resolve(), bundle=args.bundle, include_params=args.include_params)
    else:
        if not args.state or not args.key or (not args.rank and not args.stage):
            parser.error('Require --state, --key and --stage (or --rank)')
        if args.rank:
            rank_main(args, read(args.state / 'jobs' / (args.key + '.json'))['spec'])
        else:
            print(json.dumps(execute_stage(args), sort_keys=True))


if __name__ == '__main__':
    main()
