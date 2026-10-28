#!/usr/bin/env python3
"""Install pinned environments and resume the released SHA-pinned assets.

This standard-library entrypoint never loads models, rebuilds data, extracts
features, runs GPU work, or calls a scheduler.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent


def compute_visible(path, label):
    resolved = Path(path).expanduser().resolve()
    if resolved.parts[1:2] == ('projects',):
        raise ValueError(label + ' must use storage available on compute nodes')
    return resolved


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def downloads(root, eval_assets, scope=None):
    pins = json.loads((HERE / 'assets/public_pins.json').read_text())
    entries = []
    if scope in (None, 'training'):
        for source, spec in pins['raw_datasets'].items():
            entries.extend((root / 'raw' / source / row['name'], row['size']) for row in spec['files'])
        for name, spec in pins['weights'].items():
            if name != 'clip':
                entries.extend((root / 'weights' / name / row['name'], row['size']) for row in spec['files'])
    epins = json.loads((HERE / 'evaluation/pins.json').read_text())
    entries.extend((eval_assets / row['path'], row['bytes']) for row in epins['files']
                   if scope is None or row['scope'] == scope)
    return entries


def prompt_downloads(root, eval_assets, scope):
    """Small pinned sources needed before sampling, independent of scorer models."""
    return [(path, size) for path, size in downloads(root, eval_assets, scope=scope)
            if not path.relative_to(eval_assets).as_posix().startswith('models/')]


def make_plan(args):
    root = compute_visible(os.environ.get('DATA_ROOT', HERE.parent.parent / 'bigfield_data'), 'DATA_ROOT')
    eval_assets = compute_visible(os.environ.get('EVAL_ASSETS', root / 'evaluation'), 'EVAL_ASSETS')
    env = compute_visible(os.environ.get('VENV_DIR', HERE / '.venv'), 'VENV_DIR')
    entries = downloads(root, eval_assets)
    return dict(schema='bigfield-setup-v2', data_root=str(root), eval_assets=str(eval_assets),
                training_environment=str(env), jax_backend=args.jax_backend,
                download_files=len(entries), pinned_download_bytes=sum(size for _, size in entries),
                scorer_environments={name: str(compute_visible(os.environ.get(name.upper() + '_ENV', root / 'envs' / name), name.upper() + '_ENV'))
                                     for name in ('geneval', 'dpg')},
                scientific_work='Scheduled by run.sh: curated reconstruction, DINO bank, smoke, training, evaluation',
                actions=['Python 3.10 environment', 'wheel dependencies; source allowed only for pinned ml-collections', 'resume weights/data downloads and verify SHA',
                         'pinned evaluation prompts/models', 'independent GenEval and DPG environments'],
                output_root_created=False, scheduler_requested=False)


def run(command, logfile, env):
    with Path(logfile).open('a') as stream:
        stream.write('\n$ ' + ' '.join(str(value) for value in command) + '\n')
        stream.flush()
        result = subprocess.run([str(value) for value in command], cwd=HERE, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Command failed (exit ' + str(result.returncode) + '); log: ' + str(logfile))


def is_python310(python):
    try:
        result = subprocess.run([str(python), '-c', 'import sys; raise SystemExit(sys.version_info[:2] != (3,10))'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except OSError:
        return False


def make_environment(destination, logfile, env, *, current=False):
    requested = os.environ.get('PYTHON')
    if current:
        python = requested or sys.executable
        if not is_python310(python):
            raise RuntimeError('--current-environment requires PYTHON pointing to Python 3.10')
        entrypoint = Path(shutil.which(python) or python).absolute()
        compute_visible(entrypoint, 'PYTHON')
        return str(entrypoint)  # Preserve the venv entrypoint even when it is a symlink.
    destination = Path(destination)
    python = destination / 'bin/python'
    if python.exists():
        if not is_python310(python):
            raise RuntimeError('Existing environment is not Python 3.10: ' + str(destination))
        compute_visible(python, 'Environment Python')
        return str(python)
    if destination.exists():
        raise RuntimeError('Existing environment directory has no Python; preserve it and choose a fresh environment path')
    candidates = [requested, shutil.which('python3.10'), sys.executable]
    base = next((value for value in candidates if value and is_python310(value)), None)
    if base:
        compute_visible(shutil.which(base) or base, 'Python 3.10 base')
        run([base, '-m', 'venv', str(destination)], logfile, env)
    elif shutil.which('conda'):
        run(['conda', 'create', '-y', '-p', str(destination), 'python=3.10', 'pip'], logfile, env)
    else:
        raise RuntimeError('Python 3.10 or conda is required; set PYTHON to Python 3.10 and rerun setup.sh')
    if not is_python310(python):
        raise RuntimeError('Created environment does not provide Python 3.10')
    compute_visible(python, 'Environment Python')
    return str(python)


def install_training(python, args, logfile, env):
    run([python, '-m', 'pip', 'install', '--only-binary=:all:', 'pip>=25.3,<26'], logfile, env)
    run([python, '-m', 'pip', 'install', '--only-binary=:all:', 'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cpu'], logfile, env)
    packages = ['jax[cuda12]==0.4.37'] if args.jax_backend == 'cuda12' else ['jax==0.4.37', 'jaxlib==0.4.36']
    # This pinned pure-Python release has no wheel. Every other dependency,
    # including its dependencies and isolated build requirements, retains the
    # binary-only policy.
    run([python, '-m', 'pip', 'install', '--only-binary=:all:', '--no-binary=ml-collections',
         *packages, '-r', 'requirements.txt'], logfile, env)
    run([python, '-m', 'pip', 'check'], logfile, env)


def storage_gate(entries, reservations, minimum_free_inodes):
    """Check every destination filesystem, summing demands on shared mounts.

    f_fsid identifies a filesystem only for this local space calculation. It is
    never persisted as a cross-node directory or file identity.
    """
    files = []
    targets = []
    for path, size in entries:
        path = Path(path)
        present = path.stat().st_size if path.is_file() else max(
            (part.stat().st_size for part in (path.with_name(path.name + '.part'),
                                             path.with_name(path.name + '.partial')) if part.is_file()), default=0)
        row = dict(path=str(path), expected_bytes=size, present_bytes=present,
                   remaining_bytes=max(0, size - present))
        files.append(row)
        targets.append(dict(label='download', path=str(path.parent),
                            download_remaining_bytes=row['remaining_bytes'], reserve_bytes=0))
    for label, destination, reserve_bytes in reservations:
        targets.append(dict(label=label, path=str(destination), download_remaining_bytes=0,
                            reserve_bytes=reserve_bytes))
    filesystems = {}
    for row in targets:
        probe = Path(row['path']).resolve()
        while not probe.exists():
            probe = probe.parent
        if not probe.is_dir():
            probe = probe.parent
        stat = os.statvfs(probe)
        fsid = str(stat.f_fsid)
        row.update(probed_path=str(probe), filesystem=fsid,
                   free_bytes=stat.f_bavail * stat.f_frsize, free_inodes=stat.f_favail)
        fs = filesystems.setdefault(fsid, dict(filesystem=fsid, targets=[], download_remaining_bytes=0,
                                              reserve_bytes=0, free_bytes=row['free_bytes'],
                                              free_inodes=row['free_inodes'], minimum_free_inodes=minimum_free_inodes))
        fs['targets'].append(row)
        fs['download_remaining_bytes'] += row['download_remaining_bytes']
        fs['reserve_bytes'] += row['reserve_bytes']
        fs['free_bytes'] = min(fs['free_bytes'], row['free_bytes'])
        fs['free_inodes'] = min(fs['free_inodes'], row['free_inodes'])
    for fs in filesystems.values():
        fs['required_bytes'] = fs['download_remaining_bytes'] + fs['reserve_bytes']
        fs['passed'] = fs['free_bytes'] >= fs['required_bytes'] and fs['free_inodes'] >= minimum_free_inodes
    return dict(passed=all(fs['passed'] for fs in filesystems.values()),
                download_remaining_bytes=sum(row['remaining_bytes'] for row in files),
                environment_and_temporary_reserve_bytes=sum(row[2] for row in reservations),
                minimum_free_inodes=minimum_free_inodes, files=files,
                filesystems=list(filesystems.values()), quota_checked=False, reserve_is_estimate=True)


def prepare_prompts(helper, root, scope):
    """Allow sampling when optional scorer model downloads cannot complete."""
    pins = json.loads((HERE / 'evaluation/pins.json').read_text())
    for row in pins['files']:
        if row['scope'] == scope and not row['path'].startswith('models/'):
            helper['download'](row, root)
    if scope != 'dpg':
        return
    output = root / 'benchmarks/dpg/prompts.jsonl'
    if output.exists():
        helper['verify'](output, pins['dpg_prompts_sha256'])
        return
    archive = root / 'sources/ella-prompts.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_name(archive.name + '.partial')
        offset = partial.stat().st_size if partial.exists() else 0
        request = Request(pins['dpg_prompt_archive_url'], headers={'Range': f'bytes={offset}-'} if offset else {})
        with urlopen(request, timeout=120) as source:
            if offset and (source.status != 206 or not source.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                raise ValueError('Prompt archive cannot resume; partial retained')
            with partial.open('ab' if partial.exists() else 'xb') as target:
                for block in iter(lambda: source.read(1 << 20), b''):
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
        os.replace(partial, archive)
    helper['rebuild_dpg_prompts'](archive, output, pins['dpg_prompts_sha256'])


def execute(args, plan):
    root = Path(plan['data_root'])
    root.mkdir(parents=True, exist_ok=True)
    state_dir = root / 'setup'
    state_dir.mkdir(exist_ok=True)
    status_path = state_dir / 'status.json'
    invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + str(os.getpid())
    logs = state_dir / 'logs' / invocation
    logs.mkdir(parents=True, exist_ok=False)
    state = dict(plan, updated_utc=datetime.now(timezone.utc).isoformat(), status='RUNNING',
                 training_python=None, training={'status': 'PENDING'}, assets_ready=False,
                 evaluation_assets={}, scoring={}, logs=str(logs))
    atomic_json(status_path, state)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PIP_NO_COMPILE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PIP_CACHE_DIR=str(root / 'cache/pip'), HF_HOME=str(root / 'cache/huggingface'),
               TORCH_HOME=str(root / 'cache/torch'), XDG_CACHE_HOME=str(root / 'cache'),
               CONDA_PKGS_DIRS=str(root / 'cache/conda'), TMPDIR=str(root / 'setup/tmp'), JAX_PLATFORMS='cpu')
    for key in ('TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE', 'PYTORCH_PRETRAINED_BERT_CACHE',
                'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'JAX_COMPILATION_CACHE_DIR', 'JAX_ENABLE_COMPILATION_CACHE'):
        env.pop(key, None)
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    config_gate = dict(reserve_gb_input=os.environ.get('SETUP_RESERVE_GB', '32'),
                       minimum_free_inodes_input=os.environ.get('SETUP_MIN_FREE_INODES', '100000'),
                       environment_and_temporary_reserve_bytes=None, minimum_free_inodes=None,
                       passed=False, error=None)
    try:
        reserve_gb = float(config_gate['reserve_gb_input'])
        reserve = int(reserve_gb * (1 << 30))
        config_gate['environment_and_temporary_reserve_bytes'] = reserve
        inodes_min = int(config_gate['minimum_free_inodes_input'])
        config_gate.update(minimum_free_inodes=inodes_min, passed=reserve_gb >= 0 and inodes_min >= 0)
    except (ValueError, OverflowError) as error:
        config_gate['error'] = str(error)
    atomic_json(logs / 'configuration_gate.json', config_gate)
    state['configuration_gate'] = config_gate
    if not config_gate['passed']:
        state.update(status='FAILED', error='Setup reserve and minimum free inodes must be finite nonnegative numbers',
                     updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, state)
        raise ValueError(state['error'])
    atomic_json(status_path, state)
    environment_reserve = reserve // 2
    temporary_reserve = reserve - environment_reserve
    training_destination = Path(plan['training_environment'])
    if args.current_environment:
        requested = os.environ.get('PYTHON') or sys.executable
        training_destination = Path(shutil.which(requested) or requested).absolute().parent.parent
    def record_gate(name, entries, reservations):
        gate = storage_gate(entries, reservations, inodes_min)
        atomic_json(logs / (name + '.json'), gate)
        state[name] = gate
        atomic_json(status_path, state)
        return gate
    try:
        gate = record_gate('storage_gate', downloads(root, Path(plan['eval_assets']), scope='training'),
                           [('training environment', training_destination, environment_reserve),
                            ('cache and temporary files', root, temporary_reserve)])
        if not gate['passed']:
            raise RuntimeError('Insufficient filesystem space/inodes; inspect ' + str(logs / 'storage_gate.json'))
        python = make_environment(plan['training_environment'], logs / 'training.log', env,
                                  current=args.current_environment)
        state['training_python'] = python
        install_training(python, args, logs / 'training.log', env)
        state['training'] = dict(status='PASS', python=python,
                                 requirements_sha256=sha256(HERE / 'requirements.txt'), jax_backend=args.jax_backend)
        atomic_json(status_path, state)
        for kind in ('weights', 'data'):
            run([python, 'prepare_data.py', '--root', str(root), 'download', '--kind', kind], logs / (kind + '.log'), env)
    except Exception as error:
        state['status'] = 'FAILED'
        state['error'] = str(error)
        atomic_json(status_path, state)
        raise
    helper = runpy.run_path(str(HERE / 'evaluation/prepare.py'))
    eval_root = Path(plan['eval_assets'])
    # Sampling is unconditional at each readout, so its pinned prompts are
    # required even when optional scorer models/environments cannot fit.
    try:
        for scope in ('geneval', 'dpg'):
            assets = dict(status='PENDING', sampler_ready=False, error=None)
            state['evaluation_assets'][scope] = assets
            gate = record_gate(scope + '_prompts_storage_gate', prompt_downloads(root, eval_root, scope),
                               [('prompt archive and rebuild temporary files', eval_root, temporary_reserve)])
            if not gate['passed']:
                raise RuntimeError('Insufficient required prompt space/inodes; inspect ' +
                                   str(logs / (scope + '_prompts_storage_gate.json')))
            prepare_prompts(helper, eval_root, scope)
            assets['sampler_ready'] = True
            atomic_json(status_path, state)
        state['assets_ready'] = True
        atomic_json(status_path, state)
    except Exception as error:
        assets['error'] = str(error)
        state.update(status='FAILED', error=str(error), assets_ready=False,
                     updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(status_path, state)
        raise
    for scope in ('geneval', 'dpg'):
        assets = state['evaluation_assets'][scope]
        try:
            gate = record_gate(scope + '_assets_storage_gate', downloads(root, eval_root, scope=scope),
                               [('evaluation assets', eval_root, 0)])
            if not gate['passed']:
                raise RuntimeError('Insufficient optional evaluation space/inodes; inspect ' +
                                   str(logs / (scope + '_assets_storage_gate.json')))
            helper['prepare'](eval_root, scope, fetch=True)
            assets['status'] = 'PASS'
        except Exception as error:
            assets['error'] = str(error)
            print(scope + ' evaluation assets pending: ' + str(error), file=sys.stderr)
        atomic_json(status_path, state)
        scorer = dict(status='PENDING', python=str(Path(plan['scorer_environments'][scope]) / 'bin/python'), error=None)
        state['scoring'][scope] = scorer
        try:
            gate = record_gate(scope + '_environment_storage_gate', [],
                               [('scoring environment', Path(plan['scorer_environments'][scope]), environment_reserve),
                                ('cache and temporary files', root, temporary_reserve)])
            if not gate['passed']:
                raise RuntimeError('Insufficient optional scoring environment space/inodes; inspect ' +
                                   str(logs / (scope + '_environment_storage_gate.json')))
            scorer['python'] = make_environment(plan['scorer_environments'][scope], logs / (scope + '.log'), env)
            run([scorer['python'], '-m', 'pip', 'install', '--only-binary=:all:', '-r',
                 'evaluation/requirements-' + scope + '.txt'], logs / (scope + '.log'), env)
            run([scorer['python'], '-m', 'pip', 'check'], logs / (scope + '.log'), env)
            scorer['status'] = 'PASS'
        except Exception as error:
            scorer['error'] = str(error)
            print(scope + ' scoring pending: ' + str(error), file=sys.stderr)
        atomic_json(status_path, state)
    state['status'] = 'PASS' if all(state['scoring'][s]['status'] == state['evaluation_assets'][s]['status'] == 'PASS'
                                  for s in ('geneval', 'dpg')) else 'TRAINING_READY_SCORING_PENDING'
    state['updated_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(status_path, state)
    print('Training environment ready: ' + state['training_python'])
    print('Setup status: ' + str(status_path) + ' (' + state['status'] + ')')
    print('Next: bash run.sh (PLAN_ONLY; review site parameters before adding --submit)')
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', '--dry-run', action='store_true', dest='plan_only')
    parser.add_argument('--current-environment', action='store_true')
    parser.add_argument('--download-weights', '--download-data', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--jax-backend', choices=('cpu', 'cuda12'), default=os.environ.get('JAX_BACKEND', 'cuda12'))
    args = parser.parse_args(argv)
    if args.jax_backend not in ('cpu', 'cuda12'):
        parser.error('JAX_BACKEND must be cpu or cuda12')
    try:
        plan = make_plan(args)
        if args.plan_only:
            print('SETUP_PLAN: Python 3.10; JAX 0.4.37 [' + args.jax_backend + ']; no installation/download/write')
            print(json.dumps(plan, sort_keys=True, indent=2))
            return 0
        execute(args, plan)
    except Exception as error:
        print('Setup stopped: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
