#!/usr/bin/env python3
"""Train a fixed-science segment; real execution requires an allocated GPU job."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import time

# Clear inherited legacy caches before importing the first ML dependency.
for _name in ('TRANSFORMERS_CACHE','PYTORCH_TRANSFORMERS_CACHE',
              'PYTORCH_PRETRAINED_BERT_CACHE','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE',
              'JAX_COMPILATION_CACHE_DIR','JAX_ENABLE_COMPILATION_CACHE','JAX_DEFAULT_MATMUL_PRECISION'):
    os.environ.pop(_name,None)
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='bfc'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='0.90'
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('XLA_FLAGS','--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--weights-root',type=Path)
    p.add_argument('--native-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-step',type=int,required=True)
    p.add_argument('--end-step',type=int,required=True)
    p.add_argument('--nodes',type=int,required=True)
    p.add_argument('--gpus-per-node',type=int,required=True)
    p.add_argument('--coordinator')
    p.add_argument('--resume',help='latest or a full-state checkpoint directory')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    from training import load_config
    from distributed import Geometry
    config=load_config(a.config)
    geometry=Geometry(a.nodes,0,a.gpus_per_node)
    if not 0<=a.start_step<a.end_step<=config['updates']:
        p.error('Require 0 <= start < end <= 30000')
    if a.start_step and not a.resume:p.error('A nonzero segment requires --resume latest or an explicit full-state checkpoint')
    if not a.start_step and a.resume:p.error('Cold start must not specify resume')
    plan=dict(mode='PLAN_ONLY',start_step=a.start_step,end_step=a.end_step,nodes=a.nodes,
              gpus_per_node=a.gpus_per_node,global_captions=64,active_rows=1024,
              detached_reference_rows=1024,paired_real_rows=64,field_rows=1024,
              fixed_weight=config['field']['weight'],permanent_steps=config['permanent_checkpoint_steps'],
              runtime_initialized=False,weights_loaded=False)
    if a.dry_run:
        print(json.dumps(plan,sort_keys=True,indent=2));return 0
    from distributed import initialize,barrier,world_any
    geometry=initialize(a.nodes,a.gpus_per_node,a.coordinator)
    if a.start_step==0 and a.output.exists():
        existing=[p for p in a.output.iterdir() if not p.name.startswith('.coordinator_')]
        if existing:raise FileExistsError('Cold-start output already has training artifacts')
    from checkpoint import atomic_json,sha256_file,segment_report_path
    from training import Runtime
    resume=None
    if a.resume=='latest':
        latest=json.loads((a.output/'latest.json').read_text())
        resume=a.output/'checkpoints'/latest['checkpoint']
        if sha256_file(resume/'checkpoint.json')!=latest['checkpoint_sha256']:
            raise ValueError('Latest checkpoint metadata SHA differs')
    elif a.resume:resume=Path(a.resume)
    pause=[False]
    def receive(signum,frame):pause[0]=True
    signal.signal(signal.SIGUSR1,receive)
    # SIGTERM follows the same completed-update pause path when time permits.
    signal.signal(signal.SIGTERM,receive)
    rt=Runtime(config,a.data_root,a.weights_root or a.data_root/'weights',a.output,
               geometry,resume=resume,native_root=a.native_root)
    if rt.step!=a.start_step:raise ValueError('Resume clock differs from requested segment start')
    attempted=segment_report_path(a.output,a.start_step,a.end_step)
    if geometry.rank==0:
        atomic_json(attempted,dict(status='RUNNING',start_step=a.start_step,end_step=a.end_step),create_once=True)
    barrier('segment_start')
    started=time.monotonic()
    try:
        while rt.step<a.end_step:
            if world_any(pause[0]):break
            rt.update()
            if rt.step%config['checkpoint_interval']==0 or rt.step==a.end_step:rt.save()
        if rt.step!=a.end_step:rt.save()
        complete=rt.step==a.end_step
        if geometry.rank==0:atomic_json(attempted,dict(status='COMPLETE' if complete else 'PAUSED',
            start_step=a.start_step,step=rt.step,end_step=a.end_step,seconds=time.monotonic()-started,
            dose=rt.dose_status))
        return 0 if complete else 75
    except BaseException as error:
        if geometry.rank==0:atomic_json(attempted,dict(status='FAILED',start_step=a.start_step,
               step=rt.step,end_step=a.end_step,error_type=type(error).__name__,error=str(error),
               dose=rt.dose_status))
        raise

if __name__=='__main__':raise SystemExit(main())
