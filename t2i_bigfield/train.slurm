#!/usr/bin/env bash
set -euo pipefail
for name in NODES GPUS_PER_NODE DATA_ROOT OUTPUT_DIR START_STEP END_STEP BIGFIELD_CODE_ROOT; do
  [[ -n ${!name:-} ]] || { echo "Set $name explicitly" >&2; exit 2; }
done
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run this wrapper inside a Slurm GPU allocation' >&2; exit 2; }
[[ ${SLURM_NNODES:-0} == "$NODES" ]] || { echo 'Node count differs from request' >&2; exit 2; }
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd -- "$BIGFIELD_CODE_ROOT"
[[ -f launcher.py && -f train.py ]] || { echo 'Source checkout is unavailable on this node' >&2; exit 2; }
exec srun --kill-on-bad-exit=1 --ntasks="$NODES" --ntasks-per-node=1 \
  "${PYTHON:-python}" launcher.py --rank
