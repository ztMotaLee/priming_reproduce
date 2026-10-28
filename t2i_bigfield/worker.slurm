#!/usr/bin/env bash
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURMD_NODENAME:-} ]] || { echo 'An allocated compute node is required' >&2; exit 2; }
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Slurm executes a spool copy of this script. The submission's --chdir binds
# the source working directory independently of that copied script's location.
HERE=$(pwd -P)
for entry in worker.py freeze.py; do
    [[ -f "$HERE/$entry" && -r "$HERE/$entry" ]] || {
        echo "Source working directory is missing readable $entry; use run.sh with its bound --chdir" >&2
        exit 2
    }
done
exec "${BIGFIELD_PYTHON:-${PYTHON:-python3}}" "$HERE/worker.py" "$@"
