#!/usr/bin/env bash
# One process sees every local GPU. Multi-node placement is in train.slurm.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ ! -f .stage1.env ]] || source .stage1.env
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
export XLA_FLAGS='--xla_gpu_autotune_level=1'
export JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export JAX_PROCESS_COUNT="${JAX_PROCESS_COUNT:-${SLURM_NNODES:-1}}"
export JAX_PROCESS_INDEX="${JAX_PROCESS_INDEX:-${SLURM_PROCID:-0}}"
if [[ -n "${GPUS_PER_NODE:-${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}}}" ]]; then
  export STAGE1_GPUS_PER_NODE="${STAGE1_GPUS_PER_NODE:-${GPUS_PER_NODE:-${SLURM_GPUS_PER_TASK:-${SLURM_GPUS_PER_NODE:-$SLURM_GPUS_ON_NODE}}}}"
fi
DATA_DIR="${DATA_DIR:-$PWD/data}"
ASSET_ROOT="${ASSET_ROOT:-$DATA_DIR/assets}"
export HF_HOME="$ASSET_ROOT/hf-cache" TORCH_HOME="$ASSET_ROOT/torch-cache"
PYTHON="${PYTHON:-${VENV_DIR:-$PWD/.venv}/bin/python}"
command=("$PYTHON" -m t2i.train --cc12m-root "${CC12M_ROOT:-$DATA_DIR/cc12m}"
         --curated-root "${CURATED_ROOT:-$DATA_DIR/curated/sources.json}"
         --t5-model "${T5_MODEL:-$ASSET_ROOT/t5}" --perceptual-dir "${PERCEPTUAL_DIR:-$ASSET_ROOT/perceptual}"
         --output-dir "${OUTPUT_DIR:-$PWD/runs/stage1}" --microbatch "${MICROBATCH:-1}")
[[ -z "${RESUME:-}" ]] || command+=(--resume "$RESUME")
command+=("$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY_RUN process_count=%s process_index=%s global_batch=1024 local_GPU_count=%s\n' "$JAX_PROCESS_COUNT" "$JAX_PROCESS_INDEX" "${STAGE1_GPUS_PER_NODE:-auto}"
  printf 'command:'; printf ' %q' "${command[@]}"; printf '\n'
  exit 0
fi
[[ -x "$PYTHON" ]] || { printf 'Missing Python environment: %s; run bash setup.sh\n' "$PYTHON" >&2; exit 2; }
export TMPDIR
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/stage1-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}-XXXXXXXX")"
# The trainer enumerates local GPUs and atomically records all numerical
# preflight inputs before its device/batch/step gates, on every rank.
exec "${command[@]}"
