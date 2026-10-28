#!/usr/bin/env bash
# Install dependencies; payload downloads are explicit and resumable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$PWD/.venv}"
DATA_ROOT="${DATA_ROOT:-}"
JAX_BACKEND="${JAX_BACKEND:-cuda12}"
[[ "$JAX_BACKEND" == cpu || "$JAX_BACKEND" == cuda12 ]] || { echo 'JAX_BACKEND must be cpu or cuda12' >&2; exit 2; }
weights=0; data=0; dry=0; current=0
while (( $# )); do
  case "$1" in
    --download-weights) weights=1 ;;
    --download-data) data=1 ;;
    --current-environment) current=1 ;;
    --dry-run) dry=1 ;;
    -h|--help) echo 'Usage: DATA_ROOT=... bash setup.sh [--current-environment] [--download-weights] [--download-data] [--dry-run]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ -n "$DATA_ROOT" ]] || { echo 'Set DATA_ROOT to your shared data directory' >&2; exit 2; }
if (( dry )); then
  printf 'SETUP_PLAN: Python 3.10; JAX 0.4.37 [%s]; dependencies requirements.txt\n' "$JAX_BACKEND"
  printf 'DATA_ROOT=%q VENV_DIR=%q CURRENT_ENVIRONMENT=%s\n' "$DATA_ROOT" "$VENV_DIR" "$current"
  printf 'DOWNLOAD_WEIGHTS=%s DOWNLOAD_DATA=%s\n' "$weights" "$data"
  exit 0
fi
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_COMPILE=1 TOKENIZERS_PARALLELISM=false
export PIP_CACHE_DIR="$DATA_ROOT/cache/pip" HF_HOME="$DATA_ROOT/cache/huggingface"
export TORCH_HOME="$DATA_ROOT/cache/torch" XDG_CACHE_HOME="$DATA_ROOT/cache"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10 for the pinned reproduction environment"'
if (( ! current )); then
  [[ -x "$VENV_DIR/bin/python" ]] || "$PYTHON" -m venv "$VENV_DIR"
  PYTHON="$VENV_DIR/bin/python"
fi
"$PYTHON" -m pip install 'pip>=25.3,<26'
"$PYTHON" -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cpu
if [[ "$JAX_BACKEND" == cuda12 ]]; then
  "$PYTHON" -m pip install 'jax[cuda12]==0.4.37' -r requirements.txt
else
  "$PYTHON" -m pip install 'jax==0.4.37' 'jaxlib==0.4.36' -r requirements.txt
fi
"$PYTHON" -m pip check
if (( weights )); then "$PYTHON" prepare_data.py --root "$DATA_ROOT" download --kind weights; fi
if (( data )); then "$PYTHON" prepare_data.py --root "$DATA_ROOT" download --kind data; fi
printf 'Environment ready: %s\n' "$PYTHON"
printf '%s\n' 'Run curated conversion and DINO bank extraction on allocated compute resources; see README.md.'
