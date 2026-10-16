#!/usr/bin/env bash
# One-time environment/assets preparation. Dataset downloads are opt-in.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
environment_only=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) printf '%s\n' 'Usage: bash setup.sh [--dry-run] [--environment-only]' 'Set CC12M_ROOT and CURATED_ROOT for existing local shards; DOWNLOAD_DATA=1 prepares missing datasets.' 'Use --environment-only on login nodes: install dependencies without touching datasets or model assets.'; exit 0 ;;
    --dry-run) export DRY_RUN=1 ;;
    --environment-only) environment_only=1 ;;
    *) printf 'Unknown setup argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
export XLA_FLAGS='--xla_gpu_autotune_level=1'
export JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" TOKENIZERS_PARALLELISM=false
provided_cc12m="${CC12M_ROOT:+1}"
provided_curated="${CURATED_ROOT:+1}"
DATA_DIR="${DATA_DIR:-$PWD/data}"
ASSET_ROOT="${ASSET_ROOT:-$DATA_DIR/assets}"
T5_MODEL="${T5_MODEL:-$ASSET_ROOT/t5}"
PERCEPTUAL_DIR="${PERCEPTUAL_DIR:-$ASSET_ROOT/perceptual}"
CC12M_ROOT="${CC12M_ROOT:-$DATA_DIR/cc12m}"
CURATED_ROOT="${CURATED_ROOT:-$DATA_DIR/curated/sources.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/runs/stage1}"
VENV_DIR="${VENV_DIR:-$PWD/.venv}"
JAX_PLATFORM="${JAX_PLATFORM:-cuda12}"
[[ "$JAX_PLATFORM" == cuda12 || "$JAX_PLATFORM" == cpu ]] || { printf '%s\n' 'JAX_PLATFORM must be cuda12 or cpu' >&2; exit 2; }
[[ "${DOWNLOAD_DATA:-0}" == 0 || "${DOWNLOAD_DATA:-0}" == 1 ]] || { printf '%s\n' 'DOWNLOAD_DATA must be 0 or 1' >&2; exit 2; }
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'DRY_RUN setup: Python >=3.10,<3.13; venv=%q; JAX=0.4.31 [%s]\n' "$VENV_DIR" "$JAX_PLATFORM"
  printf 'ENVIRONMENT_ONLY=%s (no asset/data preparation or .stage1.env write)\n' "$environment_only"
  printf 'CC12M_ROOT=%q\nCURATED_ROOT=%q\nT5_MODEL=%q\nPERCEPTUAL_DIR=%q\n' "$CC12M_ROOT" "$CURATED_ROOT" "$T5_MODEL" "$PERCEPTUAL_DIR"
  printf 'DOWNLOAD_DATA=%s (CC12M URL metadata 3.58 GB; curated raw downloads 231.45 GB; image URLs extra)\n' "${DOWNLOAD_DATA:-0}"
  exit 0
fi
# Setup needs internet and enough CPU memory for weight conversion. Tiger uses
# its separate login download-only entry + offline allocated-compute loader.
if [[ "$environment_only" == 0 ]]; then
  case "$(hostname -s)" in
    tiger[0-9]*) printf '%s\n' 'Tiger login: use setup.sh --environment-only for dependencies; internal/download_assets_login.py for downloads; then allocated compute offline prep.' >&2; exit 2 ;;
  esac
fi
export TMPDIR
TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/stage1-setup-XXXXXXXX")"
export HF_HOME="$ASSET_ROOT/hf-cache" TORCH_HOME="$ASSET_ROOT/torch-cache"
export PIP_CACHE_DIR="$DATA_DIR/pip-cache"
export PIP_NO_COMPILE=1
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Use Python 3.10–3.12"'
if [[ ! -x "$VENV_DIR/bin/python" ]]; then "$PYTHON" -m venv "$VENV_DIR"; fi
PYTHON="$VENV_DIR/bin/python"
# Old pip releases mis-normalize names on the Torch CPU index and fall back to
# unavailable source-build dependencies. Upgrade before resolving those wheels.
"$PYTHON" -m pip install --upgrade 'pip>=25.3,<26'
# CPU Torch is used only for downloading/converting frozen perceptual assets.
"$PYTHON" -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cpu
if [[ "$JAX_PLATFORM" == cuda12 ]]; then
  "$PYTHON" -m pip install 'jax[cuda12]==0.4.31' -r t2i/requirements.txt
else
  "$PYTHON" -m pip install 'jax==0.4.31' -r t2i/requirements.txt
fi
"$PYTHON" -m pip check
if [[ "$environment_only" == 1 ]]; then
  printf 'Environment ready: %s\n' "$PYTHON"
  printf '%s\n' 'No dataset/model preparation or .stage1.env write performed. Use VENV_DIR for training after offline asset preparation.'
  exit 0
fi
# Download-only raw phase; conversion below is explicitly offline. Custom
# asset paths must already contain prepared, pinned assets.
if [[ ! -f "$T5_MODEL/assets.sha256.json" || ! -f "$PERCEPTUAL_DIR/assets.sha256.json" ]]; then
  if [[ "$T5_MODEL" != "$ASSET_ROOT/t5" || "$PERCEPTUAL_DIR" != "$ASSET_ROOT/perceptual" ]]; then
    printf '%s\n' 'Custom T5_MODEL/PERCEPTUAL_DIR must contain complete prepared assets.' >&2; exit 2
  fi
  "$PYTHON" - "$ASSET_ROOT" <<'PY'
import sys
from pathlib import Path
from t2i.asset_downloads import download_all, offline_only
from t2i.perceptual import prepare
root = Path(sys.argv[1])
if not (root / 'download_receipt.json').exists():
    download_all(root)
with offline_only():
    prepare(root / 'perceptual', raw_directory=root / 'perceptual-sources')
PY
fi
# Existing manifests alone are insufficient: require the full pinned payload set.
"$PYTHON" - "$T5_MODEL" "$PERCEPTUAL_DIR" <<'PY'
import sys
from t2i.assets import validate_t5_assets
from t2i.perceptual_assets import validate_perceptual_assets
validate_t5_assets(sys.argv[1], verify_payloads=True)
validate_perceptual_assets(sys.argv[2], verify_payloads=True)
PY
if [[ "${DOWNLOAD_DATA:-0}" == 1 ]]; then
  prep=("$PYTHON" -m t2i.prepare datasets --data-dir "$DATA_DIR" --t5-model "$T5_MODEL" --download
        --processes "${DOWNLOAD_PROCESSES:-8}" --threads "${DOWNLOAD_THREADS:-32}")
  [[ "$provided_cc12m" != 1 ]] || prep+=(--cc12m-root "$CC12M_ROOT")
  [[ "$provided_curated" != 1 ]] || prep+=(--curated-root "$CURATED_ROOT")
  "${prep[@]}"
fi
"$PYTHON" -m t2i.prepare check --cc12m-root "$CC12M_ROOT" --curated-root "$CURATED_ROOT"
# Store only local paths, shell-quoted; explicit environment settings still win.
env_tmp="$(mktemp "$PWD/.stage1.env.tmp-XXXXXXXX")"
for name in DATA_DIR ASSET_ROOT T5_MODEL PERCEPTUAL_DIR CC12M_ROOT CURATED_ROOT OUTPUT_DIR VENV_DIR; do
  printf 'if [[ -z ${%s+x} ]]; then %s=%q; fi\n' "$name" "$name" "${!name}" >> "$env_tmp"
  printf 'export %s\n' "$name" >> "$env_tmp"
done
mv "$env_tmp" .stage1.env
printf '%s\n' 'Setup complete. Run: bash train.sh'
