#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
export PYTHONDONTWRITEBYTECODE=1
if [[ -z ${PYTHON:-} && -x .venv/bin/python ]]; then PYTHON="$PWD/.venv/bin/python"; fi
exec "${PYTHON:-python3}" workflow.py "$@"
