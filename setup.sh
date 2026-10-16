#!/usr/bin/env bash
# Install the official environment and verified Stage-1 assets.
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: STORAGE=/path/to/storage bash setup.sh' >&2; exit 2; }
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STORAGE=${STORAGE:-"$REPO_ROOT/storage"}
mkdir -p "$STORAGE"
STORAGE=$(cd -- "$STORAGE" && pwd)
export STORAGE PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
unset HF_TOKEN_PATH HF_STORED_TOKENS_PATH
export CONDA_PKGS_DIRS="$STORAGE/conda-pkgs" CONDA_ENVS_PATH="$STORAGE/conda-envs"
export PIP_CACHE_DIR="$STORAGE/pip-cache" XDG_CACHE_HOME="$STORAGE/cache"
export XDG_CONFIG_HOME="$STORAGE/config" XDG_DATA_HOME="$STORAGE/share"
export HF_HOME="$STORAGE/assets/huggingface" HF_HUB_CACHE="$STORAGE/assets/huggingface/hub"
export HF_XET_CACHE="$STORAGE/assets/.xet-cache" TORCH_HOME="$STORAGE/assets/torch"
export TMPDIR="$STORAGE/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" \
    "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
cd "$REPO_ROOT"

if [[ ! -d "$STORAGE/env" ]]; then
    CONDA_BIN=${CONDA_EXE:-conda}
    if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
        echo 'Conda is required. Install Conda, or load your cluster Conda module (for example: module load anaconda3).' >&2
        exit 1
    fi
    # Conda creates a pip requirements file beside environment.yml.
    # Stage these unchanged official files so that it writes only inside STORAGE.
    ENV_SOURCE="$STORAGE/setup-source"
    mkdir -p "$ENV_SOURCE/scripts" "$ENV_SOURCE/old/environment"
    cp "$REPO_ROOT/environment.yml" "$ENV_SOURCE/environment.yml"
    cp "$REPO_ROOT/scripts/create_env.sh" "$REPO_ROOT/scripts/check_environment.py" "$ENV_SOURCE/scripts/"
    cp "$REPO_ROOT/old/environment/requirements-linux-cu124.txt" "$ENV_SOURCE/old/environment/"
    bash "$ENV_SOURCE/scripts/create_env.sh" --prefix "$STORAGE/env"
fi
PYTHON="$STORAGE/env/bin/python"
[[ -x "$PYTHON" ]] || { echo "Incomplete environment: $STORAGE/env. Repair it or select a new STORAGE." >&2; exit 1; }
export PATH="$STORAGE/env/bin:$PATH"

if [[ ${S1_SKIP_DATASET:-0} != 1 ]]; then
    if ! "$STORAGE/env/bin/hf" auth whoami >/dev/null 2>&1; then
        if [[ -n ${HF_TOKEN:-} ]]; then
            echo 'HF_TOKEN was not accepted. Check its permissions and the ImageNet dataset terms.' >&2
            exit 1
        elif [[ -t 0 ]]; then
            # Keep an optional Git-credential save inside STORAGE as well.
            GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0= \
                GIT_CONFIG_KEY_1=credential.helper \
                GIT_CONFIG_VALUE_1="store --file=$(printf '%q' "$STORAGE/hf-git-credentials")" \
                "$STORAGE/env/bin/hf" auth login
        else
            echo 'Log in with HF_TOKEN, or run setup.sh interactively. Accept the ImageNet terms on the dataset page first.' >&2
            exit 1
        fi
    fi
fi

prepare_assets() {
    local group=$1
    if "$PYTHON" scripts/download_assets.py --group "$group" --root "$STORAGE" --verify-only \
        > "$STORAGE/setup_verify_${group}.log" 2>&1; then
        echo "Verified existing $group assets."
        return
    fi
    echo "Preparing $group assets (verification log: $STORAGE/setup_verify_${group}.log)."
    "$PYTHON" - "$STORAGE" "$group" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys

storage, group = Path(sys.argv[1]), sys.argv[2]
if group == 'dataset':
    rows = json.loads(Path('old/releases/dataset.json').read_text())['files']
else:
    rows = [row for row in json.loads(Path('old/releases/encoders.json').read_text())['files']
            if group in row['groups']]
missing = sum(row['bytes'] for row in rows if not (storage / row['path']).is_file())
margin = 40_000_000_000 if group == 'dataset' else 5_000_000_000
required = missing + margin
free = shutil.disk_usage(storage).free
receipt = dict(group=group, free_bytes=free, missing_bytes=missing,
               margin_bytes=margin, required_bytes=required, passed=free >= required)
target = storage / f'setup_disk_{group}.json'
temporary = target.with_suffix('.json.tmp')
temporary.write_text(json.dumps(receipt, indent=2) + '\n')
os.replace(temporary, target)
print(f'{group}: free={free / 1e9:.1f} GB, needed={required / 1e9:.1f} GB', flush=True)
if free < required:
    raise SystemExit('Insufficient free storage. A fresh full setup needs about 300 GB plus training outputs.')
PY
    "$PYTHON" scripts/download_assets.py --group "$group" --root "$STORAGE"
    "$PYTHON" scripts/download_assets.py --group "$group" --root "$STORAGE" --verify-only
}

if [[ ${S1_SKIP_DATASET:-0} == 1 ]]; then
    echo 'S1_SKIP_DATASET=1: skipping ImageNet download and dataset login (setup test only).'
else
    prepare_assets dataset
fi
prepare_assets pretraining
"$PYTHON" scripts/check_environment.py
"$PYTHON" pretrain.py --config configs/pretrain/b80_perceptual_s100.yaml --check-config

ENV_FILE=$(mktemp "$TMPDIR/s1-env.XXXXXX")
{
    printf '# Generated by setup.sh.\n'
    for name in STORAGE CONDA_PKGS_DIRS CONDA_ENVS_PATH PIP_CACHE_DIR XDG_CACHE_HOME \
        XDG_CONFIG_HOME XDG_DATA_HOME HF_HOME HF_HUB_CACHE HF_XET_CACHE TORCH_HOME; do
        printf 'export %s=%q\n' "$name" "${!name}"
    done
    printf 'export PATH=%q:"$PATH"\n' "$STORAGE/env/bin"
    printf 'export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1\n'
} > "$ENV_FILE"
mv "$ENV_FILE" "$STORAGE/s1_env.sh"
echo "Setup complete. On a GPU node, run: STORAGE=$(printf '%q' "$STORAGE") bash $(printf '%q' "$REPO_ROOT/train.sh")"
