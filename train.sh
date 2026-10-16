#!/usr/bin/env bash
# One-node Stage-1 training; repeat the same command to resume.
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: STORAGE=/path/to/storage bash train.sh' >&2; exit 2; }
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STORAGE=${STORAGE:-"$REPO_ROOT/storage"}
[[ "$STORAGE" = /* ]] || STORAGE="$PWD/$STORAGE"
if [[ -d "$STORAGE" ]]; then STORAGE=$(cd -- "$STORAGE" && pwd); fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
if [[ -f "$STORAGE/s1_env.sh" ]]; then
    source "$STORAGE/s1_env.sh"
elif [[ ${DRY_RUN:-0} != 1 ]]; then
    echo "Run setup first: STORAGE=$(printf '%q' "$STORAGE") bash $(printf '%q' "$REPO_ROOT/setup.sh")" >&2
    exit 1
fi
# Override TMPDIR even when an older setup saved it in s1_env.sh.
export TMPDIR=${S1_TMPDIR:-/tmp}
[[ "$TMPDIR" = /* ]] || export TMPDIR="$PWD/$TMPDIR"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE

case ${NOISE_LEVEL:-1.0} in
    1|1.0) NOISE_LEVEL=1.0; SUFFIX=s100 ;;
    0.8|.8) NOISE_LEVEL=0.8; SUFFIX=s080 ;;
    *) echo 'NOISE_LEVEL must be 1.0 (pure noise) or 0.8 (control).' >&2; exit 2 ;;
esac
EPOCHS=${EPOCHS:-80}
case "$EPOCHS" in 80|160) ;; *) echo 'EPOCHS must be 80 or 160.' >&2; exit 2 ;; esac
if [[ -n ${SMOKE_STEPS:-} ]] && [[ ! "$SMOKE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'SMOKE_STEPS must be a positive integer.' >&2; exit 2
fi

if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    VISIBLE=${CUDA_VISIBLE_DEVICES//[[:space:]]/}
    if [[ -z "$VISIBLE" || "$VISIBLE" == -1 ]]; then
        GPU_COUNT=0
    else
        [[ "$VISIBLE" != ,* && "$VISIBLE" != *, && "$VISIBLE" != *,,* ]] || {
            echo 'Invalid CUDA_VISIBLE_DEVICES: empty GPU entry.' >&2; exit 2;
        }
        IFS=, read -r -a DEVICES <<< "$VISIBLE"
        GPU_COUNT=${#DEVICES[@]}
        declare -A SEEN_GPU=()
        for device in "${DEVICES[@]}"; do
            [[ "$device" =~ ^([0-9]+|GPU-[[:alnum:]-]+|MIG-[[:alnum:]/-]+)$ ]] || {
                echo 'Invalid CUDA_VISIBLE_DEVICES.' >&2; exit 2;
            }
            [[ ! ${SEEN_GPU[$device]+present} ]] || { echo 'Duplicate GPU in CUDA_VISIBLE_DEVICES.' >&2; exit 2; }
            SEEN_GPU[$device]=1
        done
    fi
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LIST=$(nvidia-smi --query-gpu=index --format=csv,noheader)
    GPU_COUNT=0
    while IFS= read -r device; do
        [[ -z "$device" ]] || GPU_COUNT=$((GPU_COUNT + 1))
    done <<< "$GPU_LIST"
else
    GPU_COUNT=0
fi
case "$GPU_COUNT" in 1|2|4|8|16) ;; *) echo "Need 1, 2, 4, 8, or 16 visible NVIDIA GPUs; found $GPU_COUNT." >&2; exit 2 ;; esac
ACCUM=$((1024 / (64 * GPU_COUNT)))
CONFIG_NAME="b${EPOCHS}_perceptual_${SUFFIX}"
SOURCE_CONFIG="$REPO_ROOT/configs/pretrain/$CONFIG_NAME.yaml"
# The optional 160-epoch control reuses the explicit 80-epoch control recipe.
if [[ "$SUFFIX" == s080 && "$EPOCHS" == 160 ]]; then
    SOURCE_CONFIG="$REPO_ROOT/configs/pretrain/b80_perceptual_s080.yaml"
fi
CONFIG="$STORAGE/configs/${CONFIG_NAME}_${GPU_COUNT}gpu.yaml"
OUTPUT=${OUTPUT_DIR:-"$STORAGE/outputs/pretrain/${CONFIG_NAME}_${GPU_COUNT}gpu"}
[[ "$OUTPUT" = /* ]] || OUTPUT="$PWD/$OUTPUT"
PYTHON="$STORAGE/env/bin/python"
TORCHRUN="$STORAGE/env/bin/torchrun"
if [[ ${DRY_RUN:-0} == 1 && ! -x "$PYTHON" ]]; then PYTHON=$(command -v python3); fi
if [[ ${DRY_RUN:-0} == 1 && ! -x "$TORCHRUN" ]]; then TORCHRUN=torchrun; fi
# Match the path lengths used by multiprocessing's resource_sharer.
"$PYTHON" - <<'PY'
import os
import socket
import tempfile

try:
    with tempfile.TemporaryDirectory(prefix='pymp-', dir=os.environ['TMPDIR']) as directory:
        address = os.path.join(directory, 'listener-01234567')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.bind(address)
except OSError as error:
    raise SystemExit(
        f"Cannot create a DataLoader AF_UNIX socket under TMPDIR={os.environ['TMPDIR']!r}: {error}. "
        'Set S1_TMPDIR to a short, writable directory; for example: S1_TMPDIR=/tmp bash train.sh.'
    ) from None
PY
RESUME=none
if [[ -f "$OUTPUT/checkpoints/latest.pth" ]]; then
    RESUME="$OUTPUT/checkpoints/latest.pth"
    if [[ -f "$OUTPUT/config.json" ]]; then
        "$PYTHON" - "$OUTPUT/config.json" "$GPU_COUNT" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1]))
train = config['train']
previous_world = 1024 // (train.get('micro_batch_size', 64) * train['gradient_accumulation_steps'])
if previous_world != int(sys.argv[2]):
    raise SystemExit(f'Resume requires the same GPU count: checkpoint used {previous_world}, requested {sys.argv[2]}.')
PY
    fi
elif [[ -e "$OUTPUT" ]]; then
    echo "Output exists without checkpoints/latest.pth: $OUTPUT. Select a new STORAGE or inspect this incomplete run." >&2
    exit 1
fi

COMMAND=("$TORCHRUN" --standalone "--nproc_per_node=$GPU_COUNT" "$REPO_ROOT/pretrain.py"
    --config "$CONFIG" --data-path "$STORAGE/data/imagenet"
    --hf-home "$STORAGE/assets/huggingface" --torch-home "$STORAGE/assets/torch"
    --lpips-weights "$STORAGE/assets/lpips/vgg.pth" --output "$OUTPUT")
if [[ "$RESUME" != none ]]; then COMMAND+=(--resume "$RESUME"); fi
if [[ -n ${SMOKE_STEPS:-} ]]; then COMMAND+=(--smoke-steps "$SMOKE_STEPS"); fi
printf 'GPU_COUNT=%s\nGRADIENT_ACCUMULATION_STEPS=%s\nNOISE_LEVEL=%s\nEPOCHS=%s\nRESUME=%s\nTMPDIR=%s\n' \
    "$GPU_COUNT" "$ACCUM" "$NOISE_LEVEL" "$EPOCHS" "$RESUME" "$TMPDIR"
printf 'COMMAND:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
[[ ${DRY_RUN:-0} != 1 ]] || exit 0

cd "$REPO_ROOT"
"$PYTHON" scripts/check_environment.py --cuda
mkdir -p "$STORAGE/configs"
"$PYTHON" - "$SOURCE_CONFIG" "$CONFIG" "$ACCUM" "$EPOCHS" <<'PY'
import os
from pathlib import Path
import sys
from omegaconf import OmegaConf
from utils.config import load_yaml

source, target, accumulation, epochs = sys.argv[1:]
config = load_yaml(source)
config['train']['gradient_accumulation_steps'] = int(accumulation)
config['train']['epochs'] = int(epochs)
temporary = Path(target).with_suffix('.yaml.tmp')
OmegaConf.save(OmegaConf.create(config), temporary)
os.replace(temporary, target)
PY
export S1_LAUNCH_ID="$(date -u +%Y%m%dT%H%M%S)-$$-$RANDOM"
set +e
"${COMMAND[@]}"
STATUS=$?
set -e
# torchrun wraps worker exit 75 as exit 1; accept only this launch's saved receipt.
if [[ "$STATUS" != 0 ]] && "$PYTHON" - "$OUTPUT" "$S1_LAUNCH_ID" <<'PY'
import json
from pathlib import Path
import sys
try:
    output = Path(sys.argv[1])
    receipt = json.loads((output / 'safe_stop.json').read_text())
    checkpoint = output / 'checkpoints' / f"step_{int(receipt['completed']):07d}.pth"
    valid = (receipt['state'] == 'SAFE_STOPPED' and receipt['launch_id'] == sys.argv[2]
             and Path(receipt['checkpoint']) == checkpoint and checkpoint.is_file())
except (OSError, ValueError, KeyError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
then
    STATUS=75
fi
if [[ "$STATUS" == 75 ]]; then
    echo 'Safely stopped at a checkpoint. Repeat the same command to continue.'
fi
exit "$STATUS"
