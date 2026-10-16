#!/usr/bin/env bash
#SBATCH --signal=B:USR1@600
# Submit from the repository root. Supply all resource/site options on the command line.
# Example: sbatch --nodes=2 --ntasks-per-node=1 --gres=gpu:8 --time=24:00:00 --export=ALL --no-requeue train.slurm
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: sbatch [site/resource options] train.slurm' >&2; exit 2; }
REPO_ROOT=${S1_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}
[[ -f "$REPO_ROOT/train.sh" ]] || { echo 'Submit from the repository root.' >&2; exit 2; }
export S1_REPO_ROOT="$REPO_ROOT"
export STORAGE=${STORAGE:-"$REPO_ROOT/storage"}
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
# A previous setup may have saved a long TMPDIR or obsolete cache variables.
export TMPDIR=${S1_TMPDIR:-/tmp}
[[ "$TMPDIR" = /* ]] || export TMPDIR="$PWD/$TMPDIR"
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
PYTHON="$STORAGE/env/bin/python"
if [[ ${DRY_RUN:-0} == 1 && ! -x "$PYTHON" ]]; then PYTHON=$(command -v python3); fi
export PREEMPT_FLAG_FILE=${PREEMPT_FLAG_FILE:-"$STORAGE/jobs/preempt-${SLURM_JOB_ID:-$$}"}
[[ "$PREEMPT_FLAG_FILE" = /* ]] || PREEMPT_FLAG_FILE="$PWD/$PREEMPT_FLAG_FILE"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    exec "$PYTHON" "$REPO_ROOT/utils/s1_launch.py" --slurm
fi
[[ ! -e "$PREEMPT_FLAG_FILE" ]] || { echo "Stop flag already exists: $PREEMPT_FLAG_FILE" >&2; exit 1; }
mkdir -p -- "$(dirname -- "$PREEMPT_FLAG_FILE")"
trap 'touch "$PREEMPT_FLAG_FILE"' USR1
"$PYTHON" "$REPO_ROOT/utils/s1_launch.py" --slurm &
CHILD=$!
set +e
while true; do
    wait "$CHILD"
    STATUS=$?
    kill -0 "$CHILD" 2>/dev/null || break
done
exit "$STATUS"
