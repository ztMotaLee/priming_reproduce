#!/usr/bin/env bash
#SBATCH --signal=B:USR1@600
# Submit from the repository root. Supply all resource/site options on the command line.
# Example: sbatch --nodes=1 --gres=gpu:4 --time=24:00:00 --export=ALL --no-requeue train.slurm
set -euo pipefail
[[ $# -eq 0 ]] || { echo 'Usage: sbatch [site/resource options] train.slurm' >&2; exit 2; }
[[ ${SLURM_NNODES:-1} == 1 ]] || { echo 'train.slurm supports one node.' >&2; exit 2; }
REPO_ROOT=${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}
[[ -f "$REPO_ROOT/train.sh" ]] || { echo 'Submit from the repository root.' >&2; exit 2; }
export STORAGE=${STORAGE:-"$REPO_ROOT/storage"}
mkdir -p "$STORAGE"
STORAGE=$(cd -- "$STORAGE" && pwd)
mkdir -p "$STORAGE/jobs"
export PREEMPT_FLAG_FILE="$STORAGE/jobs/preempt-${SLURM_JOB_ID:-$$}"
[[ ! -e "$PREEMPT_FLAG_FILE" ]] || { echo "Stop flag already exists: $PREEMPT_FLAG_FILE" >&2; exit 1; }
trap 'touch "$PREEMPT_FLAG_FILE"' USR1
bash "$REPO_ROOT/train.sh" &
CHILD=$!
set +e
while true; do
    wait "$CHILD"
    STATUS=$?
    kill -0 "$CHILD" 2>/dev/null || break
done
exit "$STATUS"
