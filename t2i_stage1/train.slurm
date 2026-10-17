#!/usr/bin/env bash
#SBATCH --job-name=priming-stage1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --no-requeue
#SBATCH --signal=USR1@180
#SBATCH --output=stage1-%j.out
set -euo pipefail
# Under Slurm this file lives in a spool directory; never use that location.
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  code_dir="${STAGE1_CODE_DIR:-${SLURM_SUBMIT_DIR:-}}"
  [[ $code_dir == /* ]] || {
    printf '%s\n' 'Set an absolute STAGE1_CODE_DIR or submit from the code directory.' >&2; exit 2;
  }
else
  code_dir="${STAGE1_CODE_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}}"
fi
cd -- "$code_dir" || exit 2
[[ -f train.sh && -f t2i/train.py ]] || {
  printf 'Invalid Stage-1 code directory: %s\n' "$PWD" >&2; exit 2;
}
[[ ! -f .stage1.env ]] || source .stage1.env
CC12M_ONLY="${CC12M_ONLY-0}"
[[ "$CC12M_ONLY" == 0 || "$CC12M_ONLY" == 1 ]] || { printf '%s\n' 'CC12M_ONLY must be 0 or 1' >&2; exit 2; }
# Export one batch-host decision; workers' .stage1.env cannot change it.
export CC12M_ONLY
unset TRANSFORMERS_CACHE PYTORCH_TRANSFORMERS_CACHE PYTORCH_PRETRAINED_BERT_CACHE
unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
export XLA_FLAGS='--xla_gpu_autotune_level=1'
export JAX_ENABLE_COMPILATION_CACHE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Validate argv without importing Python or touching GPUs. Useful in CPU CI.
want_value=0
requested_microbatch="${MICROBATCH:-1}"
requested_steps=290000
[[ "$CC12M_ONLY" == 0 ]] || requested_steps=250000
requested_smoke=0
requested_checkpoint_every=5000
for arg in "$@"; do
  if [[ "$want_value" == 1 ]]; then
    [[ -n "$arg" && "$arg" != --* ]] || { printf '%s\n' 'Missing training argument value' >&2; exit 2; }
    case "$value_kind" in
      number)
        case "$numeric_option" in
          --microbatch) requested_microbatch="$arg" ;;
          --steps) requested_steps="$arg" ;;
          --checkpoint-every) requested_checkpoint_every="$arg" ;;
        esac ;;
      noise) [[ "$arg" == logit-normal || "$arg" == fixed-0.8 || "$arg" == fixed-1.0 ]] || { printf 'Invalid noise: %s\n' "$arg" >&2; exit 2; } ;;
    esac
    want_value=0
  else
    case "$arg" in
      --cc12m-root|--curated-root|--output-dir|--resume|--t5-model|--perceptual-dir|--data-dir) want_value=1; value_kind=path ;;
      --steps|--checkpoint-every|--microbatch) want_value=1; value_kind=number; numeric_option="$arg" ;;
      --noise) want_value=1; value_kind=noise ;;
      --smoke) requested_smoke=1 ;;
      --cc12m-only) printf '%s\n' 'Use CC12M_ONLY=1 with train.slurm; --cc12m-only is the Python CLI switch.' >&2; exit 2 ;;
      *) printf 'Unsupported training argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
  fi
done
[[ "$want_value" == 0 ]] || { printf '%s\n' 'Missing training argument value' >&2; exit 2; }
nodes="${SLURM_NNODES:-${NODES:-1}}"
gpus="${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}}"
gpus="${gpus##*:}"
audit_dir="$(mktemp -d "${STAGE1_AUDIT_DIR:-${TMPDIR:-/tmp}}/stage1-launch-XXXXXXXX")"
STAGE1_GATE_NODES="$nodes" STAGE1_GATE_GPUS="$gpus" STAGE1_GATE_MICROBATCH="$requested_microbatch" \
STAGE1_GATE_STEPS="$requested_steps" STAGE1_GATE_SMOKE="$requested_smoke" \
STAGE1_GATE_CHECKPOINT_EVERY="$requested_checkpoint_every" \
"${AUDIT_PYTHON:-python3}" - "$audit_dir/preflight.json" <<'PY'
import json, os, pathlib, re, sys, tempfile
raw = {key: os.environ['STAGE1_GATE_' + key.upper()]
       for key in ('nodes', 'gpus', 'microbatch', 'steps', 'smoke', 'checkpoint_every')}
values = {key: int(value) if re.fullmatch(r'[0-9]{1,12}', value) else None
          for key, value in raw.items()}
positive = all(values[k] is not None and values[k] > 0
               for k in ('nodes', 'gpus', 'microbatch', 'steps', 'checkpoint_every'))
devices = values['nodes'] * values['gpus'] if positive else None
denominator = devices * values['microbatch'] if positive else None
cc12m_only = os.environ['CC12M_ONLY'] == '1'
maximum_steps = 250000 if cc12m_only else 290000
passed = bool(positive and values['steps'] <= maximum_steps and values['smoke'] in (0, 1)
              and (values['smoke'] == 1 or values['steps'] == maximum_steps)
              and 1024 % denominator == 0)
record = {'gate': 'launch_batch_and_dose', 'raw': raw, 'values': values,
          'global_batch': 1024, 'maximum_steps': maximum_steps, 'production_steps': maximum_steps,
          'cc12m_only': cc12m_only, 'data_plan': 'cc12m-only' if cc12m_only else 'paper-two-stage',
          'global_devices': devices, 'batch_divisor': denominator,
          'xla_flags': os.environ.get('XLA_FLAGS'), 'accepted': passed}
target = pathlib.Path(sys.argv[1])
fd, temporary = tempfile.mkstemp(prefix='.preflight-', dir=target.parent)
with os.fdopen(fd, 'w') as handle:
    json.dump(record, handle, indent=2, sort_keys=True); handle.write('\n')
    handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, target)
print('Launch preflight receipt: ' + str(target), file=sys.stderr)
if not passed:
    raise SystemExit('Invalid batch topology or update target; inspect preflight receipt')
PY
nodes=$((10#$nodes))
gpus=$((10#$gpus))
export JAX_PROCESS_COUNT="$nodes"
export STAGE1_GPUS_PER_NODE="$gpus" GPUS_PER_NODE="$gpus"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'CC12M_ONLY=%s\n' "$CC12M_ONLY"
  export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-node0:12355}"
  printf 'DRY_RUN nodes=%s GPUs_per_node=%s processes_per_node=1 global_batch=1024 effective_per_GPU=%s coordinator=%s\n' \
    "$nodes" "$gpus" "$((1024 / (nodes * gpus)))" "$JAX_COORDINATOR_ADDRESS"
  printf 'launch:'; printf ' %q' srun --export=ALL --kill-on-bad-exit=1 --ntasks="$nodes" --ntasks-per-node=1 --gpus-per-task="$gpus" --gpu-bind=none bash "$PWD/train.sh" "$@"; printf '\n'
  bash "$PWD/train.sh" "$@"
  exit 0
fi
[[ -n "${SLURM_JOB_ID:-}" ]] || { printf '%s\n' 'Use sbatch train.slurm, or DRY_RUN=1 for inspection' >&2; exit 2; }
# The batch shell runs on the first allocated compute node. All JAX workers
# use its routable hostname; override ADDRESS/PORT for site-specific networking.
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-$(hostname -f):${JAX_COORDINATOR_PORT:-12355}}"
exec srun --export=ALL --kill-on-bad-exit=1 --ntasks="$nodes" --ntasks-per-node=1 \
  --gpus-per-task="$gpus" --gpu-bind=none bash "$PWD/train.sh" "$@"
