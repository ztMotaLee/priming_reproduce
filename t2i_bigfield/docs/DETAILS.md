# Long-horizon big-field T2I post-training

Add a caption pseudo-class real-image **big field** to Priming T2I Stage-2 and test whether its advantage over the small-field baseline persists through a continuous 0–30,000 update run.

Start from `params/pt_net` of the official MiniT2I MF checkpoint: this is the frozen MiniT2I-B/16 flow-matching (diffusion) teacher that the MeanFlow distillation stores as a subtree; the MeanFlow student `params/net` is discarded at load and never trained or evaluated here. Start from it, create a new AdamW optimizer, and predict pixels in one forward pass from `2 * normal` at `tau=0`, without classifier-free guidance. This experiment uses one fixed training stream, no FD loss, and no checkpoint selection by evaluation score.

The field weight is **0.07498942093324559**, fixed throughout training. It comes from the original PASS calibration receipt, whose measured median was 42.886 seconds/update on four H100s. **Recalibrating the weight changes the experiment.**

## Contents and fixed recipe

| File | Purpose |
|---|---|
| `MENTOR_PROMPT.md` | Complete prompt for an agent on the receiving cluster |
| `configs/bigfield_tf.yaml` | Fixed scientific recipe and descriptive source mapping |
| `assets/` | Download pins and the actual curated-row/pseudo-class assignment |
| `setup.sh`, `prepare_data.py`, `data.py` | Environment, verified downloads, exact-row reconstruction, feature bank |
| `train.py`, `training.py`, `field.py`, `objective.py`, `encoders.py` | Generator training and frozen feature objectives |
| `bigfield_native/` | Namespaced model implementation, avoiding repository-root package collisions |
| `checkpoint.py`, `distributed.py` | Full-state continuation and fixed-global-batch process geometry |
| `train.slurm`, `submit_chain.sh`, `launcher.py` | Site-configured segmented multi-node launch |
| `eval_sample.py`, `score.py`, `evaluation/` | Paired sampling and pinned scorers |
| `tests/` | Lightweight CPU tests; run from this directory |

Every update uses **64 global caption groups**, each with **16 active generated images, 16 independent detached generated references, and one paired real image** at 512 pixels. The original SigLIP, ViT-MAE and Inception features, ten Sinkhorn iterations, detached full-batch normalization and AdamW settings are preserved.

The additional field uses the actual partition of 110,230 curated rows: CLIP ViT-L/14 text prefixes of 77 tokens, spherical k-means++ initialized with 1,024 clusters, dissolution of clusters smaller than 32, source-weighted grouped sampling, and 1,024 generated field rows/update. DINOv2-B features cover the whole image and four quadrants. Soft25 targets are stopped; field-noise stream offsets start at 17,000. The native and field gradients are added **before one common optimizer update**. The configuration and implementation retain the complete stream definitions.

Global batch, learning rate and random streams stay fixed as hardware changes. Each GPU owns complete caption groups; total GPUs must divide 64. With four GPUs/node, supported node counts are **1, 2, 4, 8 or 16**; with eight GPUs/node they are **1, 2, 4 or 8**. Three nodes are unsupported for either layout. These are valid batch geometries, not claims of tested hardware layouts. All ranks gather features in the original group order before evaluating the full objective. Replay gradients are summed globally; they already include the global objective's normalization.

The preregistered D5 dose diagnostic is preserved. Every update records global and parameter-group native/field/combined gradient norms, relative dose `rho_t`, gradient cosine and clipping. Every 50 updates, it compares native-only and combined AdamW updates from the same current optimizer state without changing the actual state. The fixed calibration reference is **rho0 = 0.23536276140843454**, bound to receipt SHA-256 `bf7f8dd156ed708e23df3ef30751ba95961c239435c5e8e11422954e4c42f3fa`. The original windows **901–1000** and **1901–2000** flag an unmaintained dose when their median is below `rho0 / 10` or above `10 * rho0`. This is a reported diagnostic; it does not stop training or change the fixed field weight.

## One-time setup and data

Clone branch `bigfield-posttrain` of [ztMotaLee/priming_reproduce](https://github.com/ztMotaLee/priming_reproduce), then work in `t2i_bigfield/`. Use shared storage visible at the same path on every node. Set `DATA_ROOT` to your own storage location. The generator and feature-bank preparation require an allocated GPU node; full data reconstruction and large downloads also belong on a suitable compute/data-transfer node.

```bash
git clone --branch bigfield-posttrain https://github.com/ztMotaLee/priming_reproduce.git
cd priming_reproduce/t2i_bigfield
export DATA_ROOT=/YOUR/SHARED/DATA
```

Read `assets/public_pins.json` before downloading: it records source repositories, exact revisions, expected bytes/SHA values and licensing notes. The three public sources are [BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) (Apache-2.0), [DALL-E 3 dataset](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset) (CC0-1.0), and [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image) (Apache-2.0). At the pinned revisions all three were public and ungated. Review their terms. The pinned `huggingface-hub==0.24.7` provides `huggingface-cli login` for tools using Hub authentication. This package's HTTP downloader uses public URLs and does not read saved Hub login tokens. If an endpoint starts requiring authentication, stop and ask for a supported download procedure; logging in alone does not authenticate this downloader. Do not place access credentials in scripts or this repository.

The official initialization (its `params/pt_net` teacher subtree only) is taken from [MiniT2I-B-16-MF-jax](https://huggingface.co/MiniT2I/MiniT2I-B-16-MF-jax) at revision `922ce3b5d66c28e895add8e9d93f576eea88584e`. T5, SigLIP, ViT-MAE, Inception, DINOv2 and CLIP weights have separate pins and terms. Checkpoints and encoders are downloaded as data, never committed to this repository.

The released `rows_clusters.npz` contains identifiers and hashes with the **actual cluster assignments**, without captions or images. Reconstruction must reproduce the published row order, image bytes, caption hashes, token lengths and cluster IDs. A missing source row or hash mismatch is a reason to stop; silently substituting a nearby sample changes the experiment.

Create a Python 3.10 environment and install the pinned packages. `setup.sh` uses a local virtual environment by default; alternatively activate your own Python 3.10 Conda environment and add `--current-environment`. Download flags are explicit. Transfers resume and the final bytes must match the pinned SHA.

```bash
PYTHON=python3.10 bash setup.sh --download-weights --download-data
source .venv/bin/activate
python prepare_data.py --root "$DATA_ROOT" check --kind weights
python prepare_data.py --root "$DATA_ROOT" check --kind data
python prepare_data.py --root "$DATA_ROOT" rebuild
python prepare_data.py --root "$DATA_ROOT" dino --batch 32
python prepare_data.py --root "$DATA_ROOT" check --kind prepared
python prepare_data.py --root "$DATA_ROOT" check --kind bank
```

Run reconstruction in a suitable CPU compute task and bank extraction in an allocated GPU task. Use the prepared/bank checks after building those outputs; `check --kind all` includes those outputs and is appropriate only after preparation. Inspect `--help` for individual commands. Build the real DINO feature bank on the receiving cluster. Its receipt binds the bank to the released population, model weights and feature preprocessing. The receipt's `statistics` summarize the whole bank; `probe_statistics` report the same eight fixed rows as the pins, including row-byte SHA, mean, RMS, minimum and maximum. Compare these probes with the corresponding reference rows, rather than comparing whole-bank statistics with eight-row diagnostics. A different numerical platform may require investigation before accepting a byte-SHA difference. Do not recompute the caption clusters for the paired experiment.

The optional `recluster --output NEW_PATH` command is a separate reconstruction experiment and may produce a different field. Run it on allocated GPU compute resources in a **separate Python 3.10 environment** with CUDA-enabled `torch==2.5.1` and matching `torchvision==0.20.1`, using a CUDA wheel supported by your cluster's driver. Install the pinned base requirements and `requirements-recluster.txt`, then obtain the CLIP payload with `download --kind clip`. The optional requirements do not replace Torch. The main `setup.sh` deliberately installs CPU Torch; do not run it in the optional CUDA Torch environment. Confirm `torch.cuda.is_available()` inside the allocation before reclustering. No CUDA wheel version is prescribed for an unknown receiving cluster.

The initialization/encoder downloads total about 6.33 GB; the raw data downloads total about 231.45 GB. Downloads were checked against public repository metadata, without fetching the large payloads during release preparation. The legacy Inception weight URL is a candidate: the required SHA remains pinned, and a URL resolving successfully does not prove payload identity. If that file fails SHA verification, stop and request the exact original weight; do not substitute another Inception conversion. The initialization model is labeled with custom (`other`) terms; review its model card before use or redistribution.

## Smoke test before the long run

Run the CPU suite on the login node only if its local policy permits these short tests:

```bash
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m unittest discover -s tests -v
```

Then clear `JAX_PLATFORMS` and run a **two-update smoke test on allocated H100 compute nodes**, using a new `OUTPUT_DIR`, with the topology intended for the main run. Fill the site variables in the next section first, then use:

```bash
unset JAX_PLATFORMS
export OUTPUT_DIR=/YOUR/SHARED/RUNS/new_smoke
mkdir -p "$(dirname "$OUTPUT_DIR")"
START_STEP=0 END_STEP=2 bash submit_chain.sh
START_STEP=0 END_STEP=2 bash submit_chain.sh --submit
# After successful completion, verify and continue its full state:
START_STEP=2 END_STEP=3 bash submit_chain.sh --submit
```

Inspect finite losses and gradient norms, verify all ranks completed, and inspect optimizer/RNG/cursor continuity. After the smoke continuation, save `python -m pip freeze --all`, `python -VV`, JAX/jaxlib versions, allocated device/driver information and the exact launch variables in new environment-record files under the smoke run directory. Direct dependencies are pinned, but transitive CUDA/environment resolution remains specific to the receiving cluster. Use a separate new directory for the scientific run, starting again from the official checkpoint at step zero. No numerical first-update reference is asserted for an untested GPU environment.

## Multi-node 0–30,000 launch

Fill every site-dependent value. `MEM_PER_NODE` is Slurm memory per node; `CPUS_PER_TASK` is per process, with one process per node. `GPU_TYPE` is optional: use the site's H100 resource spelling if required. The runtime verifies allocated H100 devices. Choose the coordinator port if the default 29500 is unsuitable for your network.

```bash
unset JAX_PLATFORMS
export NODES=YOUR_NODE_COUNT GPUS_PER_NODE=YOUR_GPU_COUNT
export PARTITION=YOUR_PARTITION ACCOUNT=YOUR_ACCOUNT
export CPUS_PER_TASK=YOUR_CPU_COUNT MEM_PER_NODE=YOUR_MEMORY
export TIME=YOUR_HH:MM:SS
export PYTHON=python
export OUTPUT_DIR=/YOUR/SHARED/RUNS/bigfield_0_to_30000
export START_STEP=0 END_STEP=30000 STEPS_PER_SEGMENT=1000
mkdir -p "$(dirname "$OUTPUT_DIR")"
bash submit_chain.sh                 # print the full plan; creates no jobs or run directories
bash submit_chain.sh --submit        # run only after smoke-test acceptance on your cluster
```

Create only the output parent before submission; the new `OUTPUT_DIR` itself must not already exist.

The chain uses 30 segments by default. Segment boundaries also include every required readout. The first job has no dependency; subsequent jobs use `afterok` on newly submitted preceding jobs. Submission and runtime failures stop progress; there is no automatic retry or recipe adjustment. Keep the printed submission responses and logs. If submission stops partway through a chain, inspect those job IDs before deciding how to continue; do not blindly submit another full chain.

Set segment length and walltime using your measured throughput with about 15% time margin plus initialization/checkpoint overhead. A segment exits successfully only after reaching its requested endpoint. A timeout can lose work after the latest completed checkpoint. Full-state checkpoints are written at least every 1,000 updates and at segment endpoints. Preserve checkpoints at **1,000, 2,000, 4,000, 8,000, 12,000, 16,000, 24,000 and 30,000** permanently. This implementation retains all completed checkpoints and performs no automatic cleanup.

## Continuation and checkpoint transfer

A resumable checkpoint contains parameters, AdamW moments/counters, step, RNG state, data cursor, scientific/data binding and payload SHA values. `latest.json` points to a completed checkpoint. Continuation rejects a changed recipe, data binding, missing optimizer state or inconsistent step/cursor. Do not initialize a new optimizer when resuming.

After inspecting an interrupted run and its scheduler state, set `START_STEP` to its latest completed checkpoint step and keep the same `OUTPUT_DIR`; preview a new chain before submission. Every launch writes a new segment status report and preserves previous reports, including failures. No continuation is submitted automatically. A successful CPU round-trip only verifies toy-state behavior; the allocated-node smoke continuation is still required.

Each checkpoint also has a standalone `params.msgpack`. If the scorer environments are difficult to reproduce, return the readout checkpoint's `params.msgpack` with `checkpoint.json`, its SHA-256 and the run configuration/data binding. AdamW is unnecessary for evaluation. Keep the full checkpoint on the training cluster for continuation.

## Paired evaluation

Primary metrics are **GenEval: 553 prompts × seeds 0–3, `overall_category_balanced`**, and **DPG: 1,065 prompts × seeds 0–3, `fullrow_corrected_overall`**. Prompt ordering, per-prompt seed reuse, internal key split, one-step sampler and pixel quantization are preserved for prompt-paired comparison to the original T arm. Do not replace these metrics with raw image averages or an uncorrected DPG aggregation.

Prepare the exact prompt/scorer assets with `evaluation/prepare.py`, following `evaluation/README.md`. Use the separate pinned scorer environments; do not merge incompatible training and scorer dependencies. Sampling uses one visible GPU and must run inside an allocated compute-node task.

```bash
python eval_sample.py --benchmark geneval --checkpoint "$CHECKPOINT" \
  --assets "$EVAL_ASSETS" --weights-root "$DATA_ROOT/weights" \
  --native-root "$PWD/bigfield_native" --output "$NEW_GENEVAL_OUTPUT" --plan
# Remove --plan inside a one-GPU allocation; repeat with --benchmark dpg and a new output path.
python score.py --help
```

The training **image-feature** encoders (Inception, SigLIP, ViT-MAE, DINOv2) differ from the GenEval detector/CLIP image tower and DPG mPLUG. The partition was constructed beforehand with CLIP ViT-L/14's text tower; GenEval also uses CLIP, so this is a disclosed text-partition role and does not establish complete independence of all auxiliary pretrained weights. Inception-based FID, if computed elsewhere, is descriptive because Inception is used in training; it is not an independent primary result. Do not add an FD loss.

| Arm / update | GenEval | DPG |
|---|---:|---:|
| T / 0 | 0.1345 | 26.90 |
| T / 1,000 | 0.7937 | 81.54 |
| T / 2,000 | 0.8012 | 82.06 |
| T / 4,000 | 0.8186 | 82.06 |
| Big field / 1,000 | 0.7951 | 81.80 |
| Big field / 2,000 and 4,000 | Pending | Pending |
| Priming paper / 4,000 | 0.818 | 82.16 |

The first five rows are previously measured reference runs. The final row is a paper-reported comparison, not a measurement by this package. These values are sanity references, not acceptance thresholds or guarantees.

## Compute, storage and known limits

Four-H100 calibration measured 42.886 seconds/update median, 44.053 p95 and about 24.7 GB peak device memory. At that rate 30,000 updates require about **1,430 H100-hours**, before evaluation and setup. The following are **ideal strong-scaling estimates**, with a 15% time margin; multi-node performance has not been measured for this release.

| Total H100s | Estimated training walltime with margin |
|---:|---:|
| 4 | 411 hours |
| 8 | 205.5 hours |
| 16 | 102.7 hours |
| 32 | 51.4 hours |
| 64 | 25.7 hours |

Communication, feature gathering, initialization, checkpoints and evaluation can add substantial overhead. Use the smoke test to measure your cluster and size segments. Each completed checkpoint stores about **3.1 GB of full training state plus 1.03 GB of standalone parameters**. Thirty retained checkpoints therefore need roughly **124 GB**, plus temporary save space, prepared images/caches and evaluation panels. Raw downloads add about 231.45 GB, initialization/encoder weights about 6.33 GB, and the DINO bank about 1.6 GB; these are additional to checkpoint storage. Inspect pinned byte totals and available storage before preparation. Store checkpoints/features and the required evaluation panels, not a training image dump.

Prepared images are packed into **111 tar shards** (1,000 images/shard for 110,230 rows), plus the index and manifest; their total byte size has not been measured for this release. One complete readout creates **2,212 GenEval sample PNGs** and **4,260 DPG sample PNGs plus 1,065 DPG grids**, before scorer outputs and metadata. Eight retained panels therefore contain **60,296 image files**. Panel bytes, scorer temporary space and scorer file counts are unknown until the receiving cluster's smoke/evaluation setup is measured; record free bytes and inodes before preparation and each evaluation.

The CPU suite establishes toy-input agreement with the source functions. GPU execution, multi-node collectives, throughput and generated quality require verification on the receiving cluster. Model/data terms and third-party source redistribution terms remain applicable; inspect the source notices before publishing a derivative. Report mismatches and failures without changing constants or substituting assets.

## Return at every readout

Return one row per fixed readout, the complete training/evaluation logs, configuration and data/weight hashes, checkpoint manifest and SHA values. Include failures and pending scores. Optionally transfer parameter exports for centralized scoring. Include `dose_summary.json` and the `training.jsonl` diagnostics: the dose summary carries both original windows, their median values, thresholds, calibration binding and `PENDING`/`INCOMPLETE`/`MAINTAINED`/`UNMAINTAINED` status. A `MAINTAINED` result requires both complete windows; at 1,000 the second window is still pending. Readout snapshots are stored as `readouts/step_001000.dose.json`, `readouts/step_002000.dose.json` and the corresponding later steps; return these with their readouts.

| Step | GenEval category-balanced | DPG corrected | Checkpoint SHA | Median s/update | Peak GB | D5 dose status / median rho | Status / notes |
|---:|---:|---:|---|---:|---:|---|---|
| 1000 | | | | | | | |
| 2000 | | | | | | | |
| 4000 | | | | | | | |
| 8000 | | | | | | | |
| 12000 | | | | | | | |
| 16000 | | | | | | | |
| 24000 | | | | | | | |
| 30000 | | | | | | | |


## Automated entrypoints (second release interface)

The complete original README above is retained verbatim for recipe, provenance,
manual operation and protocol reference. For the normal automated path, the short
README and this section supersede the old manual setup/submission instructions.
The `submit_chain.sh` examples in the preserved original text are historical
manual commands. Use `run.sh` for the submission journal and recovery guards.
The scientific recipe, sampler, metrics and diagnostic windows are unchanged.

### Setup

`bash setup.sh` now installs the training environment, resumes all mandatory
weight/raw-data downloads, verifies their pinned SHA/size, prepares evaluation
prompts and scorer weights, and attempts two separate scorer environments. It
never rebuilds images, extracts the DINO bank, loads a model or requests compute.
Python 3.10 is discovered from `PYTHON`, `python3.10`, or an existing Python 3.10
bootstrap; otherwise an available Conda creates it. The default training
environment is `.venv/`. `--current-environment` uses an activated Python 3.10
without replacing it; `JAX_BACKEND=cpu` selects the CPU-only dependency variant
for small tests. Production uses the default CUDA 12 JAX variant and CPU Torch.

Defaults: `DATA_ROOT` is `bigfield_data/` beside the repository, `OUTPUT_ROOT` is
`bigfield_run/` beside it, and `EVAL_ASSETS=$DATA_ROOT/evaluation`. Every node must
see these paths. `VENV_DIR`, `GENEVAL_ENV`, `DPG_ENV`, and `PYTHON` can override
environment paths. Pin and source/model license notices remain in `assets/`,
`evaluation/`, and the original sections above. Setup never creates OUTPUT_ROOT;
run reservation occurs only after the scheduler, binding and storage gates.

`bash setup.sh --plan-only` (alias `--dry-run`) prints the plan without any write,
installation, download or scheduler call. The old download flags remain accepted
for command compatibility; downloads are now enabled by default. No automatic
network retry occurs. Transfers retain partial files and resume on a deliberate
rerun; corrupt final files stop without replacement. All dependency installation
uses binary wheels except for the pinned, lightweight pure-Python
`ml-collections==0.1.1` source distribution. No large native extension is built.
If a pinned
scorer dependency has no compatible wheel, its installation is recorded pending;
do not silently upgrade it or compile a large extension on a login node.

The canonical setup state is `$DATA_ROOT/setup/status.json`. It records
`training_python`, `training.status`, `assets_ready`, `eval_assets`, independent
`evaluation_assets.{geneval,dpg}` status and `sampler_ready`, and independent
`scoring.{geneval,dpg}` status, interpreter and error. Unique setup logs preserve
each invocation under `$DATA_ROOT/setup/logs/`. Training setup/download failure
stops setup. Scorer-environment or scorer-model failure leaves training ready
and scoring pending; prompt assets are prepared first so sampling can continue.
Missing or mismatched prompt assets must still stop that benchmark's sampling.

The setup storage gate records all values before rejecting insufficient space
or inodes. Its default additional environment/temporary reserve is an estimate
of 32 GiB (`SETUP_RESERVE_GB`), with a 100,000 free-inode floor
(`SETUP_MIN_FREE_INODES`). Filesystem free space does not establish account quota;
check the quota and unknown prepared/evaluation footprint separately. Training
downloads and its environment are checked independently of optional scoring.
Each actual destination filesystem, including external environment paths, has
its byte/inode observations recorded. Insufficient optional evaluation storage
leaves training ready and scoring pending.

### Run, continuation and results

`bash run.sh` probes the site and prints a short PLAN_ONLY summary: site values
and sources, first-wave jobs/resources/dependencies, H100-hours and storage,
inode and queue checks. `bash run.sh --yes --json` prints the complete JSON plan;
node inventory is summarized by partition and GPUs/node without node names.
Approve the actual site values before `bash run.sh --submit`, which also prints
the summary and a short job-ID/held/released table. `--yes` accepts confirmed
overrides and does not imply `--submit`. PLAN_ONLY creates no files or jobs;
`sbatch --test-only` is a read-only scheduling query. Site settings persist as
`OUTPUT_ROOT/state/site.env` only when all submission gates pass. The state machine schedules data preparation,
DINO extraction, a production-topology 0-to-2 smoke and full-state 2-to-3 restore,
then segmented 0-to-30,000 production, all fixed readouts and summary. Production
restarts from the official initialization, independently of the smoke state.

QOS discovery uses account/partition associations and QOS MaxWall limits. For
CPU (01:05:00), GPU short (at most one hour) and GPU long (MAX_WALLTIME), it tries
the shortest sufficient limit first, breaking ties by descending Priority then
name, and verifies each candidate with
`sbatch --test-only` for production, applicable single-GPU bank/eval and light
CPU requests in that partition. Short-only QOS (MaxWall at most one hour) do
not count as long candidates. A sole long QOS with a shorter limit lowers
MAX_WALLTIME only after its resource probes pass, and reports the change.
Missing association QOS data or unavailable sacctmgr defaults to `none`, using
the site's default QOS; rejected candidates fall back to a test-only probe of
`none`. If every candidate's MaxWall is missing, empty or UNLIMITED, `none` is
probed first; only rejection permits probes by descending Priority then ascending name,
marked **ambiguous; mentor must confirm**. An unverified fallback stops for
confirmation. Sources distinguish `detected+test-only verified`,
`site default (QOS limits not published)`, `default none (no association QOS)`,
`user override` and `ambiguous-needs-confirmation`. `CPU_QOS`,
`GPU_QOS_SHORT` and `GPU_QOS_LONG` accept case-insensitive `none` or an explicitly
empty value to omit `--qos`. Environment variables and `--site-env` override
discovery; an existing run keeps its exact bound values. Held-job readback
checks explicit QOS before release and records the actual QOS for default requests.
Limits round up to whole minutes before segment sizing, QoS selection and GPU
hour accounting; the rounded request must remain within the partition limit.
For sites registering untyped GPU resources, explicitly set `GPU_TYPE=generic`
after confirming H100 hardware and the partition resources. Typed H100 discovery
remains the automatic path; the allocated worker verifies actual H100 devices
in both modes.

Segment sizing uses smoke-measured seconds/update with 15% margin and observed
startup/save overhead, capped by the confirmed partition walltime. The smoke must
validate finite values, all ranks, optimizer counters, RNG and cursor continuity.
A lightweight continuation job attempts post-smoke scheduling. If policy blocks
job-side submission, status prints the exact `bash run.sh --submit` continuation
command. Rerunning advances verified stages; recorded submissions are never
blindly repeated. Any failed stage stops with its reason, logs and recovery
instruction. Repair an external cause and inspect scheduler/checkpoint state
before authorizing continuation; never change the recipe to make it pass.

For an inspected, terminal failed training segment, preview an explicit recovery
with `bash run.sh --yes --recover FAILED_KEY`; add `--submit` only after the
mentor approves that continuation. It uses the latest complete verified
checkpoint and fresh generation journals while retaining the failed attempt's
evidence. Other failed stages remain stopped for diagnosis; no unattended retry
or restart from a new optimizer is allowed.

`bash status.sh` displays stages, current step, measured speed/ETA, all readout
scores and D5 status. The allocated summary job prepares the final results and
`OUTPUT_ROOT/results/return_bundle.tar`; `bash status.sh --bundle` reuses that
bundle when its state token, archive identity and manifest binding still agree,
without hashing or copying its large archive/parameter payloads. Small interim
bundles with a conservative archive-size bound at most 20 MB may be built on the
login node. The numeric size/storage gate is saved before any candidate payload
is read or a large login-node build is refused. For larger interim bundles,
preview `bash run.sh --yes --bundle` and add `--submit` only after confirmation;
this schedules an allocated packaging job without advancing training. Return
that archive plus `results/results.md`. The
archive contains the readout table, logs, dose summaries, configuration, source
and asset/checkpoint hashes, and readout parameter exports when scoring is
pending (optional otherwise). Full AdamW state remains on the training cluster.
No bulk training-image dump is created, and no automatic old-output cleanup runs.

Fake-scheduler CPU tests establish state-machine logic only. They do not establish
real-site scheduling compatibility, multi-node execution, numerical GPU behavior,
throughput, memory fit or generated quality; those require the allocated smoke
and actual production/evaluation receipts.
