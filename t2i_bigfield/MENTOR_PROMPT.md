# Prompt for the receiving cluster's coding agent

You are implementing and operating a fixed scientific experiment for my research collaborator. Work in the public repository `https://github.com/ztMotaLee/priming_reproduce`, branch `bigfield-posttrain`, directory `t2i_bigfield/`. Read its README, configuration, source notices and download pins before acting.

**Goal:** start from the official MiniT2I MF checkpoint at update zero and run 30,000 continuous Priming T2I Stage-2 updates with the released caption pseudo-class big field. Test whether the big field retains an advantage over the small-field T baseline during long training. Produce every prescribed readout, including negative results. Do not redesign the experiment.

## Ask me to fill these site values first

| Required value | My value |
|---|---|
| Shared `DATA_ROOT` and `OUTPUT_DIR` parent | |
| Slurm `PARTITION` and `ACCOUNT` | |
| `NODES` and `GPUS_PER_NODE` of H100s | |
| `CPUS_PER_TASK` and `MEM_PER_NODE` | |
| `TIME`, maximum segment walltime | |
| Optional site `GPU_TYPE` spelling / network interface / coordinator port | |
| Conda environment location and download node | |

Total GPUs must divide 64. With four GPUs/node, valid node counts are 1, 2, 4, 8 or 16; with eight GPUs/node they are 1, 2, 4 or 8. Three nodes are unsupported for either layout. These are valid batch geometries; multi-node execution still needs the smoke test. Do not invent a partition, account or resource entitlement. Discover local cluster instructions and obey them. Keep source, data, logs and checkpoints on appropriate storage. Never run GPU work, training, large downloads, full-data reconstruction or heavy processing on a login node. An allocation alone does not move a login shell: use an allocated compute-node task. Verify node identity and visible H100s there.

## Rules that you must preserve

1. Use the released `configs/bigfield_tf.yaml`. Start at official `params/pt_net`, a new AdamW at step zero, one x0 prediction from `2*normal`, `tau=0`, no CFG.
2. Keep global 64 caption groups × 16 active + 16 detached references + one paired real image per group, independent of GPU count. Preserve source sampling, every RNG stream and default production precision. Do not scale the learning rate with world size or average an already normalized replay gradient again.
3. Keep **w = 0.07498942093324559** throughout. This is an existing measured calibration result. Do not recalibrate, tune it, or change field size, encoders, Sinkhorn settings, temperatures, preprocessing or objective.
4. Use the released actual curated-row/cluster assignment. Recompute only the DINO real-feature bank from the exact rows/weights. The optional cluster rebuild is not a replacement for the paired experiment.
5. One fixed training random stream only: no multi-seed runs, no FD loss. The four prescribed evaluation seeds are the evaluation panel, not extra training seeds. The training image-feature encoders (Inception, SigLIP, ViT-MAE, DINOv2) differ from the GenEval detector/CLIP image tower and DPG mPLUG. The precomputed partition uses CLIP ViT-L/14's text tower, and GenEval also uses CLIP; disclose this text-partition role, without claiming complete independence of all auxiliary pretrained weights. Any Inception FID is descriptive only.
6. Save complete resumable state at least every 1,000 updates and segment endpoints. Keep parameters, AdamW moments/counters, step, RNG and data cursor. Never restart an optimizer when continuing.
7. Permanently retain updates **1000, 2000, 4000, 8000, 12000, 16000, 24000, 30000**. Evaluate all these checkpoints; never select a best checkpoint or suppress a readout.
8. Verify required data/weight SHA values and configuration bindings. On missing assets, nonfinite values, checkpoint disagreement or runtime failure, stop and report the exact command, log and latest complete state. Do not silently substitute assets, change the recipe, retry indefinitely or submit duplicate chains.
9. Source/data/model licensing requirements still apply. Do not copy private paths, credentials or access tokens into this repository or reports. Ask before publishing new artifacts or transferring files beyond the agreed collaborators.
10. Preserve D5: record every-update global/group gradient norms, `rho_t`, cosine and clipping, plus the every-50-update native/combined AdamW counterfactual from the same state without mutation. Keep fixed `rho0 = 0.23536276140843454` and its receipt SHA binding. Report windows 901–1000 and 1901–2000 with the original `rho0 / 10` and `10 * rho0` thresholds. A dose flag is a diagnostic; do not automatically stop, recalibrate or retune the run.

## Execute this checklist

1. Inspect the branch and local instructions. Report the repository revision and whether local changes exist. Read the README and source notices; report any unresolved third-party redistribution terms before publication.
2. Fill the site table with me. Inspect Slurm capabilities, GPU types, network and storage using lightweight read-only commands. Allocate resources only as needed. Do not assume the sending cluster's setup.
3. Set up the pinned training environment using `setup.sh`. Review dataset/model terms. The pinned Hub version uses `huggingface-cli login` for Hub-aware tools, but this package's public-URL downloader does not read saved login tokens; stop and ask if an endpoint requires authentication. Download using the pinned revisions, resumable transfers and SHA checks. Do heavy preparation in suitable compute/data-transfer tasks. Budget about 124 GB for 30 checkpoint directories (3.1 GB full state + 1.03 GB standalone parameters each), plus temporary save space, prepared data/caches and evaluation outputs; raw downloads, weights and the DINO bank require additional storage.
4. Reconstruct exactly 110,230 curated rows in the published order and validate the released partition. Build the DINO five-branch feature bank, record its SHA, shape and statistics, compare against the reference in the pins, and investigate discrepancies. Do not replace the published clusters with a fresh k-means result.
5. Run the small CPU tests from `t2i_bigfield/`, with `JAX_PLATFORMS=cpu`, one BLAS/OpenMP thread and no bytecode caches. These tests do not establish multi-node or GPU correctness.
6. Set a new smoke `OUTPUT_DIR`, create only its parent with `mkdir -p "$(dirname "$OUTPUT_DIR")"`, and use allocated H100 tasks at the intended production topology. The new run directory itself must not already exist. Run two updates, inspect finite loss and gradient logs and all ranks, restore the complete checkpoint, then continue step 2→3. Verify optimizer counters, RNG and cursor continuity. Record measured seconds/update, peak memory and startup/save overhead. Save new environment-record files under the smoke run root containing `python -m pip freeze --all`, `python -VV`, JAX/jaxlib versions, allocated device/driver details and launch variables. Transitive CUDA/environment packages are not fully locked for an unknown cluster. Record available bytes/inodes; prepared bytes and scorer temporary space must be measured locally. Do not reuse this updated smoke checkpoint for the scientific run.
7. Choose segment length and time limit from measured throughput × approximately 1.15 plus initialization/save overhead. Set a new production `OUTPUT_DIR` and create only its parent with `mkdir -p "$(dirname "$OUTPUT_DIR")"`. Preview `bash submit_chain.sh` with `START_STEP=0 END_STEP=30000`. Present job resources, endpoints, dependencies and estimated GPU-hours. Submit the reviewed production chain using `--submit` within the authorization I give you. Start from the official initialization in the new output directory.
8. Monitor progress without changing scientific settings. Keep logs and complete checkpoints. If a job fails, inspect the state and report; dependents must not continue from an incomplete segment. Before submitting a replacement chain, inspect any already submitted jobs to avoid duplicates. Resume only from a complete, verified state.
9. Prepare pinned evaluation assets/environments according to `evaluation/README.md`. At each readout, sample exactly GenEval 553 prompts × seeds 0..3 and DPG 1,065 prompts × seeds 0..3 with `eval_sample.py`. Preserve prompt order and sampling conventions. Score `overall_category_balanced` and `fullrow_corrected_overall` with `score.py`. Report pending scoring explicitly.
10. If scorer installation is impractical, preserve the checkpoint and send back its standalone `params.msgpack`, `checkpoint.json`, SHA values, configuration/data binding and logs for centralized evaluation. Do not change metrics or use an approximate replacement scorer.
11. Return the table below at every readout, the complete logs and hashes, and a final accounting of compute/time/failures. Include `dose_summary.json`, `training.jsonl` and the fixed readout snapshots such as `readouts/step_001000.dose.json` and `readouts/step_002000.dose.json`. Report D5 status and original-window medians, keeping pending/incomplete diagnostics explicit. Both windows must be complete before reporting `MAINTAINED`; the second is pending at step 1,000. Distinguish measured facts from estimates. Do not report an unrun check as passed.

| Step | GenEval category-balanced (0–1) | DPG corrected (0–100) | Checkpoint SHA-256 | Median s/update | Peak GB | D5 dose status / median rho | Status / notes |
|---:|---:|---:|---|---:|---:|---|---|
| 1000 | | | | | | | |
| 2000 | | | | | | | |
| 4000 | | | | | | | |
| 8000 | | | | | | | |
| 12000 | | | | | | | |
| 16000 | | | | | | | |
| 24000 | | | | | | | |
| 30000 | | | | | | | |

The original big-field run at 1,000 measured GenEval 0.7951 / DPG 81.80. Its 2,000/4,000 scores are pending. The original T arm measured 0.1345/26.90 at zero, 0.7937/81.54 at 1,000, 0.8012/82.06 at 2,000 and 0.8186/82.06 at 4,000. These are sanity references, not promised outcomes.

Stop and ask me when required site values or authorization are missing; terms prevent use/publication; exact assets cannot be recovered; bank checks disagree; the intended topology fails; a scientific/configuration change seems necessary; or a failed/partial chain needs a new submission decision. Continue independent preparation that does not depend on the missing answer.
