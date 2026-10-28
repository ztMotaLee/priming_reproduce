# Receiving agent instructions

Goal: reproduce 30,000 continuous one-step pixel T2I updates with the released caption pseudo-class big field and report every fixed readout, including negative results. The initialization is the flow-matching teacher `params/pt_net` from the MiniT2I-B-16-MF-jax checkpoint, not the MeanFlow student `params/net`.

Use branch `bigfield-posttrain` of `https://github.com/ztMotaLee/priming_reproduce`, directory `t2i_bigfield/`. Read README.md, docs/DETAILS.md, download pins and source/model license notices.

```bash
bash setup.sh
bash run.sh
bash status.sh
```

Have the mentor confirm the detected account, partition, node/GPU layout, CPU/memory/walltime, QOS and shared storage paths before submission. `run.sh` defaults to a concise PLAN_ONLY summary with value sources; use `bash run.sh --yes --json` for the complete plan with node counts only. QOS is automatically selected and checked with read-only `sbatch --test-only`; missing association data defaults to `none`, while all candidates lacking a finite MaxWall trigger `none` first and priority-ordered fallback marked **ambiguous; mentor must confirm**. Case-insensitive `none` or an explicitly empty QOS omits `--qos`; environment variables and `--site-env` override discovery. PLAN_ONLY creates no files or jobs. Use `bash run.sh --submit` only after confirmation; `--yes` accepts confirmed overrides and never implies submission. Existing run bindings must remain unchanged.

Setup installs independent training/GenEval/DPG environments and resumes SHA-verified downloads. Scorer installation failure is recorded pending and must not block training. All data reconstruction, DINO bank, smoke, training, sampling and scoring run on allocated compute resources. Never run heavy work on a login node.

The pipeline validates a production-topology 0-to-2 smoke and complete 2-to-3 restore, then starts production independently from official initialization. Its measured timing sets segments. If post-smoke job-side submission is blocked, rerun the printed run.sh continuation command. Repeated invocations must never duplicate jobs.

Rules:

- Do not modify configs, objectives, fields, random streams, sampling, evaluation or D5 constants. Keep w = 0.07498942093324559 and global batch independent of topology.
- Do not add training seeds, FD loss, new calibration, re-clustering, checkpoint selection or unrequested experiments.
- Preserve readouts at 1000/2000/4000/8000/12000/16000/24000/30000. GenEval: 553 × seeds 0–3, overall_category_balanced. DPG: 1065 × seeds 0–3, fullrow_corrected_overall.
- Training image encoders and primary evaluation image encoders must remain separate; disclose the existing CLIP text-partition role.
- Any asset, binding, runtime or science-check failure stops that stage. Report reason/logs; do not auto-retry or change the recipe. Resume only from a complete verified full-state checkpoint after inspecting the cause.
- Keep endpoints, features and required evaluation artifacts; do not create a bulk training-image dump or clean old outputs automatically.

At each readout, send the table from `bash status.sh`, including pending scores and D5 status. Distinguish fake-scheduler logic tests from actual allocated-node evidence; throughput and quality require measurement.

At completion run `bash status.sh --bundle` and send `results/results.md` plus `results/return_bundle.tar` under OUTPUT_ROOT. Include params/manifests/hashes when scoring is pending; keep optimizer checkpoints locally.
