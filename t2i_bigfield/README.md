# Big-field one-step T2I post-training

Run the fixed caption pseudo-class big-field experiment for 30,000 continuous updates from the official MiniT2I MF initialization, with every prescribed paired readout.

Requirements: Linux x86-64, Python 3.10 or Conda, Slurm, and H100 GPUs. Total GPUs must divide 64; the automatic default is the largest legal layout using at most 16 GPUs. Use shared storage visible on all nodes and network access to the pinned public model/data endpoints. Budget about 242 GB for downloads, about 124 GB for 30 retained checkpoints, plus environments, prepared images and evaluation workspace; their extra footprint and your quota must be checked locally.

```bash
git clone --branch bigfield-posttrain https://github.com/ztMotaLee/priming_reproduce.git
cd priming_reproduce/t2i_bigfield
bash setup.sh
bash run.sh
bash status.sh
```

`setup.sh` creates environments and resumes SHA-verified downloads on the login node. It attempts GenEval and DPG setup separately; scorer failures leave training available and scoring pending. Image reconstruction and DINO extraction run as scheduled jobs.

`run.sh` detects site settings, including QOS, and shows values with their sources in a short **PLAN_ONLY** summary. Review its resources, jobs, checks and H100-hour estimate, then run **`bash run.sh --submit`**. Use `bash run.sh --yes --json` for the full machine-readable plan, with node counts only. QOS may be `none` (case-insensitive) or explicitly empty to omit `--qos`; if every candidate lacks a finite MaxWall, verified `none` uses the site default and priority-ordered fallback is **ambiguous; mentor must confirm**; environment variables and `--site-env` override discovery. Read-only `sbatch --test-only` probes may run in PLAN_ONLY; no files or jobs are created. `--yes` accepts confirmed values and never implies submission. Repeated runs advance existing state without duplicating submissions.

The pipeline prepares data/features, checks a production-topology 2-step smoke and full-state 2-to-3 restore, measures throughput, and sizes the production segments with 15% time margin. It schedules all eight fixed readouts and the final summary. If job-side submission is disallowed, rerun the printed `run.sh --submit` command after the smoke. Failed stages stop with their logs and recovery instruction.

Defaults are `bigfield_data/` and `bigfield_run/` beside the repository; set `DATA_ROOT` and `OUTPUT_ROOT` before setup/run to choose other shared paths. Keep these settings for status and future continuation. No training or sampling runs on the login node.

Return **`OUTPUT_ROOT/results/results.md`** and **`OUTPUT_ROOT/results/return_bundle.tar`**. The final allocated summary builds the archive; `bash status.sh --bundle` reuses it without reading large payloads on the login node. It contains the readout table, logs, dose summaries, configuration and hashes, with params for pending scoring (optional otherwise). Keep full optimizer checkpoints locally for continuation.

For an interim bundle too large for the login node, preview `bash run.sh --yes --bundle` and add `--submit` after confirmation to schedule packaging only. Small bundles (at most 20 MB including archive overhead) can be created directly by status.

Readouts: 1,000 / 2,000 / 4,000 / 8,000 / 12,000 / 16,000 / 24,000 / 30,000. GenEval uses 553 prompts × seeds 0–3 (`overall_category_balanced`); DPG uses 1,065 × seeds 0–3 (`fullrow_corrected_overall`). This is one training stream, without FD loss or score-based checkpoint selection.

Common questions:

- **Download/SHA failure?** Stop and report the exact file/log; rerun setup to resume an interrupted transfer. Never substitute a different asset or replace a corrupt final file automatically.
- **Scoring pending?** Training can continue. Preserve every readout's params, checkpoint manifest and hashes for centralized scoring; do not change scorer versions or use training encoders as primary evaluators.
- **No progress or a failed job?** Use `bash status.sh`, inspect the named logs, and follow its recovery command after resolving the cause. There is no automatic retry or recipe adjustment.
- **Different GPU count or short walltime?** Confirm a layout dividing 64; the allocated smoke determines whether it fits and sizes segments. Estimates do not prove real-site performance.

The complete original protocol, manual commands, licensing, reference scores, D5 definition, storage caveats and known limits are preserved in [docs/DETAILS.md](docs/DETAILS.md). Give your coding agent [MENTOR_PROMPT.md](MENTOR_PROMPT.md).
