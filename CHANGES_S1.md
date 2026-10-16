# Stage-1 pure-noise changes

Based on official commit `107cf0459e561a3e53640ef88750ed2aaff4957b`.
The extension is opt-in through an `ablation:` block. Official configurations
retain their original execution path. No FD loss or extra training seed is added.

| File | Change and reason |
|---|---|
| `pretrain.py` | Enable explicitly validated Stage-1 noise ablations; adapt accumulation, FLOP accounting, runtime checks, and smoke/resume handling without changing official recipes. Write successful startup audits only after resume validation; write rejected startup receipts to a new sibling directory. For ablations, retain the launcher's checked socket directory after private compiler-cache setup. |
| `utils/s1_ablation.py` | Keep ablation validation and perceptual checks together. Reject invalid noise, thresholds, epochs, and global batch before training. |
| `utils/distributed.py` | Allow the ablation launcher to select its supported single-node world size; preserve eight-rank defaults for official workflows. |
| `configs/pretrain/b80_perceptual_s100.yaml` | Main 80-epoch pure-noise recipe; threshold 1.0 retains both perceptual losses. |
| `configs/pretrain/b160_perceptual_s100.yaml` | Optional 160-epoch pure-noise recipe. |
| `configs/pretrain/b80_perceptual_s080.yaml` | Explicit 80-epoch control with the original 0.8 threshold. |
| `setup.sh` | One-command official environment creation, login, storage checks, verified downloads, and recipe validation. Stage unchanged setup files under storage to keep Conda's generated files there. Keep the install-time `STORAGE/tmp` out of the generated training environment. |
| `train.sh` | Detect GPUs, keep batch 1,024 through accumulation, materialize a runtime configuration under storage, and resume automatically. Override inherited `TMPDIR` with `/tmp` or `S1_TMPDIR`, then bind a real multiprocessing-length AF_UNIX socket before launch. Print the final `TMPDIR` in dry runs. |
| `train.slurm` | Generic single-node wrapper; translate USR1 into an update-boundary checkpoint request. Site and resource settings remain command-line options. |
| `README.md` | Add a short pure-noise quick start above the complete original README. |
| `STAGE1_PURE_NOISE.md` | Explain prerequisites, the exact recipe differences, expected behavior, monitoring, and continuation with official post-training. |
| `.gitignore` | Exclude the default storage directory and its generated files. |
| `tests/test_s1_ablation.py` | Lightweight CPU regression checks for official behavior, noise/mask correctness, sampler offsets, launcher decisions, and preservation of existing output when startup or resume is rejected. |
| `tests/test_s1_tmpdir.py` | Reproduce overlong AF_UNIX paths; check legacy environments, setup output, early rejection, and real two-worker CPU DataLoader batches through the actual launcher and runtime-cache setup. Keep the official non-ablation cache path unchanged. |

`NOISE_LEVEL=0.8 EPOCHS=160` reuses the explicit control recipe and overrides its
epoch count in the generated runtime configuration. Different GPU counts keep
the global batch fixed but change random streams. GPU speed and memory use
must be checked on the target machine.
