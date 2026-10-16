# Stage-1 pure-noise changes

Based on official commit `107cf0459e561a3e53640ef88750ed2aaff4957b`.
The extension is opt-in through an `ablation:` block. Official configurations
retain their original execution path. No FD loss or extra training seed is added.

| File | Change and reason |
|---|---|
| `pretrain.py` | Enable explicitly validated Stage-1 noise ablations; adapt microbatch, accumulation, FLOP accounting, runtime CPU resources, and runtime checks, and smoke/resume handling without changing official recipes. Write successful startup audits only after resume validation; write rejected startup receipts to a new sibling directory. For ablations, retain the launcher's checked socket directory after private compiler-cache setup. |
| `utils/s1_ablation.py` | Validate eight world sizes through 128 ranks, their unique microbatch/accumulation pairs, and batch 1,024. Keep precision and dose failures auditable before raising. Cap runtime loader workers and rank threads to the CPU allocation without changing the saved recipe. |
| `utils/distributed.py` | Allow the ablation launcher to select its supported world sizes; preserve eight-rank defaults for official workflows. |
| `configs/pretrain/b80_perceptual_s100.yaml` | Main 80-epoch pure-noise recipe; threshold 1.0 retains both perceptual losses. |
| `configs/pretrain/b160_perceptual_s100.yaml` | Optional 160-epoch pure-noise recipe. |
| `configs/pretrain/b80_perceptual_s080.yaml` | Explicit 80-epoch control with the original 0.8 threshold. |
| `setup.sh` | One-command official environment creation, login, storage checks, verified downloads, and recipe validation. Stage unchanged setup files under storage to keep Conda's generated files there. Keep the install-time `STORAGE/tmp` out of the generated training environment. |
| `train.sh` | Detect GPUs, keep batch 1,024 through accumulation, materialize a runtime configuration under storage, and resume automatically. Override inherited `TMPDIR` with `/tmp` or `S1_TMPDIR`, then bind a real multiprocessing-length AF_UNIX socket before launch. Print the final `TMPDIR` in dry runs. |
| `train.slurm` | Generic Slurm wrapper: prepare the run once, validate every node, start one PyTorch launcher per node, and translate USR1 into a shared checkpoint request. Site and resource settings remain command-line options. |
| `utils/s1_launch.py` | Keep Slurm topology, microbatch selection, shared config/resume decisions, per-node preflight, and dry-run commands in one small launcher. `S1_CONFIG_DIR` can move generated configs; the default remains storage/configs. |
| `utils/pretrain_checkpoint.py` | Use a reusable CPU Gloo group for ablation B checkpoint object gathering; preserve the payload and strict restore checks, and explicitly reject changed world size or malformed counters. Official non-ablation transport is unchanged. |
| `tests/test_s1_launch.py` | Check simulated 16×8 launch commands, homogeneous node resources, config/resume preparation, and scheduler-free dry runs. |
| `tests/test_s1_multinode.py` | Check all fixed-batch layouts, CPU budgeting, per-sample real reconstruction/perceptual code with tiny fake encoders, and four-rank Gloo gradient equivalence. |
| `tests/test_s1_checkpoint_multinode.py` | Check tiny four-rank checkpoint continuation, CPU transport reuse, 128-rank shard ownership, topology rejection, and step counter types. |
| `README.md` | Add a short pure-noise quick start above the complete original README. |
| `STAGE1_PURE_NOISE.md` | Explain prerequisites, the exact recipe differences, expected behavior, monitoring, and continuation with official post-training. |
| `.gitignore` | Exclude the default storage directory and its generated files. |
| `tests/test_s1_ablation.py` | Lightweight CPU regression checks for official behavior, noise/mask correctness, sampler offsets, launcher decisions, and preservation of existing output when startup or resume is rejected. |
| `tests/test_s1_tmpdir.py` | Reproduce overlong AF_UNIX paths; check legacy environments, setup output, early rejection, and real two-worker CPU DataLoader batches through the actual launcher and runtime-cache setup. Keep the official non-ablation cache path unchanged. |

`NOISE_LEVEL=0.8 EPOCHS=160` reuses the explicit control recipe and overrides its
epoch count in the generated runtime configuration. Different GPU counts keep
the global batch fixed but change random streams. Above 16 GPUs the microbatch
is reduced, retaining the same per-sample objective and EMA batch 1,024. GPU speed and memory use
must be checked on the target machine.
