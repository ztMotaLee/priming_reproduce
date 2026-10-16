# Stage-1 pure-noise pretraining

```bash
bash setup.sh
bash train.sh
```

Run setup once. Run training on a GPU machine. This trains **Priming-B on
ImageNet-256**, using PyTorch and a single training seed. It is class conditional.

## Before setup

You need Linux x86-64, Conda, `gcc`/`g++`, and an NVIDIA driver compatible with
CUDA 12.4. Use H100 80 GB or H200 GPUs. Allow **about 300 GB plus checkpoints**.
Accept the ImageNet terms at [the dataset page](https://huggingface.co/datasets/jaykim1102/priming).
Setup asks you to log in to Hugging Face; `HF_TOKEN` also works.

## Where files go

The environment, data, assets, and outputs go under `storage/` beside this file.
To use another disk, set `STORAGE` before **both** commands:
`export STORAGE=/path/to/large/disk/priming`. The scripts work from any directory.
Two small exceptions: PyTorch compiler caches use node-local `/tmp` (set
`PRIMING_LOCAL_TMP` to move them), and Conda may register the environment in
`~/.conda/environments.txt`.
Training uses `/tmp` for temporary DataLoader sockets, including with older setup files.
Set `S1_TMPDIR` to another short, writable directory if needed; startup checks a real socket.

Each GPU count gets its own output folder, `..._<N>gpu/`. To resume a run, use
the same number of GPUs.

## What changes

The model, optimizer, losses, precision, and EMA retain the official B recipe.
There is **no FD loss**. Only the following settings change:

| Setting | Official B recipe | This run |
|---|---|---|
| Input noise level `s` | 0.8 | **1.0: pure noise** |
| Perceptual threshold | 0.8 | **1.0**, keeping LPIPS and ConvNeXt active |
| Pretraining length | 160 epochs | **80 epochs**, matching paper §4.1 |
| GPU count / accumulation | 8 / 2 | Detected automatically; global batch stays **1,024** |

With a GPU count other than eight, the random streams and numerical trajectory
differ from the eight-GPU recipe. No configuration without an explicit
`ablation:` block is changed by this extension.

## Start, stop, and resume

`bash train.sh` detects 1, 2, 4, 8, or 16 visible GPUs and uses microbatch 64.
Repeat the same command to resume `checkpoints/latest.pth`; keep the same GPU
count, code, assets, and settings. Exit code 75 means a checkpointed stop.
If an output directory has no latest checkpoint, the script asks you to inspect it.

## Optional settings

Use `NOISE_LEVEL=0.8 bash train.sh` for the 80-epoch control, or
`EPOCHS=160 bash train.sh` for longer pretraining. `SMOKE_STEPS=50 bash train.sh`
stops at update 50; resume with `SMOKE_STEPS=60` to add ten updates.
`DRY_RUN=1 bash train.sh` prints the launch command and final `TMPDIR`, and checks
a temporary socket without starting training.

## Outputs and monitoring

The default run writes `storage/outputs/pretrain/b80_perceptual_s100_<N>gpu/`.
Read `metrics.jsonl` for loss, `noise_level`, and `perceptual_active_fraction`
(expected **1.0**). Open `visualization/*.png` for fixed diagnostic panels: the
denoising row retains the official 0.8-noise diagnostic, while the generation row uses pure noise.
Checkpoints are saved each epoch; every 20-epoch milestone and the final epoch remain.

## Expected behavior

At `s=1`, the input noise is independent of the target image. The reconstruction
optimum can use the class, but has no information about the particular target.
For squared error it is the class mean. Blurry, class-average-like outputs and
a higher loss plateau are therefore expected; this is a prediction, not a measured result here.

## What the paper reports

[Paper §4.1](https://arxiv.org/abs/2610.20331) compares 80-epoch pretraining
followed by 12,500 OT steps. The final EMA FID is **65.08 for `s=1.0` versus
1.65 for `s=0.8`**. Separately, Appendix E states that perceptual reconstruction stays fixed when sweeping noise levels.
These are paper results after post-training, not results produced by this code delivery.

## Time and compute

80 epochs is **100,080 updates**. Measured on one node with **4× H100 80 GB**
(60-update smoke run, including a resume): median **1.25 s/update**, mean
**2.0 s/update** (some steps wait for data), peak GPU memory **27 GB**.
That is about **35–56 hours on 4× H100** (140–225 GPU-hours). With 8 GPUs,
expect roughly half the wall time. Check `step_seconds` in `metrics.jsonl`
after the first few hundred updates for your machine.

## Slurm clusters

From the repository root, run
`sbatch --nodes=1 --gres=gpu:4 --time=24:00:00 --export=ALL --no-requeue train.slurm`,
adding your site's partition/account options. The wrapper requests USR1 ten minutes
before timeout, saves at an update boundary, and exits 75. Submit the same command again to resume.

## Continue with official post-training

Follow [the asset and external-path instructions](docs/setup.md#assets-outside-the-checkout).
Then run official `posttrain.py` with
`--init-checkpoint "$STORAGE/outputs/pretrain/b80_perceptual_s100_8gpu/checkpoints/latest.pth"`.
Official post-training still requires eight GPUs and runs **25,000** OT steps; this delivery does not change it to the paper ablation's 12,500.

## Setup checks

Setup verifies downloaded files against the official manifests and checks the
pinned environment and new recipe. Rerunning setup verifies existing assets.
`S1_SKIP_DATASET=1 bash setup.sh` skips the dataset and its login for setup testing;
run setup normally before training. CPU tests do not establish GPU memory fit or speed.
