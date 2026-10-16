# Priming Stage 1 for MiniT2I-JAX

Reconstruction pretraining for a one-step, 512×512, text-conditioned pixel generator.
This adds `t2i/` and three root launchers to MiniT2I-JAX commit
`eeee6065217168fb553150462c8914ab65db89fe`; the upstream code stays intact.
The small separate entry point keeps Stage 1 independent of MeanFlow distillation.

From the repository root, run setup on an **internet-enabled node** with resources
for environment installation, downloads and pretrained-weight conversion. Run
training in a GPU allocation with shared access to the prepared files:

```bash
CC12M_ROOT=/shared/cc12m CURATED_ROOT=/shared/curated/sources.json DATA_DIR=/shared/stage1 bash setup.sh
bash train.sh
```

`setup.sh` saves paths in `.stage1.env`; later environment overrides take precedence.
It creates `.venv`, installs the upstream requirements with compatible pins, and
downloads frozen T5 and perceptual assets, then prepares their local payloads.
Setup requires internet; training uses those local assets offline with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. A compute node with internet can
run setup. If compute has no internet, arrange online preparation on a suitable
node and make its data/assets/environment available to training; do not depend on
a training-time download. Use Python 3.10–3.12, Linux NVIDIA GPUs,
and a CUDA 12-compatible driver. On Linux, NVIDIA's CUDA 12.x minor-compatibility
minimum is **525.60.13**; use a newer site-supported driver when required by
your GPU or installed runtime wheels. JAX/Flax/Optax/Transformers are pinned to
0.4.31/0.8.5/0.2.5/4.46.1. CUDA libraries come from `jax[cuda12]`; Torch is CPU-only
and used for asset conversion. `jax[cuda12]` installs the NVIDIA CUDA user-space
libraries with pip; the NVIDIA kernel driver must be installed by the cluster.
Avoid a conflicting CUDA-module `LD_LIBRARY_PATH`, which can override pip libraries.
See [JAX installation guidance](https://docs.jax.dev/en/latest/installation.html)
and [NVIDIA's CUDA 12.x driver compatibility table](https://docs.nvidia.com/cuda/archive/12.5.0/cuda-toolkit-release-notes/index.html).
Dependency installation and tiny CPU tests are checked separately from GPU
initialization, full pretrained loading in the new environment and multi-node
training, which still require target-cluster validation.

To download and prepare both datasets instead, replace the setup line with:

```bash
DATA_DIR=/shared/stage1 DOWNLOAD_DATA=1 bash setup.sh
```

Dataset checks accept a plain directory of `.tar` shards or a shard glob.
A directory is treated as a curated mixture only when it contains a
`sources.json` file; an explicit mixture JSON path is also accepted.

CC12M preparation invokes `img2dataset` beside the current Python interpreter,
so setup does not require activating the virtual environment or changing PATH.

This is a large, explicit download. Nothing is downloaded by DRY_RUN:
`DRY_RUN=1 bash setup.sh`. `JAX_PLATFORM=cpu` selects an environment for tiny unit tests;
the production trainer requires GPUs.

## Multiple nodes and restart

Run from the repository root. Use the partition, account and QoS accepted by your cluster:

```bash
sbatch --nodes=16 --gpus-per-node=8 --export=ALL --no-requeue train.slurm
```

There is one JAX process per node, with all eight local GPUs visible.
`jax.distributed.initialize` receives explicit local device IDs for every visible
GPU before device discovery; this avoids JAX 0.4.31 Slurm's one-GPU default.
`GPUS_PER_NODE=8` is the launcher default; override it to match a different
`--gpus-per-node` request. The first allocated
node is the coordinator; `JAX_COORDINATOR_ADDRESS=host:port` overrides its address.
The default port is 12355. A site may need to select its network interfaces for NCCL.
`STAGE1_CODE_DIR` overrides the code directory when submitting from elsewhere.

Global batch stays **1024**. At 16×8 GPUs this is 8 examples per GPU per update;
`MICROBATCH=1` accumulates 8 microbatches, and `MICROBATCH=2` accumulates 4.
At other supported GPU counts the accumulation count changes automatically.
The trainer checks requested versus visible, local, global and per-process device
counts, and verifies GPU platforms. It records every deciding value before
rejecting a mismatch. Early initialization receipts use a unique temporary
directory; the trainer writes the full runtime topology in each rank's output
preflight receipt. A standalone single-node launch discovers visible GPUs
automatically.
Launcher batch/update gates also write their complete numerical inputs and verdict
to a unique temporary `preflight.json`, including rejected configurations.

```bash
DRY_RUN=1 NODES=16 GPUS_PER_NODE=8 bash train.slurm --microbatch 2
RESUME=/shared/run/checkpoint_00005000 OUTPUT_DIR=/shared/run bash train.sh
```

The Slurm `DRY_RUN` command writes a unique temporary preflight receipt and prints
its path. It performs no training or submission, but it is not a zero-write check.

Resume uses an **exact checkpoint directory**. It may reuse its owned output
directory or write into a new empty output directory.
Each fresh run gets a unique launch ID. Its immutable run identity binds the
source, recipe, noise mode, batch layout, A/B data manifests and asset manifests.
The output's create-once `owner.json` and every checkpoint carry this identity.
An existing output accepts only its own latest checkpoint and matching identity;
another run, a changed configuration, or an older checkpoint fails before writing
that output. Use a new empty output directory for a separate run or branch.
A branch from a checkpoint gets a new identity that pins its parent's exact
receipt/manifest SHA and run identity. Source/configuration/data/assets must still
match the parent for exact continuation. Old checkpoints without this identity
are rejected; this revision does not migrate previous run outputs automatically.
Target steps and checkpoint intervals are launch controls outside the identity,
so 20→30 continuation and USR1 resubmission keep the same identity even when the
Slurm job ID changes. Saving an already existing step succeeds only when the
entire serialized state and metadata are identical; otherwise it fails closed.
Resume also verifies all payload SHA256 hashes, optimizer counters, the step,
data topology and stream cursors. USR1 or TERM requests a checkpoint
after the current update; Slurm signals job-step processes 180 seconds before
the time limit. Large-checkpoint write time must be measured at the target site.
Jobs do not requeue or retry automatically. Use a new output directory for a fresh run.

Each checkpoint contains:

- `state.msgpack`: online weights, EMA, Adam state, RNG and update count for exact restart.
- `metadata.json`, `manifest.json`, `receipt.json`: run identity, recipe, data cursors and SHA bindings.
- `minit2i/checkpoint_STEP`: online `params.net` in the native MiniT2I Flax format.
- `minit2i_ema/checkpoint_STEP`: a separate native EMA export; choose weights explicitly.

To initialize a downstream backbone, verify and transfer explicitly:

```python
from t2i.transfer import initialize_posttrain_params
from t2i.checkpoint import atomic_json
params, receipt = initialize_posttrain_params(initialized_params, checkpoint_dir,
                                              branch="net", weights="online")
atomic_json(new_run_dir / "stage1_parent.json", receipt)
```

A bare MM-JiT uses `branch=None`. A legacy MeanFlow student adds auxiliary
embeddings/tokens: only `allow_meanflow_aux=True` keeps those initialized keys.
Every trained backbone key and shape must still match. This is a weights-only
warm-start, not restoration of a MeanFlow optimizer or a guarantee of identical
forward outputs. Actual pcfield/E1 checkpoint integration remains untested.

## Frozen text assets

Setup now requires the complete official native Flax release of `google/flan-t5-large`
at revision `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`. The shared validator fixes
all seven config/tokenizer/weight files, sizes and SHA256 values. Its manifest
explicitly declares `checkpoint_format="flax"` and `from_pt=false`; config-only
manifests, omitted weights, wrong sources and alternative local weight files fail.
PyTorch/safetensors conversion is unsupported in this revision because its source
payload is not SHA-locked here. Setup and runtime hash the complete real payload;
the text loader validates it again before deserialization, even when used alone.
Run combined setup on a suitably provisioned internet-enabled node. On login
nodes, `VENV_DIR=/shared/pi_env bash setup.sh --environment-only` installs only
dependencies and runs `pip check`; it does not load/convert model assets, prepare
datasets or write `.stage1.env`. Set the same `VENV_DIR` when running `train.sh`
after asset/data preparation. Tiger login hosts reject combined setup because
conversion/full loading belong on compute. Tiger users use the separate internal
login-download and offline-compute preparation helpers supplied outside this
standalone repository. Existing Tiger asset-load receipts refer to their original
environment, not a new installation. `bash setup.sh --dry-run` remains available.

Resume rejection receipts include the actual/expected step and every optimizer
counter value/count/shape before numerical rejection. Raw optimizer structure is
audited before state deserialization, including a structural mismatch. NaN/Inf are represented as
strings so failure evidence is valid JSON. These CPU regression checks do not
establish real pretrained loading or downstream performance.

## Paper correspondence

Sources: [Priming, §3.2 and Appendix C.2](https://arxiv.org/abs/2610.20331),
[paper authors’ code](https://github.com/JaeyeonKim01/priming),
[the selected C2I reproduction](https://github.com/ztMotaLee/priming_reproduce), and
[MiniT2I dataset preparation](https://github.com/PeppaKing8/minit2i-jax/blob/d92730b2a01667a07023ff9104b9baab29b4030f/scripts/datasets/README.md).
The requested C2I reproduction reference is
[`ztMotaLee/priming_reproduce`](https://github.com/ztMotaLee/priming_reproduce);
the formula parity checks use the supplied released C2I source snapshot.
That reproduction repository is separate from the paper authors' repository.

| Item | This code | Evidence/status |
| --- | --- | --- |
| Generator | MM-JiT-B/16, 512×512, x0 prediction; no s/t argument | Paper; upstream is already time-free. Exact count: 256,548,480 (~256.55M), versus paper's rounded ~258M. No generator parameter keys removed. |
| Text | Frozen FLAN-T5-Large, 256 tokens | MiniT2I encoder/conditioning; bfloat16 compute is an implementation choice, not weight-level numerically verified here. |
| Corruption | `(1-s)*x0 + s*epsilon`, `epsilon ~ N(0,4I)` | Paper. |
| Default s | Density proportional to `s^-2 * LogitNormal(s; 0.8, 0.8^2)` | Paper; sampled exactly by a three-component Gaussian mixture in logit space. |
| Reconstruction | Adaptive pixel MSE + 0.4 LPIPS + 0.1 ConvNeXt | Paper plus released C2I per-sample normalization: stop-gradient denominator; pixel epsilon 1e-4, perceptual epsilon 1e-2. |
| Schedule | CC12M 250,000 updates → curated mix 40,000 updates | Paper; both use global batch 1024. |
| Optimizer/LR | AdamW, stagewise 4e-4 → 4e-5 | Rates/optimizer from paper; constant rate within stages, no warmup and no optimizer reset are inferred. |

`--noise fixed-0.8` and `--noise fixed-1.0` support fixed-corruption and pure-noise
ablations. They retain the default 250k+40k schedule. The paper's separate fixed-0.8
result uses **160k+40k**, so these switches alone do not reproduce that row.

Unpublished T2I details remain explicit choices, pending confirmation:

- Adam betas (0.9, 0.95), weight decay 0, EMA 0.9999 and seed 42.
- No caption dropout or CFG during reconstruction; EMA is saved separately.
- Perceptual threshold 1.0, borrowed from released C2I L recipes: `s <= 1.0`
  activates perceptual loss for every default example and for fixed s=1.0.
- Paired 224px random perceptual crops, VGG16 LPIPS v0.1 and ConvNeXt-V2-Base;
  T2I crop/backbone details are not fully specified in the paper.
- Shuffled shard order with sequential records, rather than MiniT2I's resampling
  and 1000-record shuffle buffer; this permits exact, bounded-memory restart.
- JAX production precision is used. Numerical parity gates use highest matmul
  precision; real pretrained perceptual and text weights have not been compared here.

This is **Stage 1 only**. The paper's GenEval 0.818 follows distributional
post-training; reconstruction pretraining alone does not establish that score.

## Data sources, checksums and storage

Local inputs are uncompressed WebDataset `.tar` files with contiguous `{key}.jpg`
or `{key}.png` and `{key}.txt` pairs. Directories, globs and `{00000..01096}.tar`
patterns work. The reader resizes the shorter side, center-crops to 512 and
normalizes RGB to [-1,1]. It streams one sample at a time, partitions data across
JAX hosts, and stores the next tar byte offset for seek-based restart. Keep the
same process count and local shard paths when resuming. Remote URLs/compressed
tars are not the training path; download once onto the cluster's shared storage.

The curated `sources.json` uses relative paths:

```json
{"sources": [{"root": "blip3o", "weight": 0.06}, {"root": "dalle3", "weight": 0.016}, {"root": "sharegpt4o", "weight": 0.04}]}
```

| Source | Pinned revision | Download size |
| --- | --- | ---: |
| [CC12M LLaVA recaptions](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext) | `1c24bdba3f81257082612654ba72fe18a83c7a37` | 3.58 GB URL metadata; images extra |
| [BLIP3o-60K](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) | `f7316b0aa446338ee1707484924aa59457b4bbf3` | 108.16 GB in 11 tars |
| [DALL-E 3](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset) | `96822d5c0b1338ec4bfd04b0d5909fa070e301b6` | 33.19 GB in 67 parquets |
| [ShareGPT-4o](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image) | `f9bcb9494e562f3b6d46585bda0e7f0f0d6ec43f` | 90.10 GB text-to-image subset |
| [FLAN-T5-Large](https://huggingface.co/google/flan-t5-large) | `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a` | 3.14 GB Flax weights + tokenizer |

`prepare.py` pins every downloaded file's size and SHA256; converted shards get
a preparation receipt and SHA256 inventory. Model assets have
`assets.sha256.json`, checked before loading. Existing user-provided shards are
bound by path/inode/uid/size/mtime for restart, without an automatic full-dataset
hash scan. Filesystem device IDs are deliberately excluded.

CC12M metadata SHA256 is
`da25ecab422be301215c8d7e3bd55a795375f036c1d0b7d250e0f59c9a8b284a`.
Its URL images are downloaded with `img2dataset` using `caption_llava`, then
WebDataset output and `resize_mode=no`, as in the official JAX preparation recipe.
Original URLs may be dead; inspect the generated img2dataset statistics for the
realized sample count. There is no verified complete CC12M pixel mirror here.

The **“120K” name is approximate**, not an exact-count claim. Source weights
0.06:0.016:0.04 come from MiniT2I's published B/16 fine-tuning config.
We download all 11 BLIP3o tars, convert OpenDatasets DALL-E 3 parquets, and use
ShareGPT's text-to-image subset with the official ≤256 T5-token filter and
JPEG quality 95 conversion. All declared source files must pass verification;
conversion errors stop preparation and leave receipts for inspection.

There is an upstream discrepancy: MiniT2I's acknowledgments name
`CaptionEmporium/dalle3-llama3.2-11b`, while its actual dataset guide and
`prepare_dalle3_webdataset.py` select `OpenDatasets/dalle-3-dataset`.
This package follows the **actual preparation script**, pinned at MiniT2I main
`d92730b2a01667a07023ff9104b9baab29b4030f`. Exact source identity in the original
Priming T2I run remains unconfirmed.

Preparation estimates are assumptions, not measurements: the curated raw files
total 231.45 GB (31 minutes at a sustained 1 Gbit/s, before conversion); retaining
raw and converted data needs additional space. CC12M at 100–300 kB per recovered
image would occupy roughly 1–3 TB. At 100–1000 successful image downloads/s,
~11M images would require about 3–31 hours, and URL failures change both totals.
No dataset payload was downloaded during package construction. The internal
Tiger delivery records its separate asset-only downloads and per-file SHA values;
that download record does not establish genuine compute-side loading.

## Compute estimate and validation

Priming Table 2 reports **420.16 EFLOPs for its entire logit-normal recipe**,
including subsequent OT post-training; this is not a measurement of this
Stage-1 implementation. As a compute-only scale reference, dividing that number
by 128 GPUs sustaining an assumed 100–250 TFLOP/s each gives **3.65–9.12 hours**.
This does not isolate Stage 1 or measure this JAX implementation, precision,
frozen encoders, data loading, communication, compilation or checkpoint overhead.
Measure actual step time and memory on the target cluster before reserving a full run.
The paper uses 16 H200 GPUs for reconstruction; this launch example uses
16 nodes × 8 GPUs while preserving batch 1024.

Tiny CPU tests (no full checkpoint or pretrained model download):

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

The tests cover the exact corruption law, time-free model forward, adaptive
loss/gradient parity, tiny perceptual networks, schedule and batch layout,
checkpoint/optimizer restart, data cursor restart, device-number-independent
identity, shell syntax and accepted/rejected launcher arguments. GPU, real-asset,
multi-node and full-data execution remain to be run. XLA's production flag is
`--xla_gpu_autotune_level=1`; JAX compilation caching is disabled and temporary
files use a unique job/process directory.

MiniT2I code is MIT. This JAX addition does not copy the internal pMF perceptual
implementation. Public Priming/MiniT2I code is credited above; downloaded
dataset and pretrained-weight licenses are separate and remain applicable.
