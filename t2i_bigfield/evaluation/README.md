# Paired evaluation

The two primary metrics are GenEval `overall_category_balanced` (0–1, equal weight for the six categories) and DPG `fullrow_corrected_overall` (percent, all 1065 prompts). Evaluate every prescribed readout; do not select a best checkpoint. This panel uses the original ELLA + ModelScope mPLUG evaluator, not the JAX mPLUG reproduction.

GenEval has 553 prompts × seeds 0, 1, 2, 3 = 2212 images. DPG has 1065 prompts × the same four seeds = 4260 images, 14392 questions and 57568 answers. Prompt order and the seed within each prompt are fixed. DPG uses sorted upstream prompt filenames, UTF-8 without ASCII escaping, and `strip()` on each text file. The reconstructed prompt SHA must be `b7fc9516d9ad5135664d17729cb7771aa7e3cf2bff4973dfd0053776973164eb`; the complete DPG prompt/seed/path list SHA is `e114885e4c99baeadc3a0a85a6738607fe2927e429caffe3653c7ad5728a90ef`.

Sampling uses the same `make_models` and native T5 encoder as training. Each image gets `PRNGKey(seed)`; the generator splits this key and uses the second half for a `(1,512,512,3)` normal draw scaled by 2, then predicts x0 at tau=0 in one forward pass without guidance. Text length is 256; generation and text encoding use batches of four. Quantization is exactly `clip(127.5*x+128,0,255).astype(uint8)`. The same four latent draws are reused for each prompt, matching the paired reference panel.

## Prepare once

Run downloads on an appropriate transfer or allocated node under your cluster policy. Set `EVAL_ASSETS` to a dedicated writable directory outside the source checkout. The default command only prints the download plan. `--download` explicitly fetches files, resumes weight `.partial` files, and verifies SHA-256 and byte length. It never loads a model. A corrupt final file stops preparation without replacing it. No automatic retries occur.

```bash
python evaluation/prepare.py --root "$EVAL_ASSETS"
python evaluation/prepare.py --root "$EVAL_ASSETS" --download
python evaluation/prepare.py --root "$EVAL_ASSETS" --verify
```

`pins.json` records public URLs, exact revisions, file sizes and SHA-256 for the original sources, detector, CLIP JIT archive and ModelScope mPLUG assets. Prompt source code/data retain their upstream licenses: [GenEval (MIT)](https://github.com/djghosh13/geneval/tree/af4902f24d3ca90ebbb446dd9891a59e0f82725f), [ELLA (Apache 2.0)](https://github.com/TencentQQGYLab/ELLA/tree/3c228f1dc6c4d3cad0a47493816151a419f14db3), and [MMDetection (Apache 2.0)](https://github.com/open-mmlab/mmdetection/tree/e9cae2d0787cd5c2fc6165a6061f92fa09e48fb1). Preparation downloads their license files alongside source. Review the respective model/data terms before downloading. ModelScope endpoints may require access approval or authentication; stop and arrange access if they reject a request. Credentials do not belong in scripts or this repository.

Create two separate Python 3.10 environments for scoring. These requirements preserve the reference evaluator's package versions, including Torch 2.0.0 with CUDA 11.8; they are independent of the JAX training environment. The GenEval wheel is for Linux x86-64 Python 3.10. A different platform needs an equivalent build and a new validation, not a silent package upgrade.

```bash
conda create -y -p "$GENEVAL_ENV" python=3.10 pip
"$GENEVAL_ENV/bin/python" -m pip install -r evaluation/requirements-geneval.txt
conda create -y -p "$DPG_ENV" python=3.10 pip
"$DPG_ENV/bin/python" -m pip install -r evaluation/requirements-dpg.txt
```

The requirements and model endpoints were derived from the reference assets. Installation, remote endpoint availability and CUDA scoring have not been tested by this release's CPU checks. Check those on your allocated GPU before scheduling the full panel.

## Sample and score a readout

Run each command as a separate process with exactly one allocated visible GPU. `SLURM_JOB_ID` and `SLURMD_NODENAME` must identify a compute-node task. Set `CKPT` to a retained checkpoint, `WEIGHTS` to the training weights directory, and `IMAGES`/`SCORES` to fresh working directories on storage suited to temporary image I/O. Start in `t2i_bigfield/`. Do not start the JAX sampler and Torch scorer together on the same GPU.

```bash
python eval_sample.py --benchmark geneval --checkpoint "$CKPT" \
  --assets "$EVAL_ASSETS" --weights-root "$WEIGHTS" \
  --native-root "$PWD/bigfield_native" --output "$IMAGES/geneval" --plan

python eval_sample.py --benchmark geneval --checkpoint "$CKPT" \
  --assets "$EVAL_ASSETS" --weights-root "$WEIGHTS" \
  --native-root "$PWD/bigfield_native" --output "$IMAGES/geneval"
"$GENEVAL_ENV/bin/python" score.py --benchmark geneval \
  --assets "$EVAL_ASSETS" --images "$IMAGES/geneval" --output "$SCORES/geneval"

python eval_sample.py --benchmark dpg --checkpoint "$CKPT" \
  --assets "$EVAL_ASSETS" --weights-root "$WEIGHTS" \
  --native-root "$PWD/bigfield_native" --output "$IMAGES/dpg"
"$DPG_ENV/bin/python" score.py --benchmark dpg \
  --assets "$EVAL_ASSETS" --images "$IMAGES/dpg" --output "$SCORES/dpg"
```

The scorer verifies every image SHA, prompt, seed, metadata row and native resolution before loading weights. DPG also verifies that its four-crop grids equal the four original images. Upstream ELLA skips the first CSV row and consequently loses one prompt. Our primary score restores that row, preserves repeated/self/cyclic dependencies, executes exactly one insertion-order dependency pass, and accepts only the exact answer string `yes`. The primary includes all 1065 prompts. `legacy_compatible_overall` omits the single fixed upstream prompt and is descriptive only.

On H100, GenEval retains MMCV's PyTorch reference deformable attention, with the same CPU/CUDA parity tolerances (`rtol=1e-4`, `atol=1e-5`) and temporary highest precision for that diagnostic. Production precision is restored afterward. Its original OpenAI CLIP JIT loading branch is preserved, resolving its download URL to SHA-verified local bytes. The reference site's four diagnostic control images are not bundled; they do not enter either metric. The portable scorer checks parity and actual attention calls. Scoring performs no network downloads. Invalid weights, incomplete panels, model load errors or nonfinite scores stop the run; answers already recorded remain available for diagnosis.

Archive the score JSON, GenEval per-image JSONL, DPG answers JSONL, sampling receipts, protocols and image journals at every prescribed readout. The protocols bind the checkpoint metadata, params SHA, prompt order, image bytes and evaluator pins. Large sample PNGs are temporary working artifacts; retain only the requested logs and endpoint artifacts after your own inspection and storage-policy decision.

## Params-only fallback

If scoring installation is impractical, send each retained readout's `params.msgpack` and `checkpoint.json`, plus a SHA-256 list of those two files. Also send the exact source revision, `configs/bigfield_tf.yaml`, training logs and run metadata. The params payload contains only `pt_net`; it excludes AdamW. `checkpoint.load_params` verifies both file SHA and parameter-tree SHA. Keep `state.msgpack` on your cluster for resume; it need not be transferred for evaluation. The receiver must record the two-file SHA list and use this same paired panel. Do not replace the main metrics with training-encoder scores. No FID is implemented here.
