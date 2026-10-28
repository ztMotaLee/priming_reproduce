#!/usr/bin/env python3
"""Generate the exact paired GenEval or DPG panel with the one-step teacher branch."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np

PINS = Path(__file__).parent / 'evaluation/pins.json'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    def numeric(item):
        if isinstance(item, dict):
            return {str(key): numeric(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [numeric(val) for val in item]
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        return item
    temporary.write_text(json.dumps(numeric(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def canonical_digest(rows):
    return hashlib.sha256((json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()


def rows_from_prompts(benchmark, prompts):
    """Keep upstream prompt insertion order and seeds 0,1,2,3 within each prompt."""
    if benchmark == 'geneval':
        return [dict(benchmark='geneval', index=i, seed=seed, prompt=row['prompt'], metadata=row,
                     relative_path=f'geneval/{i:05d}/samples/{seed:04d}.png')
                for i, row in enumerate(prompts) for seed in range(4)]
    if benchmark == 'dpg':
        return [dict(row, seed=seed, relative_path=f"samples/{row['key']}/{seed:04d}.png")
                for row in prompts for seed in range(4)]
    raise ValueError('Unknown benchmark')


def expected_rows(benchmark, assets):
    pins = json.loads(PINS.read_text())
    relative = f'benchmarks/{benchmark}/' + ('evaluation_metadata.jsonl' if benchmark == 'geneval' else 'prompts.jsonl')
    path = Path(assets) / relative
    expected = (next(row['sha256'] for row in pins['files'] if row['path'] == relative)
                if benchmark == 'geneval' else pins['dpg_prompts_sha256'])
    if sha256(path) != expected:
        raise ValueError('Prompt file SHA differs')
    prompts = [json.loads(line) for line in path.read_text().splitlines()]
    if len(prompts) != pins[benchmark + '_prompts']:
        raise ValueError('Prompt count differs')
    rows = rows_from_prompts(benchmark, prompts)
    if benchmark == 'dpg' and canonical_digest(rows) != pins['dpg_caption_noise_list_sha256']:
        raise ValueError('DPG prompt/seed/order digest differs')
    return rows


def keys_from_rows(rows):
    import jax
    return np.asarray([jax.random.PRNGKey(row['seed']) for row in rows])


def quantize(pixels):
    pixels = np.asarray(pixels, np.float32)
    if not np.isfinite(pixels).all():
        raise ValueError('Nonfinite generated pixels')
    return np.clip(127.5 * pixels + 128, 0, 255).astype(np.uint8)


def sample(args):
    rows = expected_rows(args.benchmark, args.assets)
    plan = dict(benchmark=args.benchmark, prompts=len(rows) // 4, images=len(rows),
                prompt_seed_order_sha256=canonical_digest(rows), resolution=512, samples_per_prompt=4,
                sampling='one model forward; teacher parameters; tau=0; 2*normal; no guidance',
                checkpoint=str(args.checkpoint), output=str(args.output))
    if args.plan:
        print(json.dumps(plan, indent=2))
        return plan
    if args.output.exists():
        raise FileExistsError('Sampling output already exists: ' + str(args.output))
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Run sampling inside an allocated compute-node task')
    for name in ('TRANSFORMERS_CACHE', 'PYTORCH_TRANSFORMERS_CACHE', 'PYTORCH_PRETRAINED_BERT_CACHE',
                 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'JAX_COMPILATION_CACHE_DIR',
                 'JAX_ENABLE_COMPILATION_CACHE', 'JAX_DEFAULT_MATMUL_PRECISION'):
        os.environ.pop(name, None)
    os.environ.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=1 --xla_gpu_deterministic_ops=true')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import jax
    import jax.numpy as jnp
    if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != 'gpu':
        raise RuntimeError('Sampling requires exactly one visible allocated GPU')
    metadata_path = args.checkpoint / 'checkpoint.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('schema') != 'bigfield-checkpoint-v1':
        raise ValueError('Unknown checkpoint metadata schema')
    plan.update(checkpoint_metadata_sha256=sha256(metadata_path), checkpoint_step=metadata['step'],
                parameter_sha256=metadata['parameter_sha256'],
                params_file_sha256=metadata['files_sha256']['params.msgpack'])
    from checkpoint import load_params
    from training import make_models, text_encoder
    from PIL import Image
    params = load_params(args.checkpoint)
    sampler = jax.jit(jax.vmap(make_models(args.native_root), in_axes=(None, 0, 0, 0)))
    tokenizer, encode, text_params = text_encoder(args.weights_root, args.native_root)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'protocol.json', plan)
    started = time.monotonic()
    try:
        with (args.output / 'images.jsonl').open('x') as stream:
            for start in range(0, len(rows), 4):
                subset = rows[start:start + 4]
                tokens = tokenizer([row['prompt'] for row in subset], max_length=256, padding='max_length',
                                   truncation=True, return_tensors='np')
                masks = np.asarray(tokens['attention_mask'], np.int32)
                texts = np.asarray(encode(text_params, jnp.asarray(tokens['input_ids'], jnp.int32),
                                          jnp.asarray(masks)), np.float32)
                keys = keys_from_rows(subset)
                pixels = np.asarray(sampler(params, texts, masks, keys), np.float32)
                observation = dict(batch_start=start, shape=list(pixels.shape),
                                   nonfinite_count=int((~np.isfinite(pixels)).sum()),
                                   elapsed_seconds=time.monotonic() - started)
                atomic_json(args.output / 'sampling_numeric.json', observation)
                if pixels.shape != (4, 512, 512, 3):
                    raise ValueError('Unexpected generated tensor shape')
                pictures = quantize(pixels)
                for row, picture in zip(subset, pictures):
                    target = args.output / row['relative_path']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(picture).save(target)
                    if args.benchmark == 'geneval':
                        metadata = target.parent.parent / 'metadata.jsonl'
                        if not metadata.exists():
                            atomic_json(metadata, row['metadata'])
                    stream.write(json.dumps(dict(row, sha256=sha256(target)), ensure_ascii=False, sort_keys=True) + '\n')
                stream.flush()
        if args.benchmark == 'dpg':
            grids = args.output / 'grids'
            grids.mkdir()
            for start in range(0, len(rows), 4):
                key = rows[start]['key']
                grid = Image.new('RGB', (1024, 1024))
                for seed in range(4):
                    with Image.open(args.output / f'samples/{key}/{seed:04d}.png') as image:
                        grid.paste(image, ((seed % 2) * 512, (seed // 2) * 512))
                grid.save(grids / (key + '.png'))
        receipt = dict(status='complete', count=len(rows), prompt_count=len(rows) // 4, native_resolution=512,
                       images_manifest_sha256=sha256(args.output / 'images.jsonl'),
                       prompt_seed_order_sha256=canonical_digest(rows), protocol_sha256=sha256(args.output / 'protocol.json'),
                       checkpoint_metadata_sha256=plan['checkpoint_metadata_sha256'],
                       parameter_sha256=plan['parameter_sha256'], elapsed_seconds=time.monotonic() - started)
        atomic_json(args.output / 'sampling_receipt.json', receipt)
        return receipt
    except BaseException as error:
        atomic_json(args.output / 'FAILED.json', dict(status='FAILED', error=type(error).__name__ + ': ' + str(error)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('geneval', 'dpg'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True, help='Evaluation assets prepared by evaluation/prepare.py')
    parser.add_argument('--weights-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', action='store_true', help='Validate prompt binding; do not load models or generate')
    return sample(parser.parse_args(argv))


if __name__ == '__main__':
    main()
