#!/usr/bin/env python3
"""Score the paired panels with original GenEval and full-row corrected ELLA DPG."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import time
import types
from eval_sample import atomic_json, expected_rows, sha256

PINS = Path(__file__).parent / 'evaluation/pins.json'
PROMPT_COUNT = 1065
QUESTION_COUNT = 14392
LEGACY_OMITTED_KEY = 'partiprompts97'
SAMPLES = 4

def questions_from_rows(rows):
    result = {}
    for row in rows:
        key, qid = row['item_id'], int(row['proposition_id'])
        q = result.setdefault(key, dict(qid2tuple={}, qid2dependency={}, qid2question={}))
        if qid <= 0 or qid in q['qid2question']:
            raise ValueError('invalid/duplicate proposition: ' + key)
        deps = [int(x.strip()) for x in row['dependency'].split(',')]
        # Official CSV contains repeated parents (e.g. 8,8); preserve them.
        if not deps or any(x < 0 for x in deps):
            raise ValueError('invalid dependencies: ' + key)
        q['qid2tuple'][qid] = row['tuple']
        q['qid2dependency'][qid] = deps
        q['qid2question'][qid] = row['question_natural_language']
    for key, q in result.items():
        # Original annotations also include self/cyclic dependencies. ELLA
        # performs one insertion-order pass, not a topological/recursive pass.
        # Do not silently repair these annotations or reject official prompts.
        for parents in q['qid2dependency'].values():
            if any(p != 0 and p not in q['qid2question'] for p in parents):
                raise ValueError('missing dependency: ' + key)
    return result

def gate_scores(raw, dependencies):
    """Exact ELLA one-pass, CSV insertion-order dependency gating."""
    if set(raw) != set(dependencies):
        raise ValueError('incomplete question answers')
    scores = dict(raw)
    for qid, parents in dependencies.items():
        if any(parent != 0 and scores[parent] == 0 for parent in parents):
            scores[qid] = 0.0
    return scores

def answer_key(row):
    return row['key'], row['seed'], row['qid']

def validate_answers(rows, questions, keys, require_complete):
    keys = set(keys)
    answers = {}
    for row in rows:
        key, seed, qid = answer_key(row)
        if (key not in keys or type(seed) is not int or not 0 <= seed < SAMPLES
                or type(qid) is not int or qid not in questions[key]['qid2question']
                or row.get('question') != questions[key]['qid2question'][qid]
                or type(row.get('answer')) is not str or answer_key(row) in answers):
            raise ValueError('invalid/duplicate VQA answer binding')
        answers[answer_key(row)] = row['answer']
    expected = sum(len(questions[k]['qid2question'])*SAMPLES for k in keys)
    if require_complete and len(answers) != expected:
        raise ValueError('missing VQA answers: %d/%d' % (len(answers), expected))
    return answers

def aggregate_answers(answers, questions, keys):
    """Return full-row diagnostic + explicit legacy-compatible subset score."""
    keys = list(keys)
    if len(set(keys)) != len(keys) or not keys:
        raise ValueError('empty or duplicate prompt coverage')
    crops, items, legacy_categories, corrected_categories = {}, {}, defaultdict(list), defaultdict(list)
    for key in keys:
        q = questions[key]
        values = []
        for seed in range(SAMPLES):
            raw = {qid: float(answers[(key, seed, qid)] == 'yes') for qid in q['qid2question']}
            gated = gate_scores(raw, q['qid2dependency'])
            score = sum(gated.values()) / len(gated)
            values.append(score)
            for qid, value in gated.items():
                category = q['qid2tuple'][qid].split('(')[0].strip()
                corrected_categories[category].append(value)
                # Original ELLA returns raw qid scores from ONLY final crop.
                if seed == SAMPLES-1 and key != LEGACY_OMITTED_KEY:
                    legacy_categories[category].append(raw[qid])
        crops[key] = values
        items[key] = sum(values) / SAMPLES
    legacy = [items[k] for k in keys if k != LEGACY_OMITTED_KEY]

    def category_means(values):
        detailed = {k: 100*sum(v)/len(v) for k, v in sorted(values.items())}
        broad = defaultdict(list)
        for key, vals in values.items():
            broad[key.split('-')[0].strip()].extend(vals)
        return {'L1': {k: 100*sum(v)/len(v) for k, v in sorted(broad.items())}, 'L2': detailed}

    return dict(full_benchmark=len(keys) == PROMPT_COUNT and set(keys) == set(questions),
        prompt_count=len(keys), answer_count=sum(len(questions[k]['qid2question'])*SAMPLES for k in keys),
        fullrow_corrected_overall=100*sum(items.values())/len(items),
        fullrow_corrected_categories_all_crops_gated=category_means(corrected_categories),
        legacy_compatible_overall=(100*sum(legacy)/len(legacy) if legacy else None),
        legacy_prompt_count=len(legacy), legacy_omitted_keys=([LEGACY_OMITTED_KEY] if LEGACY_OMITTED_KEY in keys else []),
        legacy_categories_last_crop_ungated=category_means(legacy_categories),
        item_scores=items, crop_scores=crops,
        score_units='percent', comparison='Original ELLA + ModelScope mPLUG, not MiniT2I JAX-mPLUG reproduction')


def aggregate_geneval(rows):
    if not rows:
        raise ValueError('No GenEval results')
    categories = sorted({row['tag'] for row in rows})
    values = {tag: sum(row['correct'] for row in rows if row['tag'] == tag) /
                   sum(row['tag'] == tag for row in rows) for tag in categories}
    return dict(images=len(rows), categories=values,
                overall_category_balanced=sum(values.values()) / len(values))


@contextmanager
def offline_network():
    previous = socket.socket.connect
    def denied(*args, **kwargs):
        raise RuntimeError('Network access is forbidden during scoring')
    socket.socket.connect = denied
    try:
        yield
    finally:
        socket.socket.connect = previous


def verify_assets(assets, scope):
    for row in json.loads(PINS.read_text())['files']:
        if row['scope'] != scope:
            continue
        path = assets / row['path']
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Evaluation source/model SHA differs: ' + str(path))


def verify_images(root, benchmark, assets):
    from PIL import Image
    expected = expected_rows(benchmark, assets)
    receipt = json.loads((root / 'sampling_receipt.json').read_text())
    if (receipt.get('status') != 'complete' or receipt.get('count') != len(expected)
            or receipt.get('prompt_count') != len(expected) // 4 or receipt.get('native_resolution') != 512
            or receipt.get('images_manifest_sha256') != sha256(root / 'images.jsonl')
            or receipt.get('protocol_sha256') != sha256(root / 'protocol.json')):
        raise ValueError('Incomplete or different sampling receipt')
    records = [json.loads(line) for line in (root / 'images.jsonl').read_text().splitlines()]
    if len(records) != len(expected):
        raise ValueError('Image coverage differs')
    for row, actual in zip(expected, records):
        if {k: v for k, v in actual.items() if k != 'sha256'} != row:
            raise ValueError('Image prompt/seed/order differs')
        path = root / row['relative_path']
        if sha256(path) != actual['sha256']:
            raise ValueError('Sample SHA differs')
        with Image.open(path) as image:
            if image.size != (512, 512) or image.mode != 'RGB':
                raise ValueError('Expected native 512 RGB samples')
        if benchmark == 'geneval':
            metadata = path.parent.parent / 'metadata.jsonl'
            if json.loads(metadata.read_text()) != row['metadata']:
                raise ValueError('GenEval per-prompt metadata differs')
    if benchmark == 'geneval':
        actual = {p.name for p in (root / 'geneval').iterdir() if p.is_dir() and p.name.isdigit()}
        if actual != {f'{index:05d}' for index in range(553)}:
            raise ValueError('Require exactly 553 GenEval prompt directories')
        for index in range(553):
            if {p.name for p in (root / f'geneval/{index:05d}/samples').glob('*.png')} != {f'{seed:04d}.png' for seed in range(4)}:
                raise ValueError('Extra or missing GenEval image')
    else:
        for start in range(0, len(expected), 4):
            key = expected[start]['key']
            with Image.open(root / 'grids' / (key + '.png')) as image:
                if image.size != (1024, 1024) or image.mode != 'RGB':
                    raise ValueError('DPG grid size/mode differs')
                for seed in range(4):
                    with Image.open(root / f'samples/{key}/{seed:04d}.png') as sample:
                        x, y = (seed % 2) * 512, (seed // 2) * 512
                        if image.crop((x, y, x + 512, y + 512)).tobytes() != sample.tobytes():
                            raise ValueError('DPG grid crop differs from sample')
    return expected


def reference_attention(output):
    """Same reference operation; exact precision only for CPU/CUDA parity check."""
    import torch
    import mmcv.ops.multi_scale_deform_attn as original
    reference = original.multi_scale_deformable_attn_pytorch
    state = dict(implementation='mmcv.multi_scale_deformable_attn_pytorch', calls=0)
    def apply(value, shapes, starts, locations, weights, im2col_step):
        state['calls'] += 1
        return reference(value, shapes, locations, weights)
    original.MultiScaleDeformableAttnFunction = types.SimpleNamespace(apply=apply)
    generator = torch.Generator().manual_seed(20260906)
    value = torch.randn(1, 5, 2, 4, generator=generator)
    shapes = torch.tensor([[2, 2], [1, 1]], dtype=torch.long)
    locations = torch.rand(1, 3, 2, 2, 2, 2, generator=generator)
    weights = torch.rand(1, 3, 2, 2, 2, generator=generator)
    weights = weights / weights.sum(dim=(-1, -2), keepdim=True)
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    try:
        expected = reference(value, shapes, locations, weights)
        actual = apply(value.cuda(), shapes.cuda(), torch.tensor([0, 4], device='cuda'), locations.cuda(), weights.cuda(), 64).cpu()
        passed = bool(torch.isfinite(actual).all()) and actual.abs().sum().item() > 0 and bool(torch.allclose(actual, expected, rtol=1e-4, atol=1e-5))
        atomic_json(output / 'attention_parity.json', dict(actual=actual.tolist(), expected=expected.tolist(),
            max_absolute_error=float((actual - expected).abs().max()), rtol=1e-4, atol=1e-5, passed=passed))
        if not passed:
            raise ValueError('Reference attention parity failed')
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
    return state


def load_geneval(assets):
    import torch
    import open_clip
    from open_clip import openai as openai_loader
    import mmcv
    from mmdet.apis import init_detector
    source = assets / 'sources/geneval/evaluation/evaluate_images.py'
    clip = assets / 'models/open_clip/ViT-L-14.pt'
    expected_url = openai_loader.get_pretrained_url('ViT-L-14', 'openai')
    if not expected_url:
        raise RuntimeError('Required OpenAI CLIP configuration unavailable')
    def local_openai_file(url, cache_dir=None):
        if url != expected_url:
            raise ValueError('Unpinned CLIP model requested')
        return str(clip)
    # Keep the OpenAI JIT loader; a pathname selects a different state-dict loader.
    openai_loader.download_pretrained_from_url = local_openai_file
    text = source.read_text()
    if text.count('assert DEVICE == "cuda"') != 1:
        raise ValueError('Unexpected upstream GenEval source')
    module = types.ModuleType('original_geneval')
    module.__file__ = str(source)
    exec(compile(text.replace('assert DEVICE == "cuda"', ''), str(source), 'exec'), module.__dict__)
    module.DEVICE = 'cuda'
    module.args = types.SimpleNamespace(options={})
    configuration = assets / 'sources/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py'
    config = mmcv.Config.fromfile(str(configuration))
    config.model.backbone.init_cfg = None
    module.object_detector = init_detector(config, str(assets / 'models/geneval/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth'), device='cuda')
    module.clip_model, _, module.transform = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device='cuda', cache_dir=str(clip.parent))
    module.clip_model.eval()
    module.tokenizer = open_clip.get_tokenizer('ViT-L-14')
    module.classnames = (source.parent / 'object_names.txt').read_text().splitlines()
    module.THRESHOLD, module.COUNTING_THRESHOLD = 0.3, 0.9
    module.MAX_OBJECTS, module.NMS_THRESHOLD, module.POSITION_THRESHOLD = 16, 1.0, 0.1
    color = text[text.index('def color_classification('):text.index('def compute_iou(')]
    exec(compile(color.replace('num_workers=4', 'num_workers=0'), str(source), 'exec'), module.__dict__)
    return module


def score_geneval(args, rows):
    state = reference_attention(args.output)
    module = load_geneval(args.assets)
    scored = []
    before = state['calls']
    with (args.output / 'geneval.jsonl').open('x') as stream:
        for row in rows:
            image = args.images / row['relative_path']
            result = module.evaluate_image(str(image), row['metadata'])
            result.update(prompt_index=row['index'], seed=row['seed'], image_sha256=sha256(image))
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            scored.append(result)
    if state['calls'] <= before:
        raise ValueError('Detector did not execute reference attention')
    result = aggregate_geneval(scored)
    result['reference_attention'] = state
    result['status'] = 'COMPLETE'
    return result


def dpg_metadata(assets):
    with (assets / 'benchmarks/dpg/dpg_bench.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    prompts = [json.loads(line) for line in (assets / 'benchmarks/dpg/prompts.jsonl').read_text().splitlines()]
    questions = questions_from_rows(rows)
    if (len(rows) != QUESTION_COUNT or len(prompts) != PROMPT_COUNT or len({p['key'] for p in prompts}) != PROMPT_COUNT
            or set(questions) != {p['key'] for p in prompts} or rows[0]['item_id'] != LEGACY_OMITTED_KEY
            or int(rows[0]['proposition_id']) != 1 or questions[LEGACY_OMITTED_KEY]['qid2dependency'][2] != [1]):
        raise ValueError('Unexpected DPG full-row population')
    return prompts, questions


def score_dpg(args):
    import torch
    from PIL import Image
    prompts, questions = dpg_metadata(args.assets)
    source = args.assets / 'sources/ella/compute_dpg_bench.py'
    spec = importlib.util.spec_from_file_location('original_ella_dpg', source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    load_audit = []
    original_load = torch.nn.Module.load_state_dict
    def checked_load(module, state_dict, *positional, **keywords):
        result = original_load(module, state_dict, *positional, **keywords)
        missing = sorted(set(dict(module.named_parameters())).intersection(result.missing_keys))
        load_audit.append(dict(module=type(module).__name__, missing_parameter_keys=missing,
                               missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys)))
        atomic_json(args.output / 'model_load.json', load_audit)
        if missing:
            raise ValueError('mPLUG checkpoint has uninitialized model parameters')
        return result
    torch.nn.Module.load_state_dict = checked_load
    try:
        model = upstream.MPLUG(ckpt=str((args.assets / 'models/mplug').resolve()), device='cuda:0')
    finally:
        torch.nn.Module.load_state_dict = original_load
    if not load_audit:
        raise ValueError('mPLUG checkpoint validation was not observed')
    answers = {}
    with (args.output / 'answers.jsonl').open('x') as stream:
        class Recorder:
            def vqa(self, image, question):
                key, seed, qid, expected = self.expected[self.index]
                self.index += 1
                if question != expected:
                    raise ValueError('Upstream question/crop order differs')
                answer = model.vqa(image, question)
                if type(answer) is not str:
                    raise TypeError('mPLUG returned a non-string answer')
                answers[(key, seed, qid)] = answer
                stream.write(json.dumps(dict(key=key, seed=seed, qid=qid, question=question, answer=answer), ensure_ascii=False) + '\n')
                stream.flush()
                return answer
        recorder = Recorder()
        upstream_args = types.SimpleNamespace(pic_num=4, res_path=str(args.output / 'upstream_results.txt'))
        for prompt in prompts:
            key = prompt['key']
            recorder.expected = [(key, seed, qid, question) for seed in range(4) for qid, question in questions[key]['qid2question'].items()]
            recorder.index = 0
            score, _, _ = upstream.compute_dpg_one_sample(upstream_args, questions, str(args.images / 'grids' / (key + '.png')), recorder, 512)
            checked = aggregate_answers(answers, questions, [key])['fullrow_corrected_overall'] / 100
            if recorder.index != len(recorder.expected) or abs(float(score) - checked) > 1e-12:
                raise ValueError('DPG upstream aggregation or crop order differs')
    keys = [prompt['key'] for prompt in prompts]
    records = [json.loads(line) for line in (args.output / 'answers.jsonl').read_text().splitlines()]
    validate_answers(records, questions, keys, require_complete=True)
    metrics = aggregate_answers(answers, questions, keys)
    if metrics['answer_count'] != 57568 or not metrics['full_benchmark']:
        raise ValueError('Incomplete DPG scoring coverage')
    metrics.update(status='COMPLETE', answers_sha256=sha256(args.output / 'answers.jsonl'))
    return metrics


def run(args):
    verify_assets(args.assets, args.benchmark)
    rows = verify_images(args.images, args.benchmark, args.assets)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Run scoring inside an allocated compute-node task')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Scoring requires exactly one visible allocated GPU')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / 'protocol.json', dict(benchmark=args.benchmark, images=str(args.images),
        sampling_receipt_sha256=sha256(args.images / 'sampling_receipt.json'), pins_sha256=sha256(PINS),
        device=torch.cuda.get_device_name(0)))
    started = time.monotonic()
    try:
        with offline_network():
            result = score_geneval(args, rows) if args.benchmark == 'geneval' else score_dpg(args)
        primary = 'overall_category_balanced' if args.benchmark == 'geneval' else 'fullrow_corrected_overall'
        score = float(result[primary])
        finite = math.isfinite(score)
        atomic_json(args.output / 'numeric.json', dict(primary=primary, value=score if finite else str(score), finite=finite))
        if not finite:
            raise ValueError('Nonfinite evaluation score')
        result['elapsed_seconds'] = time.monotonic() - started
        result['protocol_sha256'] = sha256(args.output / 'protocol.json')
        atomic_json(args.output / ('geneval.summary.json' if args.benchmark == 'geneval' else 'dpg.json'), result)
        print(json.dumps({primary: score, 'status': result['status']}))
        return result
    except BaseException as error:
        atomic_json(args.output / 'FAILED.json', dict(status='FAILED', error=type(error).__name__ + ': ' + str(error)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('geneval', 'dpg'), required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
