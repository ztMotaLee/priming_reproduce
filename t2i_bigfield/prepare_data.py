"""Pinned downloads, exact curated-row reconstruction, and DINO bank preparation.

Run full conversion and feature extraction on allocated compute resources.
The published row selection and cluster assignment define the reproduction.
"""
from __future__ import annotations
import argparse
import base64
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
from checkpoint import atomic_json, sha256_file
from data import (CODE, SOURCES, WEIGHTS, DatasetRows, Partition, canonical_image,
                  load_dataset, load_real, public_pins, published_rows,
                  sample_caption_indices)

K, MIN_SIZE, SEED, ITERATIONS, BATCH = 1024, 32, 20260916, 60, 512


def _verified(path, item):
    return path.is_file() and path.stat().st_size == item['size'] and sha256_file(path) == item['sha256']


def iter_downloads(root, kind):
    pins = public_pins(); root = Path(root)
    if kind in ('data', 'all'):
        for source, spec in pins['raw_datasets'].items():
            for item in spec['files']:
                url = f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/{spec['revision']}/{item['name']}"
                yield root / 'raw' / source / item['name'], item, url
    if kind in ('weights', 'clip', 'all'):
        for name, spec in pins['weights'].items():
            if (kind == 'weights' and name == 'clip') or (kind == 'clip' and name != 'clip'):
                continue
            for item in spec['files']:
                url = item.get('url') or f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/{item['name']}"
                yield root / 'weights' / name / item['name'], item, url


def download_file(path, item, url):
    """Resume an incomplete HTTP download; publish only after exact byte/SHA checks."""
    path = Path(path)
    if path.exists():
        if _verified(path, item):
            print(json.dumps(dict(status='VERIFIED', file=str(path))), flush=True); return
        raise ValueError('Existing destination differs from pinned bytes: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    start = part.stat().st_size if part.exists() else 0
    if start > item['size']:
        raise ValueError('Partial download exceeds expected bytes: ' + str(part))
    if start < item['size']:
        headers = {'User-Agent': 'bigfield-reproduction/1'}
        if start:
            headers['Range'] = 'bytes=' + str(start) + '-'
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            status = response.status
            if start and status != 206:
                raise RuntimeError('Server did not honor Range; partial file retained for inspection')
            if start and not response.headers.get('Content-Range', '').startswith('bytes ' + str(start) + '-'):
                raise RuntimeError('Server resumed at the wrong byte offset')
            with part.open('ab' if part.exists() else 'xb') as stream:
                while True:
                    block = response.read(8 << 20)
                    if not block:
                        break
                    stream.write(block)
                    if stream.tell() > item['size']:
                        raise ValueError('Download exceeds expected bytes')
                stream.flush(); os.fsync(stream.fileno())
    observed = dict(expected_sha256=item['sha256'], actual_sha256=sha256_file(part),
                    expected_bytes=item['size'], actual_bytes=part.stat().st_size)
    if observed['actual_sha256'] != observed['expected_sha256'] or observed['actual_bytes'] != observed['expected_bytes']:
        raise ValueError('Downloaded payload differs: ' + json.dumps(observed, sort_keys=True))
    os.replace(part, path)
    print(json.dumps(dict(status='VERIFIED', file=str(path), sha256=item['sha256'])), flush=True)


def check_downloads(root, kind):
    records = []
    for path, item, _ in iter_downloads(root, kind):
        entry = dict(file=str(path), expected_sha256=item['sha256'], expected_bytes=item['size'],
                     actual_bytes=path.stat().st_size if path.is_file() else None,
                     actual_sha256=sha256_file(path) if path.is_file() else None)
        entry['passed'] = entry['actual_bytes'] == item['size'] and entry['actual_sha256'] == item['sha256']
        records.append(entry)
    if not records or not all(row['passed'] for row in records):
        raise ValueError('Asset payload check failed: ' + json.dumps([r for r in records if not r['passed']], sort_keys=True))
    return records


def jpeg512(raw):
    """Exact official Stage-2 preparation: EXIF/RGB, bicubic shorter512, center, q95."""
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(raw)) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        w, h = image.size
        target = (512, int(512 * h / w)) if w <= h else (int(512 * w / h), 512)
        image = image.resize(target, Image.Resampling.BICUBIC)
        left = int(round((target[0] - 512) / 2)); top = int(round((target[1] - 512) / 2))
        image = image.crop((left, top, left + 512, top + 512))
        stream = io.BytesIO(); image.save(stream, format='JPEG', quality=95)
        return stream.getvalue()


def source_pairs(source, path, raw_root):
    """Read original captions and paired images with the original source key convention."""
    path = Path(path)
    if source == 'dalle3':
        import pyarrow.parquet as pq
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                key = str(index); index += 1
                image = row['image']
                if isinstance(image, dict): image = image.get('bytes')
                if isinstance(image, str): image = base64.b64decode(image.split(',', 1)[-1])
                if not isinstance(image, (bytes, bytearray)):
                    raise ValueError('Unsupported DALL-E image encoding')
                yield key, bytes(image), row['caption']
    elif source == 'sharegpt4o':
        captions = {}
        for row in json.loads((Path(raw_root) / source / 'text_to_image.json').read_text()):
            image = row.get('output_image', '')
            stem = image.removeprefix('image/').removesuffix('.png')
            if image.startswith('image/') and image.endswith('.png') and stem.isdigit():
                captions[int(stem)] = row.get('input_prompt', '')
        seen = set()
        with tarfile.open(path, 'r|*') as stream:
            for member in stream:
                stream.members.clear()
                stem = member.name.removeprefix('image/').removesuffix('.png')
                if not member.isfile() or not member.name.endswith('.png') or not stem.isdigit(): continue
                key = int(stem)
                if key not in captions: continue
                if key in seen: raise ValueError('Duplicate ShareGPT image id within raw tar')
                seen.add(key)
                yield stem, stream.extractfile(member).read(), captions[key]
    else:
        current = None; pair = {}
        with tarfile.open(path, 'r|*') as stream:
            for member in stream:
                stream.members.clear()
                if not member.isfile() or '.' not in member.name: continue
                key, ext = member.name.rsplit('.', 1); ext = ext.lower()
                if ext not in ('txt', 'jpg', 'jpeg', 'png', 'webp'): continue
                if current is not None and key != current:
                    if not pair.get('image') or 'caption' not in pair: raise ValueError('Incomplete BLIP3o pair')
                    yield current, pair['image'], pair['caption']
                    pair = {}
                current = key; payload = stream.extractfile(member).read()
                if ext == 'txt': pair['caption'] = payload.decode('utf-8')
                else:
                    if 'image' in pair: raise ValueError('Multiple BLIP3o images for one key')
                    pair['image'] = payload
        if current is not None:
            if not pair.get('image') or 'caption' not in pair: raise ValueError('Incomplete BLIP3o pair')
            yield current, pair['image'], pair['caption']


class ImageWriter:
    def __init__(self, data_root, shard_size=1000):
        self.root = Path(data_root).resolve()
        self.directory = self.root / 'curated/images'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.total = 0; self.shard_size = shard_size; self.tar = None; self.shards = []

    def add(self, image):
        if self.total % self.shard_size == 0:
            self.close()
            self.path = self.directory / ('shard-%05d.tar' % (self.total // self.shard_size))
            self.handle = self.path.open('xb'); self.tar = tarfile.open(fileobj=self.handle, mode='w')
        offset = self.tar.offset + 512
        info = tarfile.TarInfo('%012d.jpg' % self.total); info.size = len(image); info.mtime = 0
        self.tar.addfile(info, io.BytesIO(image)); self.total += 1
        return dict(shard=str(self.path.relative_to(self.root)), image_offset=offset,
                    image_bytes=len(image), image_sha256=hashlib.sha256(image).hexdigest())

    def close(self):
        if self.tar is not None:
            self.tar.close(); self.handle.flush(); os.fsync(self.handle.fileno()); self.handle.close()
            self.shards.append(dict(shard=str(self.path.relative_to(self.root)),
                                    bytes=self.path.stat().st_size, sha256=sha256_file(self.path)))
            self.tar = None


def rebuild_curated(root):
    root = Path(root).resolve(); pins = public_pins(); selected = published_rows()
    # Qualify all raw bytes before writing prepared outputs.
    check_downloads(root, 'data')
    destination = root / 'curated'
    if destination.exists():
        raise FileExistsError('Prepared output already exists; inspect it or choose a new data directory')
    wanted = defaultdict(dict)
    for i in range(pins['population_rows']):
        source = pins['sources'][int(selected['source'][i])]
        raw = pins['raw_filenames'][int(selected['raw_file'][i])]
        key = str(int(selected['raw_key'][i]))
        if key in wanted[(source, raw)]: raise ValueError('Duplicate published source row key')
        wanted[(source, raw)][key] = i
    rows = [None] * pins['population_rows']; writer = ImageWriter(root)
    try:
        for (source, raw_file), keys in wanted.items():
            count = 0
            for key, image, caption in source_pairs(source, root / 'raw' / source / raw_file, root / 'raw'):
                if key not in keys: continue
                i = keys[key]
                if rows[i] is not None: raise ValueError('Duplicate raw row selected')
                if hashlib.sha256(caption.encode()).digest() != selected['caption_sha256'][i].tobytes():
                    raise ValueError('Original caption differs at published row ' + str(i))
                prepared = jpeg512(image)
                if hashlib.sha256(prepared).digest() != selected['image_sha256'][i].tobytes():
                    raise ValueError('Prepared JPEG differs at row ' + str(i) + '; use the pinned image library version')
                rows[i] = dict(caption=caption, source=source, raw_file=raw_file, raw_key=key,
                    t5_tokens=int(selected['t5_tokens'][i]), geneval_train=bool(selected['geneval_train'][i]),
                    **writer.add(prepared))
                count += 1
            if count != len(keys): raise ValueError('Selected raw rows are missing from ' + raw_file)
            print(json.dumps(dict(source=source, raw_file=raw_file, selected=count)), flush=True)
    finally:
        writer.close()
    if any(row is None for row in rows): raise ValueError('Incomplete prepared population')
    index = destination / 'index.jsonl'
    with index.open('x') as stream:
        for row in rows: stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    atomic_json(destination / 'manifest.json', dict(schema='bigfield-curated-v1', status='PASS',
        index_sha256=sha256_file(index), published_rows_sha256=pins['index']['sha256'],
        population_rows=len(rows), shards=writer.shards), create_once=True)
    return dict(status='PASS', rows=len(rows), index_sha256=sha256_file(index))


def bank_probe(value, row):
    """Diagnostics for one published probe row, using float64 reductions."""
    value = np.ascontiguousarray(value, dtype=np.float32)
    precise = value.astype(np.float64)
    return dict(row=int(row), sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                mean=float(precise.mean()), rms=float(np.sqrt(np.square(precise).mean())),
                minimum=float(value.min()), maximum=float(value.max()))


def build_bank(root, batch=32):
    """Recompute all five DINO branches with the same encoder used in training."""
    import jax
    from encoders import FrozenDINO
    if jax.default_backend() != 'gpu':
        raise RuntimeError('Full real feature bank extraction requires an allocated GPU')
    root = Path(root).resolve(); pins = public_pins()
    rows, binding = load_dataset(root)
    check_downloads(root, 'weights')
    encoder = FrozenDINO(root / 'weights')
    parameter_sha = encoder.parameter_digest()
    if parameter_sha != pins['bank_reference']['encoder_parameter_sha256']:
        raise ValueError('DINO converted parameter SHA differs')
    destination = root / 'bank'
    destination.mkdir(parents=True, exist_ok=False)
    features = np.lib.format.open_memmap(destination / 'real_features.npy', mode='w+',
                                        dtype=np.float32, shape=(len(rows), 5, 768))
    encode = jax.jit(encoder.pooled)
    image_hash = hashlib.sha256(); row_hash = hashlib.sha256()
    stats = dict(count=0, sum=0., sum_squares=0., minimum=float('inf'), maximum=float('-inf'))
    probe_rows = {item['row'] for item in pins['bank_reference']['probe_statistics']}
    probes = []
    try:
        for start in range(0, len(rows), batch):
            images = load_real(rows, np.arange(start, min(start + batch, len(rows))))
            value = np.asarray(jax.device_get(encode(images)), np.float32)
            if value.shape != (len(images), 5, 768) or not np.isfinite(value).all():
                raise ValueError('DINO bank produced wrong shape or nonfinite values')
            features[start:start + len(images)] = value
            for row in sorted(probe_rows.intersection(range(start, start + len(images)))):
                probes.append(bank_probe(value[row - start], row))
            image_hash.update(images.tobytes())
            for i, image in enumerate(images, start):
                row_hash.update((json.dumps(dict(row_index=i, canonical_image_sha256=hashlib.sha256(image.tobytes()).hexdigest())) + '\n').encode())
            stats['count'] += value.size
            stats['sum'] += float(value.sum(dtype=np.float64)); stats['sum_squares'] += float(np.square(value.astype(np.float64)).sum())
            stats['minimum'] = min(stats['minimum'], float(value.min())); stats['maximum'] = max(stats['maximum'], float(value.max()))
            if start % (batch * 200) == 0:
                print(json.dumps(dict(rows=start + len(images), total=len(rows))), flush=True)
        features.flush()
    finally:
        del features
    if encoder.parameter_digest() != parameter_sha:
        raise ValueError('Frozen DINO parameters changed during bank preparation')
    actual = sha256_file(destination / 'real_features.npy')
    reference = pins['bank_reference']
    images_match = image_hash.hexdigest() == reference['real_images_sha256']
    receipt = dict(schema='bigfield-bank-v1', status='PASS' if images_match else 'FAIL', population_rows=len(rows),
        published_rows_sha256=pins['index']['sha256'], data_binding=binding,
        encoder_parameter_sha256=parameter_sha, real_features_sha256=actual,
        real_images_sha256=image_hash.hexdigest(), canonical_image_rows_sha256=row_hash.hexdigest(),
        shape=[len(rows), 5, 768], dtype='float32', statistics=stats,
        statistics_scope='Whole bank; count, sum, sum_squares, minimum and maximum.',
        probe_scope=reference['probe_scope'], probe_statistics=probes,
        reference_bank_sha256=reference['real_features_sha256'],
        reference_bank_bitwise_equal=actual == reference['real_features_sha256'])
    atomic_json(destination / 'receipt.json', receipt, create_once=True)
    if not images_match:
        raise ValueError('Canonical image population differs; bank receipt retained')
    return receipt


def embed_captions(captions, device, weights_directory):
    import torch
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device=device,
                                                       cache_dir=str(weights_directory))
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model.eval()
    out = np.empty((len(captions), 768), np.float32)
    with torch.no_grad():
        for start in range(0, len(captions), BATCH):
            tokens = tokenizer(captions[start:start + BATCH]).to(device)
            features = model.encode_text(tokens).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            out[start:start + BATCH] = features.cpu().numpy()
    if not np.isfinite(out).all(): raise RuntimeError('non-finite CLIP text embeddings')
    return out


def kmeans_plus_plus(x, k, generator):
    import torch
    n = x.shape[0]
    first = int(torch.randint(n, (1,), generator=generator, device=x.device))
    centroids = [x[first]]
    closest = 2. - 2. * (x @ x[first])
    for _ in range(1, k):
        probabilities = closest.clamp(min=0) / closest.clamp(min=0).sum()
        index = int(torch.multinomial(probabilities, 1, generator=generator))
        centroids.append(x[index])
        closest = torch.minimum(closest, 2. - 2. * (x @ x[index]))
    return torch.stack(centroids)


def spherical_kmeans(embeddings, k, device, receipt):
    import torch
    x = torch.from_numpy(embeddings).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    centroids = kmeans_plus_plus(x, k, generator)
    history = []
    for iteration in range(ITERATIONS):
        assignment = (x @ centroids.T).argmax(1)
        summed = torch.zeros_like(centroids).index_add_(0, assignment, x)
        counts = torch.bincount(assignment, minlength=centroids.shape[0]).float()
        updated = summed / counts.clamp(min=1.)[:, None]
        updated[counts == 0] = centroids[counts == 0]
        updated = updated / updated.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        shift = float((updated - centroids).norm(dim=-1).max())
        centroids = updated; history.append(shift)
        if shift < 1e-5: break
    merges = 0
    while True:
        assignment = (x @ centroids.T).argmax(1)
        counts = torch.bincount(assignment, minlength=centroids.shape[0])
        keep = counts >= MIN_SIZE
        if bool(keep.all()): break
        merges += int((~keep).sum()); centroids = centroids[keep]
        if centroids.shape[0] < 2: raise RuntimeError('cluster dissolution collapsed the partition')
    assignment = (x @ centroids.T).argmax(1)
    cosine = (x * centroids[assignment]).sum(1)
    counts = torch.bincount(assignment, minlength=centroids.shape[0])
    receipt['kmeans'] = dict(requested_k=k, final_k=int(centroids.shape[0]), min_size=MIN_SIZE,
        iterations=len(history), final_shift=history[-1], dissolved_clusters=merges,
        size_min=int(counts.min()), size_median=float(counts.float().median()), size_max=int(counts.max()),
        mean_cosine_to_centroid=float(cosine.mean()), seed=SEED)
    return assignment.cpu().numpy().astype(np.int32), centroids.cpu().numpy().astype(np.float32)


def rebuild_partition(root, output):
    """A separate experiment: CUDA k-means is not bitwise reproducible."""
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Full CLIP/k-means rebuild requires an allocated GPU')
    rows, _ = load_dataset(root)
    lock = public_pins()['weights']['clip']['files'][0]
    path = Path(root) / 'weights/clip' / lock['name']
    if not _verified(path, lock): raise ValueError('CLIP weight SHA differs')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    embeddings = embed_captions([r['caption'] for r in rows], 'cuda', path.parent)
    receipt = dict(experiment_changes=True, note='This assignment is not the published measured partition.')
    ids, centroids = spherical_kmeans(embeddings, K, 'cuda', receipt)
    np.save(output / 'cluster_ids.npy', ids, allow_pickle=False)
    np.save(output / 'centroids.npy', centroids, allow_pickle=False)
    atomic_json(output / 'receipt.json', receipt, create_once=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path, help='Shared data/weight directory')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('download'); p.add_argument('--kind', choices=('weights', 'data', 'clip', 'all'), default='weights')
    p.add_argument('--list-only', action='store_true')
    p = sub.add_parser('check'); p.add_argument('--kind', choices=('weights', 'data', 'clip', 'prepared', 'bank', 'all'), default='weights')
    sub.add_parser('rebuild')
    p = sub.add_parser('dino'); p.add_argument('--batch', type=int, default=32)
    p = sub.add_parser('recluster'); p.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'download':
        items = list(iter_downloads(args.root, args.kind))
        if args.list_only:
            print(json.dumps(dict(files=len(items), bytes=sum(i['size'] for _, i, _ in items),
                destinations=[str(p) for p, _, _ in items]), indent=2)); return
        for path, item, url in items: download_file(path, item, url)
        result = dict(status='PASS', files=len(items))
    elif args.command == 'check':
        if args.kind in ('weights', 'data', 'clip', 'all'): result = dict(status='PASS', files=len(check_downloads(args.root, args.kind)))
        if args.kind in ('prepared', 'all'):
            rows, binding = load_dataset(args.root, verify_payload=True); result = dict(status='PASS', rows=len(rows), binding=binding)
        if args.kind in ('bank', 'all'):
            rows, _ = load_dataset(args.root); result = dict(status='PASS', bank=Partition(args.root, rows).identity)
    elif args.command == 'rebuild': result = rebuild_curated(args.root)
    elif args.command == 'dino':
        if args.batch < 1: parser.error('--batch must be positive')
        result = build_bank(args.root, args.batch)
    else: result = rebuild_partition(args.root, args.output)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__': main()
