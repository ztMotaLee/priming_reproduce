"""Portable exact-row curated data and the published pseudo-class partition."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from checkpoint import atomic_json, sha256_file

CODE = Path(__file__).resolve().parent
SOURCES = ('blip3o', 'dalle3', 'sharegpt4o')
WEIGHTS = {'blip3o': .06, 'dalle3': .016, 'sharegpt4o': .04}


def public_pins():
    return json.loads((CODE / 'assets/public_pins.json').read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def published_rows():
    pins = public_pins()
    path = CODE / 'assets' / pins['index']['file']
    if sha256_file(path) != pins['index']['sha256']:
        raise ValueError('Published row/cluster payload SHA differs')
    with np.load(path, allow_pickle=False) as value:
        arrays = {name: value[name].copy() for name in value.files}
    if len(arrays['source']) != pins['population_rows']:
        raise ValueError('Published row population differs')
    return arrays


def row_identity(row):
    return dict(source=row['source'], raw_file=row['raw_file'], raw_key=str(row['raw_key']),
                caption_sha256=hashlib.sha256(row['caption'].encode()).hexdigest(),
                image_sha256=row['image_sha256'], t5_tokens=int(row['t5_tokens']),
                geneval_train=bool(row['geneval_train']))


def validate_rows(rows, arrays=None):
    pins = public_pins(); arrays = published_rows() if arrays is None else arrays
    if len(rows) != pins['population_rows']:
        raise ValueError('Curated population must contain exactly 110230 rows')
    for i, row in enumerate(rows):
        expected = dict(source=pins['sources'][int(arrays['source'][i])],
            raw_file=pins['raw_filenames'][int(arrays['raw_file'][i])],
            raw_key=str(int(arrays['raw_key'][i])),
            caption_sha256=arrays['caption_sha256'][i].tobytes().hex(),
            image_sha256=arrays['image_sha256'][i].tobytes().hex(),
            t5_tokens=int(arrays['t5_tokens'][i]), geneval_train=bool(arrays['geneval_train'][i]))
        if row_identity(row) != expected:
            raise ValueError('Curated row identity/order differs at row ' + str(i))
    return True


class DatasetRows(list):
    def __init__(self, rows, *, source='curated', weights=None, data_root=None):
        super().__init__(rows)
        self.source = source
        self.weights = dict(weights or WEIGHTS)
        self.data_root = None if data_root is None else Path(data_root).resolve()
        self.source_indices = {}
        seen = set()
        for index, row in enumerate(self):
            caption = row['caption']
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError('Empty or non-string caption')
            if caption in seen:
                continue
            seen.add(caption)
            self.source_indices.setdefault(row.get('source', 'curated'), []).append(index)
        if len(seen) < 64:
            raise ValueError('At least 64 distinct literal captions required')
        if any(not self.source_indices.get(name) for name in self.weights):
            raise ValueError('Positive-weight source has no distinct captions')


def sample_caption_indices(rows, step, seed=20261022):
    """64 unique captions; exact categorical-source then uniform-within-source stream."""
    if not isinstance(rows, DatasetRows):
        rows = DatasetRows(rows)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(step), 64512])))
    sources = tuple(rows.weights)
    weights = np.asarray([rows.weights[k] for k in sources], np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Nonfinite or invalid frozen source weights')
    weights /= weights.sum()
    chosen = []; seen = set()
    pools = {name: np.asarray(rows.source_indices[name], np.int64) for name in sources}
    remaining = {name: len(pools[name]) for name in sources}
    while len(chosen) < 64:
        current = weights * np.asarray([remaining[name] > 0 for name in sources])
        current /= current.sum()
        name = sources[int(rng.choice(len(sources), p=current))]
        pool = pools[name]
        index = int(pool[int(rng.integers(len(pool)))])
        if rows[index]['caption'] in seen:
            continue
        seen.add(rows[index]['caption']); chosen.append(index); remaining[name] -= 1
    return np.asarray(chosen, np.int64)


def _relative(root, name):
    root = Path(root).resolve(); path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Data payload path escapes its data directory')
    return path


def load_dataset(data_root, verify_payload=False):
    root = Path(data_root).resolve(); prepared = root / 'curated'
    manifest_path = prepared / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    index_path = prepared / 'index.jsonl'
    if manifest.get('status') != 'PASS' or manifest.get('index_sha256') != sha256_file(index_path):
        raise ValueError('Prepared dataset manifest/index SHA differs')
    pins = public_pins()
    if manifest.get('published_rows_sha256') != pins['index']['sha256']:
        raise ValueError('Prepared dataset uses another population/partition')
    rows = [json.loads(line) for line in index_path.open()]
    validate_rows(rows)
    for item in manifest['shards']:
        path = _relative(root, item['shard'])
        if path.stat().st_size != item['bytes'] or (verify_payload and sha256_file(path) != item['sha256']):
            raise ValueError('Prepared image shard SHA/size differs: ' + item['shard'])
    binding = dict(schema='bigfield-data-v1', published_rows_sha256=pins['index']['sha256'],
                   index_sha256=manifest['index_sha256'], manifest_sha256=sha256_file(manifest_path),
                   source_counts=pins['index']['source_counts'], weights=WEIGHTS)
    binding['binding_sha256'] = digest(binding)
    return DatasetRows(rows, data_root=root), binding


def canonical_image(image):
    from PIL import ImageOps
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512), transforms.ToTensor(), transforms.Normalize([.5], [.5])])
    return np.asarray(transform(ImageOps.exif_transpose(image).convert('RGB'))).transpose(1, 2, 0)


def load_real(rows, ids, data_root=None):
    from PIL import Image
    root = data_root if data_root is not None else getattr(rows, 'data_root', None)
    if root is None:
        raise ValueError('A prepared data directory is required')
    batch = []
    for i in ids:
        row = rows[int(i)]; path = _relative(root, row['shard'])
        offset, count = int(row['image_offset']), int(row['image_bytes'])
        if offset < 0 or count <= 0 or offset + count > path.stat().st_size:
            raise ValueError('Prepared JPEG bounds differ')
        with path.open('rb') as stream:
            stream.seek(offset); payload = stream.read(count)
        if len(payload) != count or hashlib.sha256(payload).hexdigest() != row['image_sha256']:
            raise ValueError('Prepared JPEG SHA differs')
        with Image.open(io.BytesIO(payload)) as image:
            batch.append(canonical_image(image))
    return np.stack(batch)


class Partition:
    """Load the complete real feature bank into RAM, preserving published row order."""
    def __init__(self, data_root, rows, verify_bank=True):
        from field import caption_probabilities
        pins = public_pins(); arrays = published_rows()
        self.cluster_ids = arrays['cluster_ids']
        self.population_rows = pins['population_rows']
        self.clusters = int(self.cluster_ids.max()) + 1
        order = np.argsort(self.cluster_ids, kind='stable')
        boundaries = np.searchsorted(self.cluster_ids[order], np.arange(self.clusters + 1))
        self.members = [order[boundaries[c]:boundaries[c + 1]].astype(np.int64) for c in range(self.clusters)]
        self.sizes = np.asarray([len(m) for m in self.members], np.int64)
        if (self.cluster_ids.dtype != np.int32 or self.clusters < 64 or self.sizes.min() < 32
                or not np.array_equal(np.unique(self.cluster_ids), np.arange(self.clusters))):
            raise ValueError('Published cluster shape/contiguity/minimum-size differs')
        validate_rows(rows, arrays)
        self.probabilities = caption_probabilities(rows)
        directory = Path(data_root) / 'bank'
        receipt = json.loads((directory / 'receipt.json').read_text())
        path = directory / 'real_features.npy'
        if (receipt.get('status') != 'PASS' or receipt.get('published_rows_sha256') != pins['index']['sha256']
                or receipt.get('encoder_parameter_sha256') != pins['bank_reference']['encoder_parameter_sha256']
                or (verify_bank and sha256_file(path) != receipt['real_features_sha256'])):
            raise ValueError('Real feature bank binding differs')
        self.real_features = np.load(path, allow_pickle=False)
        if (self.real_features.shape != (self.population_rows, 5, 768)
                or self.real_features.dtype != np.float32 or not np.isfinite(self.real_features).all()):
            raise ValueError('Real feature bank shape/dtype/finiteness differs')
        self.identity = dict(published_rows_sha256=pins['index']['sha256'],
            real_features_sha256=receipt['real_features_sha256'],
            encoder_parameter_sha256=receipt['encoder_parameter_sha256'],
            clusters=self.clusters, population_rows=self.population_rows, size_min=int(self.sizes.min()),
            size_max=int(self.sizes.max()), bank_loading='whole_array_in_memory')
