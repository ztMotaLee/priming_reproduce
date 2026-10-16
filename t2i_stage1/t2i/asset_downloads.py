"""Pinned raw weights: stdlib-only download/SHA phase, no tensor loading.

Training never calls these download functions. Tiger login nodes use the
internal download-only wrapper; conversion and genuine loading happen offline
in an allocated compute process. Partial or completed outputs are never reused.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import urllib.request

from .assets import T5_SOURCE_LOCK, T5_MANIFEST_FORMAT

CONVNEXT_REVISION = '758ff0922dc09136abb55774e7f8b1e1bd0dc344'
CONVNEXT_REPO = 'facebook/convnextv2-base-22k-224'
PERCEPTUAL_RAW_LOCK = [
    {'name': 'vgg16-397923af.pth', 'size': 553433881,
     'sha256': '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0',
     'url': 'https://download.pytorch.org/models/vgg16-397923af.pth'},
    {'name': 'lpips-vgg.pth', 'size': 7289,
     'sha256': 'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868',
     'url': 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth'},
    {'name': 'convnext/config.json', 'size': 69727,
     'sha256': 'dac818f34972086f8171c444adac7752e13466b201eee2e248768597aa114732',
     'url': f'https://huggingface.co/{CONVNEXT_REPO}/resolve/{CONVNEXT_REVISION}/config.json'},
    {'name': 'convnext/model.safetensors', 'size': 354917376,
     'sha256': '8937dc20b1d7a63ea628752315a89b9a4fc0f58bf78b3299041e99c0ad3f93bb',
     'url': f'https://huggingface.co/{CONVNEXT_REPO}/resolve/{CONVNEXT_REVISION}/model.safetensors'},
]


def atomic_json(path, value, *, create=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.download-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        if create:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def t5_download_lock():
    source = T5_SOURCE_LOCK
    prefix = f'https://huggingface.co/{source["repo_id"]}/resolve/{source["revision"]}/'
    return [{**item, 'url': prefix + item['name']} for item in source['files']]


def download_file(root, item, *, opener=None):
    """One request, bounded bytes, hash while writing; retain partials on failure."""
    root = Path(root)
    target = root / item['name']
    partial = target.with_name(target.name + '.part')
    if os.path.lexists(target) or os.path.lexists(partial):
        raise FileExistsError('No retry/overwrite of an existing download: ' + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    progress = root / 'download.progress.json'
    values = {'file': item['name'], 'expected_size': item['size'],
              'expected_sha256': item['sha256'], 'actual_size': 0,
              'actual_sha256': None, 'accepted': False, 'status': 'DOWNLOADING'}
    atomic_json(progress, values)
    digest, size = hashlib.sha256(), 0
    try:
        with partial.open('xb') as stream, (opener or urllib.request.urlopen)(item['url'], timeout=60) as response:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                size += len(block); digest.update(block); stream.write(block)
                if size > item['size']:
                    break
            stream.flush(); os.fsync(stream.fileno())
        values.update(actual_size=size, actual_sha256=digest.hexdigest())
        values['accepted'] = size == item['size'] and digest.hexdigest() == item['sha256']
        values['status'] = 'VERIFIED' if values['accepted'] else 'FAILED'
        atomic_json(progress, values)
        if not values['accepted']:
            raise ValueError('Pinned raw download size/SHA differs: ' + item['name'])
        os.link(partial, target)
        partial.unlink()  # Only this invocation's successful temporary download.
        return item['sha256']
    except Exception as error:
        values.update(actual_size=size, actual_sha256=digest.hexdigest(),
                      accepted=False, status='FAILED', error=str(error))
        atomic_json(progress, values)
        raise


def download_all(asset_root):
    """Fresh destination required; a partial download is evidence, not a retry."""
    asset_root = Path(asset_root)
    if any(os.path.lexists(asset_root / kind) for kind in ('t5', 'perceptual-sources', 'perceptual')):
        raise FileExistsError('Asset output exists; use a fresh asset root, no automatic retry')
    total = sum(item['size'] for item in t5_download_lock() + PERCEPTUAL_RAW_LOCK)
    atomic_json(asset_root / 'download_budget.progress.json',
                {'download_bytes': total, 'maximum_bytes': 5_000_000_000,
                 'accepted': total < 5_000_000_000})
    if total >= 5_000_000_000:
        raise ValueError('Asset download budget must be below 5 GB')
    t5, raw = asset_root / 't5', asset_root / 'perceptual-sources'
    t5_files = {item['name']: download_file(t5, item) for item in t5_download_lock()}
    atomic_json(t5 / 'assets.sha256.json',
        {'format': T5_MANIFEST_FORMAT, 'checkpoint_format': 'flax', 'from_pt': False,
         'source': T5_SOURCE_LOCK, 'files': t5_files}, create=True)
    raw_files = {item['name']: download_file(raw, item) for item in PERCEPTUAL_RAW_LOCK}
    atomic_json(raw / 'raw.sha256.json',
                {'schema': 'priming-perceptual-raw-v1', 'source': PERCEPTUAL_RAW_LOCK,
                 'files': raw_files}, create=True)
    summary = {'status': 'RAW_DOWNLOADS_SHA_VERIFIED', 'download_bytes': total,
               't5_files': t5_files, 'perceptual_raw_files': raw_files,
               'real_loader_verified': False}
    atomic_json(asset_root / 'download_receipt.json', summary, create=True)
    return summary


def validate_raw_perceptual(raw):
    """Full offline payload check before any tensor deserialization/conversion."""
    raw = Path(raw)
    manifest = json.loads((raw / 'raw.sha256.json').read_text())
    expected = {item['name']: item['sha256'] for item in PERCEPTUAL_RAW_LOCK}
    if (manifest.get('schema') != 'priming-perceptual-raw-v1'
            or manifest.get('source') != PERCEPTUAL_RAW_LOCK or manifest.get('files') != expected):
        raise ValueError('Raw perceptual source lock differs')
    for item in PERCEPTUAL_RAW_LOCK:
        path = raw / item['name']
        values = {'file': item['name'], 'expected_size': item['size'],
                  'expected_sha256': item['sha256'], 'actual_size': path.stat().st_size,
                  'actual_sha256': sha(path)}
        values['accepted'] = (not path.is_symlink() and values['actual_size'] == item['size']
                              and values['actual_sha256'] == item['sha256'])
        atomic_json(raw / 'validation.progress.json', values)
        if not values['accepted']:
            raise ValueError('Offline raw perceptual size/SHA differs: ' + item['name'])
    return expected


@contextmanager
def offline_only():
    """Fail closed on network attempts, including unexpected library fallbacks."""
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    def denied(*args, **kwargs):
        raise RuntimeError('Network forbidden during offline asset preparation')
    originals = [(socket.socket, 'connect', socket.socket.connect),
                 (socket.socket, 'connect_ex', socket.socket.connect_ex),
                 (socket, 'create_connection', socket.create_connection),
                 (socket, 'getaddrinfo', socket.getaddrinfo)]
    for owner, name, _ in originals:
        setattr(owner, name, denied)
    try:
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
