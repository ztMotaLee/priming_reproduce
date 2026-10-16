"""Small, dependency-free binding checks for prepared Priming features."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_perceptual_assets(path, *, verify_payloads=True):
    path = Path(path).resolve()
    binding_path = path / 'assets.sha256.json'
    binding = json.loads(binding_path.read_text())
    required = {'weights.npz', 'manifest.json'}
    if set(binding.get('files', {})) != required:
        raise ValueError('Perceptual binding must cover weights.npz and manifest.json')
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = manifest.get('sources', {})
    if (manifest.get('format') != 'priming-perceptual-v1'
            or sources.get('convnext_id') != 'facebook/convnextv2-base-22k-224'
            or sources.get('convnext_revision') != '758ff0922dc09136abb55774e7f8b1e1bd0dc344'
            or sources.get('lpips_md5') != 'd507d7349b931f0638a25a48a722f98a'
            or manifest.get('depths') != [3, 3, 27, 3]
            or manifest.get('hidden_sizes') != [128, 256, 512, 1024]
            or binding['files']['weights.npz'] != manifest.get('sha256')):
        raise ValueError('Perceptual recipe/source binding differs')
    for name, digest in binding['files'].items():
        item = path / name
        if (not item.is_file() or item.is_symlink()
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Missing/invalid perceptual payload: ' + name)
        if (verify_payloads or name == 'manifest.json') and sha(item) != digest:
            raise ValueError('Perceptual payload SHA differs: ' + name)
    return {'path': str(path), 'manifest': str(binding_path),
            'manifest_sha256': sha(binding_path), 'files': binding['files'],
            'payloads_verified': bool(verify_payloads),
            'status': 'PERCEPTUAL_PAYLOADS_VERIFIED' if verify_payloads else 'PERCEPTUAL_METADATA_VERIFIED'}
