"""Create or verify the portable source manifest; never loads model assets."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def source_paths():
    skipped = {'__pycache__', '.pytest_cache', '.venv', '.git'}
    files = []
    for directory, subdirectories, names in os.walk(HERE):
        subdirectories[:] = [name for name in subdirectories if not name.startswith('.') and name not in skipped]
        files.extend(Path(directory) / name for name in names
                     if not name.startswith('.') and name != 'sources.sha256' and not name.endswith('.pyc'))
    return sorted(files)


def check():
    manifest = HERE / 'sources.sha256'
    expected = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        if len(digest) != 64 or name in expected:
            raise ValueError('Invalid source manifest')
        expected[name] = digest
    actual = {p.relative_to(HERE).as_posix(): sha256(p) for p in source_paths()}
    if expected != actual:
        raise ValueError('Source manifest differs: ' + ', '.join(sorted(
            name for name in expected.keys() | actual.keys() if expected.get(name) != actual.get(name))))
    return sha256(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.check:
        target = HERE / 'sources.sha256'
        temporary = HERE / '.sources.sha256.pending'
        with temporary.open('x') as stream:
            stream.write(''.join(sha256(p) + '  ' + p.relative_to(HERE).as_posix() + '\n' for p in source_paths()))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    print('SOURCE_MANIFEST_OK ' + check())


if __name__ == '__main__':
    main()
