#!/usr/bin/env python3
"""Prepare SHA-locked evaluation sources, prompts, and original model weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from urllib.request import Request, urlopen

PINS = Path(__file__).with_name('pins.json')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected, size=None):
    path = Path(path)
    if (not path.is_file() or (size is not None and path.stat().st_size != size)
            or sha256(path) != expected):
        raise ValueError('Missing or different SHA/size: ' + str(path))
    return path


def download(row, root):
    """Explicit invocation resumes only its .partial file; never replace a final file."""
    path = root / row['path']
    if path.exists():
        return verify(path, row['sha256'], row['bytes'])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > row['bytes']:
        raise ValueError('Oversized partial download: ' + str(partial))
    if offset < row['bytes']:
        request = Request(row['url'], headers={'Range': f'bytes={offset}-'} if offset else {})
        with urlopen(request, timeout=120) as response:
            if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                raise ValueError('Server cannot resume this file; preserve partial and report')
            with partial.open('ab' if partial.exists() else 'xb') as stream:
                for block in iter(lambda: response.read(8 << 20), b''):
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
    verify(partial, row['sha256'], row['bytes'])
    os.replace(partial, path)
    return path


def rebuild_dpg_prompts(archive, output, expected):
    """Canonical sorted upstream .txt, strip(), UTF-8 JSON lines, no ASCII escaping."""
    rows = []
    with tarfile.open(archive, 'r:gz') as members:
        for member in members:
            pieces = Path(member.name).parts
            if len(pieces) != 4 or pieces[1:3] != ('dpg_bench', 'prompts') or not pieces[3].endswith('.txt'):
                continue
            if not member.isfile() or member.size > (1 << 20):
                raise ValueError('Unexpected prompt archive member')
            stream = members.extractfile(member)
            rows.append((pieces[3], stream.read().decode('utf-8').strip()))
    if len(rows) != 1065 or len({name for name, _ in rows}) != 1065:
        raise ValueError('Prompt archive must contain exactly 1065 unique text files')
    raw = ''.join(json.dumps(dict(key=Path(name).stem, prompt=text), ensure_ascii=False) + '\n'
                  for name, text in sorted(rows)).encode()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Rebuilt DPG prompt order/content SHA differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(raw)


def prepare(root, scope, *, fetch=False, verify_only=False):
    pins = json.loads(PINS.read_text())
    rows = [row for row in pins['files'] if scope == 'all' or row['scope'] == scope]
    plan = dict(root=str(root), scope=scope, files=len(rows), bytes=sum(row['bytes'] for row in rows),
                network_requested=fetch, paths=[row['path'] for row in rows])
    print(json.dumps(plan, indent=2))
    if not fetch and not verify_only:
        return plan
    for row in rows:
        if fetch:
            download(row, root)
        else:
            verify(root / row['path'], row['sha256'], row['bytes'])
    if scope in ('all', 'dpg'):
        output = root / 'benchmarks/dpg/prompts.jsonl'
        if output.exists() or not fetch:
            verify(output, pins['dpg_prompts_sha256'])
        else:
            archive = root / 'sources/ella-prompts.tar.gz'
            # The generated 1065-row payload SHA verifies all selected archive content.
            # This archive is small source data; no model is loaded here.
            if not archive.exists():
                temporary = archive.with_name(archive.name + '.partial')
                offset = temporary.stat().st_size if temporary.exists() else 0
                request = Request(pins['dpg_prompt_archive_url'], headers={'Range': f'bytes={offset}-'} if offset else {})
                with urlopen(request, timeout=120) as source:
                    if offset and (source.status != 206 or not source.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise ValueError('Archive server cannot resume; preserve partial and report')
                    with temporary.open('ab' if temporary.exists() else 'xb') as target:
                        for block in iter(lambda: source.read(1 << 20), b''):
                            target.write(block)
                os.replace(temporary, archive)
            rebuild_dpg_prompts(archive, output, pins['dpg_prompts_sha256'])
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scope', choices=('geneval', 'dpg', 'all'), default='all')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--download', action='store_true')
    action.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    prepare(args.root.resolve(), args.scope, fetch=args.download, verify_only=args.verify)


if __name__ == '__main__':
    main()
