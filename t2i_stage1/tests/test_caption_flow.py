"""Public setup/root mapping and strict caption-binding CPU regressions."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from t2i import prepare, train
from t2i.recaption import ensure_caption_receipt, validate_caption_receipt


CODE = Path(__file__).resolve().parents[1]


def prepared_root(root):
    root.mkdir(parents=True)
    shard = root / 'one.tar'
    with tarfile.open(shard, 'w') as archive:
        for name, payload in [('a.jpg', b'unchanged-image'), ('a.txt', b'llava caption')]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    prepare.atomic_json(root / 'preparation.json', dict(caption_source='caption_llava',
        source=prepare.REMOTE_LOCK['cc12m'], files={'one.tar': prepare.sha256(shard)}), create_once=True)
    return root


class CaptionFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='caption-flow-', dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / 'original'
        self.original.mkdir()
        (self.original / 'keep.txt').write_text('untouched original')
        self.data = self.root / 'data'
        self.resolved = prepared_root(self.data / 'cc12m_llava')
        self.caption = ensure_caption_receipt(self.resolved, self.data)

    def test_train_requires_receipt_and_binds_actual_sha(self):
        args = SimpleNamespace(cc12m_root=str(self.resolved), data_dir=str(self.data))
        value = train.caption_data_binding(args)
        self.assertEqual(value, {'A': self.caption})
        self.assertEqual(value['A']['caption_source'], 'caption_llava')
        self.assertEqual(value['A']['receipt_sha256'], prepare.sha256(value['A']['receipt_path']))
        args.cc12m_root = str(self.original)
        with self.assertRaisesRegex(ValueError, 'DOWNLOAD_DATA=1 bash setup.sh'):
            train.caption_data_binding(args)

    def test_exported_original_resolves_and_changed_receipt_rejects(self):
        before = {p.name: p.read_bytes() for p in self.original.iterdir()}
        prepare.record_cc12m_mapping(self.original, self.resolved, self.data)
        args = SimpleNamespace(cc12m_root=str(self.original), data_dir=str(self.data))
        self.assertEqual(train.caption_data_binding(args), {'A': self.caption})
        self.assertEqual(args.cc12m_root, str(self.resolved.resolve()))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.original.iterdir()})
        self.assertNotIn('device', json.dumps(json.loads((self.data / 'cc12m_root_mapping.json').read_text())))
        receipt = Path(self.caption['receipt_path'])
        receipt.write_text(receipt.read_text() + ' ')
        with self.assertRaises(ValueError):
            prepare.resolve_cc12m_root(self.original, self.data)

    def test_existing_preparation_needs_no_metadata_download(self):
        fresh = prepared_root(self.data / 'fresh-prepared')
        with mock.patch.object(prepare, 'download_group', side_effect=AssertionError('network')):
            actual = prepare.prepare_existing_cc12m(fresh, self.data, 1)
        self.assertEqual(Path(actual), fresh)
        self.assertEqual(validate_caption_receipt(fresh, self.data)['caption_source'], 'caption_llava')

    def test_setup_rejects_cache_inside_input_before_download(self):
        before = {p.name: p.read_bytes() for p in self.original.iterdir()}
        with mock.patch.object(prepare, 'download_group', side_effect=AssertionError('network')):
            with self.assertRaisesRegex(ValueError, 'DATA_DIR must be outside'):
                prepare.prepare_existing_cc12m(self.original, self.original / 'new-cache', 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.original.iterdir()})

    def test_setup_then_still_exported_root_and_resume_for_both_plans(self):
        # The fake environment skips dependency/asset operations and stands in
        # for datasets conversion. Root mapping, check, resolution, receipt and
        # trainer binding use their real production implementations.
        work = self.root / 'code'
        work.mkdir()
        shutil.copy2(CODE / 'setup.sh', work / 'setup.sh')
        shutil.copy2(CODE / 'train.sh', work / 'train.sh')
        fake_bin = self.root / 'bin'
        fake_bin.mkdir()
        host = fake_bin / 'hostname'
        host.write_text('#!/usr/bin/env bash\nprintf "compute-fixture\\n"\n')
        host.chmod(0o755)
        venv_bin = self.root / 'venv' / 'bin'
        venv_bin.mkdir(parents=True)
        fake_python = venv_bin / 'python'
        fake_python.write_text(f'#!{sys.executable}\n' + '''import os, sys
from t2i import prepare, train
args = sys.argv[1:]
if args[:3] == ['-m', 't2i.prepare', 'datasets']:
    prepare.record_cc12m_mapping(os.environ['CC12M_ROOT'], os.environ['FAKE_RESOLVED'], os.environ['DATA_DIR'])
elif args[:2] == ['-m', 't2i.prepare']:
    sys.argv = ['prepare', *args[2:]]
    prepare.main()
elif args[:2] == ['-m', 't2i.train']:
    parsed = train.parse_args(args[2:])
    binding = train.caption_data_binding(parsed)
    print('TRAIN_BINDING ' + __import__('json').dumps(dict(root=parsed.cc12m_root, captions=binding)))
''')
        fake_python.chmod(0o755)
        for name in ('t5', 'perceptual'):
            target = self.data / 'assets' / name
            target.mkdir(parents=True)
            (target / 'assets.sha256.json').write_text('{}')
        curated = prepared_root(self.data / 'curated')
        env = dict(os.environ, PYTHONPATH=str(CODE), PYTHON=str(fake_python), CC12M_ROOT=str(self.original),
                   DATA_DIR=str(self.data), CURATED_ROOT=str(curated), VENV_DIR=str(venv_bin.parent),
                   OUTPUT_DIR=str(self.root / 'run'), DOWNLOAD_DATA='1', DRY_RUN='0',
                   FAKE_RESOLVED=str(self.resolved), PATH=str(fake_bin) + os.pathsep + os.environ['PATH'])
        for plan in ('0', '1'):
            with self.subTest(plan=plan):
                env['CC12M_ONLY'] = plan
                result = subprocess.run(['bash', 'setup.sh'], cwd=work, env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                saved = (work / '.stage1.env').read_text()
                self.assertIn(str(self.resolved), saved)
                for command in (['bash', 'train.sh'],
                                ['bash', '-c', 'source .stage1.env; export RESUME="$OUTPUT_DIR/checkpoint_00000020"; bash train.sh']):
                    result = subprocess.run(command, cwd=work, env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    binding = json.loads(next(line.removeprefix('TRAIN_BINDING ')
                        for line in result.stdout.splitlines() if line.startswith('TRAIN_BINDING ')))
                    self.assertEqual(binding['root'], str(self.resolved))
                    self.assertEqual(binding['captions']['A'], self.caption)
                    self.assertIn('CC12M caption mapping:', result.stderr)


if __name__ == '__main__':
    unittest.main()
