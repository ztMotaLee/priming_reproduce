"""Tiny stdlib raw-download regression tests; never downloads real weights."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

from t2i import asset_downloads as module


class RawDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='raw-assets-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = b'unit fixture, never production weights'
        self.item = {'name': 'weight', 'size': len(self.payload),
                     'sha256': hashlib.sha256(self.payload).hexdigest(),
                     'url': 'https://example.invalid/unit'}

    def test_streamed_sha_and_create_once_no_overwrite(self):
        opener = mock.Mock(return_value=io.BytesIO(self.payload))
        self.assertEqual(module.download_file(self.root, self.item, opener=opener), self.item['sha256'])
        self.assertEqual((self.root / 'weight').read_bytes(), self.payload)
        self.assertTrue(json.loads((self.root / 'download.progress.json').read_text())['accepted'])
        with self.assertRaises(FileExistsError):
            module.download_file(self.root, self.item, opener=opener)
        self.assertEqual(opener.call_count, 1)

    def test_bad_size_or_hash_retains_exact_gate_and_partial_without_retry(self):
        for name, payload in [('short', b'x'), ('hash', b'x' * len(self.payload)),
                              ('oversize', self.payload + b'extra')]:
            with self.subTest(name=name):
                root = self.root / name
                opener = mock.Mock(return_value=io.BytesIO(payload))
                with self.assertRaises(ValueError):
                    module.download_file(root, self.item, opener=opener)
                value = json.loads((root / 'download.progress.json').read_text())
                self.assertEqual(value['actual_size'], len(payload))
                self.assertEqual(value['actual_sha256'], hashlib.sha256(payload).hexdigest())
                self.assertFalse(value['accepted'])
                self.assertFalse((root / 'weight').exists())
                self.assertEqual((root / 'weight.part').read_bytes(), payload)
                with self.assertRaises(FileExistsError):
                    module.download_file(root, self.item, opener=opener)
                self.assertEqual(opener.call_count, 1)

    def test_network_failure_is_audited_no_retry(self):
        opener = mock.Mock(side_effect=OSError('unit network failure'))
        with self.assertRaises(OSError):
            module.download_file(self.root, self.item, opener=opener)
        value = json.loads((self.root / 'download.progress.json').read_text())
        self.assertEqual(value['actual_size'], 0)
        self.assertEqual(value['status'], 'FAILED')
        self.assertEqual(opener.call_count, 1)

    def test_raw_full_sha_validation_precedes_deserialization(self):
        item = copy.deepcopy(self.item)
        (self.root / 'weight').write_bytes(self.payload)
        module.atomic_json(self.root / 'raw.sha256.json',
            {'schema': 'priming-perceptual-raw-v1', 'source': [item],
             'files': {'weight': item['sha256']}}, create=True)
        with mock.patch.object(module, 'PERCEPTUAL_RAW_LOCK', [item]):
            self.assertEqual(module.validate_raw_perceptual(self.root), {'weight': item['sha256']})
            (self.root / 'weight').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                module.validate_raw_perceptual(self.root)
        value = json.loads((self.root / 'validation.progress.json').read_text())
        self.assertEqual(value['actual_sha256'], hashlib.sha256(b'bad').hexdigest())
        self.assertFalse(value['accepted'])

    def test_offline_guard_blocks_network_and_restores_socket_methods(self):
        original = socket.create_connection
        with module.offline_only():
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
            self.assertEqual(os.environ['TRANSFORMERS_OFFLINE'], '1')
            with self.assertRaisesRegex(RuntimeError, 'Network forbidden'):
                socket.create_connection(('example.invalid', 443))
            with self.assertRaisesRegex(RuntimeError, 'Network forbidden'):
                socket.getaddrinfo('example.invalid', 443)
            with socket.socket() as connection:
                with self.assertRaisesRegex(RuntimeError, 'Network forbidden'):
                    connection.connect(('127.0.0.1', 9))
        self.assertIs(socket.create_connection, original)

    def test_full_asset_budget_is_below_5gb(self):
        entries = module.t5_download_lock() + module.PERCEPTUAL_RAW_LOCK
        self.assertLess(sum(item['size'] for item in entries), 5_000_000_000)
        self.assertEqual(len(entries), 11)
        self.assertTrue(all(len(item['sha256']) == 64 for item in entries))


if __name__ == '__main__':
    unittest.main()
