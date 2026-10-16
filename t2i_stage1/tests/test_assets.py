"""Tiny T5 binding tests; no real model payload, download or JAX import.

The happy-path fixtures contain deliberately tiny placeholder bytes. Only the
payload-size and payload-hash I/O are mocked to the fixed official values; the
production lock and validator are real. These fixtures are never model assets.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from t2i import assets, text


class T5AssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="t5-binding-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.lock = {entry["name"]: entry for entry in assets.T5_SOURCE_LOCK["files"]}
        for name in self.lock:
            (self.root / name).write_bytes(b"tiny unit fixture, never model weights\n")
        self.value = {
            "format": assets.T5_MANIFEST_FORMAT, "checkpoint_format": "flax", "from_pt": False,
            "source": copy.deepcopy(assets.T5_SOURCE_LOCK),
            "files": {name: entry["sha256"] for name, entry in self.lock.items()},
        }
        self.write_manifest()

    def write_manifest(self):
        (self.root / "assets.sha256.json").write_text(json.dumps(self.value))

    @contextmanager
    def pinned_payload_io(self, *, sizes=None, digests=None):
        sizes, digests = sizes or {}, digests or {}
        real_sha256 = assets._sha256

        def size(path):
            return sizes.get(path.name, self.lock[path.name]["size"])

        def digest(path):
            if path.name in self.lock:
                return digests.get(path.name, self.lock[path.name]["sha256"])
            return real_sha256(path)

        with mock.patch.object(assets, "_file_size", side_effect=size) as stat, \
             mock.patch.object(assets, "_sha256", side_effect=digest) as sha:
            yield stat, sha

    def test_full_pinned_flax_binding_verifies_every_payload(self):
        with self.pinned_payload_io() as (stat, sha):
            binding = assets.validate_t5_assets(self.root)
        self.assertEqual(binding["status"], "T5_PAYLOADS_VERIFIED")
        self.assertTrue(binding["payloads_verified"])
        self.assertFalse(binding["from_pt"])
        self.assertEqual(binding["checkpoint_format"], "flax")
        self.assertEqual(set(binding["sha256_verified_files"]), set(self.lock))
        self.assertEqual(stat.call_count, len(self.lock))
        self.assertEqual({call.args[0].name for call in sha.call_args_list},
                         set(self.lock) | {"assets.sha256.json"})
        self.assertEqual(binding["manifest_sha256"], hashlib.sha256(
            (self.root / "assets.sha256.json").read_bytes()).hexdigest())

    def test_metadata_mode_skips_only_full_weight_hash(self):
        with self.pinned_payload_io() as (stat, sha):
            binding = assets.validate_t5_assets(self.root, verify_payloads=False)
        self.assertEqual(binding["status"], "T5_METADATA_VERIFIED")
        self.assertFalse(binding["payloads_verified"])
        self.assertEqual(stat.call_count, len(self.lock))
        self.assertEqual(set(binding["sha256_verified_files"]), set(self.lock) - {assets.T5_WEIGHT_FILE})
        self.assertNotIn(assets.T5_WEIGHT_FILE, {call.args[0].name for call in sha.call_args_list})

    def test_wrong_official_origin_or_revision_is_rejected_before_payload_io(self):
        for field, wrong in (("repo_id", "untrusted/flan-t5-large"),
                             ("revision", "0" * 40), ("repo_type", "dataset")):
            with self.subTest(field=field), self.pinned_payload_io() as (stat, sha):
                original = self.value["source"][field]
                self.value["source"][field] = wrong
                self.write_manifest()
                with self.assertRaisesRegex(assets.AssetValidationError, "source differs"):
                    assets.validate_t5_assets(self.root)
                self.value["source"][field] = original
                stat.assert_not_called()
                sha.assert_not_called()

    def test_config_only_self_consistent_manifest_is_rejected(self):
        digest = hashlib.sha256((self.root / "config.json").read_bytes()).hexdigest()
        self.value["files"] = {"config.json": digest}
        self.value["source"]["files"] = [{"name": "config.json", "size": 38, "sha256": digest}]
        self.write_manifest()
        with self.assertRaises(assets.AssetValidationError):
            assets.validate_t5_assets(self.root)

    def test_omitted_weight_manifest_entry_is_rejected_even_if_file_exists(self):
        del self.value["files"][assets.T5_WEIGHT_FILE]
        self.write_manifest()
        with self.assertRaisesRegex(assets.AssetValidationError, "exactly every pinned"):
            assets.validate_t5_assets(self.root)

    def test_listed_but_absent_weight_is_rejected(self):
        (self.root / assets.T5_WEIGHT_FILE).unlink()
        with self.pinned_payload_io():
            with self.assertRaisesRegex(assets.AssetValidationError, "Missing regular pinned"):
                assets.validate_t5_assets(self.root, verify_payloads=False)

    def test_extra_manifest_file_is_rejected(self):
        self.value["files"]["model.safetensors"] = "0" * 64
        self.write_manifest()
        with self.assertRaisesRegex(assets.AssetValidationError, "exactly every pinned"):
            assets.validate_t5_assets(self.root)

    def test_unlisted_local_weight_or_index_is_rejected(self):
        for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
                     "flax_model-00001-of-00002.msgpack"):
            with self.subTest(name=name):
                target = self.root / name
                target.write_bytes(b"unlisted tiny fixture")
                with self.assertRaisesRegex(assets.AssetValidationError, "Unlisted T5 weight"):
                    assets.validate_t5_assets(self.root)
                target.unlink()

    def test_unsupported_format_and_from_pt_fail_closed(self):
        for checkpoint_format, from_pt in (("safetensors", True), ("pytorch", True), ("flax", True)):
            with self.subTest(checkpoint_format=checkpoint_format, from_pt=from_pt):
                self.value.update(checkpoint_format=checkpoint_format, from_pt=from_pt)
                self.write_manifest()
                with self.assertRaisesRegex(assets.AssetValidationError, "from_pt is unsupported"):
                    assets.validate_t5_assets(self.root)

    def test_real_tiny_weight_cannot_pass_expected_size_check(self):
        with self.assertRaisesRegex(assets.AssetValidationError, "size differs") as caught:
            assets.validate_t5_assets(self.root)
        self.assertEqual(caught.exception.values["actual_size"],
                         (self.root / "config.json").stat().st_size)
        self.assertEqual(caught.exception.values["expected_size"], self.lock["config.json"]["size"])

    def test_truncated_weight_deciding_sizes_survive_in_error(self):
        with self.pinned_payload_io(sizes={assets.T5_WEIGHT_FILE: 123}):
            with self.assertRaises(assets.AssetValidationError) as caught:
                assets.validate_t5_assets(self.root, verify_payloads=False)
        self.assertEqual(caught.exception.values, {
            "file": assets.T5_WEIGHT_FILE, "expected_size": 3132624407, "actual_size": 123})

    def test_weight_hash_is_required_in_full_mode(self):
        with self.pinned_payload_io(digests={assets.T5_WEIGHT_FILE: "0" * 64}):
            with self.assertRaisesRegex(assets.AssetValidationError, "SHA256 differs") as caught:
                assets.validate_t5_assets(self.root)
        self.assertEqual(caught.exception.values["file"], assets.T5_WEIGHT_FILE)

    def test_metadata_mode_still_hashes_small_config_and_tokenizer(self):
        for name in ("config.json", "tokenizer.json", "spiece.model"):
            with self.subTest(name=name), self.pinned_payload_io(digests={name: "0" * 64}):
                with self.assertRaisesRegex(assets.AssetValidationError, "SHA256 differs") as caught:
                    assets.validate_t5_assets(self.root, verify_payloads=False)
                self.assertEqual(caught.exception.values["file"], name)

    def test_wrong_declared_digest_is_rejected(self):
        self.value["files"][assets.T5_WEIGHT_FILE] = "0" * 64
        self.write_manifest()
        with self.assertRaisesRegex(assets.AssetValidationError, "exactly every pinned"):
            assets.validate_t5_assets(self.root)

    def test_missing_explicit_format_is_rejected(self):
        del self.value["checkpoint_format"]
        self.write_manifest()
        with self.assertRaisesRegex(assets.AssetValidationError, "fields differ"):
            assets.validate_t5_assets(self.root)

    def test_loader_rejects_wrong_origin_before_transformers_import(self):
        self.value["source"]["repo_id"] = "same-shape/random-t5"
        self.write_manifest()
        with mock.patch.dict(sys.modules, {"transformers": None}):
            with self.assertRaisesRegex(assets.AssetValidationError, "source differs"):
                text.load_text_encoder(self.root)

    def test_loader_requires_full_validation_and_disables_from_pt(self):
        fake_jax = ModuleType("jax")
        fake_numpy = ModuleType("jax.numpy")
        fake_numpy.bfloat16 = object()
        fake_jax.numpy = fake_numpy
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=mock.Mock(return_value=object()))
        model = SimpleNamespace(config=SimpleNamespace(d_model=1024, num_layers=24), params={})
        fake_transformers.FlaxT5EncoderModel = SimpleNamespace(from_pretrained=mock.Mock(return_value=model))
        with mock.patch.dict(sys.modules, {"jax": fake_jax, "jax.numpy": fake_numpy,
                                           "transformers": fake_transformers}), \
             mock.patch.object(text, "validate_t5_assets", wraps=assets.validate_t5_assets) as validate, \
             self.pinned_payload_io() as (_, sha):
            _, params, _ = text.load_text_encoder(self.root)
        self.assertEqual(params, {})
        validate.assert_called_once_with(self.root, verify_payloads=True)
        self.assertIn(assets.T5_WEIGHT_FILE, {call.args[0].name for call in sha.call_args_list})
        fake_transformers.FlaxT5EncoderModel.from_pretrained.assert_called_once_with(
            str(self.root), from_pt=False, dtype=fake_numpy.bfloat16, local_files_only=True)


if __name__ == "__main__":
    unittest.main()
