"""Tiny CPU fixtures for exact-URL, byte-preserving CC12M recaptioning."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
from unittest import mock

from t2i import recaption as rc


class RecaptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cc12m-recaption-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "cache"

    def metadata(self, rows):
        path = self.root / "metadata.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        self.expected = {"size": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.meta = path
        return path

    def row(self, key="a", *, url=None, caption=None, llava=None):
        return {"url": "https://example.invalid/" + key if url is None else url,
                "caption": "old " + key if caption is None else caption,
                "caption_llava": "LLaVA " + key if llava is None else llava,
                "status": "success"}

    def shard(self, records, *, name="00000.tar", use_json=True, parquet=False):
        root = self.root / "input"
        root.mkdir(exist_ok=True)
        path = root / name
        rows = []
        with tarfile.open(path, "w") as archive:
            for key, row in records:
                members = [(key + ".jpg", b"JPEG original bytes " + key.encode()),
                           (key + ".txt", row["caption"].encode())]
                if use_json:
                    members.append((key + ".json", json.dumps({"url": row["url"],
                                                               "caption": row["caption"]}).encode()))
                rows.append({"key": key, "url": row["url"]})
                for member_name, raw in members:
                    member = tarfile.TarInfo(member_name)
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
        if parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
            pq.write_table(pa.Table.from_pylist(rows), path.with_suffix(".parquet"))
        return root

    def run_recaption(self, source, **kwargs):
        return rc.recaption(source, self.root / "output", metadata=self.meta,
                            data_dir=self.data, expected=self.expected, **kwargs)

    def members(self, path):
        with tarfile.open(path) as archive:
            return [(member.name, archive.extractfile(member).read()) for member in archive if member.isfile()]

    def test_json_exact_url_image_bytes_order_idempotence(self):
        rows = [self.row("b"), self.row("a")]
        self.metadata(rows)
        source = self.shard(list(zip(["b", "a"], rows)))
        before = self.members(source / "00000.tar")
        result = self.run_recaption(source)
        after = self.members(self.root / "output" / "00000.tar")
        self.assertEqual([name for name, _ in before], [name for name, _ in after])
        for (name, old), (_, new) in zip(before, after):
            self.assertEqual(new, b"LLaVA " + name[0].encode() if name.endswith(".txt") else old)
        self.assertEqual((result["samples"], result["input_samples"], result["match_rate"]), (2, 2, 1.0))
        binding = rc.validate_caption_receipt(self.root / "output")
        self.assertEqual(binding["caption_source"], "caption_llava")
        identities = rc._snapshot(self.root / "output")
        self.assertEqual(self.run_recaption(source), result)
        self.assertEqual(rc._snapshot(self.root / "output"), identities)
        self.assertEqual(self.members(source / "00000.tar"), before)

    def test_parquet_fallback(self):
        row = self.row()
        self.metadata([row])
        source = self.shard([("a", row)], use_json=False, parquet=True)
        self.assertEqual(self.run_recaption(source)["samples"], 1)
        self.assertIn(("a.txt", b"LLaVA a"), self.members(self.root / "output" / "00000.tar"))

    def test_parquet_url_sidecar_identity_is_bound_on_resume(self):
        row = self.row()
        self.metadata([row])
        source = self.shard([("a", row)], use_json=False, parquet=True)
        result = self.run_recaption(source)
        self.assertEqual(result["input_url_sidecars"], [rc.shard_identity(source / "00000.parquet")])
        (source / "00000.parquet").write_bytes(b"changed sidecar")
        with self.assertRaisesRegex(ValueError, "input/metadata differs"):
            self.run_recaption(source)

    def test_json_url_precedes_parquet(self):
        row = self.row()
        self.metadata([row])
        source = self.shard([("a", row)], parquet=True)
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist([{"key": "a", "url": "wrong"}]), source / "00000.parquet")
        self.assertEqual(self.run_recaption(source)["samples"], 1)

    def test_missing_url_fails_shard_and_no_completion(self):
        row = self.row()
        self.metadata([row])
        source = self.shard([("a", row)], use_json=False)
        with self.assertRaisesRegex(ValueError, "No URL"):
            self.run_recaption(source)
        output = self.root / "output"
        self.assertFalse((output / "00000.tar").exists())
        self.assertFalse((output / "preparation.json").exists())
        progress = json.loads((output / ".recaption_shards" / "00000.tar.progress.json").read_text())
        self.assertEqual(progress["counts"]["input"], 1)
        self.assertEqual(progress["status"], "FAILED")

    def test_duplicates_disambiguate_original_identical_and_ambiguous(self):
        base = self.row()
        self.metadata([base, dict(base), dict(base, caption="other", caption_llava="other llava"),
                       self.row("b"), self.row("b", caption="alternate"),
                       self.row("c", caption="left", llava="one"), self.row("c", caption="right", llava="two")])
        source = self.shard([("a", base), ("b", self.row("b")), ("c", self.row("c"))])
        result = self.run_recaption(source)
        counts = result["shards"][0]["counts"]
        self.assertEqual((counts["matched"], counts["matched_disambiguated"], counts["ambiguous_url"]), (2, 1, 1))

    def test_disambiguation_accepts_txt_when_json_caption_differs(self):
        row = self.row()
        self.metadata([row, dict(row, caption="other old", caption_llava="other llava")])
        source = self.shard([("a", row)])
        path = source / "00000.tar"
        members = self.members(path)
        with tarfile.open(path, "w") as archive:
            for name, raw in members:
                if name.endswith(".json"):
                    raw = json.dumps({"url": row["url"], "caption": "unrelated JSON caption"}).encode()
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        result = self.run_recaption(source)
        self.assertEqual(result["shards"][0]["counts"]["matched_disambiguated"], 1)
        self.assertIn(("a.txt", b"LLaVA a"), self.members(self.root / "output" / "00000.tar"))

    def test_unmatched_exact_url_drops_no_text_guess(self):
        rows = [self.row("a"), self.row("b")]
        self.metadata(rows)
        source = self.shard([("a", rows[0]), ("b", dict(rows[1], url=rows[1]["url"] + "/"))])
        result = self.run_recaption(source)
        self.assertEqual(result["match_rate"], 0.5)
        self.assertEqual(result["shards"][0]["counts"]["unmatched_url"], 1)
        self.assertEqual([name for name, _ in self.members(self.root / "output" / "00000.tar")],
                         ["a.jpg", "a.txt", "a.json"])

    def test_match_gate_audits_before_failure(self):
        self.metadata([self.row()])
        source = self.shard([("a", self.row("x")), ("b", self.row("y"))])
        with self.assertRaisesRegex(ValueError, "below 0.5"):
            self.run_recaption(source)
        progress = json.loads((self.root / "output" / "recaption_progress.json").read_text())
        self.assertEqual((progress["preparation"]["samples"], progress["preparation"]["input_samples"],
                          progress["preparation"]["match_rate"]), (0, 2, 0.0))
        self.assertFalse((self.root / "output" / "caption_receipt.json").exists())

    def test_space_gate_audits_before_index_or_tar(self):
        self.metadata([self.row()])
        source = self.shard([("a", self.row())])
        usage = type("Usage", (), {"free": 0})()
        with mock.patch.object(rc.shutil, "disk_usage", return_value=usage), \
             mock.patch.object(rc, "build_index") as build:
            with self.assertRaisesRegex(ValueError, "1.05"):
                self.run_recaption(source)
            build.assert_not_called()
        progress = json.loads((self.root / "output" / "recaption_progress.json").read_text())
        self.assertEqual(progress["space"]["free_bytes"], 0)
        self.assertGreater(progress["space"]["required_free_bytes"], 0)
        self.assertFalse(progress["space"]["passed"])

    def test_resume_only_completed_shards_and_ignore_temporary(self):
        row = self.row()
        self.metadata([row])
        source = self.shard([("a", row)])
        self.shard([("a", row)], name="00001.tar")
        original = rc._recaption_shard
        calls = []
        def interrupt(arguments):
            calls.append(arguments[0])
            if len(calls) == 2:
                raise RuntimeError("fixture interrupted")
            return original(arguments)
        with mock.patch.object(rc, "_recaption_shard", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_recaption(source)
        first_identity = rc.shard_identity(self.root / "output" / "00000.tar")
        (self.root / "output" / "00001.tar.tmp-fixture").write_bytes(b"not complete")
        result = self.run_recaption(source)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(first_identity, rc.shard_identity(self.root / "output" / "00000.tar"))

    def test_atomic_shard_writer_does_not_publish_on_exception(self):
        self.metadata([self.row()])
        source = self.shard([("a", self.row())])
        with mock.patch.object(rc, "resolve_caption", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                self.run_recaption(source)
        output = self.root / "output"
        self.assertFalse((output / "00000.tar").exists())
        self.assertFalse((output / ".recaption_shards" / "00000.tar.json").exists())
        self.assertFalse(list(output.glob("00000.tar.tmp-*")))

    def test_recovers_preparation_to_caption_receipt_crash_window(self):
        self.metadata([self.row()])
        source = self.shard([("a", self.row())])
        with mock.patch.object(rc, "_write_caption_receipt", side_effect=RuntimeError("receipt crash")):
            with self.assertRaisesRegex(RuntimeError, "receipt crash"):
                self.run_recaption(source)
        output = self.root / "output"
        self.assertTrue((output / "preparation.json").exists())
        self.assertFalse((output / "caption_receipt.json").exists())
        original_preparation = (output / "preparation.json").read_bytes()
        original_shard = rc.shard_identity(output / "00000.tar")
        result = self.run_recaption(source)
        self.assertEqual(result["samples"], 1)
        self.assertEqual((output / "preparation.json").read_bytes(), original_preparation)
        self.assertEqual(rc.shard_identity(output / "00000.tar"), original_shard)
        rc.validate_caption_receipt(output)

    def test_zero_retained_shard_fails_with_counts_even_if_global_rate_passes(self):
        row = self.row()
        self.metadata([row])
        source = self.shard([("a", row), ("b", row)])
        self.shard([("missing", self.row("missing"))], name="00001.tar")
        with self.assertRaisesRegex(ValueError, "empty shards"):
            self.run_recaption(source)
        progress = json.loads((self.root / "output" / "recaption_progress.json").read_text())
        self.assertEqual(progress["preparation"]["match_rate"], 2 / 3)
        self.assertEqual(len(progress["preparation"]["empty_shards"]), 1)
        self.assertFalse(progress["preparation"]["nonempty_shard_gate_passed"])
        self.assertFalse((self.root / "output" / "preparation.json").exists())

    def test_sampled_parquet_is_loaded_once_per_shard(self):
        row = self.row()
        self.metadata([row])
        source = self.shard([("a", dict(row, caption=row["caption_llava"])),
                             ("b", dict(row, caption=row["caption_llava"]))], use_json=False, parquet=True)
        with mock.patch.object(rc, "_parquet_urls", wraps=rc._parquet_urls) as lookup:
            rc.ensure_caption_receipt(source, self.data, metadata=self.meta, expected=self.expected)
            self.assertEqual(lookup.call_count, 1)

    def test_index_stream_receipt_identity_and_pinned_sha(self):
        self.metadata([self.row(), self.row("ignored", llava="")])
        index = rc.build_index(self.meta, self.data, expected=self.expected)
        receipt = json.loads(rc._index_receipt(index).read_text())
        self.assertEqual((receipt["rows"], receipt["indexed_rows"]), (2, 1))
        self.assertEqual(receipt["actual_sha256"], self.expected["sha256"])
        self.assertLess(receipt["peak_rss_bytes"], 8_000_000_000)
        self.assertEqual(receipt["max_peak_rss_bytes"], 8_000_000_000)
        self.assertNotIn("device", receipt["index_identity"])
        self.assertEqual(rc.build_index(self.meta, self.data, expected=self.expected), index)
        with sqlite3.connect(index) as connection:
            self.assertEqual(rc.resolve_caption(connection, self.row()["url"], "old a"), ("LLaVA a", "matched"))
        self.meta.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "identity"):
            rc.build_index(self.meta, self.data, expected=self.expected)

    def test_index_bad_size_and_sha_are_audited_no_receipt(self):
        self.metadata([self.row()])
        expected = dict(self.expected, size=self.expected["size"] + 1)
        with self.assertRaisesRegex(ValueError, "size"):
            rc.build_index(self.meta, self.data, expected=expected)
        progress_path = self.data / "cc12m_caption_index" / (expected["sha256"] + ".sqlite.progress.json")
        self.assertFalse(json.loads(progress_path.read_text())["size_matches"])
        expected = dict(self.expected, sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            rc.build_index(self.meta, self.data, expected=expected)
        progress_path = self.data / "cc12m_caption_index" / (expected["sha256"] + ".sqlite.progress.json")
        self.assertFalse(json.loads(progress_path.read_text())["sha256_matches"])
        self.assertFalse(Path(str(progress_path).replace(".progress.json", ".receipt.json")).exists())

    def test_sampled_check_at_99_percent_and_root_read_only(self):
        rows = [self.row(str(i)) for i in range(100)]
        self.metadata(rows)
        records = [(str(i), dict(row, caption=row["caption_llava"] if i else "one wrong"))
                   for i, row in enumerate(rows)]
        source = self.shard(records)
        source_files = set(source.iterdir())
        binding = rc.ensure_caption_receipt(source, self.data, metadata=self.meta, expected=self.expected,
                                            sample_size=100)
        receipt = json.loads(Path(binding["receipt_path"]).read_text())
        self.assertEqual(receipt["details"]["match_rate"], 0.99)
        self.assertEqual(set(source.iterdir()), source_files)
        self.assertEqual(rc.validate_caption_receipt(source, self.data), binding)

    def test_sampled_check_failure_audits_and_guard_fails(self):
        self.metadata([self.row()])
        source = self.shard([("a", self.row())])
        with self.assertRaisesRegex(rc.CaptionVerificationError, "DOWNLOAD_DATA=1 bash setup.sh"):
            rc.ensure_caption_receipt(source, self.data, metadata=self.meta, expected=self.expected)
        failures = list((self.data / "caption_receipts").glob("*.failed.json"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(json.loads(failures[0].read_text())["details"]["match_rate"], 0.0)
        with self.assertRaises(rc.CaptionVerificationError):
            rc.validate_caption_receipt(source, self.data)

    def test_own_preparation_provenance_trusted_without_metadata(self):
        self.metadata([self.row()])
        source = self.shard([("a", self.row())])
        lock, atomic_json, sha256 = rc._helpers()
        atomic_json(source / "preparation.json", {"caption_source": "caption_llava", "source": lock,
                                                  "files": {"00000.tar": sha256(source / "00000.tar")}})
        binding = rc.ensure_caption_receipt(source, self.data)
        self.assertTrue(Path(binding["receipt_path"]).is_relative_to(self.data))
        self.assertFalse((source / "caption_receipt.json").exists())
        self.assertEqual(rc.validate_caption_receipt(source, self.data), binding)

    def test_missing_receipt_stale_shards_and_preparation_fail_closed(self):
        self.metadata([self.row()])
        source = self.shard([("a", self.row())])
        with self.assertRaises(rc.CaptionVerificationError):
            rc.validate_caption_receipt(source, self.data)
        self.run_recaption(source)
        output = self.root / "output"
        (output / "preparation.json").write_text("{}")
        with self.assertRaisesRegex(rc.CaptionVerificationError, "preparation"):
            rc.validate_caption_receipt(output)

    def test_root_device_change_is_not_identity_change(self):
        self.metadata([self.row()])
        source = self.shard([("a", self.row())])
        before = rc.root_identity(source)
        original_stat = Path.stat
        def changed_stat(path, *args, **kwargs):
            value = original_stat(path, *args, **kwargs)
            # Change only the device, retaining every shared-file invariant.
            fields = list(value)
            fields[2] += 100
            import os
            return os.stat_result(fields)
        with mock.patch.object(Path, "stat", changed_stat):
            self.assertEqual(rc.root_identity(source), before)

    def test_output_input_overlap_refused(self):
        self.metadata([self.row()])
        source = self.shard([("a", self.row())])
        with self.assertRaisesRegex(ValueError, "separate"):
            rc.recaption(source, source / "copy", metadata=self.meta, data_dir=self.data,
                         expected=self.expected)
        with self.assertRaisesRegex(ValueError, "DATA_DIR"):
            rc.recaption(source, self.root / "output", metadata=self.meta, data_dir=source,
                         expected=self.expected)
        with self.assertRaisesRegex(ValueError, "DATA_DIR"):
            rc.ensure_caption_receipt(source, source, metadata=self.meta, expected=self.expected)
        self.assertEqual({path.name for path in source.iterdir()}, {"00000.tar"})


if __name__ == "__main__":
    unittest.main()
