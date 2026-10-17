"""Regression fixtures for output admission and duplicate-member rejection."""
from contextlib import ExitStack
import builtins
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from t2i import prepare
from t2i import recaption as rc


class OutputAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="recaption-admission-",
                                                 dir=os.environ.get("TMPDIR"))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"
        self.data = self.root / "cache"
        self.rows = [{"url": "https://example.invalid/" + key,
                      "caption": "old " + key, "caption_llava": "LLaVA " + key,
                      "status": "success"} for key in ("a", "b")]
        self.metadata = self.root / "metadata.jsonl.gz"
        with gzip.open(self.metadata, "wt", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row) + "\n")
        self.expected = {"size": self.metadata.stat().st_size,
                         "sha256": hashlib.sha256(self.metadata.read_bytes()).hexdigest()}
        self.write_members(self.record("a", self.rows[0]))

    @staticmethod
    def record(key, row):
        return [(key + ".jpg", b"original image " + row["url"].encode()),
                (key + ".txt", row["caption"].encode()),
                (key + ".json", json.dumps({"url": row["url"],
                                            "caption": row["caption"]}).encode())]

    def write_members(self, members):
        with tarfile.open(self.source / "00000.tar", "w") as archive:
            for name, raw in members:
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))

    def run_recaption(self):
        return rc.recaption(self.source, self.output, metadata=self.metadata,
                            data_dir=self.data, expected=self.expected)

    def plan(self):
        return dict(rc._snapshot(self.source), format=rc.FORMAT,
                    source=prepare.REMOTE_LOCK["cc12m"], metadata_lock=self.expected)

    def snapshot(self):
        result = {}
        for path in [self.root, *sorted(self.root.rglob("*"))]:
            stat = path.lstat()
            result[str(path.relative_to(self.root))] = (
                "directory" if path.is_dir() else path.read_bytes(),
                stat.st_ino, stat.st_mtime_ns, stat.st_uid)
        return result

    def assert_readonly_refusal(self):
        """Refusal must precede even an attempted mkdir or atomic writer call."""
        before = self.snapshot()
        forbidden = []
        writer_attempts = []
        original_open, original_io_open, original_os_open = builtins.open, io.open, os.open

        def file_open(original):
            def checked(file, mode="r", *args, **kwargs):
                if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
                    writer_attempts.append((str(file), mode))
                    raise AssertionError("Output admission attempted a file write")
                return original(file, mode, *args, **kwargs)
            return checked

        def os_open(file, flags, *args, **kwargs):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                writer_attempts.append((str(file), flags))
                raise AssertionError("Output admission attempted os.open for writing")
            return original_os_open(file, flags, *args, **kwargs)

        with ExitStack() as stack:
            for owner, name in ((prepare, "atomic_json"), (Path, "mkdir"), (os, "mkdir"),
                                (tempfile, "mkstemp"), (os, "replace"), (os, "rename"),
                                (os, "link"), (os, "unlink")):
                forbidden.append(stack.enter_context(mock.patch.object(
                    owner, name, side_effect=AssertionError("Output admission attempted " + name))))
            stack.enter_context(mock.patch.object(builtins, "open", side_effect=file_open(original_open)))
            stack.enter_context(mock.patch.object(io, "open", side_effect=file_open(original_io_open)))
            stack.enter_context(mock.patch.object(os, "open", side_effect=os_open))
            with self.assertRaises((ValueError, FileExistsError)):
                self.run_recaption()
            for operation in forbidden:
                operation.assert_not_called()
            self.assertEqual(writer_attempts, [])
        self.assertEqual(self.snapshot(), before)

    def test_foreign_tar_without_plan_is_refused_before_any_write(self):
        self.output.mkdir()
        (self.output / "old.tar").write_bytes(b"old tar must be untouched")
        self.assert_readonly_refusal()

    def test_ordinary_output_file_is_refused_before_any_write(self):
        self.output.write_bytes(b"old ordinary file must be untouched")
        self.assert_readonly_refusal()

    def test_foreign_checkpoint_manifest_progress_is_refused_before_any_write(self):
        self.output.mkdir()
        (self.output / "checkpoint").mkdir()
        (self.output / "checkpoint" / "state.bin").write_bytes(b"old checkpoint")
        (self.output / "manifest.json").write_bytes(b'{"old":true}\n')
        (self.output / "recaption_progress.json").write_bytes(b'{"old_progress":true}\n')
        self.assert_readonly_refusal()

    def test_malformed_plan_is_refused_before_any_write(self):
        self.output.mkdir()
        (self.output / ".recaption_plan.json").write_text("{not valid json")
        self.assert_readonly_refusal()

    def test_non_object_plan_is_refused_before_any_write(self):
        self.output.mkdir()
        (self.output / ".recaption_plan.json").write_text("[]")
        self.assert_readonly_refusal()

    def test_mismatched_plan_is_refused_before_any_write(self):
        self.output.mkdir()
        plan = self.plan()
        plan["metadata_lock"] = dict(self.expected, sha256="0" * 64)
        (self.output / ".recaption_plan.json").write_text(json.dumps(plan))
        self.assert_readonly_refusal()

    def test_empty_output_is_allowed(self):
        self.output.mkdir()
        self.assertEqual(self.run_recaption()["samples"], 1)
        rc.validate_caption_receipt(self.output)

    def test_matching_own_plan_is_resumed_and_completed_shard_is_untouched(self):
        self.output.mkdir()
        plan_path = self.output / ".recaption_plan.json"
        plan_path.write_text(json.dumps(self.plan()))
        original_plan = plan_path.read_bytes()
        self.assertEqual(self.run_recaption()["samples"], 1)
        self.assertEqual(plan_path.read_bytes(), original_plan)
        before = self.snapshot()
        with mock.patch.object(rc, "_recaption_shard", side_effect=AssertionError("completed shard rerun")):
            self.assertEqual(self.run_recaption()["samples"], 1)
        self.assertEqual(self.snapshot(), before)

    def assert_duplicate_refused_without_publication(self, members):
        self.write_members(members)
        source_before = (self.source / "00000.tar").read_bytes()
        with mock.patch.object(rc, "_recaption_shard", wraps=rc._recaption_shard) as worker:
            with self.assertRaisesRegex(ValueError, "(?i)duplicate|repeated|conflict"):
                self.run_recaption()
            worker.assert_called_once()
        self.assertEqual((self.source / "00000.tar").read_bytes(), source_before)
        self.assertFalse((self.output / "00000.tar").exists())
        self.assertFalse((self.output / ".recaption_shards" / "00000.tar.json").exists())
        self.assertFalse((self.output / "preparation.json").exists())
        self.assertFalse((self.output / "caption_receipt.json").exists())
        self.assertFalse(list(self.output.glob("00000.tar.tmp-*")))
        progress = json.loads((self.output / "recaption_progress.json").read_text())
        self.assertEqual(progress["status"], "FAILED")

    def test_consecutive_complete_records_with_same_key_do_not_merge(self):
        self.assert_duplicate_refused_without_publication(
            self.record("same", self.rows[0]) + self.record("same", self.rows[1]))

    def test_conflicting_duplicate_json_cannot_overwrite_sample_url(self):
        members = self.record("a", self.rows[0])
        members.append(("a.json", json.dumps({"url": self.rows[1]["url"],
                                               "caption": self.rows[1]["caption"]}).encode()))
        self.assert_duplicate_refused_without_publication(members)

    def test_sample_fields_rejects_case_insensitive_conflicting_json_directly(self):
        members = self.record("a", self.rows[0])
        members.append(("a.JSON", json.dumps({"url": self.rows[1]["url"],
                                               "caption": self.rows[1]["caption"]}).encode()))
        sample = [(tarfile.TarInfo(name), raw) for name, raw in members]
        with self.assertRaisesRegex(ValueError, "(?i)duplicate.*json"):
            rc.sample_fields(sample, self.source / "00000.tar")

    def test_multiple_image_extensions_are_duplicate_critical_members(self):
        members = self.record("a", self.rows[0])
        members.append(("a.PNG", b"second image must not share the first caption"))
        self.assert_duplicate_refused_without_publication(members)


if __name__ == "__main__":
    unittest.main()
