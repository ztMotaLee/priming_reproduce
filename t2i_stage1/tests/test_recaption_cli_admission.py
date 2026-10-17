"""CLI admission and exact parquet sample keys."""
from contextlib import ExitStack, redirect_stdout
import builtins
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

from t2i import prepare
from t2i import recaption as rc


class CliAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="recaption-cli-",
                                                 dir=os.environ.get("TMPDIR"))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"
        self.data = self.root / "cache"
        self.rows = [{"url": "https://example.invalid/" + key,
                      "caption": "old " + key, "caption_llava": "LLaVA " + key,
                      "status": "success"} for key in ("cat", "dog")]
        self.metadata = self.root / "metadata.jsonl.gz"
        with gzip.open(self.metadata, "wt", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row) + "\n")
        self.expected = {"size": self.metadata.stat().st_size,
                         "sha256": hashlib.sha256(self.metadata.read_bytes()).hexdigest()}
        self.lock = copy.deepcopy(prepare.REMOTE_LOCK["cc12m"])
        self.lock["files"][0].update(self.expected)
        self.write_shard([("cat", self.rows[0])], json_urls=True)

    def write_shard(self, records, *, json_urls=False):
        with tarfile.open(self.source / "00000.tar", "w") as archive:
            for key, row in records:
                members = [(key + ".jpg", b"original image " + key.encode()),
                           (key + ".txt", row["caption"].encode())]
                if json_urls:
                    members.append((key + ".json", json.dumps({"url": row["url"],
                                                               "caption": row["caption"]}).encode()))
                for name, raw in members:
                    member = tarfile.TarInfo(name)
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))

    def write_parquet(self, rows):
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), self.source / "00000.parquet")

    @staticmethod
    def members(path):
        with tarfile.open(path) as archive:
            return [(member.name, archive.extractfile(member).read())
                    for member in archive if member.isfile()]

    def snapshot(self):
        result = {}
        for path in [self.root, *sorted(self.root.rglob("*"))]:
            stat = path.lstat()
            result[str(path.relative_to(self.root))] = (
                "directory" if path.is_dir() else path.read_bytes(),
                stat.st_ino, stat.st_mtime_ns, stat.st_uid)
        return result

    def cli(self, data_dir):
        argv = ["prepare", "recaption", "--cc12m-root", str(self.source),
                "--output", str(self.output), "--data-dir", str(data_dir),
                "--processes", "1"]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()) as output:
            prepare.main()
        return json.loads(output.getvalue())

    def assert_cli_readonly_refusal(self, data_dir):
        """Arm every write guard only after constructing the old-output fixture."""
        before = self.snapshot()
        forbidden, writer_attempts = [], []
        original_open, original_io_open, original_os_open = builtins.open, io.open, os.open

        def file_open(original):
            def checked(file, mode="r", *args, **kwargs):
                if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
                    writer_attempts.append((str(file), mode))
                    raise AssertionError("CLI admission attempted a file write")
                return original(file, mode, *args, **kwargs)
            return checked

        def os_open(file, flags, *args, **kwargs):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                writer_attempts.append((str(file), flags))
                raise AssertionError("CLI admission attempted os.open for writing")
            return original_os_open(file, flags, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(prepare.REMOTE_LOCK, {"cc12m": self.lock}))
            for owner, name in ((prepare, "download_group"), (prepare, "atomic_json"),
                                (Path, "mkdir"), (os, "mkdir"), (tempfile, "mkstemp"),
                                (os, "replace"), (os, "rename"), (os, "link"), (os, "unlink")):
                forbidden.append(stack.enter_context(mock.patch.object(
                    owner, name, side_effect=AssertionError("CLI admission attempted " + name))))
            stack.enter_context(mock.patch.object(builtins, "open", side_effect=file_open(original_open)))
            stack.enter_context(mock.patch.object(io, "open", side_effect=file_open(original_io_open)))
            stack.enter_context(mock.patch.object(os, "open", side_effect=os_open))
            with self.assertRaises((ValueError, FileExistsError)):
                self.cli(data_dir)
            for operation in forbidden:
                operation.assert_not_called()
            self.assertEqual(writer_attempts, [])
        self.assertEqual(self.snapshot(), before)

    def test_cli_old_tar_data_dir_equals_output_refuses_before_download_or_write(self):
        self.output.mkdir()
        (self.output / "old.tar").write_bytes(b"old tar must remain unchanged")
        self.assert_cli_readonly_refusal(self.output)

    def test_cli_old_tar_data_dir_inside_output_refuses_before_download_or_write(self):
        self.output.mkdir()
        (self.output / "old.tar").write_bytes(b"old tar must remain unchanged")
        self.assert_cli_readonly_refusal(self.output / "nested" / "cache")

    def test_cli_sidecar_only_output_refuses_before_download_or_write(self):
        self.output.mkdir()
        (self.output / ".recaption_shards").mkdir()
        (self.output / ".recaption_shards" / "old.progress.json").write_text('{"status":"FAILED"}')
        (self.output / "recaption_progress.json").write_text('{"old":true}')
        for data_dir in (self.output, self.output / "nested" / "cache"):
            with self.subTest(data_dir=str(data_dir)):
                self.assert_cli_readonly_refusal(data_dir)

    def test_cli_existing_output_file_refuses_before_download_or_write(self):
        self.output.write_bytes(b"old ordinary file")
        self.assert_cli_readonly_refusal(self.output / "cache")

    def test_cli_malformed_plan_refuses_before_download_or_write(self):
        self.output.mkdir()
        (self.output / ".recaption_plan.json").write_text("{not valid json")
        self.assert_cli_readonly_refusal(self.output)

    def test_cli_mismatched_plan_refuses_before_download_or_write(self):
        self.output.mkdir()
        plan = dict(rc._snapshot(self.source), format=rc.FORMAT, source=self.lock,
                    metadata_lock=dict(self.expected, sha256="0" * 64))
        (self.output / ".recaption_plan.json").write_text(json.dumps(plan))
        self.assert_cli_readonly_refusal(self.output / "cache")

    def fake_download(self, group, directory):
        self.assertEqual(group, "cc12m")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "train.jsonl.gz").write_bytes(self.metadata.read_bytes())
        return directory

    def test_cli_new_output_allows_metadata_download_then_conversion(self):
        with mock.patch.dict(prepare.REMOTE_LOCK, {"cc12m": self.lock}), mock.patch.object(
                prepare, "download_group", side_effect=self.fake_download) as download:
            result = self.cli(self.data)
            download.assert_called_once()
            self.assertEqual(result["samples"], 1)
            self.assertEqual(rc.validate_caption_receipt(self.output)["caption_source"], "caption_llava")

    def test_cli_empty_output_and_matching_plan_resume_remain_usable(self):
        self.output.mkdir()
        with mock.patch.dict(prepare.REMOTE_LOCK, {"cc12m": self.lock}), mock.patch.object(
                prepare, "download_group", side_effect=self.fake_download) as download:
            first = self.cli(self.data)
            download.assert_called_once()
            before = self.snapshot()
            with mock.patch.object(rc, "_recaption_shard", side_effect=AssertionError("completed shard rerun")):
                self.assertEqual(self.cli(self.data), first)
            self.assertEqual(self.snapshot(), before)
            download.assert_called_once()

    def test_parquet_basename_collision_fails_without_final_tar_or_receipt(self):
        self.write_shard([("cats/0001", self.rows[0]), ("dogs/0001", self.rows[1])])
        self.write_parquet([{"key": "0001", "url": self.rows[0]["url"]}])
        before = self.members(self.source / "00000.tar")
        with self.assertRaisesRegex(ValueError, "(?i)No URL|exact.*key|missing.*key"):
            rc.recaption(self.source, self.output, metadata=self.metadata,
                         data_dir=self.data, expected=self.expected)
        self.assertEqual(self.members(self.source / "00000.tar"), before)
        self.assertFalse((self.output / "00000.tar").exists())
        self.assertFalse((self.output / ".recaption_shards" / "00000.tar.json").exists())
        self.assertFalse((self.output / "preparation.json").exists())
        self.assertFalse((self.output / "caption_receipt.json").exists())
        self.assertFalse(list(self.output.glob("00000.tar.tmp-*")))
        progress = json.loads((self.output / "recaption_progress.json").read_text())
        self.assertEqual(progress["status"], "FAILED")

    def test_parquet_exact_nested_keys_keep_distinct_captions_bytes_and_order(self):
        self.write_shard([("dogs/0001", self.rows[1]), ("cats/0001", self.rows[0])])
        self.write_parquet([{"key": "cats/0001", "url": self.rows[0]["url"]},
                            {"key": "dogs/0001", "url": self.rows[1]["url"]},
                            {"key": "0001", "url": "https://example.invalid/decoy"}])
        before = self.members(self.source / "00000.tar")
        result = rc.recaption(self.source, self.output, metadata=self.metadata,
                              data_dir=self.data, expected=self.expected)
        after = self.members(self.output / "00000.tar")
        self.assertEqual([name for name, _ in after], [name for name, _ in before])
        expected = {"cats/0001.txt": b"LLaVA cat", "dogs/0001.txt": b"LLaVA dog"}
        for (name, original), (_, actual) in zip(before, after):
            self.assertEqual(actual, expected[name] if name in expected else original)
        self.assertEqual(self.members(self.source / "00000.tar"), before)
        self.assertEqual((result["samples"], result["input_samples"], result["match_rate"]), (2, 2, 1.0))
        rc.validate_caption_receipt(self.output)


if __name__ == "__main__":
    unittest.main()
