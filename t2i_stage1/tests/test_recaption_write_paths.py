"""Derived caption-write paths must be admitted before any write or download."""
from contextlib import ExitStack, redirect_stdout
import builtins
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


class DerivedWriteAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="recaption-write-paths-",
                                                 dir=os.environ.get("TMPDIR"))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.output = self.root / "output"
        self.row = dict(url="https://example.invalid/a", caption="old a",
                        caption_llava="LLaVA a", status="success")
        self.metadata = self.root / "metadata.jsonl.gz"
        with gzip.open(self.metadata, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(self.row) + "\n")
        self.expected = dict(size=self.metadata.stat().st_size,
                             sha256=hashlib.sha256(self.metadata.read_bytes()).hexdigest())
        self.lock = dict(prepare.REMOTE_LOCK["cc12m"], files=[dict(
            name="train.jsonl.gz", **self.expected)])

    def source(self, relative, *, metadata=False, prepared=False):
        root = self.data / relative
        root.mkdir(parents=True)
        with tarfile.open(root / "00000.tar", "w") as archive:
            for name, payload in (("a.jpg", b"unchanged image"),
                                  ("a.txt", self.row["caption"].encode()),
                                  ("a.json", json.dumps(self.row).encode())):
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        # Existing audit files must survive both missing and cached metadata.
        (root / "download.progress.json").write_text('{"old":true}\n')
        (root / "assets.sha256.json").write_text('{"old":true}\n')
        if metadata:
            (root / "train.jsonl.gz").write_bytes(self.metadata.read_bytes())
        if prepared:
            (root / "preparation.json").write_text(json.dumps(dict(
                caption_source="caption_llava", source=self.lock,
                files={"00000.tar": prepare.sha256(root / "00000.tar")})))
        return root

    def snapshot(self):
        result = {}
        for path in [self.root, *sorted(self.root.rglob("*"))]:
            stat = path.lstat()
            result[str(path.relative_to(self.root))] = (
                os.readlink(path) if path.is_symlink() else
                "directory" if path.is_dir() else path.read_bytes(),
                stat.st_ino, stat.st_mtime_ns, stat.st_uid)
        return result

    def invoke(self, entry, source, metadata):
        if entry == "setup":
            return prepare.prepare_existing_cc12m(source, self.data, 1, metadata)
        if entry == "api":
            return rc.recaption(source, self.output, metadata=metadata,
                                data_dir=self.data, expected=self.expected)
        if entry == "check":
            return rc.ensure_caption_receipt(source, self.data, metadata=metadata,
                                             expected=self.expected)
        argv = ["prepare", "recaption", "--cc12m-root", str(source),
                "--output", str(self.output), "--data-dir", str(self.data),
                "--processes", "1"]
        if metadata is not None:
            argv += ["--metadata", str(metadata)]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            return prepare.main()

    def assert_zero_write_refusal(self, entry, source, metadata=None):
        before = self.snapshot()
        attempted, guards = [], []
        original_open, original_io_open, original_os_open = builtins.open, io.open, os.open

        def file_open(original):
            def checked(file, mode="r", *args, **kwargs):
                if isinstance(mode, str) and any(flag in mode for flag in "wax+"):
                    attempted.append((str(file), mode))
                    raise AssertionError("Derived-path admission attempted file write")
                return original(file, mode, *args, **kwargs)
            return checked

        def os_open(file, flags, *args, **kwargs):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                attempted.append((str(file), flags))
                raise AssertionError("Derived-path admission attempted os.open write")
            return original_os_open(file, flags, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(prepare.REMOTE_LOCK, {"cc12m": self.lock}))
            for owner, name in ((prepare, "download_group"), (prepare, "atomic_json"),
                                (rc, "build_index"), (Path, "mkdir"), (os, "mkdir"),
                                (tempfile, "mkstemp"), (os, "replace"), (os, "rename"),
                                (os, "link"), (os, "unlink")):
                guards.append(stack.enter_context(mock.patch.object(owner, name,
                    side_effect=AssertionError("Derived-path admission attempted " + name))))
            stack.enter_context(mock.patch.object(builtins, "open", side_effect=file_open(original_open)))
            stack.enter_context(mock.patch.object(io, "open", side_effect=file_open(original_io_open)))
            stack.enter_context(mock.patch.object(os, "open", side_effect=os_open))
            with self.assertRaisesRegex(ValueError, "(?i)input root|separate|DATA_DIR"):
                self.invoke(entry, source, metadata)
            for guard in guards:
                guard.assert_not_called()
            self.assertEqual(attempted, [])
        self.assertEqual(self.snapshot(), before)

    def test_setup_download_collision_cached_metadata_is_zero_write(self):
        self.assert_zero_write_refusal("setup", self.source("raw/cc12m", metadata=True))

    def test_setup_download_collision_missing_metadata_is_zero_write(self):
        self.assert_zero_write_refusal("setup", self.source("raw/cc12m"))

    def test_cli_download_collision_cached_metadata_is_zero_write(self):
        self.assert_zero_write_refusal("cli", self.source("raw/cc12m", metadata=True))

    def test_cli_download_collision_missing_metadata_is_zero_write(self):
        self.assert_zero_write_refusal("cli", self.source("raw/cc12m"))

    def test_setup_index_collision_is_zero_write(self):
        self.assert_zero_write_refusal("setup", self.source("cc12m_caption_index"), self.metadata)

    def test_cli_index_collision_is_zero_write(self):
        self.assert_zero_write_refusal("cli", self.source("cc12m_caption_index"), self.metadata)

    def test_api_index_collision_is_zero_write(self):
        self.assert_zero_write_refusal("api", self.source("cc12m_caption_index"), self.metadata)

    def test_check_index_collision_is_zero_write(self):
        self.assert_zero_write_refusal("check", self.source("cc12m_caption_index"), self.metadata)

    def test_receipt_directory_collision_before_trusted_preparation_write(self):
        self.assert_zero_write_refusal("check", self.source("caption_receipts", prepared=True))

    def test_setup_receipt_directory_collision_is_zero_write(self):
        self.assert_zero_write_refusal("setup", self.source("caption_receipts"), self.metadata)

    def test_setup_output_collision_is_zero_write(self):
        self.assert_zero_write_refusal("setup", self.source("cc12m_llava"), self.metadata)

    def test_setup_mapping_collision_is_zero_write(self):
        self.assert_zero_write_refusal("setup", self.source("cc12m_root_mapping.json"), self.metadata)

    def test_stale_local_receipt_refused_before_building_index(self):
        source = self.source("ordinary")
        (source / "caption_receipt.json").write_text("{}")
        self.assert_zero_write_refusal("check", source, self.metadata)

    def test_index_directory_symlink_into_input_is_zero_write(self):
        source = self.source("ordinary")
        (self.data / "cc12m_caption_index").symlink_to(source, target_is_directory=True)
        for entry in ("setup", "cli", "api", "check"):
            with self.subTest(entry=entry):
                self.assert_zero_write_refusal(entry, source, self.metadata)

    def test_download_directory_symlink_into_input_is_zero_write(self):
        source = self.source("ordinary")
        (self.data / "raw").mkdir()
        (self.data / "raw/cc12m").symlink_to(source, target_is_directory=True)
        for entry in ("setup", "cli"):
            with self.subTest(entry=entry):
                self.assert_zero_write_refusal(entry, source)

    def test_download_directory_below_input_is_zero_write(self):
        source = self.source("raw")
        for entry in ("setup", "cli"):
            with self.subTest(entry=entry):
                self.assert_zero_write_refusal(entry, source)

    def test_receipt_and_failure_leaf_symlinks_into_input_are_zero_write(self):
        source = self.source("ordinary")
        receipt = rc._receipt_path(source, self.data)
        receipt.parent.mkdir()
        (source / "old.json").write_text("{}")
        receipt.symlink_to(source / "old.json")
        self.assert_zero_write_refusal("check", source, self.metadata)
        # The stale receipt symlink remains untouched; use a separate data root
        # so the failed-receipt leaf is independently the first collision.
        self.data = self.root / "other-data"
        receipt = rc._receipt_path(source, self.data)
        receipt.parent.mkdir(parents=True)
        receipt.with_name(receipt.stem + ".failed.json").symlink_to(source / "old.json")
        self.assert_zero_write_refusal("check", source, self.metadata)

    def test_mapping_leaf_symlink_into_input_is_zero_write(self):
        source = self.source("ordinary")
        (source / "old-mapping.json").write_text('{"original":{"path":"/unrelated"}}')
        (self.data / "cc12m_root_mapping.json").symlink_to(source / "old-mapping.json")
        self.assert_zero_write_refusal("setup", source, self.metadata)

    def test_valid_receipt_fast_path_needs_no_download_or_write(self):
        source = self.source("raw/cc12m", prepared=True)
        with mock.patch.dict(prepare.REMOTE_LOCK, {"cc12m": self.lock}):
            rc.ensure_caption_receipt(source, self.data)
            before = self.snapshot()
            with mock.patch.object(prepare, "download_group", side_effect=AssertionError("download")), \
                 mock.patch.object(prepare, "atomic_json", side_effect=AssertionError("write")), \
                 mock.patch.object(rc, "build_index", side_effect=AssertionError("index")), \
                 mock.patch.object(Path, "mkdir", side_effect=AssertionError("mkdir")):
                self.assertEqual(prepare.prepare_existing_cc12m(source, self.data, 1), str(source))
                self.assertEqual(rc.ensure_caption_receipt(source, self.data)["caption_source"], "caption_llava")
            self.assertEqual(self.snapshot(), before)

    def test_valid_ancestor_data_dir_and_readonly_metadata_inside_input(self):
        source = self.source("raw/cc12m", metadata=True)
        metadata = source / "train.jsonl.gz"
        before = {path.name: path.read_bytes() for path in source.iterdir()}
        with mock.patch.dict(prepare.REMOTE_LOCK, {"cc12m": self.lock}), mock.patch.object(
                prepare, "download_group", side_effect=AssertionError("explicit metadata is read-only")):
            resolved = prepare.prepare_existing_cc12m(source, self.data, 1, metadata)
            self.assertEqual(Path(resolved), self.data / "cc12m_llava")
            self.assertEqual(prepare.resolve_cc12m_root(source, self.data), resolved)
            self.assertEqual(rc.validate_caption_receipt(resolved)["caption_source"], "caption_llava")
            self.assertEqual(prepare.prepare_existing_cc12m(source, self.data, 1, metadata), resolved)
        self.assertEqual(before, {path.name: path.read_bytes() for path in source.iterdir()})


if __name__ == "__main__":
    unittest.main()
