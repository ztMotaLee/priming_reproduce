"""Exercise the CC12M CLI boundary without network, GPU, or real downloads."""
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import ModuleType
import unittest
from unittest import mock

from PIL import Image

from t2i import prepare


class PrepareCliTests(unittest.TestCase):
    def fixtures(self, root, *, environment_cli=True, stale_cli=False):
        """Use fake metadata/Parquet and a real tiny executable subprocess."""
        raw = root / "raw"
        raw.mkdir()
        with gzip.open(raw / "train.jsonl.gz", "wt") as handle:
            handle.write(json.dumps({"url": "https://example.invalid/image.jpg", "key": "0",
                                     "caption_llava": "long recaption", "caption": "wrong caption"}) + "\n")
        environment_bin = root / "venv" / "bin"
        environment_bin.mkdir(parents=True)
        # A real venv commonly has exactly this kind of symlink. The production
        # lookup must retain its lexical bin parent instead of resolving it.
        python = environment_bin / "python"
        python.symlink_to(sys.executable)
        invocation = root / "invocation.json"
        if environment_cli:
            cli = environment_bin / "img2dataset"
            cli.write_text(f"#!{sys.executable}\n" +
                           "import io, json, pathlib, sys, tarfile\n" +
                           f"pathlib.Path({str(invocation)!r}).write_text(json.dumps(sys.argv))\n" +
                           "output = pathlib.Path(sys.argv[sys.argv.index('--output_folder') + 1])\n" +
                           "output.mkdir(parents=True)\n" +
                           "with tarfile.open(output / 'part-00000.tar', 'w') as archive:\n" +
                           "    for name, value in [('0.jpg', b'tiny fixture'), ('0.txt', b'long recaption')]:\n" +
                           "        member = tarfile.TarInfo(name)\n" +
                           "        member.size = len(value)\n" +
                           "        archive.addfile(member, io.BytesIO(value))\n")
            cli.chmod(0o755)
        isolated_path = root / "path"
        isolated_path.mkdir()
        stale_marker = root / "stale-invoked"
        if stale_cli:
            stale = isolated_path / "img2dataset"
            stale.write_text(f"#!{sys.executable}\n" +
                             "import pathlib\n" +
                             f"pathlib.Path({str(stale_marker)!r}).write_text('wrong CLI')\n" +
                             "raise SystemExit(73)\n")
            stale.chmod(0o755)

        # Avoid an optional PyArrow install while retaining the actual JSONL ->
        # row conversion and preparation/subprocess/receipt path under test.
        arrow = ModuleType("pyarrow")
        arrow.Table = type("Table", (), {"from_pylist": staticmethod(lambda rows: list(rows))})
        parquet = ModuleType("pyarrow.parquet")
        parquet.write_table = lambda rows, path: path.write_text(json.dumps(rows))
        arrow.parquet = parquet
        return raw, python, isolated_path, invocation, stale_marker, arrow, parquet

    def run_preparation(self, root, *, environment_cli=True, stale_cli=False):
        fixture = self.fixtures(root, environment_cli=environment_cli, stale_cli=stale_cli)
        raw, python, isolated_path, invocation, stale_marker, arrow, parquet = fixture
        with mock.patch.object(prepare, "download_group", return_value=raw) as download, \
             mock.patch.object(prepare.sys, "executable", str(python)), \
             mock.patch.dict(os.environ, {"PATH": str(isolated_path)}), \
             mock.patch.dict(sys.modules, {"pyarrow": arrow, "pyarrow.parquet": parquet}):
            # subprocess.run is deliberately real: a bare CLI or resolved
            # interpreter symlink fails these cases through OS command lookup.
            prepare.prepare_cc12m(root / "data", processes=1, threads=1)
        download.assert_called_once_with("cc12m", root / "data" / "raw" / "cc12m")
        argv = json.loads(invocation.read_text())
        self.assertEqual(argv[0], str(python.parent / "img2dataset"))
        self.assertEqual(argv[argv.index("--processes_count") + 1], "1")
        self.assertEqual(argv[argv.index("--thread_count") + 1], "1")
        self.assertEqual(json.loads((raw / "parquet" / "part-00000.parquet").read_text()),
                         [{"url": "https://example.invalid/image.jpg", "caption": "long recaption", "key": "0"}])
        receipt = json.loads((root / "data" / "cc12m" / "preparation.json").read_text())
        self.assertEqual(receipt["files"], {"part-00000.tar": prepare.sha256(root / "data" / "cc12m" / "part-00000.tar")})
        self.assertFalse(stale_marker.exists())

    def test_unactivated_venv_with_isolated_path(self):
        with tempfile.TemporaryDirectory(prefix="prepare-cli-") as directory:
            self.run_preparation(Path(directory))

    def test_stale_path_cli_cannot_override_environment(self):
        with tempfile.TemporaryDirectory(prefix="prepare-cli-") as directory:
            self.run_preparation(Path(directory), stale_cli=True)

    def test_missing_environment_cli_fails_closed_without_path_fallback(self):
        with tempfile.TemporaryDirectory(prefix="prepare-cli-") as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError) as caught:
                self.run_preparation(root, environment_cli=False, stale_cli=True)
            self.assertEqual(caught.exception.filename, str(root / "venv" / "bin" / "img2dataset"))
            self.assertFalse((root / "stale-invoked").exists())
            self.assertFalse((root / "data" / "cc12m" / "preparation.json").exists())


class PrepareCheckCliTests(unittest.TestCase):
    def write_shard(self, path):
        """Write a real, tiny image-caption tar without network preparation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        image = io.BytesIO()
        Image.new("RGB", (2, 2), color="red").save(image, format="JPEG")
        with tarfile.open(path, "w") as archive:
            for name, value in (("0.jpg", image.getvalue()), ("0.txt", b"tiny red image")):
                member = tarfile.TarInfo(name)
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
        return path

    def write_curated(self, root):
        curated = root / "curated"
        chosen = self.write_shard(curated / "chosen" / "part-00000.tar")
        # The manifest selects only one source. Recursive tar expansion would
        # include this extra shard and must not replace manifest handling.
        self.write_shard(curated / "excluded" / "part-00000.tar")
        spec = curated / "sources.json"
        spec.write_text(json.dumps({"sources": [{"name": "chosen", "root": "chosen", "weight": 1.0}]}))
        return curated, spec, chosen

    def run_check(self, cc12m, curated, *, output=None):
        output = io.StringIO() if output is None else output
        argv = ["prepare", "check", "--cc12m-root", str(cc12m), "--curated-root", str(curated)]
        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stdout", output):
            # Keep main(), argument parsing, shard discovery and file identity
            # checks real so directory dispatch failures cross the CLI boundary.
            prepare.main()
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([row["root"] for row in rows], [str(cc12m), str(curated)])
        for row in rows:
            self.assertGreater(row["bytes"], 0)
            self.assertRegex(row["identity_sha256"], r"^[0-9a-f]{64}$")
        return rows

    def test_check_accepts_plain_tar_directory(self):
        for sources_is_directory in (False, True):
            with self.subTest(sources_is_directory=sources_is_directory), \
                 tempfile.TemporaryDirectory(prefix="prepare-check-") as directory:
                root = Path(directory)
                first = self.write_shard(root / "cc12m" / "part-00000.tar")
                second = self.write_shard(root / "cc12m" / "nested" / "part-00001.tar")
                curated = self.write_shard(root / "curated" / "part-00000.tar")
                if sources_is_directory:
                    (root / "cc12m" / "sources.json").mkdir()
                rows = self.run_check(root / "cc12m", curated.parent)
                self.assertEqual([row["shards"] for row in rows], [2, 1])
                self.assertEqual(rows[0]["bytes"], first.stat().st_size + second.stat().st_size)

    def test_check_accepts_tar_glob(self):
        with tempfile.TemporaryDirectory(prefix="prepare-check-") as directory:
            root = Path(directory)
            self.write_shard(root / "cc12m" / "part-00000.tar")
            self.write_shard(root / "cc12m" / "part-00001.tar")
            self.write_shard(root / "cc12m" / "excluded.tar")
            curated = self.write_shard(root / "curated" / "part-00000.tar")
            rows = self.run_check(root / "cc12m" / "part-*.tar", curated)
            self.assertEqual([row["shards"] for row in rows], [2, 1])

    def test_check_accepts_explicit_sources_json(self):
        with tempfile.TemporaryDirectory(prefix="prepare-check-") as directory:
            root = Path(directory)
            cc12m = self.write_shard(root / "cc12m" / "part-00000.tar")
            _, spec, chosen = self.write_curated(root)
            rows = self.run_check(cc12m, spec)
            self.assertEqual([row["shards"] for row in rows], [1, 1])
            self.assertEqual(rows[1]["bytes"], chosen.stat().st_size)

    def test_check_prefers_sources_json_in_directory(self):
        with tempfile.TemporaryDirectory(prefix="prepare-check-") as directory:
            root = Path(directory)
            cc12m = self.write_shard(root / "cc12m" / "part-00000.tar")
            curated, _, chosen = self.write_curated(root)
            rows = self.run_check(cc12m.parent, curated)
            self.assertEqual([row["shards"] for row in rows], [1, 1])
            self.assertEqual(rows[1]["bytes"], chosen.stat().st_size)

    def test_check_missing_shards_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="prepare-check-") as directory:
            root = Path(directory)
            (root / "empty").mkdir()
            curated = self.write_shard(root / "curated" / "part-00000.tar")
            output = io.StringIO()
            with self.assertRaises(FileNotFoundError):
                self.run_check(root / "empty", curated, output=output)
            self.assertEqual(output.getvalue(), "")

    def test_check_missing_explicit_sources_json_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="prepare-check-") as directory:
            root = Path(directory)
            cc12m = self.write_shard(root / "cc12m" / "part-00000.tar")
            output = io.StringIO()
            with self.assertRaises(FileNotFoundError):
                self.run_check(cc12m, root / "missing.json", output=output)
            self.assertEqual([json.loads(line)["root"] for line in output.getvalue().splitlines()],
                             [str(cc12m)])

    def test_check_missing_manifest_source_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="prepare-check-") as directory:
            root = Path(directory)
            cc12m = self.write_shard(root / "cc12m" / "part-00000.tar")
            curated, spec, _ = self.write_curated(root)
            spec.write_text(json.dumps({"sources": [{"name": "missing", "root": "missing", "weight": 1.0}]}))
            output = io.StringIO()
            with self.assertRaises(FileNotFoundError):
                self.run_check(cc12m, curated, output=output)
            self.assertEqual([json.loads(line)["root"] for line in output.getvalue().splitlines()],
                             [str(cc12m)])


if __name__ == "__main__":
    unittest.main()
