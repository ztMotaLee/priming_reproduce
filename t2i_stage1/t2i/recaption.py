"""Replace existing CC12M alt-text with the pinned MiniT2I LLaVA captions.

The SQLite index is built by streaming compressed metadata with a bounded cache.
Images and sample order are preserved, and only receipt-complete shards resume.
No input files are changed. Filesystem identities deliberately exclude st_dev.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import tarfile
import tempfile
import time

from .data import expand_shards, shard_identity


FORMAT = "cc12m-caption-llava-v1"
GUIDANCE = "Run DOWNLOAD_DATA=1 bash setup.sh to verify or create the LLaVA-caption copy."


class CaptionVerificationError(ValueError):
    """Missing caption provenance or an observed caption-source mismatch."""


def _helpers():
    # prepare owns the remote lock and common atomic writer; import lazily.
    from .prepare import REMOTE_LOCK, atomic_json, sha256
    return REMOTE_LOCK["cc12m"], atomic_json, sha256


def root_identity(root):
    path = Path(root).expanduser().resolve()
    anchor = path
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    stat = anchor.stat()
    result = {"path": str(path), "inode": stat.st_ino, "uid": stat.st_uid}
    if anchor != path:
        result["anchor"] = str(anchor)
    return result


def _json(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _snapshot(root):
    shards = expand_shards(root)
    return {"root_identity": root_identity(root),
            "shards": [shard_identity(path) for path in shards],
            "url_sidecars": [shard_identity(path.with_suffix(".parquet")) for path in shards
                             if path.with_suffix(".parquet").exists()]}


def _index_receipt(index):
    return Path(str(index) + ".receipt.json")


def _metadata_path(metadata, data_dir):
    return Path(metadata) if metadata is not None else Path(data_dir) / "raw" / "cc12m" / "train.jsonl.gz"


class _HashReader(io.RawIOBase):
    def __init__(self, handle):
        self.handle, self.digest = handle, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.handle.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


def build_index(metadata, data_dir, expected=None):
    """Build one immutable index for a size/SHA-pinned compressed JSONL table.

    ``expected`` is an explicit alternative lock for small offline fixtures.
    Production callers omit it, using the checked-in REMOTE_LOCK unconditionally.
    A failed or interrupted build retains an audit journal and requires an
    explicit new DATA_DIR; it is never retried automatically.
    """
    lock, atomic_json, sha256 = _helpers()
    expected = dict(lock["files"][0] if expected is None else expected)
    expected = {"size": int(expected["size"]), "sha256": expected["sha256"]}
    metadata, data_dir = Path(metadata).resolve(), Path(data_dir).resolve()
    directory = data_dir / "cc12m_caption_index"
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / (expected["sha256"] + ".sqlite")
    receipt_path, progress_path = _index_receipt(index), Path(str(index) + ".progress.json")
    if receipt_path.exists():
        receipt = _json(receipt_path)
        if (receipt.get("format") != FORMAT or receipt.get("status") != "PASS"
                or receipt.get("metadata_lock") != expected
                or receipt.get("index_identity") != shard_identity(index)
                or receipt.get("metadata_identity") != shard_identity(metadata)):
            raise ValueError("Caption index receipt or metadata identity differs")
        return index
    if progress_path.exists() or index.exists():
        raise FileExistsError(f"Incomplete caption index; choose a new DATA_DIR: {progress_path}")
    started = time.monotonic()
    actual_size = metadata.stat().st_size
    progress = {"format": FORMAT, "status": "BUILDING", "build_token": f"{os.getpid()}-{time.time_ns()}", "metadata_lock": expected,
                "metadata_identity": shard_identity(metadata), "actual_size": actual_size,
                "size_matches": actual_size == expected["size"], "rows": 0, "indexed_rows": 0,
                "max_peak_rss_bytes": 8_000_000_000}
    atomic_json(progress_path, progress, create_once=True)
    if not progress["size_matches"]:
        progress.update(status="FAILED", reason="metadata_size")
        atomic_json(progress_path, progress)
        raise ValueError("Caption metadata size differs from the pinned lock")
    fd, temporary = tempfile.mkstemp(prefix=index.name + ".tmp-", dir=directory)
    os.close(fd)
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA threads=1")
        connection.execute("CREATE TABLE captions (url TEXT NOT NULL, original TEXT, llava TEXT NOT NULL)")
        with metadata.open("rb") as raw:
            hashed = _HashReader(raw)
            with io.BufferedReader(hashed, 1 << 20) as buffered:
                with gzip.GzipFile(fileobj=buffered) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as reader:
                        batch = []
                        for line in reader:
                            row = json.loads(line)
                            progress["rows"] += 1
                            if (isinstance(row.get("url"), str) and row["url"]
                                    and isinstance(row.get("caption_llava"), str) and row["caption_llava"]
                                    and row.get("status", "success") == "success"):
                                original = row.get("caption")
                                batch.append((row["url"], original if isinstance(original, str) else None,
                                              row["caption_llava"]))
                            if len(batch) >= 5000:
                                connection.executemany("INSERT INTO captions VALUES (?,?,?)", batch)
                                progress["indexed_rows"] += len(batch)
                                batch.clear()
                                connection.commit()
                            if progress["rows"] % 100000 == 0:
                                progress.update(elapsed_seconds=time.monotonic() - started,
                                                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                                atomic_json(progress_path, progress)
                                if progress["peak_rss_bytes"] > progress["max_peak_rss_bytes"]:
                                    raise ValueError("Caption index peak RSS exceeded 8 GB")
                        if batch:
                            connection.executemany("INSERT INTO captions VALUES (?,?,?)", batch)
                            progress["indexed_rows"] += len(batch)
                # Read any compressed trailing bytes before accepting its SHA.
                while buffered.read(1 << 20):
                    pass
            actual_sha = hashed.digest.hexdigest()
        progress.update(actual_sha256=actual_sha, sha256_matches=actual_sha == expected["sha256"],
                        elapsed_seconds=time.monotonic() - started,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        atomic_json(progress_path, progress)
        if not progress["sha256_matches"]:
            raise ValueError("Caption metadata SHA256 differs from the pinned lock")
        if progress["peak_rss_bytes"] > progress["max_peak_rss_bytes"]:
            raise ValueError("Caption index peak RSS exceeded 8 GB")
        connection.execute("CREATE INDEX exact_url ON captions(url COLLATE BINARY)")
        connection.execute("PRAGMA user_version=1")
        connection.commit()
        connection.close()
        connection = None
        progress.update(elapsed_seconds=time.monotonic() - started,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        atomic_json(progress_path, progress)
        if progress["peak_rss_bytes"] > progress["max_peak_rss_bytes"]:
            raise ValueError("Caption index peak RSS exceeded 8 GB")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, index)  # create-once, including concurrent callers
        index_sha = sha256(index)
        receipt = dict(progress, status="PASS", index_sha256=index_sha,
                       elapsed_seconds=time.monotonic() - started,
                       index_identity=shard_identity(index), format=FORMAT)
        atomic_json(receipt_path, receipt, create_once=True)
        atomic_json(progress_path, dict(receipt))
        return index
    except BaseException as error:
        progress.update(status="FAILED", error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.monotonic() - started,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        atomic_json(progress_path, progress)
        raise
    finally:
        if connection is not None:
            connection.close()
        if os.path.exists(temporary):
            os.unlink(temporary)  # only this invocation's temporary database


def open_index(index):
    connection = sqlite3.connect(Path(index).resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA cache_size=-8192")
    return connection


def resolve_caption(connection, url, original=None):
    """Exact URL comparison; use exact original caption only to resolve conflict."""
    originals = set(original if isinstance(original, (list, tuple)) else [original])
    originals.discard(None)
    all_values, matched = set(), set()
    for source, llava in connection.execute("SELECT original,llava FROM captions WHERE url=? COLLATE BINARY", (url,)):
        if len(all_values) < 2:
            all_values.add(llava)
        if source in originals and len(matched) < 2:
            matched.add(llava)
    if len(all_values) == 1:
        return next(iter(all_values)), "matched"
    if not all_values:
        return None, "unmatched_url"
    if len(matched) == 1:
        return next(iter(matched)), "matched_disambiguated"
    return None, "ambiguous_url"


def iter_samples(shard):
    """Yield contiguous samples as [(TarInfo, original bytes), ...]."""
    with tarfile.open(shard, "r|") as archive:
        key, sample = None, []
        for member in archive:
            # Streaming tarfile still retains each TarInfo unless cleared.
            archive.members.clear()
            if not member.isfile() or "." not in member.name:
                continue
            next_key = member.name.rsplit(".", 1)[0]
            if key is not None and next_key != key:
                yield sample
                sample = []
            key = next_key
            sample.append((member, archive.extractfile(member).read()))
        if sample:
            yield sample


def _parquet_urls(shard):
    path = Path(shard).with_suffix(".parquet")
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path)
    if not {"key", "url"}.issubset(table.schema_arrow.names):
        raise ValueError(f"Parquet needs key and url columns: {path}")
    result = {}
    for batch in table.iter_batches(batch_size=4096, columns=["key", "url"]):
        for row in batch.to_pylist():
            key, url = str(row["key"]), row["url"]
            if key in result and result[key] != url:
                raise ValueError(f"Conflicting URL for parquet key {key}: {path}")
            result[key] = url
            if len(result) > 500000:
                raise ValueError(f"Parquet shard exceeds bounded key-map limit (500000): {path}")
    return result


def _has_json_url(sample):
    for member, raw in sample:
        if member.name.rsplit(".", 1)[-1].lower() == "json":
            try:
                value = json.loads(raw)
                if isinstance(value, dict) and isinstance(value.get("url"), str) and value["url"]:
                    return True
            except (ValueError, UnicodeDecodeError):
                pass
    return False


def sample_fields(sample, shard, parquet=None):
    key = sample[0][0].name.rsplit(".", 1)[0]
    by_extension, critical_members = {}, set()
    for member, raw in sample:
        extension = member.name.rsplit(".", 1)[-1].lower()
        role = "image" if extension in {"jpg", "jpeg", "png"} else extension
        if role in {"image", "txt", "json"}:
            if role in critical_members:
                raise ValueError(f"Duplicate {role} member for key {key!r}: {shard}; "
                                 "consecutive records with the same key are unsupported")
            critical_members.add(role)
        by_extension[extension] = raw
    metadata = {}
    if "json" in by_extension:
        try:
            metadata = json.loads(by_extension["json"])
            if not isinstance(metadata, dict):
                metadata = {}
        except (ValueError, UnicodeDecodeError):
            metadata = {}
    url = metadata.get("url")
    if not isinstance(url, str) or not url:
        if parquet is None:
            parquet = _parquet_urls(shard)
        url = None if parquet is None else parquet.get(key)
    if not isinstance(url, str) or not url:
        raise ValueError(f"No URL in sample {key!r} JSON or same-name shard parquet: {shard}")
    original = metadata.get("caption")
    text = by_extension.get("txt", b"").decode("utf-8")
    if not isinstance(original, str):
        original = text
    elif original != text:
        original = (original, text)
    return url, original, by_extension


def _recaption_shard(arguments):
    shard, target, index, binding = arguments
    shard, target, index = Path(shard), Path(target), Path(index)
    lock, atomic_json, sha256 = _helpers()
    receipt_path = target.parent / ".recaption_shards" / (target.name + ".json")
    progress_path = target.parent / ".recaption_shards" / (target.name + ".progress.json")
    source = shard_identity(shard)
    if receipt_path.exists():
        receipt = _json(receipt_path)
        if (receipt.get("source") != source or receipt.get("binding") != binding
                or receipt.get("output_identity") != shard_identity(target)
                or receipt.get("output_sha256") != sha256(target)):
            raise ValueError(f"Completed recaption shard binding differs: {target}")
        return receipt
    started = time.monotonic()
    counts = {"input": 0, "matched": 0, "matched_disambiguated": 0,
              "unmatched_url": 0, "ambiguous_url": 0, "missing_members": 0}
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".tmp-", dir=target.parent)
    os.close(fd)
    connection = open_index(index)
    try:
        parquet = None
        parquet_loaded = False
        with tarfile.open(temporary, "w") as output:
            for sample in iter_samples(shard):
                counts["input"] += 1
                # Read the companion parquet at most once, only if needed.
                has_url = _has_json_url(sample)
                if not has_url and not parquet_loaded:
                    parquet, parquet_loaded = _parquet_urls(shard) or {}, True
                url, original, fields = sample_fields(sample, shard, parquet)
                if not {"jpg", "jpeg", "png"}.intersection(fields) or "txt" not in fields:
                    counts["missing_members"] += 1
                    continue
                caption, reason = resolve_caption(connection, url, original)
                if caption is None:
                    counts[reason] += 1
                    continue
                counts["matched"] += 1
                if reason == "matched_disambiguated":
                    counts["matched_disambiguated"] += 1
                for member, raw in sample:
                    if member.name.rsplit(".", 1)[-1].lower() == "txt":
                        raw = caption.encode("utf-8")
                    new_member = copy.copy(member)
                    new_member.size = len(raw)
                    output.addfile(new_member, io.BytesIO(raw))
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        digest = sha256(temporary)
        os.replace(temporary, target)
        receipt = {"format": FORMAT, "status": "PASS", "source": source,
                   "binding": binding, "counts": counts, "output_sha256": digest,
                   "output_identity": shard_identity(target), "elapsed_seconds": time.monotonic() - started}
        atomic_json(receipt_path, receipt, create_once=True)
        atomic_json(progress_path, receipt)
        return receipt
    except BaseException as error:
        atomic_json(progress_path, {"format": FORMAT, "status": "FAILED", "source": source,
                                   "binding": binding, "counts": counts,
                                   "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        connection.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


def _receipt_path(root, data_dir):
    root = Path(root).expanduser().resolve()
    if root.is_dir() and (root / "caption_receipt.json").exists():
        return root / "caption_receipt.json"
    if data_dir is None:
        raise CaptionVerificationError(f"No caption verification receipt for {root}. {GUIDANCE}")
    return Path(data_dir).resolve() / "caption_receipts" / (_digest(root_identity(root)) + ".json")


def _binding(path, receipt):
    _, _, sha256 = _helpers()
    return {"caption_source": "caption_llava", "receipt_path": str(Path(path).resolve()),
            "receipt_sha256": sha256(path), "root_identity": receipt["root_identity"],
            "source": receipt["source"]}


def validate_caption_receipt(root, data_dir=None):
    """Read-only training guard: root and every shard must retain their identity."""
    lock, _, sha256 = _helpers()
    path = _receipt_path(root, data_dir)
    if not path.exists():
        raise CaptionVerificationError(f"Missing caption verification receipt for {root}. {GUIDANCE}")
    receipt = _json(path)
    snapshot = _snapshot(root)
    if (receipt.get("format") != FORMAT or receipt.get("status") != "PASS"
            or receipt.get("caption_source") != "caption_llava" or receipt.get("source") != lock
            or receipt.get("root_identity") != root_identity(root)
            or receipt.get("shards") != snapshot["shards"]
            or receipt.get("url_sidecars", []) != snapshot["url_sidecars"]):
        raise CaptionVerificationError(f"Caption receipt is stale or differs from the data root. {GUIDANCE}")
    if receipt.get("preparation_sha256") is not None:
        preparation = Path(root) / "preparation.json"
        if not preparation.exists() or sha256(preparation) != receipt["preparation_sha256"]:
            raise CaptionVerificationError(f"Caption preparation binding differs. {GUIDANCE}")
    return _binding(path, receipt)


def _write_caption_receipt(root, data_dir, *, kind, preparation_sha256=None, details=None, local=False):
    lock, atomic_json, _ = _helpers()
    path = Path(root) / "caption_receipt.json" if local else _receipt_path(root, data_dir)
    if not local and (path.resolve() == Path(root).resolve() or Path(root).resolve() in path.resolve().parents):
        raise ValueError("Caption verification DATA_DIR must not write inside the input root")
    receipt = dict(_snapshot(root), format=FORMAT, status="PASS", caption_source="caption_llava",
                   source=lock, kind=kind, preparation_sha256=preparation_sha256, details=details or {})
    atomic_json(path, receipt, create_once=True)
    return _binding(path, receipt)


def check_caption_write_paths(root, data_dir, *, output=None, download=False,
                              verify=False, mapping=False):
    """Read-only admission of the actual derived writers, including symlinks.

    DATA_DIR may be an input ancestor. Its reserved write paths must stay
    outside the input, even when metadata is already cached. Explicit metadata
    is only read, so its location does not itself require admission.
    """
    source, cache = Path(root).expanduser().resolve(), Path(data_dir).expanduser().resolve()
    if cache == source or source in cache.parents:
        raise ValueError("DATA_DIR must be outside the input root, with separate write paths")
    paths = {"caption index": cache / "cc12m_caption_index"}
    if download:
        paths["metadata download"] = cache / "raw" / "cc12m"
    if verify:
        paths["caption receipts"] = cache / "caption_receipts"
        receipt = _receipt_path(root, cache)
        paths["caption receipt"] = receipt
        paths["failed caption receipt"] = receipt.with_name(receipt.stem + ".failed.json")
    if mapping:
        paths["root mapping"] = cache / "cc12m_root_mapping.json"
    if output is not None:
        target = Path(output).expanduser().resolve()
        if target == source or source in target.parents or target in source.parents:
            raise ValueError("Recaption output must be separate from the input root")
        paths["recaption output"] = target
    for role, path in paths.items():
        target = path.resolve()
        if target == source or source in target.parents:
            raise ValueError(f"Caption {role} write path overlaps the input root: {target}")


def check_recaption_output(cc12m_root, output, data_dir=None, expected=None, *, download=False):
    """Read-only admission shared by the CLI before downloads and conversion."""
    lock, _, _ = _helpers()
    output = Path(output).expanduser().resolve()
    data_dir = Path(data_dir).resolve() if data_dir is not None else output.parent
    check_caption_write_paths(cc12m_root, data_dir, output=output, download=download)
    plan_path = output / ".recaption_plan.json"
    plan = dict(_snapshot(cc12m_root), format=FORMAT, source=lock,
                metadata_lock={key: (lock["files"][0] if expected is None else expected)[key]
                               for key in ("size", "sha256")})
    # Admission is entirely read-only: even mkdir(exist_ok=True) belongs after
    # it. A matching exclusive plan is the only ownership proof for a nonempty
    # output, including one left by an interrupted conversion.
    if output.exists() and not output.is_dir():
        raise FileExistsError("Recaption output must be a directory")
    if plan_path.exists():
        if _json(plan_path) != plan:
            raise ValueError("Recaption input/metadata differs from its exclusive output plan")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Recaption refuses a nonempty output directory without its own plan")
    return plan


def recaption(cc12m_root, output, metadata=None, processes=1, data_dir=None, expected=None):
    """Create/resume an image-byte-identical LLaVA caption copy of existing tars."""
    lock, atomic_json, sha256 = _helpers()
    output = Path(output).expanduser().resolve()
    data_dir = Path(data_dir).resolve() if data_dir is not None else output.parent
    if not 1 <= int(processes) <= 16:
        raise ValueError("Recaption processes must be between 1 and 16")
    plan = check_recaption_output(cc12m_root, output, data_dir=data_dir, expected=expected)
    snapshot = plan
    source_paths = expand_shards(cc12m_root)
    source_root = Path(cc12m_root).expanduser().resolve()
    # Keep original relative shard names and subdirectories.
    base = source_root if source_root.is_dir() else Path(os.path.commonpath([str(p.parent) for p in source_paths]))
    plan_path = output / ".recaption_plan.json"
    output.mkdir(parents=True, exist_ok=True)
    if not plan_path.exists():
        atomic_json(plan_path, plan, create_once=True)
    (output / ".recaption_shards").mkdir(exist_ok=True)
    if (output / "preparation.json").exists():
        preparation = _json(output / "preparation.json")
        if not (output / "caption_receipt.json").exists():
            # Recover the narrow crash window after preparation was published.
            # Verify the immutable plan, exact shard set and all file hashes;
            # never reinterpret a foreign or incomplete output as complete.
            files = {str(path.relative_to(output)): sha256(path) for path in expand_shards(output)}
            if (preparation.get("source") != lock or preparation.get("caption_source") != "caption_llava"
                    or preparation.get("metadata_lock") != plan["metadata_lock"]
                    or preparation.get("input_root_identity") != snapshot["root_identity"]
                    or preparation.get("input_shards") != snapshot["shards"]
                    or preparation.get("input_url_sidecars", []) != snapshot["url_sidecars"]
                    or preparation.get("files") != files or preparation.get("match_gate_passed") is not True
                    or preparation.get("empty_shards")):
                raise ValueError("Incomplete recaption preparation cannot be recovered safely")
            _write_caption_receipt(output, data_dir, kind="recaption", local=True,
                                   preparation_sha256=sha256(output / "preparation.json"),
                                   details={"index_receipt_sha256": preparation["index_receipt_sha256"],
                                            "plan_sha256": sha256(plan_path)})
        validate_caption_receipt(output, data_dir)
        return preparation
    total_bytes = sum(path.stat().st_size for path in source_paths)
    free_bytes = shutil.disk_usage(output).free
    space = {"input_tar_bytes": total_bytes, "free_bytes": free_bytes,
             "required_free_bytes": math.ceil(1.05 * total_bytes), "space_multiplier": 1.05}
    space["passed"] = free_bytes >= space["required_free_bytes"]
    progress = {"format": FORMAT, "status": "PREFLIGHT", "space": space}
    atomic_json(output / "recaption_progress.json", progress)
    if not space["passed"]:
        progress.update(status="FAILED", reason="insufficient_space")
        atomic_json(output / "recaption_progress.json", progress)
        raise ValueError("Recaption needs free space >= 1.05 times the input tar size")
    index = build_index(_metadata_path(metadata, data_dir), data_dir, expected=expected)
    index_binding = {"index_receipt_sha256": sha256(_index_receipt(index)), "plan_sha256": sha256(plan_path)}
    arguments = []
    for shard in source_paths:
        target = output / shard.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        (target.parent / ".recaption_shards").mkdir(exist_ok=True)
        arguments.append((str(shard), str(target), str(index), index_binding))
    try:
        if int(processes) == 1:
            receipts = [_recaption_shard(argument) for argument in arguments]
        else:
            with ProcessPoolExecutor(max_workers=int(processes)) as executor:
                receipts = list(executor.map(_recaption_shard, arguments))
        input_samples = sum(item["counts"]["input"] for item in receipts)
        samples = sum(item["counts"]["matched"] for item in receipts)
        rate = samples / input_samples if input_samples else 0.0
        empty_shards = [item["output_identity"]["path"] for item in receipts if item["counts"]["matched"] == 0]
        preparation = {"format": FORMAT, "caption_source": "caption_llava", "source": lock,
                       "input_root_identity": snapshot["root_identity"], "input_shards": snapshot["shards"],
                       "input_url_sidecars": snapshot["url_sidecars"],
                       "metadata_lock": plan["metadata_lock"], "index_receipt_sha256": index_binding["index_receipt_sha256"],
                       "samples": samples, "input_samples": input_samples, "match_rate": rate,
                       "minimum_match_rate": 0.5, "match_gate_passed": rate >= 0.5,
                       "empty_shards": empty_shards, "minimum_samples_per_shard": 1,
                       "nonempty_shard_gate_passed": not empty_shards,
                       "shards": receipts, "files": {str(Path(item["output_identity"]["path"]).relative_to(output)):
                                                       item["output_sha256"] for item in receipts}}
        progress.update(status="PASS" if rate >= 0.5 and not empty_shards else "FAILED", preparation=preparation,
                        reason="match_rate" if rate < 0.5 else "empty_shard" if empty_shards else None)
        atomic_json(output / "recaption_progress.json", progress)
        if rate < 0.5:
            raise ValueError(f"Recaption match rate {samples}/{input_samples}={rate:.6f} is below 0.5")
        if empty_shards:
            raise ValueError(f"Recaption produced {len(empty_shards)} empty shards; training needs at least one sample in each")
        atomic_json(output / "preparation.json", preparation, create_once=True)
        _write_caption_receipt(output, data_dir, kind="recaption", local=True,
                               preparation_sha256=sha256(output / "preparation.json"), details=index_binding)
        return preparation
    except BaseException as error:
        progress.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        atomic_json(output / "recaption_progress.json", progress)
        raise


def ensure_caption_receipt(root, data_dir, metadata=None, expected=None, sample_size=2000):
    """Trust checked preparation provenance, otherwise verify deterministic samples."""
    lock, atomic_json, sha256 = _helpers()
    try:
        return validate_caption_receipt(root, data_dir)
    except CaptionVerificationError:
        pass
    check_caption_write_paths(root, data_dir, verify=True)
    preparation_path = Path(root) / "preparation.json"
    if preparation_path.exists():
        preparation = _json(preparation_path)
        if preparation.get("caption_source") == "caption_llava" and preparation.get("source") == lock:
            paths = expand_shards(root)
            actual = {str(path.relative_to(Path(root).resolve())): sha256(path) for path in paths}
            if preparation.get("files") != actual:
                raise ValueError("CC12M preparation file SHA map differs from its root")
            if preparation.get("match_gate_passed") is False:
                raise ValueError("CC12M recaption match gate did not pass")
            return _write_caption_receipt(root, data_dir, kind="preparation",
                                           preparation_sha256=sha256(preparation_path))
    metadata = _metadata_path(metadata, data_dir)
    if not metadata.exists():
        raise CaptionVerificationError(f"Pinned metadata is needed to verify CC12M captions. {GUIDANCE}")
    if int(sample_size) < 1:
        raise ValueError("Caption sample size must be positive")
    index = build_index(metadata, data_dir, expected=expected)
    paths = expand_shards(root)
    count = min(len(paths), int(sample_size))
    selected = [paths[(i * len(paths)) // count] for i in range(count)]
    quota, remainder = divmod(int(sample_size), count)
    sampled = equal = missing = 0
    connection = open_index(index)
    try:
        for number, shard in enumerate(selected):
            parquet = None
            parquet_loaded = False
            for offset, sample in enumerate(iter_samples(shard)):
                if offset >= quota + int(number < remainder):
                    break
                sampled += 1
                try:
                    # The fallback table is bounded and loaded once per shard.
                    if not _has_json_url(sample) and not parquet_loaded:
                        parquet, parquet_loaded = _parquet_urls(shard) or {}, True
                    url, original, fields = sample_fields(sample, shard, parquet)
                except ValueError:
                    missing += 1
                    continue
                caption, _ = resolve_caption(connection, url, original)
                if caption is not None and fields.get("txt") == caption.encode("utf-8"):
                    equal += 1
        rate = equal / sampled if sampled else 0.0
        verification = dict(_snapshot(root), format=FORMAT, status="PASS" if rate >= 0.99 else "FAILED",
                            caption_source="caption_llava", source=lock, kind="sampled",
                            preparation_sha256=None,
                            details={"requested_samples": int(sample_size), "sampled": sampled, "equal": equal,
                                     "missing_url": missing, "match_rate": rate, "minimum_match_rate": 0.99,
                                     "passed": rate >= 0.99, "index_receipt_sha256": sha256(_index_receipt(index))})
        # Record every value used by the precision/source gate before raising.
        path = _receipt_path(root, data_dir)
        if Path(root).resolve() in path.resolve().parents:
            raise ValueError("Caption verification DATA_DIR must not write inside the input root")
        if verification["status"] != "PASS":
            failure_path = path.with_name(path.stem + ".failed.json")
            atomic_json(failure_path, verification)
            raise CaptionVerificationError(f"CC12M captions match LLaVA in {equal}/{sampled} samples ({rate:.2%}); need >=99%. {GUIDANCE}")
        atomic_json(path, verification, create_once=True)
        return _binding(path, verification)
    finally:
        connection.close()
