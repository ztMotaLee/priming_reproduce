"""Strict, standard-library-only binding for the released FLAN-T5 assets.

Metadata validation hashes the small tokenizer/config files and checks the
complete checkpoint's size. Full validation also hashes the 3.13 GB checkpoint
and belongs on an allocated compute node. Neither mode downloads assets.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


T5_MANIFEST_FORMAT = "priming-t5-assets-v1"
T5_WEIGHT_FILE = "flax_model.msgpack"
# The official metadata snapshot already used by prepare.py, 2026-10-16.
# A PyTorch/safetensors conversion has no supported SHA lock in this release.
T5_SOURCE_LOCK = {
    "repo_id": "google/flan-t5-large",
    "repo_type": "model",
    "revision": "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a",
    "files": [
        {"name": "config.json", "size": 662,
         "sha256": "bfa5beeb5a4630a97f043f071b9b5d858c842604cff5db874680f33b56090c8c"},
        {"name": T5_WEIGHT_FILE, "size": 3132624407,
         "sha256": "44c5f419a8cb7b2c7d2f89c49ea324220ae403178405be0c9ead68d76445e558"},
        {"name": "generation_config.json", "size": 147,
         "sha256": "f5a1c7e2be8092018d8835128987edf0111637dd98e90599cc80310fef75d95a"},
        {"name": "special_tokens_map.json", "size": 2201,
         "sha256": "5c87151ef0f72a99d1f766a4c418bd2a1f90aaa30a8e22fe5eca9641daebb64f"},
        {"name": "spiece.model", "size": 791656,
         "sha256": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86"},
        {"name": "tokenizer.json", "size": 2424064,
         "sha256": "fe2ebbbbde2985be723e0ce18217853e4020c5e9d35bd07be2c27ab9d3ead57a"},
        {"name": "tokenizer_config.json", "size": 2539,
         "sha256": "5d19985330a9123285cc583fc60616d083aa9df7435812b5d8bb3e749f435d56"},
    ],
}


class AssetValidationError(ValueError):
    """Validation failure with the exact deciding values for an outer receipt."""

    def __init__(self, message, **values):
        super().__init__(message)
        self.values = values


def _require(condition, message, **values):
    if not condition:
        raise AssetValidationError(message, **values)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_size(path):
    return Path(path).stat().st_size


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssetValidationError("Duplicate T5 manifest key", key=key)
        result[key] = value
    return result


def _is_weight_file(path):
    name = path.name.lower()
    return (path.suffix.lower() in {".msgpack", ".safetensors", ".bin", ".h5", ".ckpt", ".pt", ".pth", ".npz"}
            or name.endswith((".bin.index.json", ".safetensors.index.json", ".msgpack.index.json")))


def validate_t5_assets(path, *, verify_payloads=True):
    """Return a pinned binding; reject any incomplete or alternative checkpoint.

    ``verify_payloads=False`` still verifies all six small files' SHA256 values.
    Only the full checkpoint SHA is deferred, so its returned status is explicitly
    metadata-only. Runtime callers must use the default full verification.
    """
    path = Path(path).resolve()
    _require(path.is_dir(), "Missing local T5 assets", path=str(path))
    manifest = path / "assets.sha256.json"
    _require(manifest.is_file() and not manifest.is_symlink(),
             "T5 assets require a regular assets.sha256.json from setup", path=str(manifest))
    try:
        value = json.loads(manifest.read_text(), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssetValidationError("Unreadable T5 asset manifest", path=str(manifest),
                                   error=str(error)) from error
    _require(isinstance(value, dict), "T5 asset manifest must be an object",
             actual_type=type(value).__name__)
    expected_fields = {"format", "checkpoint_format", "from_pt", "source", "files"}
    _require(set(value) == expected_fields, "T5 manifest fields differ from the pinned format",
             expected=sorted(expected_fields), actual=sorted(value))
    _require(value["format"] == T5_MANIFEST_FORMAT,
             "Unsupported T5 asset manifest format", expected=T5_MANIFEST_FORMAT, actual=value["format"])
    _require(value["checkpoint_format"] == "flax" and value["from_pt"] is False,
             "Only pinned official Flax T5 weights are supported; from_pt is unsupported",
             expected={"checkpoint_format": "flax", "from_pt": False},
             actual={"checkpoint_format": value["checkpoint_format"], "from_pt": value["from_pt"]})
    _require(value["source"] == T5_SOURCE_LOCK,
             "T5 source differs from the pinned official google/flan-t5-large release",
             expected=T5_SOURCE_LOCK, actual=value["source"])
    expected_files = {entry["name"]: entry["sha256"] for entry in T5_SOURCE_LOCK["files"]}
    _require(value["files"] == expected_files,
             "T5 manifest must list exactly every pinned config, tokenizer and complete weight file",
             expected=expected_files, actual=value["files"])
    # Transformers resolves weights in this directory. Reject any alternative
    # local checkpoint or shard index, including an unlisted from_pt source.
    unlisted = sorted(child.name for child in path.iterdir()
                      if _is_weight_file(child) and child.name not in expected_files)
    _require(not unlisted, "Unlisted T5 weight files are unsupported", files=unlisted)
    verified = []
    for entry in T5_SOURCE_LOCK["files"]:
        target = path / entry["name"]
        _require(target.is_file() and not target.is_symlink(),
                 "Missing regular pinned T5 asset file", path=str(target), file=entry["name"])
        size = _file_size(target)
        _require(size == entry["size"], "Pinned T5 asset size differs",
                 file=entry["name"], expected_size=entry["size"], actual_size=size)
        if verify_payloads or entry["name"] != T5_WEIGHT_FILE:
            digest = _sha256(target)
            _require(digest == entry["sha256"], "Pinned T5 asset SHA256 differs",
                     file=entry["name"], expected_sha256=entry["sha256"], actual_sha256=digest)
            verified.append(entry["name"])
    return {
        "path": str(path), "manifest": str(manifest), "manifest_sha256": _sha256(manifest),
        "source": copy.deepcopy(T5_SOURCE_LOCK), "files": dict(expected_files),
        "checkpoint_format": "flax", "from_pt": False,
        "payloads_verified": bool(verify_payloads), "sha256_verified_files": verified,
        "status": "T5_PAYLOADS_VERIFIED" if verify_payloads else "T5_METADATA_VERIFIED",
    }
