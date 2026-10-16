"""Bounded-memory local WebDataset reader with exact, seek-based restart.

Images and captions must share a key and be contiguous in an uncompressed tar.
The reader never extracts files and does not build an in-memory shard index.
Shard order is deterministically shuffled once per epoch. Different JAX hosts
receive disjoint shards; when there are fewer shards than hosts, those hosts
partition records within a shard. Restore requires the same host count.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image
import tarfile


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _braces(pattern):
    match = re.search(r"\{(\d+)\.\.(\d+)\}", pattern)
    if match is None:
        return [pattern]
    start, stop = map(int, match.groups())
    width = len(match.group(1))
    if stop < start:
        raise ValueError("Descending shard ranges are unsupported")
    return [expanded for index in range(start, stop + 1)
            for expanded in _braces(pattern[:match.start()] + f"{index:0{width}d}" + pattern[match.end():])]


def expand_shards(root):
    """Resolve a directory, glob, brace pattern, or list to sorted local tars."""
    roots = [root] if isinstance(root, (str, os.PathLike)) else list(root)
    paths = []
    for item in roots:
        for pattern in _braces(os.fspath(item)):
            path = Path(pattern).expanduser()
            matches = sorted(path.rglob("*.tar")) if path.is_dir() else sorted(glob.glob(str(path)))
            if not matches:
                raise FileNotFoundError(f"No WebDataset shards match {pattern!r}")
            paths.extend(Path(p).resolve() for p in matches)
    paths = sorted(set(paths))
    if not paths or any(not p.is_file() or p.suffix != ".tar" for p in paths):
        raise ValueError("Use existing, uncompressed local .tar shards")
    return paths


def shard_identity(path):
    """Filesystem identity portable across login/compute GPFS mount devices."""
    path = Path(path).resolve()
    stat = path.stat()
    # st_dev deliberately excluded: the same shared file has different device
    # numbers on different cluster nodes. No same-UID adversary is assumed.
    return {"path": str(path), "inode": stat.st_ino, "uid": stat.st_uid,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _image(raw, image_size):
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= height:
            target = (image_size, int(image_size * height / width))
        else:
            target = (int(image_size * width / height), image_size)
        image = image.resize(target, Image.Resampling.BICUBIC)
        left = int(round((target[0] - image_size) / 2))
        top = int(round((target[1] - image_size) / 2))
        image = image.crop((left, top, left + image_size, top + image_size))
        return np.asarray(image, dtype=np.float32) / 127.5 - 1.0


class _ShardStream:
    def __init__(self, root, process_index, process_count, seed, image_size, state=None):
        self.paths = expand_shards(root)
        self.identities = [shard_identity(p) for p in self.paths]
        self.manifest_sha256 = _digest(self.identities)
        self.rank, self.world = int(process_index), int(process_count)
        if self.world < 1 or not 0 <= self.rank < self.world:
            raise ValueError("Invalid data process index/count")
        self.seed, self.image_size = int(seed), int(image_size)
        self.epoch, self.shard_cursor, self.byte_offset, self.record_index = 0, 0, 0, 0
        self.samples_emitted = 0
        self._tar = None
        if state is not None:
            expected = {"manifest_sha256": self.manifest_sha256, "process_index": self.rank,
                        "process_count": self.world, "seed": self.seed, "image_size": self.image_size}
            if any(state.get(key) != value for key, value in expected.items()):
                raise ValueError("Data restart differs in shards, process topology, seed, or image size")
            for key in ("epoch", "shard_cursor", "byte_offset", "record_index", "samples_emitted"):
                value = int(state[key])
                if value < 0:
                    raise ValueError("Negative data cursor")
                setattr(self, key, value)
        self._set_epoch()
        if self.shard_cursor >= len(self._assigned):
            raise ValueError("Data shard cursor is outside its epoch")

    def _set_epoch(self):
        order = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(len(self.paths))
        if len(order) >= self.world:
            self._assigned = list(order[self.rank::self.world])
            self._record_rank, self._record_world = 0, 1
        else:
            slot = self.rank % len(order)
            self._assigned = [int(order[slot])]
            self._record_rank = self.rank // len(order)
            self._record_world = (self.world - 1 - slot) // len(order) + 1

    def _open(self):
        path = self.paths[self._assigned[self.shard_cursor]]
        if shard_identity(path) != self.identities[self._assigned[self.shard_cursor]]:
            raise ValueError(f"Dataset shard changed since startup: {path}")
        # r: rejects compressed tars; saved byte offsets are directly seekable.
        self._tar = tarfile.open(path, "r:")
        if self.byte_offset:
            if self.byte_offset % tarfile.BLOCKSIZE or self.byte_offset > path.stat().st_size:
                raise ValueError("Invalid saved tar byte offset")
            self._tar.firstmember = None
            self._tar.offset = self.byte_offset

    def _advance(self):
        self.close()
        self.shard_cursor += 1
        if self.shard_cursor == len(self._assigned):
            self.epoch += 1
            self.shard_cursor = 0
            self._set_epoch()
        self.byte_offset, self.record_index = 0, 0

    def next_sample(self):
        empty_shards = 0
        while True:
            if self._tar is None:
                self._open()
            key, image, caption = None, None, None
            while True:
                member = self._tar.next()
                # TarFile otherwise retains every prior TarInfo in a list.
                self._tar.members.clear()
                if member is None:
                    if image is not None or caption is not None:
                        raise ValueError(f"Unpaired image/caption at end of {self._tar.name}")
                    if self.record_index == 0:
                        raise ValueError(f"Empty WebDataset shard: {self._tar.name}")
                    self._advance()
                    empty_shards += 1
                    if empty_shards > len(self._assigned) + 1:
                        raise ValueError("This host has no records in its dataset partition")
                    break
                self.byte_offset = self._tar.offset
                if not member.isfile() or "." not in member.name:
                    continue
                stem, extension = member.name.rsplit(".", 1)
                extension = extension.lower()
                if extension not in ("jpg", "jpeg", "png", "txt"):
                    continue
                if key is not None and stem != key:
                    raise ValueError(f"WebDataset members must form contiguous image/txt pairs: {key}")
                key = stem
                selected = self.record_index % self._record_world == self._record_rank
                if extension == "txt":
                    if caption is not None:
                        raise ValueError(f"Duplicate caption for {key}")
                    caption = self._tar.extractfile(member).read() if selected else b""
                else:
                    if image is not None:
                        raise ValueError(f"Duplicate image for {key}")
                    image = self._tar.extractfile(member).read() if selected else b""
                if image is not None and caption is not None:
                    self.record_index += 1
                    if selected:
                        self.samples_emitted += 1
                        return _image(image, self.image_size), caption.decode("utf-8").strip(), key
                    key, image, caption = None, None, None

    def state_dict(self):
        return {"manifest_sha256": self.manifest_sha256, "process_index": self.rank,
                "process_count": self.world, "seed": self.seed, "image_size": self.image_size,
                "epoch": self.epoch, "shard_cursor": self.shard_cursor,
                "byte_offset": self.byte_offset, "record_index": self.record_index,
                "samples_emitted": self.samples_emitted}

    def close(self):
        if self._tar is not None:
            self._tar.close()
            self._tar = None


class WebDatasetStream:
    """Local shard stream, or weighted sources from a prepared sources.json.

    next_batch(n) returns float32 NHWC images, Python captions, and sample keys.
    State records the exact next tar offset, stream counters and metadata binding.
    No sample is silently dropped, no prefetch state is lost on checkpoint, and
    restoring never replays previous epochs. Compression and remote URLs are
    intentionally outside this storage-backed training path.
    """
    def __init__(self, root, process_index=0, process_count=1, seed=42, image_size=512, state=None):
        self._root = root
        self.rank, self.world = int(process_index), int(process_count)
        self.seed, self.image_size, self.mix_counter = int(seed), int(image_size), 0
        path = Path(root) if isinstance(root, (str, os.PathLike)) else None
        if path is not None and path.is_dir() and (path / "sources.json").is_file():
            path = path / "sources.json"
        if path is not None and path.suffix == ".json":
            spec = json.loads(path.read_text())
            sources = spec["sources"]
            roots = [str((path.parent / s["root"]).resolve()) if not Path(s["root"]).is_absolute()
                     else s["root"] for s in sources]
            self.weights = np.asarray([s["weight"] for s in sources], dtype=np.float64)
        else:
            roots, self.weights = [root], np.asarray([1.0])
        if not len(roots) or np.any(self.weights <= 0) or not np.all(np.isfinite(self.weights)):
            raise ValueError("Source mixture requires finite, positive weights")
        self.weights /= self.weights.sum()
        if state is not None:
            if state.get("version") != 1 or len(state["streams"]) != len(roots):
                raise ValueError("Unsupported data restart state")
            if not np.array_equal(np.asarray(state["weights"]), self.weights):
                raise ValueError("Mixture weights changed at restart")
            self.mix_counter = int(state["mix_counter"])
            if self.mix_counter < 0:
                raise ValueError("Negative mixture counter")
        self.streams = [_ShardStream(r, self.rank, self.world, self.seed + i * 1009, self.image_size,
                                    None if state is None else state["streams"][i])
                        for i, r in enumerate(roots)]
        self.manifest_sha256 = _digest({"streams": [s.manifest_sha256 for s in self.streams],
                                       "weights": self.weights.tolist()})
        if state is not None and state.get("manifest_sha256") != self.manifest_sha256:
            raise ValueError("Dataset mixture manifest changed at restart")

    def next_batch(self, count):
        if int(count) != count or count < 1:
            raise ValueError("Batch size must be positive")
        images, captions, keys = [], [], []
        for _ in range(int(count)):
            # Counter-based choice is reproducible without serializing an RNG.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.mix_counter]))
            source = int(rng.choice(len(self.streams), p=self.weights))
            image, caption, key = self.streams[source].next_sample()
            self.mix_counter += 1
            images.append(image)
            captions.append(caption)
            keys.append(f"{source}:{key}")
        return {"images": np.stack(images), "captions": captions, "keys": keys}

    def state_dict(self):
        return {"version": 1, "manifest_sha256": self.manifest_sha256,
                "weights": self.weights.tolist(), "mix_counter": self.mix_counter,
                "streams": [s.state_dict() for s in self.streams]}

    def load_state_dict(self, state):
        replacement = WebDatasetStream(self._root, self.rank, self.world, self.seed,
                                       self.image_size, state=state)
        self.close()
        self.__dict__.update(replacement.__dict__)

    def close(self):
        for stream in self.streams:
            stream.close()
