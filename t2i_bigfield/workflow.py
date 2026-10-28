"""Resumable, site-configured Slurm orchestration. Default: read-only PLAN_ONLY.

Only --submit creates a run or invokes a job-creating scheduler command. Each
logical job has one immutable submission journal; ambiguous or failed attempts
are never retried automatically. The scientific implementation is unchanged.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

HERE = Path(__file__).resolve().parent
PLAN_ONLY_CONTEXT = ContextVar("plan_only", default=False)
READOUTS = (1000, 2000, 4000, 8000, 12000, 16000, 24000, 30000)
TRAIN_SIGNAL_SECONDS = 180
SIGNAL_EARLY_SECONDS = 60
SIGNAL_BUDGET_SECONDS = TRAIN_SIGNAL_SECONDS + SIGNAL_EARLY_SECONDS
CACHE_KEYS = ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
              "PYTORCH_PRETRAINED_BERT_CACHE", "HF_HUB_CACHE",
              "HUGGINGFACE_HUB_CACHE", "JAX_COMPILATION_CACHE_DIR",
              "JAX_ENABLE_COMPILATION_CACHE")
FAILED = {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL",
          "BOOT_FAIL", "PREEMPTED", "DEADLINE", "REVOKED"}
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
          "RESIZING", "REQUEUED", "SIGNALING", "STAGE_OUT"}
SITE_KEYS = ("ACCOUNT", "PARTITION", "NODES", "GPUS_PER_NODE", "CPUS_PER_TASK",
             "MEM_PER_NODE", "MAX_WALLTIME", "CPU_PARTITION", "CPU_QOS",
             "CPU_TIME", "GPU_TYPE", "GPU_QOS_SHORT", "GPU_QOS_LONG", "PYTHON",
             "CPU_CPUS_PER_TASK", "CPU_MEM_PER_NODE", "DATA_ROOT", "OUTPUT_ROOT",
             "SMOKE_TIME", "BANK_TIME", "EVAL_TIME", "MIN_FREE_BYTES", "MIN_FREE_INODES")
QOS_KEYS = ("CPU_QOS", "GPU_QOS_SHORT", "GPU_QOS_LONG")


class SubmissionRejected(RuntimeError):
    """An explicit scheduler rejection, distinct from an ambiguous response."""


class ManualAdvanceRequired(RuntimeError):
    """The compute-node submission was explicitly rejected before any job existed."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value, *, create_once=False):
    """Publish fully fsynced JSON; create-once uses atomic link, never overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if create_once:
        os.link(temporary, path)
        temporary.unlink()
    else:
        os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path):
    return json.loads(Path(path).read_text())


def duration(value):
    """Accept the two walltime spellings returned by Slurm."""
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", str(value))
    if not match:
        # sinfo commonly prints minutes:seconds for short partition limits.
        short = re.fullmatch(r"(\d+):(\d{2})", str(value))
        if short:
            minutes, seconds = map(int, short.groups())
            if seconds < 60 and minutes * 60 + seconds > 0:
                return minutes * 60 + seconds
        raise ValueError("Walltime must use [days-]HH:MM:SS")
    days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
    if minutes >= 60 or seconds >= 60 or (days and hours >= 24):
        raise ValueError("Invalid walltime")
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise ValueError("Walltime must be positive")
    return total


def walltime(seconds):
    seconds = requested_seconds(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, second = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{second:02d}"


def requested_seconds(seconds):
    """Slurm rounds time limits up to minutes; bind every budget to that value."""
    seconds = float(seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Requested walltime must be finite and positive")
    return int(math.ceil(seconds / 60)) * 60


def memory_megabytes(value):
    match = re.fullmatch(r"([1-9]\d*)([KMGT]?)", str(value), re.I)
    if not match:
        raise ValueError("Invalid Slurm memory quantity")
    scale = {"K": 1 / 1024, "": 1, "M": 1, "G": 1024, "T": 1024 ** 2}
    return max(1, int(int(match[1]) * scale[match[2].upper()]))


def clean_environment(env=None):
    result = dict(os.environ if env is None else env)
    for key in CACHE_KEYS:
        result.pop(key, None)
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def command(command, *, check=True, env=None):
    result = subprocess.run([str(x) for x in command], text=True, capture_output=True,
                            env=clean_environment(env), timeout=45)
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(map(str, command[:3]))}: {result.stderr.strip() or result.stdout.strip()}")
    return result


def select_layout(gpus_per_node, maximum_nodes=16):
    gpus_per_node = int(gpus_per_node)
    layouts = [nodes for nodes in range(1, min(16, int(maximum_nodes)) + 1)
               if 0 < nodes * gpus_per_node <= 16 and 64 % (nodes * gpus_per_node) == 0]
    if not layouts:
        raise ValueError("No <=16-GPU layout divides the fixed 64 caption groups")
    return max(layouts)


def partition_inventory():
    """Read one row per physical node, including nodes that are currently busy."""
    result = command(["sinfo", "-h", "-N", "-o", "%P|%N|%G|%c|%m|%l"], check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Partition discovery unavailable")
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 6:
            continue
        name, node, gres, cpus, memory, limit = fields
        if not node or not cpus.isdecimal() or not memory.isdecimal():
            continue
        unlimited = limit.strip().lower() in {"infinite", "unlimited"}
        try:
            seconds = None if unlimited else duration(limit)
        except ValueError:
            continue
        row = {"name": name.rstrip("*"), "node": node, "gres": gres,
               "cpus": int(cpus), "memory": int(memory), "limit": limit,
               "limit_seconds": seconds, "limit_unlimited": unlimited}
        identity = (row["name"], node)
        if identity in rows and rows[identity] != row:
            raise RuntimeError("Conflicting resource rows for one physical node")
        rows[identity] = row
    if not rows:
        raise RuntimeError("Cannot parse physical-node partition inventory")
    return list(rows.values())


def gpu_capacity(row, gpu_type):
    """Return allocatable GPUs of the requested type on this physical node."""
    total = 0
    for item in row["gres"].split(","):
        typed = re.fullmatch(r"gpu:([^:()]+):(\d+)(?:\([^)]*\))?", item, re.I)
        untyped = re.fullmatch(r"gpu:(\d+)(?:\([^)]*\))?", item, re.I)
        if typed and (gpu_type == "generic" or typed[1].lower() == gpu_type.lower()):
            total += int(typed[2])
        elif untyped and gpu_type == "generic":
            total += int(untyped[1])
    return total


def discover_site(env=None):
    """Read-only discovery; unknown values remain explicit questions."""
    env = dict(os.environ if env is None else env)
    found, notes = {}, []
    associations = []
    try:
        account = command(["sacctmgr", "-n", "-P", "show", "assoc",
                           "where", f"user={getpass.getuser()}",
                           "format=Account,Partition,QOS"], check=False)
        if account.returncode == 0:
            for line in account.stdout.splitlines():
                fields = line.split("|")
                if fields[0].strip():
                    associations.append({"account": fields[0].strip(),
                                         "partition": fields[1].strip() if len(fields) > 1 else "",
                                         "qos": fields[2].strip() if len(fields) > 2 else ""})
        choices = sorted({row["account"] for row in associations})
        if len(choices) == 1:
            found["ACCOUNT"] = choices[0]
        elif choices:
            notes.append("Accounts: " + ", ".join(choices))
        else:
            notes.append("Account discovery unavailable; set ACCOUNT (root 待核).")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        notes.append("Account discovery unavailable; set ACCOUNT (root 待核).")
    partitions = []
    try:
        partitions = partition_inventory()
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired):
        notes.append("Partition discovery unavailable; set site parameters (root 待核).")
    # Untyped GRES never establishes the hardware model. A mentor can explicitly
    # confirm generic scheduling using GPU_TYPE=generic (or an empty value);
    # the allocated worker still independently requires physical H100 devices.
    generic = "GPU_TYPE" in env and env["GPU_TYPE"].lower() in ("", "generic")
    groups = {}
    for row in partitions:
        typed = re.search(r"gpu:(h100[^:,()]*):(\d+)", row["gres"], re.I)
        untyped = re.search(r"(?:^|,)gpu:(\d+)(?:$|[,()])", row["gres"], re.I)
        if typed or (generic and untyped):
            gpu_type = "generic" if generic else typed[1]
            physical_count = int(typed[2] if typed else untyped[1])
            gpu_count = min(physical_count, int(env.get("GPUS_PER_NODE", physical_count)))
            if env.get("PARTITION") and row["name"] != env["PARTITION"]:
                continue
            if gpu_capacity(row, gpu_type) >= gpu_count:
                groups.setdefault((row["name"], gpu_type, gpu_count), []).append(row)
    candidates = []
    for (name, gpu_type, gpu_count), rows in groups.items():
        try:
            nodes = select_layout(gpu_count, len({row["node"] for row in rows}))
        except ValueError:
            continue
        candidates.append((nodes * gpu_count, name, gpu_type, gpu_count, nodes, rows))
    if candidates:
        _, name, gpu_type, gpu_count, nodes, rows = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        found.update(PARTITION=name, GPU_TYPE=gpu_type, GPUS_PER_NODE=str(gpu_count),
                     NODES=str(nodes), CPUS_PER_TASK=str(min(row["cpus"] for row in rows)),
                     MEM_PER_NODE=str(max(1, int(min(row["memory"] for row in rows) * 0.9))) + "M",
                     MAX_WALLTIME=walltime(min(86400, min(row["limit_seconds"] if row["limit_seconds"] is not None else 86400
                                                         for row in rows)) // 60 * 60))
    cpu = [row for row in partitions if "gpu:" not in row["gres"].lower()
           and (row["limit_seconds"] is None or row["limit_seconds"] >= 3900)
           and (not env.get("CPU_PARTITION") or row["name"] == env["CPU_PARTITION"])]
    if cpu:
        row = sorted(cpu, key=lambda row: row["name"])[0]
        found.update(CPU_PARTITION=row["name"], CPU_CPUS_PER_TASK=str(min(4, row["cpus"])),
                     CPU_MEM_PER_NODE=str(max(1, min(32768, int(row["memory"] * 0.8)))) + "M")
    # Used only during first configuration, never stored in the run or plan.
    found["_association_records"] = associations
    return found, notes


def qos_disabled(value):
    """Keep the legacy '-' spelling while accepting explicit empty/none values."""
    return str(value).strip().lower() in ("", "none", "-")


def qos_names(associations, account, partition):
    names = set()
    for row in associations:
        if row["account"] != account or row["partition"] not in ("", partition):
            continue
        for value in row["qos"].split(","):
            value = value.strip().lstrip("+")
            if value and re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                names.add(value)
    return names


def qos_limits(priorities=None):
    """Read MaxWall and Priority; None means no published finite limit."""
    result = command(["sacctmgr", "-n", "-P", "show", "qos", "format=Name,MaxWall,Priority"], check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "QOS MaxWall discovery unavailable")
    limits = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 2 or not re.fullmatch(r"[A-Za-z0-9_.-]+", fields[0].strip()):
            continue
        value = fields[1].strip()
        try:
            seconds = None if value.lower() in ("", "infinite", "unlimited") else duration(value)
        except ValueError:
            continue
        name = fields[0].strip()
        limits[name] = seconds
        if priorities is not None:
            priority = fields[2].strip() if len(fields) > 2 else ""
            priorities[name] = int(priority) if priority.isdecimal() else 0
    return limits


def qos_probe(site, key, qos, seconds, *, shape="primary"):
    """Ask the controller to validate resources without creating a job or file."""
    cpu = key == "CPU_QOS"
    nodes = 1 if cpu else int(site["NODES"])
    request = {"partition": site["CPU_PARTITION"] if cpu else site["PARTITION"],
               "nodes": nodes, "gpus_per_node": 0 if cpu else int(site["GPUS_PER_NODE"]),
               "cpus_per_task": int(site["CPU_CPUS_PER_TASK"] if cpu else site["CPUS_PER_TASK"]),
               "mem_per_node": site["CPU_MEM_PER_NODE"] if cpu else site["MEM_PER_NODE"],
               "walltime_seconds": seconds}
    if shape == "light_cpu":
        request.update(cpus_per_task=1, mem_per_node=str(min(4096, memory_megabytes(site["CPU_MEM_PER_NODE"]))) + "M")
    elif shape == "single_gpu":
        nodes = 1
        request.update(nodes=1, gpus_per_node=1, cpus_per_task=min(8, int(site["CPUS_PER_TASK"])),
                       mem_per_node=str(min(131072, memory_megabytes(site["MEM_PER_NODE"]))) + "M")
    args = ["sbatch", "--test-only", "--parsable", "--export=ALL", "--no-requeue",
            f"--chdir={HERE}", f"--account={site['ACCOUNT']}", f"--partition={request['partition']}",
            f"--nodes={nodes}", f"--ntasks={nodes}", "--ntasks-per-node=1",
            f"--cpus-per-task={request['cpus_per_task']}", f"--mem={request['mem_per_node']}",
            f"--time={walltime(seconds)}"]
    if not qos_disabled(qos):
        args.append(f"--qos={qos}")
    if not cpu:
        gpu_type = site["GPU_TYPE"]
        prefix = gpu_type + ":" if gpu_type.lower() not in ("", "generic") else ""
        args.append(f"--gpus-per-node={prefix}{request['gpus_per_node']}")
    args.append("--wrap=:")
    probe = {"qos": qos, "shape": shape, "resources": request, "accepted": False}
    try:
        result = command(args, check=False)
        # Controller diagnostics can enumerate thousands of node names. Keep
        # structured decisions/resources; exclude that inventory from plans.
        probe.update(accepted=result.returncode == 0, returncode=result.returncode,
                     detail="accepted" if result.returncode == 0 else "scheduler rejected request")
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        probe.update(returncode=None, detail=type(error).__name__, unavailable=True)
    return probe


def qos_candidate_probes(site, key, name, seconds):
    """Validate production and the smaller requests using this resource class."""
    shapes = [("primary", seconds)]
    if key == "CPU_QOS":
        shapes.append(("light_cpu", seconds))
    else:
        single_times = [min(duration(site[stage + "_TIME"]), seconds)
                        for stage in ("BANK", "EVAL")]
        # Long-QOS selection uses the production maximum; short-QOS selection
        # only covers single-GPU jobs whose actual request is at most an hour.
        actual = [min(duration(site[stage + "_TIME"]), duration(site["MAX_WALLTIME"]))
                  for stage in ("BANK", "EVAL")]
        applicable = [value for value, actual_value in zip(single_times, actual)
                      if (actual_value <= 3600) == (key == "GPU_QOS_SHORT")]
        if applicable:
            shapes.append(("single_gpu", max(applicable)))
    probes = []
    for shape, required_seconds in shapes:
        probe = qos_probe(site, key, name, required_seconds, shape=shape)
        probes.append(probe)
        if not probe["accepted"]:
            break
    return probes


def discover_qos(site, associations, overridden):
    """Select permitted, sufficient QOS values and verify each resource class."""
    notes, evidence, failures = [], {}, []
    original_maximum = duration(site["MAX_WALLTIME"])
    automatic = [key for key in QOS_KEYS if key not in overridden]
    for key in QOS_KEYS:
        if key in overridden:
            # Preserve '-' exactly, for compatibility with existing site.env files.
            if str(site[key]).strip().lower() in ("", "none"):
                site[key] = "none"
            site["_sources"][key] = "user override"
    if not automatic:
        return notes
    allowed = {key: qos_names(associations, site["ACCOUNT"],
                             site["CPU_PARTITION"] if key == "CPU_QOS" else site["PARTITION"])
               for key in automatic}
    limits, priorities = {}, {}
    limits_available = False
    if any(allowed.values()):
        try:
            limits = qos_limits(priorities)
            limits_available = True
        except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as error:
            notes.append("QOS MaxWall discovery unavailable; use site-default none: " + str(error))
    long_cap = None
    if "GPU_QOS_LONG" in automatic:
        # A <=1h QOS belongs to short jobs and must not hide the sole QOS
        # capable of longer training segments. Commit a lower cap only after
        # that candidate passes the production and applicable single-GPU probes.
        known = {name: limits[name] for name in allowed["GPU_QOS_LONG"]
                 if name in limits and (limits[name] is None or limits[name] > 3600)}
        target = duration(site["MAX_WALLTIME"])
        if len(known) == 1:
            name, limit = next(iter(known.items()))
            if limit is not None and limit < target:
                long_cap = limit // 60 * 60
    ordered = (["GPU_QOS_LONG"] + [key for key in automatic if key != "GPU_QOS_LONG"]
               if long_cap else automatic)
    for key in ordered:
        seconds = (3900 if key == "CPU_QOS" else
                   long_cap if key == "GPU_QOS_LONG" and long_cap else duration(site["MAX_WALLTIME"]))
        if key == "GPU_QOS_SHORT":
            seconds = min(3600, seconds)
        unpublished = bool(allowed[key]) and limits_available and not any(
            limits.get(name) is not None for name in allowed[key])
        if unpublished:
            # Without finite limits, a named QOS cannot be justified by job
            # duration. Let site defaults/plugins assign it whenever possible.
            eligible = sorted(allowed[key], key=lambda name: (-priorities.get(name, 0), name))
        else:
            eligible = [name for name in allowed[key] if name in limits
                        and (limits[name] is None or limits[name] >= seconds)]
            eligible.sort(key=lambda name: (float("inf") if limits[name] is None else limits[name],
                                             -priorities.get(name, 0), name))
        if not allowed[key]:
            default_source = "default none (no association QOS)"
        elif unpublished:
            default_source = "site default (QOS limits not published)"
        elif not limits_available:
            default_source = "default none (QOS discovery unavailable)"
        else:
            default_source = "default none (no accepted sufficient QOS)"
        row = {"requested_seconds": seconds, "partition": site["CPU_PARTITION"] if key == "CPU_QOS" else site["PARTITION"],
               "allowed_qos": sorted(allowed[key]),
               "max_wall_seconds": {name: limits[name] for name in sorted(allowed[key]) if name in limits},
               "priorities": {name: priorities.get(name, 0) for name in sorted(allowed[key])},
               "limits_not_published": unpublished,
               "original_gpu_max_walltime_seconds": original_maximum,
               "effective_gpu_max_walltime_seconds": duration(site["MAX_WALLTIME"]),
               "candidates": [{"qos": name, "max_wall_seconds": limits.get(name),
                               "priority": priorities.get(name, 0)} for name in eligible], "probes": []}
        evidence[key] = row
        attempts = ["none"] + eligible if unpublished else eligible + ["none"]
        for name in attempts:
            probe_seconds = original_maximum if name == "none" and key == "GPU_QOS_LONG" and long_cap else seconds
            probes = qos_candidate_probes(site, key, name, probe_seconds)
            row["probes"].extend(probes)
            probe = probes[-1]
            if all(value["accepted"] for value in probes):
                if key == "GPU_QOS_LONG" and long_cap and name != "none":
                    notes.append(f"MAX_WALLTIME reduced from {site['MAX_WALLTIME']} to {walltime(long_cap)} "
                                 f"by the sole accepted long GPU QOS {name} MaxWall={limits[name]}s.")
                    site["MAX_WALLTIME"] = walltime(long_cap)
                    site["_sources"]["MAX_WALLTIME"] = "test-only verified"
                source = (default_source if name == "none" else
                          "ambiguous-needs-confirmation" if unpublished else "detected+test-only verified")
                site[key], site["_sources"][key] = name, source
                row.update(selected=name, source=source, selected_requested_seconds=probe_seconds,
                           effective_gpu_max_walltime_seconds=duration(site["MAX_WALLTIME"]))
                if unpublished and name != "none":
                    notes.append(f"{key}={name}: ambiguous; mentor must confirm "
                                 "(QOS limits not published; site default rejected).")
                elif name == "none":
                    reason = ("association has no QOS list or sacctmgr is unavailable" if not allowed[key]
                              else "QOS limits not published" if unpublished
                              else "no associated QOS with sufficient MaxWall was test-only accepted")
                    notes.append(f"{key}=none ({reason}); site default QOS, test-only verified.")
                break
            # An unavailable executable cannot validate another named candidate;
            # still attempt the required default-none fallback exactly once.
            if probe.get("unavailable"):
                if name != "none":
                    fallback_seconds = original_maximum if key == "GPU_QOS_LONG" and long_cap else seconds
                    fallback_probes = qos_candidate_probes(site, key, "none", fallback_seconds)
                    row["probes"].extend(fallback_probes)
                    if all(value["accepted"] for value in fallback_probes):
                        site[key], site["_sources"][key] = "none", default_source
                        row.update(selected="none", source=default_source)
                        notes.append(f"{key}=none (QOS probing unavailable); site default QOS, test-only verified.")
                break
        if "selected" not in row:
            failures.append(key)
    site["_qos_discovery"] = evidence
    if failures:
        gate_failure({"run_prefix": run_prefix(site), "site": site, "qos_discovery": evidence},
                     ["QOS discovery incomplete; confirm " + ", ".join(failures)
                      + " with an explicit override (none uses the site default); root 待核"])
    return notes


def load_site_env(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            raise ValueError(f"Invalid site.env line: {line}")
        key, raw = line.split("=", 1)
        if key not in SITE_KEYS:
            raise ValueError(f"Unknown site.env key: {key}")
        values = shlex.split(raw, comments=True)
        if len(values) > 1:
            raise ValueError(f"Use one shell-quoted value for {key}")
        result[key] = values[0] if values else ""
    return result


def interpreter_path(value):
    value = str(value)
    candidate = shutil.which(value) if "/" not in value else None
    return str(Path(candidate or value).expanduser().absolute())


def site_configuration(args, env=None):
    env = dict(os.environ if env is None else env)
    supplied = load_site_env(args.site_env) if args.site_env else {}
    root = Path(args.output_root or env.get("OUTPUT_ROOT", supplied.get("OUTPUT_ROOT", HERE.parent.parent / "bigfield_run"))).expanduser().resolve()
    saved = root / "state" / "site.json"
    defaults = {"DATA_ROOT": str(Path(env.get("DATA_ROOT", HERE.parent.parent / "bigfield_data")).expanduser().resolve()),
                "OUTPUT_ROOT": str(root), "CPU_TIME": "01:05:00", "GPU_TYPE": "h100",
                "SMOKE_TIME": "01:00:00", "BANK_TIME": "04:00:00", "EVAL_TIME": "08:00:00",
                "PYTHON": str(HERE / ".venv/bin/python") if (HERE / ".venv/bin/python").is_file() else sys.executable,
                "MIN_FREE_BYTES": env.get("BIGFIELD_MIN_FREE_BYTES", str(500 * 1024 ** 3)),
                "MIN_FREE_INODES": env.get("BIGFIELD_MIN_FREE_INODES", "100000")}
    if saved.is_file():
        site = read_json(saved)
        site.setdefault("_sources", {key: "saved binding" for key in SITE_KEYS if key in site})
        associations = []
        notes = []
    else:
        # Site-file values participate in discovery, including account and
        # partition selection; process environment remains the final override.
        discovered, notes = discover_site(dict(supplied, **env))
        associations = discovered.pop("_association_records", [])
        site = dict(defaults, **discovered)
        site["_sources"] = dict({key: "default" for key in defaults},
                                **{key: "detected" for key in discovered})
    if args.site_env:
        site.update(supplied)
        site["_sources"].update({key: "user override" for key in supplied})
    for key in SITE_KEYS:
        if key in env:
            value = env[key]
            if key == "PYTHON":
                value = interpreter_path(value)
            elif key == "GPU_TYPE" and value.lower() in ("", "generic"):
                value = "generic"
            elif key in ("DATA_ROOT", "OUTPUT_ROOT"):
                value = str(Path(value).expanduser().resolve())
            elif key in ("NODES", "GPUS_PER_NODE", "CPUS_PER_TASK", "CPU_CPUS_PER_TASK") and value.isdecimal():
                value = str(int(value))
            if saved.is_file() and str(site.get(key, "")) != value:
                raise ValueError(f"Existing run binds {key}={site.get(key)!r}; use its site.env or a new OUTPUT_ROOT")
            site[key] = value
            if not saved.is_file():
                site["_sources"][key] = "user override"
    if args.data_root:
        site["DATA_ROOT"] = str(Path(args.data_root).expanduser().resolve())
        if not saved.is_file():
            site["_sources"]["DATA_ROOT"] = "user override"
    if not saved.exists() and "PYTHON" not in env and not args.site_env:
        setup = Path(site["DATA_ROOT"]) / "setup/status.json"
        if setup.is_file() and read_json(setup).get("training_python"):
            site["PYTHON"] = read_json(setup)["training_python"]
            site["_sources"]["PYTHON"] = "detected"
    site["OUTPUT_ROOT"] = str(root)
    if not saved.is_file() and (args.output_root or "OUTPUT_ROOT" in env or "OUTPUT_ROOT" in supplied):
        site["_sources"]["OUTPUT_ROOT"] = "user override"
    required = ("ACCOUNT", "PARTITION", "GPUS_PER_NODE", "CPUS_PER_TASK", "MEM_PER_NODE", "MAX_WALLTIME", "CPU_PARTITION",
                )
    interactive = sys.stdin.isatty() and not args.yes and not saved.exists()
    if interactive:
        for note in notes:
            print(note, file=sys.stderr)
        print("Confirm site parameters. Press Enter to accept each detected value.", file=sys.stderr)
        for key in required + ("NODES", "GPU_TYPE", "SMOKE_TIME", "BANK_TIME", "EVAL_TIME"):
            print(f"{key} [{site.get(key, '')}; {site['_sources'].get(key, 'default')}]: ",
                  file=sys.stderr, end="", flush=True)
            answer = input().strip()
            if answer:
                site[key] = answer
                site["_sources"][key] = "user override"
    elif not args.yes and not saved.exists():
        raise ValueError("First site configuration requires interactive confirmation or --yes after the mentor confirms ACCOUNT/PARTITION/NODES")
    missing = [key for key in required if not str(site.get(key, "")).strip()]
    if missing:
        raise ValueError("Discovery incomplete; provide " + ", ".join(missing) + " and use --yes after confirmation")
    site["GPU_TYPE"] = site.get("GPU_TYPE") or "generic"
    site.setdefault("NODES", str(select_layout(site["GPUS_PER_NODE"])))
    site.setdefault("CPU_CPUS_PER_TASK", str(min(4, int(site["CPUS_PER_TASK"]))))
    site.setdefault("CPU_MEM_PER_NODE", str(min(32768, memory_megabytes(site["MEM_PER_NODE"]))) + "M")
    for key in ("NODES", "GPUS_PER_NODE", "CPUS_PER_TASK", "CPU_CPUS_PER_TASK"):
        if not str(site[key]).isdecimal() or int(site[key]) <= 0:
            raise ValueError(f"{key} must be a positive integer")
        site[key] = str(int(site[key]))
    total = int(site["NODES"]) * int(site["GPUS_PER_NODE"])
    if 64 % total:
        raise ValueError("Total GPUs must divide the fixed 64 global caption groups")
    for key in ("ACCOUNT", "PARTITION", "CPU_PARTITION", "GPU_TYPE"):
        if site.get(key) and not re.fullmatch(r"[A-Za-z0-9_.-]+", site[key]):
            raise ValueError(f"Invalid {key}")
    for key in ("MEM_PER_NODE", "CPU_MEM_PER_NODE"):
        if not re.fullmatch(r"[1-9]\d*[KMGT]?", str(site[key]), re.I):
            raise ValueError(f"{key} must be a positive Slurm memory quantity")
    duration(site["MAX_WALLTIME"])
    duration(site["CPU_TIME"])
    for key in ("SMOKE_TIME", "BANK_TIME", "EVAL_TIME"):
        duration(site[key])
    if duration(site["CPU_TIME"]) != 3900:
        raise ValueError("CPU_TIME must be exactly 01:05:00 to avoid short-request rewriting")
    if not saved.is_file():
        notes.extend(discover_qos(site, associations, set(supplied) | set(env)))
        if interactive:
            for key in QOS_KEYS:
                print(f"{key} [{site[key]}; {site['_sources'][key]}]: ",
                      file=sys.stderr, end="", flush=True)
                answer = input().strip()
                if answer:
                    site[key] = "none" if qos_disabled(answer) and answer != "-" else answer
                    site["_sources"][key] = "user override"
    for key in QOS_KEYS:
        if key not in site or (not qos_disabled(site[key]) and not re.fullmatch(r"[A-Za-z0-9_.-]+", site[key])):
            raise ValueError(f"Invalid {key}")
    for key in ("MIN_FREE_BYTES", "MIN_FREE_INODES"):
        if not str(site[key]).isdecimal():
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("OUTPUT_ROOT", "DATA_ROOT", "PYTHON"):
        path = Path(site[key]).expanduser()
        # Keep the virtualenv entry point: resolving its symlink selects the
        # base interpreter and silently loses the pinned environment.
        site[key] = interpreter_path(path) if key == "PYTHON" else str(path.resolve())
        if "\n" in site[key] or "\x00" in site[key]:
            raise ValueError(f"Invalid {key}")
    if Path(site["OUTPUT_ROOT"]) == Path(site["DATA_ROOT"]) or Path(site["DATA_ROOT"]) in Path(site["OUTPUT_ROOT"]).parents:
        raise ValueError("OUTPUT_ROOT must be outside DATA_ROOT")
    if saved.exists():
        original = read_json(saved)
        if any(site.get(key) != original.get(key) for key in SITE_KEYS):
            raise ValueError("Existing site configuration is immutable; use a new OUTPUT_ROOT for changes")
    return site, notes


def run_prefix(site):
    return "b2_" + hashlib.sha256(site["OUTPUT_ROOT"].encode()).hexdigest()[:10] + "_"


def queue_snapshot():
    result = command(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j|%T"])
    rows = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) != 3 or not fields[0] or re.search(r"\s", fields[0]):
            raise RuntimeError("Cannot parse scheduler queue; root 待核")
        rows[fields[0]] = {"name": fields[1], "status": fields[2].split()[0].upper()}
    return rows


def accounting(identifiers):
    if not identifiers:
        return {}
    result = command(["sacct", "-n", "-P", "-j", ",".join(identifiers), "--format=JobIDRaw,State"])
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0] in identifiers:
            rows[fields[0]] = fields[1].split()[0].rstrip("+").upper()
    return rows


def existing_jobs(state):
    directory = Path(state) / "jobs"
    return {path.stem: read_json(path) for path in sorted(directory.glob("*.json"))} if directory.is_dir() else {}


def refresh_jobs(state, jobs, queue, *, submit=False):
    missing = []
    for key, job in jobs.items():
        if job.get("superseded_by"):
            continue
        if job.get("status") in ("submission_failed", "submitting", "failed"):
            raise RuntimeError(f"Stage {key} stopped: {job.get('error', job['status'])}. Inspect journals/logs; use explicit recovery for a terminal training failure; no automatic retry")
        path = Path(state) / "receipts" / (key + ".json")
        if path.exists():
            receipt = read_json(path)
            if receipt.get("status") not in ("complete", "failed"):
                raise RuntimeError(f"Invalid receipt status for {key}")
            if receipt.get("source_manifest_sha256") != job["spec"]["source_manifest_sha256"]:
                raise RuntimeError(f"Source binding differs in receipt {key}")
            job["status"] = receipt["status"]
            if receipt["status"] == "failed":
                job["error"] = receipt.get("error", "Worker reported failure")
        elif job.get("status") in ("skipped", "manual_required"):
            continue
        elif not job.get("job_id"):
            raise RuntimeError(f"Ambiguous or failed submission {key}; inspect its journal and logs; no automatic retry")
        elif str(job["job_id"]) in queue:
            row = queue[str(job["job_id"])]
            if row["name"] != job["spec"]["job_name"]:
                raise RuntimeError(f"Queue job ID/JobName binding differs for {key}; no new jobs")
            if row["status"] not in ACTIVE | FAILED:
                raise RuntimeError(f"{key}: unknown scheduler state {row['status']}; root 待核; no new jobs. Inspect journals/logs, then rerun bash run.sh --yes --submit --output-root {shlex.quote(job['spec']['output_root'])}")
            # PENDING alone does not distinguish a held root from a released
            # one. Keep the durable release obligation until scontrol confirms
            # its disposition after the entire submission wave is verified.
            pending_release = job.get("release_pending") is True or job.get("status") == "held"
            job["status"] = "held" if pending_release and row["status"] in ACTIVE else row["status"].lower()
        else:
            missing.append(str(job["job_id"]))
    if missing:
        try:
            records = accounting(missing)
        except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("Accounting unavailable; root 待核. No new jobs: " + str(error)) from error
        for key, job in jobs.items():
            if str(job.get("job_id")) not in missing:
                continue
            status = records.get(str(job["job_id"]))
            if status in FAILED:
                job["status"] = "failed"
                job["error"] = status
            elif status in ACTIVE:
                job["status"] = status.lower()
            elif status == "COMPLETED":
                raise RuntimeError(f"{key} completed without a verified receipt; inspect logs; no new jobs")
            else:
                raise RuntimeError(f"{key}: scheduler state unavailable; root 待核; no new jobs")
    for key, job in jobs.items():
        if job.get("superseded_by"):
            continue
        if submit:
            atomic_json(Path(state) / "jobs" / (key + ".json"), job)
        if job["status"] == "failed":
            raise RuntimeError(f"Stage {key} failed: {job.get('error', 'unknown')}. Logs: {job['spec']['output_root']}/logs. Inspect the latest verified checkpoint and report; no automatic retry.")
    return jobs


def nearest_existing(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def gate_failure(report, errors):
    if PLAN_ONLY_CONTEXT.get():
        # Read-only plans retain all gate inputs in the diagnostic stream.
        print("GATE_REPORT " + json.dumps(dict(report, errors=errors), sort_keys=True), file=sys.stderr, flush=True)
        raise RuntimeError("; ".join(errors) + ". PLAN_ONLY; no files or run root/journal reserved")
    # A failing first preflight must not consume the run root or any journal.
    evidence = Path(tempfile.gettempdir()) / ("bigfield_preflight_" + report["run_prefix"].strip("_"))
    evidence.mkdir(parents=True, exist_ok=True)
    target = evidence / ("gate_" + uuid.uuid4().hex + ".json")
    atomic_json(target, dict(report, errors=errors), create_once=True)
    raise RuntimeError("; ".join(errors) + f". Gate evidence: {target}")


def preflight(site, jobs, queue, *, submit=False):
    root, data = Path(site["OUTPUT_ROOT"]), Path(site["DATA_ROOT"])
    report = {"run_prefix": run_prefix(site), "storage": [], "queue": queue,
              "min_free_bytes": int(site["MIN_FREE_BYTES"]),
              "min_free_inodes": int(site["MIN_FREE_INODES"])}
    errors = []
    requested = {"gpu_partition": site["PARTITION"], "gpu_type": site["GPU_TYPE"],
                 "nodes": int(site["NODES"]), "gpus_per_node": int(site["GPUS_PER_NODE"]),
                 "gpu_cpus_per_node": int(site["CPUS_PER_TASK"]),
                 "gpu_memory_megabytes": memory_megabytes(site["MEM_PER_NODE"]),
                 "gpu_walltime_seconds": duration(site["MAX_WALLTIME"]),
                 "cpu_partition": site["CPU_PARTITION"],
                 "cpu_cpus_per_node": int(site["CPU_CPUS_PER_TASK"]),
                 "cpu_memory_megabytes": memory_megabytes(site["CPU_MEM_PER_NODE"]),
                 "cpu_walltime_seconds": duration(site["CPU_TIME"])}
    report["requested_capacity"] = requested
    try:
        inventory = partition_inventory()
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["resource_inventory"] = {"status": "root 待核", "error": str(error)}
        if submit:
            errors.append("Physical-node resource gate unavailable; root 待核")
    else:
        gpu_nodes = {row["node"] for row in inventory if row["name"] == site["PARTITION"]
                     and gpu_capacity(row, site["GPU_TYPE"]) >= requested["gpus_per_node"]
                     and row["cpus"] >= requested["gpu_cpus_per_node"]
                     and row["memory"] >= requested["gpu_memory_megabytes"]
                     and (row["limit_seconds"] is None or row["limit_seconds"] >= requested["gpu_walltime_seconds"])}
        cpu_nodes = {row["node"] for row in inventory if row["name"] == site["CPU_PARTITION"]
                     and "gpu:" not in row["gres"].lower()
                     and row["cpus"] >= requested["cpu_cpus_per_node"]
                     and row["memory"] >= requested["cpu_memory_megabytes"]
                     and (row["limit_seconds"] is None or row["limit_seconds"] >= requested["cpu_walltime_seconds"])}
        report["resource_inventory"] = {"status": "checked", "physical_nodes": inventory,
                                        "compatible_gpu_nodes": sorted(gpu_nodes),
                                        "compatible_cpu_nodes": sorted(cpu_nodes)}
        if len(gpu_nodes) < requested["nodes"]:
            errors.append(f"GPU request needs {requested['nodes']} compatible physical nodes; only {len(gpu_nodes)} exist")
        if not cpu_nodes:
            errors.append("CPU partition has no physical node that fits CPUs/memory and the 3900-second request")
    if root.exists() and not (root / "state" / "site.json").is_file():
        errors.append("OUTPUT_ROOT already exists without this workflow's site binding; choose a new run root")
    if (root / "state" / "binding.json").is_file():
        saved = read_json(root / "state" / "binding.json").get("output_identity")
        identity = root.stat()
        actual = {"path": str(root), "inode": identity.st_ino, "uid": identity.st_uid}
        if saved != actual:
            errors.append("Run directory path/inode/uid identity differs")
    for path in (root, data, HERE, Path(site["PYTHON"])):
        if path.parts[1:2] == ("projects",) or path.resolve().parts[1:2] == ("projects",):
            errors.append("Runtime paths must use compute-visible shared storage")
    for path in (root, data):
        ancestor = nearest_existing(path)
        stat = os.statvfs(ancestor)
        free_bytes, free_inodes = stat.f_bavail * stat.f_frsize, stat.f_favail
        identity = ancestor.stat()
        report["storage"].append({"path": str(path), "checked_path": str(ancestor),
                                  "inode": identity.st_ino, "uid": identity.st_uid,
                                  "free_bytes": free_bytes, "free_inodes": free_inodes})
        if free_bytes < report["min_free_bytes"] or free_inodes < report["min_free_inodes"]:
            errors.append(f"Storage gate failed for {path}: bytes={free_bytes}, inodes={free_inodes}")
    known = {str(job.get("job_id")) for job in jobs.values() if job.get("job_id")}
    for identifier, row in queue.items():
        if row["name"].startswith(report["run_prefix"]) and identifier not in known:
            errors.append(f"Unjournaled job with this run prefix: {identifier} {row['name']}")
    if submit and not data.is_dir():
        errors.append("DATA_ROOT is absent; run setup.sh first")
    if submit and not Path(site["PYTHON"]).is_file():
        errors.append("Training Python is absent; run setup.sh or set PYTHON")
    if errors:
        gate_failure(report, errors)
    return report


def parse_scontrol(text):
    # Colon-bearing keys are real Slurm fields, e.g. AllocNode:Sid, ReqB:S:C:T.
    matches = list(re.finditer(r"(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=", text.strip()))
    return {match[1]: text.strip()[match.end():matches[index + 1].start() if index + 1 < len(matches) else None].strip()
            for index, match in enumerate(matches)}


def validate_scontrol(text, spec, identifier, dependency_ids, *, held_root=False):
    fields = parse_scontrol(text)
    resources = spec["resources"]
    checks = {"JobId": str(identifier), "JobName": spec["job_name"],
              "Account": resources["account"], "Partition": resources["partition"],
              "Requeue": "0"}
    if resources.get("qos"):
        checks["QOS"] = resources["qos"]
    mismatches = [f"{key}={fields.get(key)!r}, expected {value!r}" for key, value in checks.items()
                  if fields.get(key) != value]
    node = fields.get("NumNodes", "")
    if node not in (str(resources["nodes"]), f"{resources['nodes']}-{resources['nodes']}"):
        mismatches.append("NumNodes differs")
    try:
        if duration(fields.get("TimeLimit", "")) != resources["walltime_seconds"]:
            mismatches.append("TimeLimit differs")
    except ValueError:
        mismatches.append("TimeLimit is invalid")
    if fields.get("NumCPUs") and int(fields["NumCPUs"]) < resources["nodes"] * resources["cpus_per_task"]:
        mismatches.append("NumCPUs is below request")
    if fields.get("NumTasks") != str(resources["nodes"]):
        mismatches.append("NumTasks differs")
    if fields.get("CPUs/Task") is not None and fields["CPUs/Task"] != str(resources["cpus_per_task"]):
        mismatches.append("CPUs/Task differs")
    if fields.get("MinMemoryNode") is not None:
        try:
            if memory_megabytes(fields["MinMemoryNode"]) != memory_megabytes(resources["mem_per_node"]):
                mismatches.append("MinMemoryNode differs")
        except ValueError:
            mismatches.append("MinMemoryNode is invalid")
    dependency = fields.get("Dependency", "")
    expected = {str(identifier) for identifier in dependency_ids}
    try:
        actual_dependencies = afterok_ids(dependency)
    except ValueError:
        actual_dependencies = None
    if actual_dependencies != expected:
        mismatches.append(f"Dependency={dependency!r}, expected afterok IDs {sorted(expected)!r}")
    if resources["gpus_per_node"]:
        gpu_type = resources["gpu_type"].lower()
        expected_gpu = "gres/gpu:" + (gpu_type + ":" if gpu_type not in ("", "generic") else "") + str(resources["gpus_per_node"])
        actual = fields.get("TresPerNode", "").lower()
        if expected_gpu not in actual.split(","):
            mismatches.append("TresPerNode GPU request differs")
    if held_root and (fields.get("JobState") != "PENDING" or fields.get("Reason") != "JobHeldUser"):
        mismatches.append("Root job is not held pending as requested")
    if mismatches:
        raise RuntimeError("Scheduler readback differs: " + "; ".join(mismatches))
    return fields


def afterok_ids(value):
    """Accept compact/expanded AND-afterok readback; reject OR/extra clauses."""
    if value in ("", "(null)", "None"):
        return set()
    result = set()
    for clause in value.split(","):
        clause = re.sub(r"\((?:unfulfilled|fulfilled)\)", "", clause)
        match = re.fullmatch(r"afterok:(\d+(?::\d+)*)", clause)
        if not match:
            raise ValueError("Dependency is not an AND of afterok job IDs")
        result.update(match[1].split(":"))
    return result


def segment_plan(smoke, maximum_seconds, *, start=0):
    required = ("seconds_per_step", "startup_seconds", "checkpoint_seconds")
    values = {key: float(smoke[key]) for key in required}
    if smoke.get("validated") is not True or any(not math.isfinite(value) or value < 0 for value in values.values()) or values["seconds_per_step"] <= 0:
        raise ValueError("Smoke timing requires validated finite positive throughput and nonnegative measured overheads")
    maximum_seconds = int(maximum_seconds)
    schedulable_seconds = maximum_seconds // 60 * 60
    overhead = values["startup_seconds"] + values["checkpoint_seconds"] + SIGNAL_BUDGET_SECONDS
    capacity = math.floor((schedulable_seconds - overhead) / (1.15 * values["seconds_per_step"]))
    if capacity < 1:
        raise ValueError("Partition walltime cannot fit one measured update plus 15% margin, measured overhead and signal budget")
    jobs, cursor = [], int(start)
    while cursor < 30000:
        end = min(cursor + capacity, next(readout for readout in READOUTS if readout > cursor))
        def estimate(stop):
            saves = stop // 1000 - cursor // 1000 + int(stop % 1000 != 0)
            return requested_seconds((stop - cursor) * values["seconds_per_step"] * 1.15 + values["startup_seconds"] + saves * values["checkpoint_seconds"] + SIGNAL_BUDGET_SECONDS), saves
        low, high = cursor + 1, end
        while low <= high:
            middle = (low + high) // 2
            if estimate(middle)[0] <= maximum_seconds:
                low = middle + 1
            else:
                high = middle - 1
        end = high
        if end <= cursor:
            raise ValueError("Partition walltime cannot fit the next update and required save overhead")
        seconds, saves = estimate(end)
        jobs.append({"start_step": cursor, "end_step": end, "walltime_seconds": seconds,
                     "checkpoint_saves": saves})
        cursor = end
    return {"basis": "validated receiving-cluster smoke; throughput x 1.15 plus measured startup/save overhead and signal budget",
            "seconds_per_step": values["seconds_per_step"], "startup_seconds": values["startup_seconds"],
            "checkpoint_seconds": values["checkpoint_seconds"], "signal_budget_seconds": SIGNAL_BUDGET_SECONDS,
            "steps_per_segment": capacity,
            "partition_max_seconds": maximum_seconds, "schedulable_seconds": schedulable_seconds, "jobs": jobs}


def resources(site, stage, seconds=None):
    cpu = stage in ("data", "advance", "summary")
    if seconds is None:
        seconds = duration(site["CPU_TIME"]) if cpu else min(86400, duration(site["MAX_WALLTIME"])) // 60 * 60
        if stage in ("smoke", "bank", "eval"):
            seconds = min(seconds, duration(site[stage.upper() + "_TIME"]))
    seconds = requested_seconds(seconds)
    if not cpu and seconds > min(86400, duration(site["MAX_WALLTIME"])):
        raise ValueError("Rounded GPU walltime exceeds the partition maximum")
    qos = site["CPU_QOS"] if cpu else site["GPU_QOS_SHORT" if seconds <= 3600 else "GPU_QOS_LONG"]
    if qos_disabled(qos):
        qos = ""
    cpus = int(site["CPU_CPUS_PER_TASK"] if cpu else site["CPUS_PER_TASK"])
    memory = site["CPU_MEM_PER_NODE"] if cpu else site["MEM_PER_NODE"]
    if stage in ("advance", "summary"):
        cpus, memory = 1, str(min(4096, memory_megabytes(memory))) + "M"
    elif stage in ("bank", "eval"):
        cpus, memory = min(8, cpus), str(min(131072, memory_megabytes(memory))) + "M"
    return {"nodes": 1 if cpu or stage in ("bank", "eval") else int(site["NODES"]),
            "gpus_per_node": 0 if cpu else (1 if stage in ("bank", "eval") else int(site["GPUS_PER_NODE"])),
            "cpus_per_task": cpus, "mem_per_node": memory,
            "partition": site["CPU_PARTITION"] if cpu else site["PARTITION"],
            "qos": qos,
            "account": site["ACCOUNT"], "gpu_type": site["GPU_TYPE"], "walltime_seconds": seconds}


def reusable_readout(site, state, jobs, readout, checkpoint, source_sha):
    """Reuse one live/completed evaluation only after verifying its producer pin."""
    matches = []
    for key, old in jobs.items():
        spec = old["spec"]
        if spec["stage"] != "eval" or old.get("superseded_by") or spec.get("end_step") != readout:
            continue
        if old["status"].upper() not in ACTIVE | {"COMPLETE"}:
            raise RuntimeError(f"Existing readout {key} is failed or uncertain; inspect it before recovery")
        expected = {"source_manifest_sha256": source_sha, "output_root": site["OUTPUT_ROOT"],
                    "data_root": site["DATA_ROOT"], "code_root": str(HERE),
                    "train_output": str(Path(site["OUTPUT_ROOT"]) / "train"), "readout_step": readout}
        if any(spec.get(name) != value for name, value in expected.items()):
            raise RuntimeError(f"Existing readout {key} source/output binding differs")
        parent = spec.get("parent_checkpoint")
        if parent is None:
            producer = spec.get("parent_key", spec.get("checkpoint_key"))
            pin = spec.get("dependency_receipts", {}).get(producer, {})
            receipt = Path(state) / "receipts" / (str(producer) + ".json")
            journal = Path(state) / "journals" / (str(producer) + ".json")
            if (pin.get("producer") != producer or pin.get("source_manifest_sha256") != source_sha
                    or pin.get("path") != str(receipt) or not receipt.is_file() or not journal.is_file()
                    or pin.get("journal_sha256") != sha256_file(journal)
                    or (pin.get("sha256") and pin["sha256"] != sha256_file(receipt))):
                raise RuntimeError(f"Existing readout {key} producer receipt/journal binding differs")
            value = read_json(receipt)
            if value.get("status") != "complete" or value.get("source_manifest_sha256") != source_sha:
                raise RuntimeError(f"Existing readout {key} producer is not verified complete")
            parent = value.get("checkpoint")
        if parent != checkpoint:
            raise RuntimeError(f"Existing readout {key} checkpoint binding differs")
        if old["status"] == "complete":
            receipt = Path(state) / "receipts" / (key + ".json")
            if not receipt.is_file() or read_json(receipt).get("checkpoint") != checkpoint:
                raise RuntimeError(f"Completed readout {key} checkpoint receipt differs")
        matches.append(key)
    if len(matches) > 1:
        raise RuntimeError("Multiple live/completed evaluations exist for the recovered readout; inspect journals")
    return matches


def continuation_graph(graph, jobs, site):
    """A login-node rerun gives an explicitly rejected nested attempt a new key."""
    replacements = {}
    for spec in graph:
        old = jobs.get(spec["key"], {})
        if (old.get("status") == "manual_required" and old.get("nested_policy_rejected") is True
                and old.get("no_job_created") is True and old.get("job_id") is None):
            replacements[spec["key"]] = spec["key"] + "_c001"
    for spec in graph:
        original = spec["key"]
        if original in replacements:
            spec.update(key=replacements[original], job_name=run_prefix(site) + replacements[original],
                        continuation_of=original, continuation_attempt=1)
        spec["dependencies"] = [replacements.get(key, key) for key in spec["dependencies"]]
        for name in ("parent_key", "checkpoint_key"):
            if spec.get(name) in replacements:
                spec[name] = replacements[spec[name]]
    return graph


def stage_plan(site, state, jobs=None, source_sha=None, recovery=None):
    jobs = jobs or {}
    source_sha = source_sha or (sha256_file(HERE / "sources.sha256") if (HERE / "sources.sha256").exists() else "UNFROZEN")
    graph = []
    if recovery is None and (Path(state) / "recovery.json").is_file():
        recovery = read_json(Path(state) / "recovery.json")
    def add(key, stage, dependencies=(), seconds=None, **extra):
        try:
            allocation = resources(site, stage, seconds)
        except (ValueError, TypeError, OverflowError) as error:
            # Planning failures precede reservation and retain the actual
            # inputs/limits used by the timing and resource gate.
            limits = {name: duration(site[name]) for name in
                      ("CPU_TIME", "MAX_WALLTIME", "SMOKE_TIME", "BANK_TIME", "EVAL_TIME")}
            gpu_maximum = min(86400, limits["MAX_WALLTIME"])
            gate_failure({"run_prefix": run_prefix(site), "stage": stage, "site": site,
                          "seconds_argument": seconds, "walltime_limits_seconds": limits,
                          "minute_seconds": 60, "gpu_maximum_seconds": gpu_maximum,
                          "schedulable_gpu_seconds": gpu_maximum // 60 * 60}, [str(error)])
        graph.append(dict(key=key, stage=stage, dependencies=list(dependencies),
                          job_name=run_prefix(site) + key, resources=allocation,
                          data_root=site["DATA_ROOT"], output_root=site["OUTPUT_ROOT"],
                          code_root=str(HERE), python=site["PYTHON"],
                          source_manifest_sha256=source_sha, train_output=str(Path(site["OUTPUT_ROOT"]) / "train"),
                          smoke_output=str(Path(site["OUTPUT_ROOT"]) / "smoke"), **extra))
    add("data", "data")
    add("bank", "bank", ["data"])
    add("smoke", "smoke", ["bank"])
    smoke_path = Path(state) / "receipts" / "smoke.json"
    measured = None
    if smoke_path.exists() and read_json(smoke_path).get("status") == "complete":
        smoke = read_json(smoke_path)
        try:
            measured = segment_plan(smoke, min(86400, duration(site["MAX_WALLTIME"])),
                                    start=recovery["checkpoint"]["step"] if recovery else 0)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            gate_failure({"run_prefix": run_prefix(site), "smoke": smoke,
                          "partition_max_seconds": duration(site["MAX_WALLTIME"]),
                          "gpu_maximum_seconds": min(86400, duration(site["MAX_WALLTIME"])),
                          "schedulable_gpu_seconds": min(86400, duration(site["MAX_WALLTIME"])) // 60 * 60,
                          "safety_multiplier": 1.15, "minute_seconds": 60,
                          "train_signal_seconds": TRAIN_SIGNAL_SECONDS,
                          "signal_early_seconds": SIGNAL_EARLY_SECONDS,
                          "signal_budget_seconds": SIGNAL_BUDGET_SECONDS}, [str(error)])
        parent = None
        readout_keys = []
        suffix = "_" + recovery["generation"] if recovery else ""
        if recovery:
            for key, old in jobs.items():
                if old["spec"]["stage"] == "eval" and not old.get("superseded_by") and old["spec"].get("end_step", 30001) < recovery["checkpoint"]["step"]:
                    readout_keys.append(key)
        for segment in measured["jobs"]:
            key = f"train_{segment['start_step']:06d}_{segment['end_step']:06d}" + suffix
            extra = {"parent_checkpoint": recovery["checkpoint"]} if recovery and parent is None else {}
            add(key, "train", [parent] if parent else ["smoke"], segment["walltime_seconds"],
                start_step=segment["start_step"], end_step=segment["end_step"], parent_key=parent, **extra)
            parent = key
            if segment["end_step"] in READOUTS:
                eval_key = f"eval_{segment['end_step']:06d}" + suffix
                add(eval_key, "eval", [key], checkpoint_key=key, parent_key=key,
                    end_step=segment["end_step"], readout_step=segment["end_step"])
                readout_keys.append(eval_key)
        # A recovered checkpoint can itself be a previously missing readout.
        if recovery and recovery["checkpoint"]["step"] in READOUTS:
            readout = recovery["checkpoint"]["step"]
            matching = reusable_readout(site, state, jobs, readout, recovery["checkpoint"], source_sha)
            if matching:
                readout_keys.extend(matching)
            else:
                eval_key = f"eval_{readout:06d}" + suffix
                add(eval_key, "eval", ["smoke"], parent_checkpoint=recovery["checkpoint"],
                    end_step=readout, readout_step=readout)
                readout_keys.append(eval_key)
        add("summary" + suffix, "summary", readout_keys)
    else:
        add("advance", "advance", ["smoke"])
    graph = continuation_graph(graph, jobs, site)
    planned = [spec for spec in graph if spec["key"] not in jobs]
    return {"mode": "PLAN_ONLY", "run_prefix": run_prefix(site), "site": site, "jobs": planned,
            "existing_jobs": {key: {"job_id": job.get("job_id"), "status": job["status"]} for key, job in jobs.items()},
            "readout_steps": list(READOUTS), "measured_segmentation": measured,
            "h100_hours_requested": sum(spec["resources"]["nodes"] * spec["resources"]["gpus_per_node"] * spec["resources"]["walltime_seconds"] / 3600 for spec in planned),
            "production_h100_hours": None if measured is None else sum(segment["walltime_seconds"] for segment in measured["jobs"]) * int(site["NODES"]) * int(site["GPUS_PER_NODE"]) / 3600,
            "recovery": recovery,
            "production_note": "Production segmentation and H100-hours require the successful allocated-node smoke; no unmeasured throughput is asserted." if measured is None else measured["basis"]}


def bundle_plan(site, state, jobs, source_sha):
    """Plan a CPU archive job using only small state/checkpoint metadata."""
    state = Path(state)
    if not (state / "binding.json").is_file():
        raise ValueError("No reserved run exists; run setup.sh and run.sh first")
    snapshot = {}
    for directory in (state / "jobs", state / "receipts"):
        for path in sorted(directory.glob("*.json")):
            if not path.stem.startswith("bundle_"):
                snapshot[str(path.relative_to(state))] = sha256_file(path)
    training = Path(site["OUTPUT_ROOT"]) / "train"
    for name in ("latest.json", "progress.json", "dose_summary.json"):
        path = training / name
        if path.is_file():
            snapshot["train/" + name] = sha256_file(path)
    fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    key = "bundle_" + fingerprint[:12]
    if key in jobs:
        receipt = state / "receipts" / (key + ".json")
        status = read_json(receipt).get("status") if receipt.is_file() else jobs[key]["status"]
        if status in ("failed", "submission_failed", "submitting"):
            raise RuntimeError("This archive attempt failed or is ambiguous; inspect its immutable journal/log before choosing another snapshot")
    spec = dict(key=key, stage="summary", dependencies=[], job_name=run_prefix(site) + key,
                resources=resources(site, "summary"), data_root=site["DATA_ROOT"],
                output_root=site["OUTPUT_ROOT"], code_root=str(HERE), python=site["PYTHON"],
                source_manifest_sha256=source_sha, train_output=str(training),
                smoke_output=str(Path(site["OUTPUT_ROOT"]) / "smoke"),
                snapshot_fingerprint=fingerprint, snapshot_metadata_sha256=snapshot,
                note="Explicit allocated CPU bundle of currently available artifacts; no dependency on unfinished stages.")
    return dict(mode="PLAN_ONLY", run_prefix=run_prefix(site), site=site,
                jobs=[] if key in jobs else [spec], readout_steps=list(READOUTS),
                existing_jobs={name: {"job_id": job.get("job_id"), "status": job["status"]} for name, job in jobs.items()},
                h100_hours_requested=0, production_h100_hours=None,
                production_note="Archive-only CPU snapshot; training is not advanced.")


def prepare_recovery(key, state, jobs, queue, source_sha):
    """Explicitly authorize one new attempt from a terminal train checkpoint.

    This never edits a journal, cancels a job, or reads a checkpoint payload on
    the login node. The allocated worker verifies payloads and optimizer state.
    """
    if key not in jobs or jobs[key]["spec"]["stage"] != "train":
        raise ValueError("--recover requires the exact key of a failed training stage")
    target = jobs[key]
    if target.get("superseded_by"):
        raise ValueError("That failure already has an explicit recovery; rerun without --recover")
    identifier = str(target.get("job_id", ""))
    if not identifier or identifier in queue:
        raise ValueError("Recovery requires a terminal allocation and its known job ID; inspect queue/journal")
    try:
        status = accounting([identifier]).get(identifier)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Recovery accounting unavailable; root 待核; no new attempt") from error
    receipt = Path(state) / "receipts" / (key + ".json")
    if status not in FAILED or (receipt.exists() and read_json(receipt).get("status") == "complete"):
        raise ValueError("Recovery requires confirmed failed scheduler state and no complete stage receipt")
    superseded = {key}
    changed = True
    while changed:
        changed = False
        for child, job in jobs.items():
            if child not in superseded and set(job["spec"].get("dependencies", [])) & superseded:
                superseded.add(child)
                changed = True
    for child in superseded:
        job = jobs[child]
        queued = queue.get(str(job.get("job_id")), {})
        if queued and queued["status"] != "PENDING":
            raise ValueError(f"Recovery blocked by active downstream allocation {child}; inspect scheduler state")
    latest = Path(target["spec"]["train_output"]) / "latest.json"
    if not latest.is_file():
        raise ValueError("No complete checkpoint is available; report the failure instead of automatically restarting")
    pointer = read_json(latest)
    checkpoint = latest.parent / "checkpoints" / pointer["checkpoint"]
    manifest = checkpoint / "checkpoint.json"
    actual = sha256_file(manifest)
    meta = read_json(manifest)
    step = int(meta.get("step", -1))
    if (actual != pointer.get("checkpoint_sha256") or meta.get("schema") != "bigfield-checkpoint-v1"
            or step != pointer.get("step") or meta.get("data_cursor") != 64 * step
            or not target["spec"]["start_step"] <= step <= target["spec"]["end_step"]):
        raise ValueError("Latest checkpoint manifest/step/cursor binding differs")
    # The immutable generation record pins every subsequent attempt spec.
    previous = list((Path(state) / "recoveries").glob("*.json"))
    generation = f"r{len(previous) + 1:03d}"
    value = dict(generation=generation, failed_key=key, failed_job_id=identifier,
                 scheduler_state=status, source_manifest_sha256=source_sha,
                 checkpoint={"path": str(checkpoint), "manifest_sha256": actual, "step": step},
                 latest_pointer_sha256=sha256_file(latest), superseded_keys=sorted(superseded),
                 note="Explicit recovery only; old failed/dependent jobs and all journals are retained. Pending afterok jobs are left untouched.")
    copied = {name: dict(job) for name, job in jobs.items()}
    for child in superseded:
        copied[child]["superseded_by"] = generation
    return value, copied


def sbatch_command(spec, state, dependency_ids=(), *, hold=False, test_only=False):
    res = spec["resources"]
    args = ["sbatch", "--parsable", "--export=ALL", "--no-requeue", f"--chdir={HERE}",
            f"--job-name={spec['job_name']}", f"--account={res['account']}",
            f"--partition={res['partition']}", f"--nodes={res['nodes']}", f"--ntasks={res['nodes']}",
            "--ntasks-per-node=1", f"--cpus-per-task={res['cpus_per_task']}",
            f"--mem={res['mem_per_node']}", f"--time={walltime(res['walltime_seconds'])}",
            f"--output={spec['output_root']}/logs/{spec['key']}_%j.log"]
    if res["qos"]:
        args.append(f"--qos={res['qos']}")
    if res["gpus_per_node"]:
        gpu_type = res["gpu_type"]
        request = (gpu_type + ":" if gpu_type.lower() not in ("", "generic") else "") + str(res["gpus_per_node"])
        args.append(f"--gpus-per-node={request}")
    if spec["stage"] == "train":
        args.append(f"--signal=USR1@{TRAIN_SIGNAL_SECONDS}")
    if dependency_ids:
        args.append("--dependency=afterok:" + ":".join(dependency_ids))
    if hold:
        args.append("--hold")
    if test_only:
        args.append("--test-only")
    return args + [str(HERE / "worker.slurm"), "--stage", spec["stage"], "--state", str(state), "--key", spec["key"]]


def reserve_root(site, preflight_report, source_sha):
    root = Path(site["OUTPUT_ROOT"])
    if root.exists():
        saved = read_json(root / "state" / "binding.json")
        if saved.get("source_manifest_sha256") != source_sha:
            raise RuntimeError("Run source manifest differs; use the frozen source that created this run")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()  # all queue, prefix, storage, source and wrapper gates ran first
    state = root / "state"
    state.mkdir()
    atomic_json(state / "site.json", site, create_once=True)
    # Keep shell-readable parameters beside the JSON binding, without sourcing it.
    content = "# Confirmed site parameters; this run binds these values.\n" + "".join(f"{key}={shlex.quote(str(site.get(key, '')))}\n" for key in SITE_KEYS)
    with (state / "site.env").open("x") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    identity = root.stat()
    atomic_json(state / "binding.json", {"source_manifest_sha256": source_sha,
                "code_root": str(HERE), "output_identity": {"path": str(root), "inode": identity.st_ino, "uid": identity.st_uid}}, create_once=True)
    atomic_json(state / "gates" / "initial.json", preflight_report, create_once=True)
    (root / "logs").mkdir()


def bind_dependencies(spec, state, jobs):
    bound = dict(spec, dependency_receipts={})
    for key in spec["dependencies"]:
        receipt = Path(state) / "receipts" / (key + ".json")
        journal = Path(state) / "journals" / (key + ".json")
        dependency = {"path": str(receipt), "producer": key,
                      "source_manifest_sha256": spec["source_manifest_sha256"],
                      "journal_sha256": sha256_file(journal)}
        if jobs[key]["status"] == "complete":
            dependency["sha256"] = sha256_file(receipt)
            parent = read_json(receipt).get("checkpoint")
            if parent and key == spec.get("parent_key", spec.get("checkpoint_key")):
                bound["parent_checkpoint"] = parent
        bound["dependency_receipts"][key] = dependency
    return bound


def explicit_policy_rejection(result):
    """Only an unambiguous no-job policy denial permits a manual continuation."""
    if result.returncode == 0 or result.stdout.strip():
        return False
    message = result.stderr.strip().lower()
    message = re.sub(r"^(?:sbatch:\s*)?(?:error:\s*)?", "", message)
    return bool(re.fullmatch(r"job-origin submission disabled[.!]?", message) or
                re.fullmatch(r"batch job submission failed:\s*job submit from compute nodes is disabled[.!]?", message))


def manual_advance(state, site, reason):
    fallback = {"auto_advance": "manual_required", "reason": reason,
                "resume_command": f"bash run.sh --yes --submit --output-root {shlex.quote(site['OUTPUT_ROOT'])}",
                "root_review": "root 待核"}
    atomic_json(Path(state) / "advance_fallback.json", fallback)
    print(json.dumps(fallback), flush=True)
    return 69


def pending_release_roots(jobs, keys=None):
    return [key for key, job in jobs.items() if not job.get("superseded_by")
            and (keys is None or key in keys)
            and (job.get("release_pending") is True or job.get("status") == "held")]


def verify_release_waves(state, jobs, planned=(), *, release_keys=None):
    """Re-read every registered member before completing any held-root release.

    The immutable root journal names the full intended wave, so interruption
    cannot turn a partially registered wave into permission to release its root.
    Missing members are allowed only when this invocation will submit them.
    """
    planned = set(planned)
    roots, checked = pending_release_roots(jobs, release_keys), {}
    for root in roots:
        resume = "bash run.sh --yes --submit --output-root " + shlex.quote(jobs[root]["spec"]["output_root"])
        try:
            journal = read_json(Path(state) / "journals" / (root + ".json"))
            wave = journal.get("wave_keys")
            if (journal.get("held_root") is not True or not isinstance(wave, list)
                    or not wave or root not in wave or any(not isinstance(key, str) for key in wave)
                    or len(set(wave)) != len(wave)):
                raise RuntimeError("Held-root wave binding is missing or invalid")
            for key in wave:
                if key not in jobs:
                    if key in planned:
                        continue
                    raise RuntimeError(f"Submission wave is incomplete: missing {key}")
                if key in checked:
                    continue
                job = jobs[key]
                entry = read_json(Path(state) / "journals" / (key + ".json"))
                if entry.get("spec") != job["spec"] or entry.get("key") != key:
                    raise RuntimeError(f"Journal/spec binding differs for {key}")
                if (job["spec"]["stage"] == "advance" and job["status"] == "manual_required"
                        and job.get("no_job_created") is True and not job.get("job_id")):
                    checked[key] = {"manual_required": True}
                    continue
                receipt = Path(state) / "receipts" / (key + ".json")
                if job["status"] == "complete" or (receipt.is_file() and read_json(receipt).get("status") == "complete"):
                    if (not receipt.is_file() or read_json(receipt).get("status") != "complete"
                            or read_json(receipt).get("source_manifest_sha256") != job["spec"]["source_manifest_sha256"]):
                        raise RuntimeError(f"Complete receipt binding differs for {key}")
                    if key == root and job.get("release_state") not in ("requested", "released"):
                        raise RuntimeError(f"Held root {key} completed without a recorded release request")
                    job["status"] = "complete"
                    checked[key] = {"complete": True}
                    continue
                identifier = str(job.get("job_id") or "")
                if not identifier.isdecimal() or job["status"] in ("submitting", "submission_failed", "failed", "manual_required"):
                    raise RuntimeError(f"Submission state is unconfirmed or failed for {key}; no automatic retry")
                dependency_ids = list(entry["dependency_ids"])
                # A lost release response may be followed by real progress.
                # Slurm can remove satisfied afterok edges; only a verified
                # complete producer receipt permits removing its original ID.
                for parent in job["spec"]["dependencies"]:
                    producer = jobs.get(parent, {})
                    parent_id = str(producer.get("job_id") or "")
                    if producer.get("status") != "complete" or parent_id not in dependency_ids:
                        continue
                    parent_path = Path(state) / "receipts" / (parent + ".json")
                    receipt_value = read_json(parent_path)
                    if (receipt_value.get("status") != "complete" or receipt_value.get("source_manifest_sha256")
                            != job["spec"]["source_manifest_sha256"]):
                        raise RuntimeError(f"Satisfied dependency receipt differs for {parent}")
                    dependency_ids.remove(parent_id)
                text = command(["scontrol", "show", "job", "-o", identifier]).stdout
                # Controllers may retain fulfilled afterok text or clear it.
                # Both must describe exactly the original binding after only
                # verified-complete producers are removed.
                actual_dependencies = afterok_ids(parse_scontrol(text).get("Dependency", ""))
                expected_dependencies = set(entry["dependency_ids"])
                if not set(dependency_ids) <= actual_dependencies <= expected_dependencies:
                    raise RuntimeError(f"Scheduler dependency binding differs for {key}")
                fields = validate_scontrol(text, job["spec"], identifier, actual_dependencies)
                if fields.get("JobState") not in ACTIVE:
                    raise RuntimeError(f"Unknown or terminal readback state for {key}: {fields.get('JobState')!r}")
                if entry.get("held_root"):
                    held = fields.get("JobState") == "PENDING" and fields.get("Reason") == "JobHeldUser"
                    if ((held and job.get("release_state") == "released")
                            or (not held and (job.get("release_state") not in ("requested", "released")
                                             or fields.get("Reason", "").startswith("JobHeld")))):
                        raise RuntimeError(f"Held root {key} has an unexplained scheduler disposition")
                    fields["release_needed"] = held
                checked[key] = fields
        except (ValueError, KeyError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Held wave stopped at {root}: {error}. Inspect journals/logs; after resolving, resume with {resume}. No release was attempted.") from error
    return roots, checked


def submit_plan(plan, state, jobs, queue, *, nested=False, release_keys=None, output_stream=None):
    output_stream = sys.stdout if output_stream is None else output_stream
    print("Submission | job ID | held/released", file=output_stream, flush=True)
    batch = []
    # Existing held roots are checked before consuming another journal. The
    # same check runs after all missing members have been registered below.
    verify_release_waves(state, jobs, (spec["key"] for spec in plan["jobs"]), release_keys=release_keys)
    wave_keys = [spec["key"] for spec in plan["jobs"]]
    for unbound in plan["jobs"]:
        spec = bind_dependencies(unbound, state, jobs)
        key = spec["key"]
        dependency_ids = []
        for parent in spec["dependencies"]:
            job = jobs[parent]
            if job["status"] == "complete":
                continue  # Never attach afterok to a completed job.
            if job["status"] in ("failed", "manual_required", "skipped"):
                raise RuntimeError(f"Dependency {parent} is not successful; stop")
            identifier = str(job.get("job_id", ""))
            if identifier not in queue and parent not in {item[0] for item in batch}:
                raise RuntimeError(f"Dependency {parent} is no longer active; rerun to refresh before submission")
            dependency_ids.append(identifier)
        hold = not dependency_ids
        args = sbatch_command(spec, state, dependency_ids, hold=hold)
        journal = {"schema": "bigfield-submission-v1", "key": key, "created": time.time(),
                   "spec": spec, "command": args, "dependency_ids": dependency_ids,
                   "held_root": hold, "wave_keys": wave_keys}
        atomic_json(Path(state) / "journals" / (key + ".json"), journal, create_once=True)
        record = {"spec": spec, "status": "submitting", "job_id": None}
        atomic_json(Path(state) / "jobs" / (key + ".json"), record, create_once=True)
        child = clean_environment(dict(os.environ, BIGFIELD_PYTHON=spec["python"], PYTHON=spec["python"]))
        try:
            result = command(args, env=child, check=False)
            if result.returncode:
                message = result.stderr.strip() or result.stdout.strip() or f"sbatch exited {result.returncode}"
                if explicit_policy_rejection(result):
                    raise SubmissionRejected(message)
                raise RuntimeError(message + "; submission outcome is unconfirmed")
            response = result.stdout.strip()
            if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", response):
                raise RuntimeError("Unexpected submission response; inspect queue; do not retry")
            identifier = response.split(";")[0]
            record.update(job_id=identifier, status="held" if hold else "pending",
                          release_pending=hold, readback_verified=False)
            atomic_json(Path(state) / "jobs" / (key + ".json"), record)
            readback = command(["scontrol", "show", "job", "-o", identifier]).stdout
            atomic_json(Path(state) / "readbacks" / (key + ".json"), {"text": readback,
                        "parsed": parse_scontrol(readback), "requested_resources": spec["resources"],
                        "dependency_ids": dependency_ids}, create_once=True)
            validate_scontrol(readback, spec, identifier, dependency_ids, held_root=hold)
            record["readback_verified"] = True
            atomic_json(Path(state) / "jobs" / (key + ".json"), record)
        except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as error:
            record.update(status="submission_failed", error=str(error))
            if nested and not batch and isinstance(error, SubmissionRejected) and not record.get("job_id"):
                record.update(status="manual_required", nested_policy_rejected=True, no_job_created=True)
                atomic_json(Path(state) / "jobs" / (key + ".json"), record)
                jobs[key] = record
                raise ManualAdvanceRequired(str(error)) from error
            # An unsupported optional advance submission can fall back to run.sh.
            if spec["stage"] == "advance" and isinstance(error, SubmissionRejected) and not record.get("job_id"):
                record.update(status="manual_required", no_job_created=True,
                              resume_command=f"bash run.sh --yes --submit --output-root {shlex.quote(spec['output_root'])}")
                atomic_json(Path(state) / "jobs" / (key + ".json"), record)
                print(f"{key} | - | manual required: {record['resume_command']}", file=output_stream, flush=True)
                jobs[key] = record
                break
            atomic_json(Path(state) / "jobs" / (key + ".json"), record)
            raise RuntimeError(f"Submission stopped at {key}: {error}. Held roots remain held; inspect journals and logs. No retry or release was attempted.") from error
        jobs[key] = record
        batch.append((key, identifier, hold))
        print(f"{key} | {identifier} | {'held' if hold else 'dependency pending'}", file=output_stream, flush=True)
    # Release only after the entire wave has been read back successfully.
    roots, checked = verify_release_waves(state, jobs, release_keys=release_keys)
    for key in roots:
        job = jobs[key]
        fields = checked[key]
        if fields.get("release_needed"):
            # Durable intent distinguishes a lost release response from an
            # unexplained externally changed hold on the next explicit rerun.
            job.update(release_state="requested", release_pending=True)
            atomic_json(Path(state) / "jobs" / (key + ".json"), job)
            try:
                command(["scontrol", "release", str(job["job_id"])])
            except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired) as error:
                job["release_error"] = str(error)
                atomic_json(Path(state) / "jobs" / (key + ".json"), job)
                raise RuntimeError(f"Release stopped at {key}: {error}. No job was resubmitted. Inspect journals/logs, then resume with bash run.sh --yes --submit --output-root {shlex.quote(job['spec']['output_root'])}") from error
        job.update(status="complete" if fields.get("complete") else fields.get("JobState", "PENDING").lower(),
                   release_pending=False, release_state="released", readback_verified=True)
        atomic_json(Path(state) / "jobs" / (key + ".json"), job)
        print(f"{key} | {job['job_id']} | released", file=output_stream, flush=True)
    return jobs


def inventory_summary(report, site):
    """Keep plan structure while replacing physical-node names with counts."""
    report = dict(report)
    inventory = dict(report["resource_inventory"])
    if "physical_nodes" in inventory:
        groups = {}
        for row in inventory["physical_nodes"]:
            group = groups.setdefault(row["name"], {"partition": row["name"], "node_count": 0, "gpus_per_node": {}})
            group["node_count"] += 1
            count = str(gpu_capacity(row, "generic"))
            group["gpus_per_node"][count] = group["gpus_per_node"].get(count, 0) + 1
        inventory["physical_nodes"] = [groups[key] for key in sorted(groups)]
        for kind in ("gpu", "cpu"):
            inventory["compatible_" + kind + "_node_count"] = len(inventory.pop("compatible_" + kind + "_nodes"))
    report["resource_inventory"] = inventory
    return report


def print_plan_summary(plan, args):
    """One-screen default; detailed plans remain available through --json."""
    site, checks = plan["site"], plan["checks"]
    sources = site.get("_sources", {})
    print("SUBMIT" if args.submit else "PLAN_ONLY (read-only; --test-only probes create no jobs)")
    print("Parameter | value | source")
    for key in ("ACCOUNT", "PARTITION", "NODES", "GPUS_PER_NODE", "GPU_TYPE", "CPUS_PER_TASK", "MEM_PER_NODE",
                "MAX_WALLTIME", "CPU_PARTITION", "CPU_CPUS_PER_TASK", "CPU_MEM_PER_NODE", "CPU_TIME",
                "CPU_QOS", "GPU_QOS_SHORT", "GPU_QOS_LONG", "PYTHON", "DATA_ROOT", "OUTPUT_ROOT"):
        print(f"{key} | {site[key]} | {sources.get(key, 'saved binding' if plan.get('existing_jobs') else 'default')}")
    ambiguous = [key for key in QOS_KEYS if sources.get(key) == "ambiguous-needs-confirmation"]
    if ambiguous:
        print("WARNING: " + ", ".join(ambiguous) + ": ambiguous; mentor must confirm (QOS limits not published).")
    print("Job key | resources (per node) | time | QOS | dependencies")
    for spec in plan["jobs"][:8]:
        res = spec["resources"]
        resources_text = f"{res['nodes']}n x {res['gpus_per_node']}GPU/{res['cpus_per_task']}CPU/{res['mem_per_node']}"
        print(f"{spec['key']} | {resources_text} | {walltime(res['walltime_seconds'])} | {res['qos'] or 'none'} | {','.join(spec['dependencies']) or 'root (held)'}")
    if len(plan["jobs"]) > 8:
        print(f"... {len(plan['jobs']) - 8} more jobs; use --json for all jobs")
    print(f"First/current wave: {plan['h100_hours_requested']:.3f} H100h")
    production = plan.get("production_h100_hours")
    print("Later training segments/H100h: determined by validated smoke timing (throughput x 1.15)."
          if production is None else f"Measured production request: {production:.3f} H100h; full segmentation: --json")
    queue_status = checks.get("queue_status", "PASS")
    capacity_status = "PASS" if checks["resource_inventory"]["status"] == "checked" else "root 待核"
    print(f"Checks: storage PASS (minimum {checks['min_free_bytes']} bytes); inode PASS (minimum {checks['min_free_inodes']}); queue {queue_status}; capacity {capacity_status}")
    notes = plan.get("notes", [])
    if notes:
        # Values and provenance above remain complete; long query diagnostics
        # are kept in --json. Prioritize changed walltime and pending checks.
        priority = sorted(notes, key=lambda note: ("MAX_WALLTIME" not in note, "root 待核" not in note))
        print("Notes: " + " / ".join(priority)[:450])
    next_command = ["bash", "run.sh", "--submit"]
    if args.yes:
        next_command.append("--yes")
    for option, value in (("--output-root", args.output_root), ("--data-root", args.data_root),
                          ("--site-env", args.site_env), ("--recover", args.recover)):
        if value:
            next_command.extend((option, str(value)))
    if args.bundle:
        next_command.append("--bundle")
    print("Next: " + shlex.join(next_command), flush=True)


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="Create jobs; omitted by default")
    parser.add_argument("--yes", action="store_true", help="Accept already confirmed site values; does not submit")
    parser.add_argument("--json", action="store_true", help="Print the complete machine-readable plan with node counts")
    parser.add_argument("--output-root")
    parser.add_argument("--data-root")
    parser.add_argument("--site-env", help="Read explicit, shell-quoted KEY=value site settings")
    parser.add_argument("--recover", metavar="FAILED_KEY", help="Explicit new attempt from a terminal failed training stage's latest bound checkpoint")
    parser.add_argument("--bundle", action="store_true", help="Plan an allocated CPU return bundle; --submit creates its job")
    parser.add_argument("--advance", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    PLAN_ONLY_CONTEXT.set(not args.submit)
    if args.bundle and (args.recover or args.advance):
        parser.error("--bundle cannot be combined with --recover or --advance")
    site, notes = site_configuration(args)
    state = Path(site["OUTPUT_ROOT"]) / "state"
    jobs = existing_jobs(state)
    source_path = HERE / "sources.sha256"
    source_sha = sha256_file(source_path) if source_path.is_file() else "UNFROZEN"
    recovery = read_json(state / "recovery.json") if (state / "recovery.json").exists() else None
    if recovery:
        generation_path = state / "recoveries" / (recovery["generation"] + ".json")
        if not generation_path.is_file() or read_json(generation_path) != recovery or recovery["source_manifest_sha256"] != source_sha:
            raise RuntimeError("Recovery generation/source binding differs")
        for name in recovery["superseded_keys"]:
            jobs[name]["superseded_by"] = recovery["generation"]
    queue_status = "PASS"
    try:
        queue = queue_snapshot()
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as error:
        if args.submit:
            raise RuntimeError("Queue gate unavailable; root 待核; no run root/journal reserved: " + str(error)) from error
        queue = {}
        queue_status = "root 待核"
        notes.append("Queue check unavailable; root 待核; PLAN_ONLY only: " + str(error))
    report = preflight(site, jobs, queue, submit=args.submit)
    report["queue_status"] = queue_status
    if args.recover:
        recovery, jobs = prepare_recovery(args.recover, state, jobs, queue, source_sha)
    if not args.bundle:
        jobs = refresh_jobs(state, jobs, queue, submit=False)
    if (state / "binding.json").exists() and read_json(state / "binding.json").get("source_manifest_sha256") != source_sha:
        raise RuntimeError("Run source binding differs; restore its exact frozen sources")
    plan = bundle_plan(site, state, jobs, source_sha) if args.bundle else stage_plan(site, state, jobs, source_sha, recovery=recovery)
    plan["checks"] = inventory_summary(report, site)
    plan["notes"] = notes
    for spec in plan["jobs"]:
        spec["command_preview"] = sbatch_command(spec, state, ["<" + key + ">" for key in spec["dependencies"]], hold=not spec["dependencies"])
    if args.json:
        print(json.dumps(plan, indent=2), flush=True)
    else:
        print_plan_summary(plan, args)
    # An explicit archive request must not advance a held training wave.
    release_keys = ({key for key in jobs if key.startswith("bundle_")} |
                    {spec["key"] for spec in plan["jobs"]}) if args.bundle else None
    if not args.submit or (not plan["jobs"] and not pending_release_roots(jobs, release_keys)):
        return plan
    # Source and bash guards precede run-root reservation and every journal.
    if source_sha == "UNFROZEN":
        raise RuntimeError("Missing sources.sha256; freeze/check the release before submitting")
    command([site["PYTHON"], str(HERE / "freeze.py"), "--check"])
    command(["bash", "-n", str(HERE / "worker.slurm")])
    nested = bool(args.advance and os.environ.get("SLURM_JOB_ID"))
    if nested and plan["jobs"]:
        if any(job.get("nested_policy_rejected") and job.get("no_job_created")
               and job.get("status") == "manual_required" for job in jobs.values()):
            return manual_advance(state, site, "A previous nested submission was explicitly rejected; continue from the login node")
        # Detect nested submission policy before consuming any production journal.
        reason = None
        try:
            probe = command(sbatch_command(plan["jobs"][0], state, test_only=True), check=False)
            if probe.returncode:
                reason = probe.stderr.strip() or probe.stdout.strip() or f"sbatch --test-only exited {probe.returncode}"
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            # This probe cannot create a job. Real submission timeouts below
            # remain ambiguous, immutable submission_failed attempts.
            reason = str(error)
        if reason is not None:
            return manual_advance(state, site, reason)
    reserve_root(site, report, source_sha)
    if args.recover:
        atomic_json(state / "recoveries" / (recovery["generation"] + ".json"), recovery, create_once=True)
        atomic_json(state / "recovery.json", recovery)
        for name in recovery["superseded_keys"]:
            atomic_json(state / "jobs" / (name + ".json"), jobs[name])
    try:
        submit_plan(plan, state, jobs, queue, nested=nested, release_keys=release_keys,
                    output_stream=sys.stderr if args.json else sys.stdout)
    except ManualAdvanceRequired as error:
        return manual_advance(state, site, str(error))
    return plan


def main(argv=None):
    token = PLAN_ONLY_CONTEXT.set(False)
    try:
        return _main(argv)
    finally:
        PLAN_ONLY_CONTEXT.reset(token)


if __name__ == "__main__":
    try:
        result = main()
        if result == 69:
            sys.exit(69)
    except (ValueError, RuntimeError, FileExistsError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"STOP: {error}", file=sys.stderr)
        sys.exit(2)
