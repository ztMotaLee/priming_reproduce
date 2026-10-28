"""Portable Slurm chain planner and one-process-per-node rendezvous.

Planning reads configuration and prints JSON. Only --submit invokes Slurm.
Every dependency refers to a preceding job submitted by this invocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

READOUTS = (1000, 2000, 4000, 8000, 12000, 16000, 24000, 30000)
HERE = Path(__file__).resolve().parent


def required(env, name):
    value = env.get(name, "").strip()
    if not value or value.startswith("YOUR_"):
        raise ValueError(f"Set {name} explicitly")
    if "\n" in value or "\x00" in value:
        raise ValueError(f"Invalid {name}")
    return value


def positive(env, name):
    value = required(env, name)
    if not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def duration(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if not match:
        raise ValueError("TIME must use [days-]HH:MM:SS")
    days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
    if minutes >= 60 or seconds >= 60 or (days and hours >= 24):
        raise ValueError("Invalid TIME")
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise ValueError("TIME must be positive")
    return total


def plan(env):
    nodes, gpus = positive(env, "NODES"), positive(env, "GPUS_PER_NODE")
    total = nodes * gpus
    if 64 % total:
        raise ValueError("Total GPUs must divide the fixed 64 caption groups")
    partition, account = required(env, "PARTITION"), required(env, "ACCOUNT")
    cpus, memory = positive(env, "CPUS_PER_TASK"), required(env, "MEM_PER_NODE")
    if not re.fullmatch(r"[1-9]\d*[KMGT]?", memory, flags=re.I):
        raise ValueError("MEM_PER_NODE must be a positive Slurm memory quantity")
    wall = required(env, "TIME")
    seconds = duration(wall)
    data = str(Path(required(env, "DATA_ROOT")).expanduser().resolve())
    output = str(Path(required(env, "OUTPUT_DIR")).expanduser().resolve())
    if Path(data) == Path(output) or Path(data) in Path(output).parents:
        raise ValueError("OUTPUT_DIR must be outside DATA_ROOT")
    start = int(env.get("START_STEP", "0"))
    end = int(env.get("END_STEP", "30000"))
    segment = int(env.get("STEPS_PER_SEGMENT", "1000"))
    if not (0 <= start < end <= 30000) or segment <= 0:
        raise ValueError("Require 0 <= START_STEP < END_STEP <= 30000 and positive segment size")
    python = env.get("PYTHON", "python")
    gpu_type = env.get("GPU_TYPE", "").strip()
    if gpu_type and not re.fullmatch(r"[A-Za-z0-9_-]+", gpu_type):
        raise ValueError("Invalid GPU_TYPE")
    base = ["sbatch", "--parsable", f"--chdir={HERE}", f"--nodes={nodes}", f"--ntasks={nodes}",
            "--ntasks-per-node=1", f"--gpus-per-node={gpu_type + ':' if gpu_type else ''}{gpus}",
            f"--cpus-per-task={cpus}", f"--mem={memory}", f"--time={wall}",
            f"--partition={partition}", f"--account={account}", "--export=ALL", "--no-requeue",
            "--signal=USR1@180"]
    jobs = []
    cursor = start
    while cursor < end:
        stop = min(cursor + segment, end, next((r for r in READOUTS if r > cursor), end))
        command = base + [f"--job-name=bigfield_{cursor:06d}_{stop:06d}",
                          f"--output={output}.segment_{cursor:06d}_%j.log"]
        if jobs:
            command += ["--dependency=afterok:<previous-job-from-this-chain>"]
        command += [str(HERE / "train.slurm")]
        jobs.append({"start_step": cursor, "end_step": stop,
                     "resume": "latest" if cursor else None,
                     "dependency": len(jobs) - 1 if jobs else None,
                     "command": command})
        cursor = stop
    # Ideal strong scaling is a planning estimate, never a measured guarantee.
    updates = end - start
    ideal_seconds = updates * 42.886401559982914 * 4 / total
    return {"mode": "PLAN_ONLY", "nodes": nodes, "gpus_per_node": gpus,
            "global_caption_groups": 64, "data_root": data, "output_dir": output,
            "python": python, "walltime_seconds_per_segment": seconds,
            "jobs": jobs, "readout_steps": list(READOUTS),
            "estimates": {"basis": "4-GPU measured median with ideal strong scaling; unverified",
                          "compute_hours": ideal_seconds / 3600,
                          "compute_hours_with_15_percent_margin": ideal_seconds / 3600 * 1.15,
                          "h100_hours": ideal_seconds / 3600 * total,
                          "requested_h100_hours": len(jobs) * seconds / 3600 * total}}


def submit(chain, env):
    output = Path(chain["output_dir"])
    first = chain["jobs"][0]["start_step"]
    if first == 0 and output.exists():
        raise ValueError("A new run requires a new OUTPUT_DIR; use a separate smoke directory")
    if first and not (output / "latest.json").is_file():
        raise ValueError("Resuming requires OUTPUT_DIR/latest.json")
    if not output.parent.is_dir():
        raise ValueError("Create the parent of OUTPUT_DIR before submission")
    if not Path(chain["data_root"]).is_dir():
        raise ValueError("DATA_ROOT does not exist")
    previous = None
    for job in chain["jobs"]:
        command = list(job["command"])
        command = [f"--dependency=afterok:{previous}" if x.startswith("--dependency=") else x
                   for x in command]
        child = dict(env, DATA_ROOT=chain["data_root"], OUTPUT_DIR=chain["output_dir"],
                     START_STEP=str(job["start_step"]), END_STEP=str(job["end_step"]),
                     PYTHON=chain["python"], BIGFIELD_CODE_ROOT=str(HERE))
        result = subprocess.run(command, env=child, text=True, capture_output=True, check=True)
        identifier = result.stdout.strip()
        if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", identifier):
            raise RuntimeError("Unexpected submission response; stop and inspect scheduler state")
        previous = identifier.split(";")[0]
        print(json.dumps({"submitted_job": previous, "start_step": job["start_step"],
                          "end_step": job["end_step"]}), flush=True)


def run_rank(env):
    """Publish rank-zero's address on shared storage before JAX initialization."""
    rank, nodes = int(env["SLURM_PROCID"]), positive(env, "NODES")
    if int(env["SLURM_NTASKS"]) != nodes or not 0 <= rank < nodes:
        raise ValueError("Launch exactly one task per node")
    output = Path(required(env, "OUTPUT_DIR"))
    output.mkdir(parents=True, exist_ok=True)
    job = required(env, "SLURM_JOB_ID")
    rendezvous = output / f".coordinator_{job}.json"
    if rank == 0:
        port = int(env.get("COORDINATOR_PORT", "29500"))
        if not 1024 <= port <= 65535:
            raise ValueError("COORDINATOR_PORT must be in 1024..65535")
        address = f"{socket.gethostname()}:{port}"
        temporary = rendezvous.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump({"address": address}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, rendezvous)
    deadline = time.monotonic() + 120
    while not rendezvous.exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Timed out waiting for coordinator; stop the chain")
        time.sleep(0.2)
    address = json.loads(rendezvous.read_text())["address"]
    command = [env.get("PYTHON", "python"), str(HERE / "train.py"),
               "--data-root", required(env, "DATA_ROOT"), "--output", str(output),
               "--start-step", required(env, "START_STEP"),
               "--end-step", required(env, "END_STEP"), "--nodes", str(nodes),
               "--gpus-per-node", required(env, "GPUS_PER_NODE"), "--coordinator", address]
    if int(env["START_STEP"]) > 0:
        command += ["--resume", "latest"]
    os.execvpe(command[0], command, dict(env))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--rank", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rank:
        run_rank(os.environ)
    else:
        chain = plan(os.environ)
        print(json.dumps(chain, indent=2), flush=True)
        if args.submit:
            submit(chain, os.environ)


if __name__ == "__main__":
    main()
