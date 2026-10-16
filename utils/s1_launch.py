"""Small Stage-1 Slurm launcher; allocation discovery does not call Slurm."""
import sys
from pathlib import Path

# A direct script must not let utils/random.py shadow the standard library.
if __package__ in (None, ""):
    sys.path[0] = str(Path(__file__).resolve().parents[1])

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
import time


WORLD_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
CACHE_VARIABLES = (
    "TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE",
    "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE",
)


def batch_shape(world):
    if world not in WORLD_SIZES:
        raise ValueError(f"World size must be one of {WORLD_SIZES}; found {world}.")
    return (64, 16 // world) if world <= 16 else (1024 // world, 1)


def split_outside_brackets(value):
    result, start, depth = [], 0, 0
    for index, character in enumerate(value):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(value[start:index])
            start = index + 1
        if depth < 0:
            raise ValueError(f"Invalid node list: {value!r}.")
    if depth:
        raise ValueError(f"Invalid node list: {value!r}.")
    result.append(value[start:])
    return result


def expand_nodelist(value):
    """Expand the bracket/range syntax used by SLURM_JOB_NODELIST."""
    result = []
    for expression in split_outside_brackets(value):
        if not expression:
            raise ValueError("An empty host is not a valid node list.")
        match = re.search(r"\[([^\[\]]+)\]", expression)
        if match is None:
            if "[" in expression or "]" in expression:
                raise ValueError(f"Invalid node list: {value!r}.")
            result.append(expression)
            continue
        choices = []
        for item in match[1].split(","):
            if item.isdigit():
                choices.append(item)
            else:
                interval = re.fullmatch(r"(\d+)-(\d+)", item)
                if interval is None or int(interval[1]) > int(interval[2]):
                    raise ValueError(f"Invalid node range: {item!r}.")
                width = len(interval[1]) if interval[1].startswith("0") else 0
                choices.extend(str(index).zfill(width) for index in range(int(interval[1]), int(interval[2]) + 1))
        for choice in choices:
            result.extend(expand_nodelist(expression[:match.start()] + choice + expression[match.end():]))
    return result


def visible_gpu_count(value):
    value = re.sub(r"\s+", "", value)
    if value in ("", "-1"):
        return 0
    devices = value.split(",")
    if any(not re.fullmatch(r"\d+|GPU-[\w-]+|MIG-[\w/-]+", device) for device in devices):
        raise ValueError("Invalid CUDA_VISIBLE_DEVICES.")
    if len(set(devices)) != len(devices):
        raise ValueError("Duplicate GPU in CUDA_VISIBLE_DEVICES.")
    return len(devices)


def slurm_gpu_count(environment, nodes):
    for name in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        if environment.get(name):
            counts = []
            for part in environment[name].split(","):
                match = re.fullmatch(r"(?:gpu(?::[\w-]+)?:)?(\d+)(?:\(x(\d+)\))?", part.strip())
                if match is None:
                    raise ValueError(f"Cannot parse {name}={environment[name]!r}.")
                counts.extend([int(match[1])] * int(match[2] or 1))
            if len(set(counts)) != 1 or (len(counts) != 1 and len(counts) != nodes):
                raise ValueError("Every node must have the same GPU count; allocation metadata differs.")
            return counts[0]
    if environment.get("SLURM_JOB_GPUS"):
        devices = set()
        for part in environment["SLURM_JOB_GPUS"].split(","):
            interval = re.fullmatch(r"(\d+)-(\d+)", part)
            if interval:
                if int(interval[1]) > int(interval[2]):
                    raise ValueError("Invalid SLURM_JOB_GPUS range.")
                devices.update(str(index) for index in range(int(interval[1]), int(interval[2]) + 1))
            elif re.fullmatch(r"\d+|GPU-[\w-]+|MIG-[\w/-]+", part):
                devices.add(part)
            else:
                raise ValueError("Cannot parse SLURM_JOB_GPUS.")
        return len(devices)
    if "CUDA_VISIBLE_DEVICES" in environment:
        return visible_gpu_count(environment["CUDA_VISIBLE_DEVICES"])
    result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True)
    return sum(bool(line.strip()) for line in result.stdout.splitlines())


def allocation(environment):
    nodelist = environment.get("SLURM_JOB_NODELIST") or environment.get("SLURM_NODELIST")
    hosts = expand_nodelist(nodelist) if nodelist else []
    raw_nodes = environment.get("SLURM_NNODES") or environment.get("SLURM_JOB_NUM_NODES") or str(len(hosts) or 1)
    if not raw_nodes.isdigit() or int(raw_nodes) < 1:
        raise ValueError("SLURM_NNODES must be a positive integer.")
    nodes = int(raw_nodes)
    if not hosts and nodes == 1:
        hosts = [socket.gethostname()]
    if len(hosts) != nodes or len(set(hosts)) != nodes:
        raise ValueError("SLURM_JOB_NODELIST must contain exactly SLURM_NNODES distinct hosts.")
    gpus = slurm_gpu_count(environment, nodes)
    batch_shape(nodes * gpus)
    job = environment.get("SLURM_JOB_ID", "0")
    job_number = int(job) if job.isdigit() else int(hashlib.sha256(job.encode()).hexdigest()[:8], 16)
    return dict(nodes=nodes, gpus=gpus, world=nodes * gpus, hosts=hosts,
                master_addr=hosts[0], master_port=15000 + job_number % 45000)


def check_tmpdir(directory):
    try:
        with tempfile.TemporaryDirectory(prefix="pymp-", dir=directory) as temporary:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.bind(os.path.join(temporary, "listener-01234567"))
    except OSError as error:
        raise ValueError(
            f"Cannot create a DataLoader AF_UNIX socket under TMPDIR={directory!r}: {error}. "
            "Set S1_TMPDIR to a short, writable directory; for example S1_TMPDIR=/tmp."
        ) from None


def resume_path(output, world):
    latest = output / "checkpoints/latest.pth"
    if latest.is_file():
        metadata = output / "config.json"
        if metadata.is_file():
            train = json.loads(metadata.read_text())["train"]
            previous = 1024 // (train.get("micro_batch_size", 64) * train["gradient_accumulation_steps"])
            if previous != world:
                raise ValueError(f"Resume requires the same world size: checkpoint used {previous}, requested {world}.")
        return str(latest)
    if output.exists():
        raise ValueError(f"Output exists without checkpoints/latest.pth: {output}. Select a new output or inspect the incomplete run.")
    return None


def write_config(source, target, micro, accumulation, epochs):
    from omegaconf import OmegaConf
    from utils.config import load_yaml

    config = load_yaml(source)
    config["train"].update(micro_batch_size=micro, gradient_accumulation_steps=accumulation, epochs=epochs)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".yaml.tmp")
    OmegaConf.save(OmegaConf.create(config), temporary)
    os.replace(temporary, target)


def worker_command(python, repository, topology, node_rank, arguments):
    return [python, "-m", "torch.distributed.run", f"--nnodes={topology['nodes']}",
            f"--nproc_per_node={topology['gpus']}", f"--node_rank={node_rank}",
            f"--master_addr={topology['master_addr']}", f"--master_port={topology['master_port']}",
            str(repository / "pretrain.py"), *arguments]


def preflight(expected):
    check_tmpdir(os.environ["TMPDIR"])
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = visible_gpu_count(os.environ["CUDA_VISIBLE_DEVICES"])
        if visible != expected:
            raise ValueError(f"Node {socket.gethostname()} exposes {visible} GPUs in CUDA_VISIBLE_DEVICES; expected {expected} GPUs.")
    import torch

    actual = torch.cuda.device_count()
    print(f"PREFLIGHT_NODE node_rank={os.environ.get('SLURM_NODEID')} host={socket.gethostname()} "
          f"gpus={actual} expected_gpus={expected}", flush=True)
    if actual != expected:
        raise ValueError(f"Node {socket.gethostname()} exposes {actual} CUDA GPUs; every node must expose {expected}.")
    for index in range(actual):
        torch.cuda.set_device(index)
        name = torch.cuda.get_device_name(index)
        value = torch.ones(1, device=f"cuda:{index}").item()
        print(f"PREFLIGHT_GPU host={socket.gethostname()} device={index} name={name!r} probe_value={value}", flush=True)
        if value != 1:
            raise ValueError(f"Node {socket.gethostname()} CUDA GPU {index} probe returned {value}; expected 1.")
    print(f"PREFLIGHT_OK node_rank={os.environ.get('SLURM_NODEID')} host={socket.gethostname()} gpus={actual}", flush=True)


def safe_stop(output, launch_id):
    try:
        receipt = json.loads((output / "safe_stop.json").read_text())
        checkpoint = output / "checkpoints" / f"step_{int(receipt['completed']):07d}.pth"
        return (receipt["state"] == "SAFE_STOPPED" and receipt["launch_id"] == launch_id
                and Path(receipt["checkpoint"]) == checkpoint and checkpoint.is_file())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_slurm():
    environment = os.environ
    dry_run = environment.get("DRY_RUN") == "1"
    repository = Path(environment["S1_REPO_ROOT"])
    storage = Path(environment["STORAGE"])
    python = str(storage / "env/bin/python")
    if dry_run and not os.access(python, os.X_OK):
        python = sys.executable
    topology = allocation(environment)
    micro, accumulation = batch_shape(topology["world"])
    noise = environment.get("NOISE_LEVEL", "1.0")
    if noise in ("1", "1.0"):
        noise, suffix = "1.0", "s100"
    elif noise in ("0.8", ".8"):
        noise, suffix = "0.8", "s080"
    else:
        raise ValueError("NOISE_LEVEL must be 1.0 (pure noise) or 0.8 (control).")
    epochs = environment.get("EPOCHS", "80")
    if epochs not in ("80", "160"):
        raise ValueError("EPOCHS must be 80 or 160.")
    smoke = environment.get("SMOKE_STEPS")
    if smoke is not None and not re.fullmatch(r"[1-9][0-9]*", smoke):
        raise ValueError("SMOKE_STEPS must be a positive integer.")
    name = f"b{epochs}_perceptual_{suffix}"
    source_name = "b80_perceptual_s080" if suffix == "s080" and epochs == "160" else name
    source = repository / "configs/pretrain" / f"{source_name}.yaml"
    config_directory = Path(environment.get("S1_CONFIG_DIR", str(storage / "configs"))).absolute()
    config = config_directory / f"{name}_{topology['world']}gpu.yaml"
    output = Path(environment.get("OUTPUT_DIR", str(storage / "outputs/pretrain" / f"{name}_{topology['world']}gpu"))).absolute()
    check_tmpdir(environment["TMPDIR"])
    resume = resume_path(output, topology["world"])
    arguments = ["--config", str(config), "--data-path", str(storage / "data/imagenet"),
                 "--hf-home", str(storage / "assets/huggingface"), "--torch-home", str(storage / "assets/torch"),
                 "--lpips-weights", str(storage / "assets/lpips/vgg.pth"), "--output", str(output)]
    if resume:
        arguments.extend(["--resume", resume])
    if smoke:
        arguments.extend(["--smoke-steps", smoke])
    print(f"NNODES={topology['nodes']}\nGPUS_PER_NODE={topology['gpus']}\nGPU_COUNT={topology['world']}\n"
          f"MICRO_BATCH_SIZE={micro}\nGRADIENT_ACCUMULATION_STEPS={accumulation}\n"
          f"NOISE_LEVEL={noise}\nEPOCHS={epochs}\nRESUME={resume or 'none'}\nTMPDIR={environment['TMPDIR']}", flush=True)
    for rank, host in enumerate(topology["hosts"]):
        print(f"NODE_COMMAND[{rank}] host={host}: {shlex.join(worker_command(python, repository, topology, rank, arguments))}", flush=True)
    if dry_run:
        return 0
    helper = str(repository / "utils/s1_launch.py")
    srun = ["srun", f"--nodes={topology['nodes']}", f"--ntasks={topology['nodes']}",
            "--ntasks-per-node=1", "--kill-on-bad-exit=1", "--export=ALL"]
    # Each node checks its CUDA visibility and its local socket path before training starts.
    subprocess.run([*srun, python, helper, "--preflight", str(topology["gpus"])], check=True, cwd=repository)
    subprocess.run([python, "scripts/check_environment.py", "--cuda"], check=True, cwd=repository)
    write_config(source, config, micro, accumulation, int(epochs))
    environment["S1_LAUNCH_ID"] = f"{time.time_ns()}-{os.getpid()}"
    environment["S1_TOPOLOGY"] = json.dumps(topology)
    environment["S1_TRAIN_ARGUMENTS"] = json.dumps(arguments)
    environment["S1_TRAIN_PYTHON"] = python
    status = subprocess.run([*srun, python, helper, "--node-launch"], cwd=repository).returncode
    if status != 0 and safe_stop(output, environment["S1_LAUNCH_ID"]):
        status = 75
    if status == 75:
        print("Safely stopped at a checkpoint. Submit the same command again to continue.", flush=True)
    return status


def main():
    for name in CACHE_VARIABLES:
        os.environ.pop(name, None)
    # Direct execution must also allow importing utils.config from the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--slurm", action="store_true")
    modes.add_argument("--preflight", type=int)
    modes.add_argument("--node-launch", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.preflight is not None:
            preflight(arguments.preflight)
            return 0
        if arguments.node_launch:
            topology = json.loads(os.environ["S1_TOPOLOGY"])
            rank = int(os.environ["SLURM_NODEID"])
            if not 0 <= rank < topology["nodes"]:
                raise ValueError("SLURM_NODEID lies outside the allocation.")
            command = worker_command(os.environ["S1_TRAIN_PYTHON"], Path(os.environ["S1_REPO_ROOT"]),
                                     topology, rank, json.loads(os.environ["S1_TRAIN_ARGUMENTS"]))
            os.execv(command[0], command)
        return run_slurm()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Stage-1 launch failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
