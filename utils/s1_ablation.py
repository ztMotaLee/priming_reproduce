"""Explicit Stage-1 noise ablation checks; official recipes do not use these."""
import json
import math
import os
from pathlib import Path


NOISE_LEVELS = (.5, .6, .7, .8, .9, 1.)
ACCUMULATIONS = (1, 2, 4, 8, 16)
WORLD_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
BATCH_LAYOUTS = {world: (64, 16 // world) if world <= 16 else (1024 // world, 1)
                 for world in WORLD_SIZES}


def validate_config(config, family):
    ablation, train, model = config["ablation"], config["train"], config["model"]
    if not isinstance(ablation, dict) or set(ablation) != {"name", "noise_level"}:
        raise ValueError("ablation requires only name and noise_level")
    noise = ablation["noise_level"]
    if ablation["name"] != "stage1_noise_level" or family != "b":
        raise ValueError("Only Priming-B stage1_noise_level ablation is supported")
    if isinstance(noise, bool) or noise not in NOISE_LEVELS:
        raise ValueError(f"noise_level must be one of {NOISE_LEVELS}")
    if (model["t_min"], model["t_max"]) != (noise, noise):
        raise ValueError("Ablation requires t_min = t_max = noise_level")
    micro, accumulation = train["micro_batch_size"], train["gradient_accumulation_steps"]
    if (type(micro) is not int or type(accumulation) is not int
            or (micro, accumulation) not in BATCH_LAYOUTS.values()):
        raise ValueError(f"Ablation (microbatch, accumulation) must be one of {tuple(BATCH_LAYOUTS.values())}")
    if train["dtype"] != "bf16":
        raise ValueError("Ablation preserves BF16")
    if train["steps_per_epoch"] != 1251 or type(train["epochs"]) is not int or train["epochs"] not in (80, 160):
        raise ValueError("Ablation requires 1251 updates/epoch and 80 or 160 epochs")
    if model["label_drop_prob"] != 0 or model["t_dist"] != "uniform" or model["corruption"] != "denoising":
        raise ValueError("Ablation preserves fixed-noise denoising and no label dropout")
    threshold = config["loss"]["perceptual_threshold"]
    if not config["loss"]["aux"]["enabled"] or not math.isfinite(threshold) or threshold < noise:
        raise ValueError("Ablation requires perceptual losses enabled and perceptual_threshold >= noise_level")
    if train["milestone_interval"] != 20:
        raise ValueError("Ablation preserves a checkpoint every 20 epochs")


def write_receipt(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(values, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def audit(output, gate, values, *, rank=0):
    """Write every gate input before a runtime failure, including nonfinite values."""
    if output is not None:
        write_receipt(Path(output) / f"s1_audit_rank{rank}.json", dict(gate=gate, **values))


def validate_runtime(config, world, *, output=None, rank=0, record_success=False):
    train = config["train"]
    micro, accumulation = train["micro_batch_size"], train["gradient_accumulation_steps"]
    global_batch = world * micro * accumulation if all(type(value) is int for value in (world, micro, accumulation)) else None
    expected = BATCH_LAYOUTS.get(world) if type(world) is int else None
    layout_valid = (type(micro) is int and type(accumulation) is int
                    and expected == (micro, accumulation))
    precision_valid = train["dtype"] == "bf16"
    passed = layout_valid and global_batch == 1024 and precision_valid
    values = dict(world_size=world, micro_batch_size=micro, accumulation=accumulation,
                  global_batch=global_batch, required_global_batch=1024,
                  expected_micro_batch_size=expected[0] if expected else None,
                  expected_accumulation=expected[1] if expected else None,
                  allowed_world_sizes=WORLD_SIZES, layout_valid=layout_valid,
                  dtype=train["dtype"], required_dtype="bf16", precision_valid=precision_valid,
                  passed=passed)
    if record_success or not passed:
        audit(output, "global_batch", values, rank=rank)
    if not passed:
        receipt = f"; audit: {output}" if output is not None else ""
        raise ValueError(f"Stage-1 requires the supported BF16 layout with world_size * microbatch * accumulation = 1024; got {world} * {micro} * {accumulation} = {global_batch}, dtype={train['dtype']}{receipt}")
    return global_batch


def loader_resources(requested_workers, *, environ=None, output=None, rank=0):
    """Keep CPU limits out of the recipe and checkpoint identity."""
    environ = os.environ if environ is None else environ
    workers, threads, cpu_per_rank = requested_workers, 4, None
    if environ.get("SLURM_CPUS_PER_TASK"):
        raw_cpus = environ["SLURM_CPUS_PER_TASK"]
        raw_local_gpus = environ.get("LOCAL_WORLD_SIZE", environ.get("WORLD_SIZE", "1"))
        cpus, local_gpus = None, None
        try:
            cpus, local_gpus = int(raw_cpus), int(raw_local_gpus)
            valid = local_gpus >= 1 and cpus >= local_gpus
        except ValueError:
            valid = False
        if not valid:
            audit(output, "cpu_budget", dict(slurm_cpus_per_task=raw_cpus, local_world_size=raw_local_gpus,
                  cpus=cpus, local_gpus=local_gpus, required_minimum=1,
                  required_cpus=local_gpus,
                  requested_workers=requested_workers, passed=False), rank=rank)
            raise ValueError("SLURM_CPUS_PER_TASK must provide at least one CPU per local GPU")
        cpu_per_rank = cpus // local_gpus
        workers = min(workers, max(0, cpu_per_rank - 1))
        threads = min(threads, max(1, cpu_per_rank - workers))
    return dict(num_workers=workers, threads=threads, cpu_per_rank=cpu_per_rank)


def validate_mask(mask, *, output=None, rank=0, step=0, micro=0, noise_level=None, perceptual_threshold=None):
    fraction = mask.detach().float().mean()
    if not bool(mask.all().item()):
        audit(output, "perceptual_mask", dict(step=step, micro=micro,
              active_samples=int(mask.sum().item()), total_samples=mask.numel(),
              perceptual_active_fraction=float(fraction.item()), required_fraction=1.,
              noise_level=noise_level, perceptual_threshold=perceptual_threshold, passed=False), rank=rank)
        raise ValueError("Stage-1 perceptual mask must include every sample")
    return fraction
