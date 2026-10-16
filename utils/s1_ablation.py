"""Explicit Stage-1 noise ablation checks; official recipes do not use these."""
import json
import math
import os
from pathlib import Path


NOISE_LEVELS = (.5, .6, .7, .8, .9, 1.)
ACCUMULATIONS = (1, 2, 4, 8, 16)
WORLD_SIZES = (1, 2, 4, 8, 16)


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
    if train["micro_batch_size"] != 64 or train["dtype"] != "bf16":
        raise ValueError("Ablation preserves microbatch 64 and BF16")
    if type(train["gradient_accumulation_steps"]) is not int or train["gradient_accumulation_steps"] not in ACCUMULATIONS:
        raise ValueError(f"Ablation accumulation must be one of {ACCUMULATIONS}")
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
    global_batch = world * micro * accumulation
    passed = world in WORLD_SIZES and micro == 64 and accumulation in ACCUMULATIONS and global_batch == 1024
    values = dict(world_size=world, micro_batch_size=micro, accumulation=accumulation,
                  global_batch=global_batch, required_global_batch=1024, passed=passed)
    if record_success or not passed:
        audit(output, "global_batch", values, rank=rank)
    if not passed:
        receipt = f"; audit: {output}" if output is not None else ""
        raise ValueError(f"Stage-1 requires world_size * 64 * accumulation = 1024; got {world} * {micro} * {accumulation} = {global_batch}{receipt}")
    return global_batch


def validate_mask(mask, *, output=None, rank=0, step=0, micro=0, noise_level=None, perceptual_threshold=None):
    fraction = mask.detach().float().mean()
    if not bool(mask.all().item()):
        audit(output, "perceptual_mask", dict(step=step, micro=micro,
              active_samples=int(mask.sum().item()), total_samples=mask.numel(),
              perceptual_active_fraction=float(fraction.item()), required_fraction=1.,
              noise_level=noise_level, perceptual_threshold=perceptual_threshold, passed=False), rank=rank)
        raise ValueError("Stage-1 perceptual mask must include every sample")
    return fraction
