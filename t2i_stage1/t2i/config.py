"""Paper constants and explicitly chosen implementation defaults."""
from dataclasses import asdict, dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class Recipe:
    stage_a_steps: int = 250_000
    stage_b_steps: int = 40_000
    global_batch: int = 1024
    lr_a: float = 4e-4
    lr_b: float = 4e-5
    # The paper does not specify the following optimizer/EMA details.
    beta1: float = .9
    beta2: float = .95
    weight_decay: float = 0.
    ema_decay: float = .9999
    seed: int = 42
    image_size: int = 512
    text_length: int = 256
    noise_std: float = 2.
    perceptual_threshold: float = 1.

    def stage(self, completed_steps):
        return 'A' if completed_steps < self.stage_a_steps else 'B'

    def learning_rate(self, completed_steps):
        # Inference: stagewise constants, no warmup. No optimizer reset at B.
        return jnp.where(completed_steps < self.stage_a_steps, self.lr_a, self.lr_b)

    def as_dict(self):
        return asdict(self)


def batch_layout(global_batch, processes, local_devices, microbatch=1):
    if min(global_batch, processes, local_devices, microbatch) <= 0:
        raise ValueError('Batch topology values must be positive')
    devices = processes * local_devices
    if global_batch % (devices * microbatch):
        raise ValueError('Global batch must divide devices * microbatch exactly')
    return dict(global_batch=global_batch, processes=processes, local_devices=local_devices,
                devices=devices, per_device=global_batch // devices,
                per_process=global_batch // processes, microbatch=microbatch,
                accumulation_steps=global_batch // (devices * microbatch))
