"""Per-sample adaptive Priming reconstruction objective.

The pixel epsilon 1e-4 and perceptual epsilon 1e-2 follow the released C2I
implementation. T2I's threshold is unpublished: 1.0 adopts the released L160
setting and applies perceptual supervision to every logit-normal example.
"""
import jax
import jax.numpy as jnp


def adaptive_weight(loss, norm_eps=1e-4, norm_p=1.0):
    if norm_eps <= 0 or norm_p < 0:
        raise ValueError("norm_eps must be positive; norm_p must be nonnegative")
    return loss / jax.lax.stop_gradient((loss + norm_eps) ** norm_p)


def reconstruction_loss(prediction, target, noise_levels, *, perceptual_fn=None,
                        perceptual_rng=None, perceptual_threshold=1.0,
                        norm_eps=1e-4, norm_p=1.0, perceptual_norm_eps=1e-2,
                        lpips_weight=0.4, convnext_weight=0.1):
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have the same NHWC shape")
    if noise_levels.shape != (prediction.shape[0],):
        raise ValueError("noise_levels must have shape (B,)")
    if not 0 <= perceptual_threshold <= 1:
        raise ValueError("perceptual_threshold must lie in [0,1]")
    pixel = jnp.mean((prediction.astype(jnp.float32) - target) ** 2, axis=(1, 2, 3))
    weighted_pixel = adaptive_weight(pixel, norm_eps, norm_p)
    if perceptual_fn is None:
        lpips = convnext = jnp.zeros_like(pixel)
    else:
        lpips, convnext = perceptual_fn(prediction, target, perceptual_rng)
        if lpips.shape != pixel.shape or convnext.shape != pixel.shape:
            raise ValueError("perceptual callable must return two (B,) arrays")
        mask = noise_levels <= perceptual_threshold  # official uses <=, including s=1
        lpips = jnp.where(mask, lpips, 0.0)
        convnext = jnp.where(mask, convnext, 0.0)
    auxiliary = (
        lpips_weight * adaptive_weight(lpips, perceptual_norm_eps, norm_p)
        + convnext_weight * adaptive_weight(convnext, perceptual_norm_eps, norm_p)
    )
    loss = jnp.mean(weighted_pixel + auxiliary)
    return loss, {
        "loss": loss, "loss_mse": jnp.mean(pixel),
        "loss_pixel_adaptive": jnp.mean(weighted_pixel),
        "aux_loss": jnp.mean(auxiliary), "aux_loss_lpips": jnp.mean(lpips),
        "aux_loss_convnext": jnp.mean(convnext),
        "perceptual_active_fraction": jnp.mean(noise_levels <= perceptual_threshold),
    }
