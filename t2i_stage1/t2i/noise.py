"""Exact sampling of p(s) proportional to s^-2 LogitNormal(s; .8, .8^2)."""
import math

import jax
import jax.numpy as jnp


def mixture_parameters(mu=0.8, sigma=0.8):
    """Normal-mixture law for z=logit(s), not an importance approximation.

    s^-2=(1+exp(-z))^2=sum_k C(2,k) exp(-k*z). Exponential
    tilting turns each component into N(mu-k*sigma^2,sigma^2), with
    unnormalized mass C(2,k)*exp(-k*mu+k^2*sigma^2/2).
    """
    if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("mu must be finite and sigma must be positive and finite")
    k = jnp.arange(3, dtype=jnp.float32)
    logits = jnp.asarray([0.0, math.log(2.0), 0.0]) - k * mu + 0.5 * k**2 * sigma**2
    return jax.nn.softmax(logits), mu - k * sigma**2


def sample_noise_levels(key, batch_size, *, mode="logit_normal", fixed_s=0.8,
                        mu=0.8, sigma=0.8):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if mode in ("fixed", "fixed_0.8", "fixed_1.0"):
        value = {"fixed_0.8": 0.8, "fixed_1.0": 1.0}.get(mode, fixed_s)
        if not 0 < value <= 1:
            raise ValueError("fixed_s must lie in (0,1]")
        return jnp.full((batch_size,), value, dtype=jnp.float32)
    if mode != "logit_normal":
        raise ValueError("mode must be logit_normal, fixed, fixed_0.8, or fixed_1.0")
    weights, means = mixture_parameters(mu, sigma)
    component_key, normal_key = jax.random.split(key)
    component = jax.random.categorical(component_key, jnp.log(weights), shape=(batch_size,))
    logits = means[component] + sigma * jax.random.normal(normal_key, (batch_size,))
    return jax.nn.sigmoid(logits)


def corrupt_inputs(key, images, noise_levels, *, noise_std=2.0):
    if noise_std <= 0 or not math.isfinite(noise_std):
        raise ValueError("noise_std must be positive and finite")
    if images.ndim != 4 or noise_levels.shape != (images.shape[0],):
        raise ValueError("expected NHWC images and a (B,) noise-level vector")
    s = noise_levels.reshape((-1, 1, 1, 1))
    epsilon = jax.random.normal(key, images.shape, dtype=jnp.float32) * noise_std
    return (1.0 - s) * images + s * epsilon
