"""Pseudo-class Soft25 field and unchanged fixed random streams.

The force is the JAX translation distributed by its pMF derivative as MIT.
Its original W-Flow snapshot did not include a license; upstream redistribution
terms require review. Provenance commits: 7778a25ff4f97945162b5d0fa7b0e951778e4373
and b382451f973a08da1d3fd654f86d91500860e44a.
"""
from __future__ import annotations
import math
from typing import Iterable, Tuple
import numpy as np
import jax
import jax.numpy as jnp
DEFAULT_RADII = RADII = (.02,.05,.2)
SEED=20260906
SOURCES=('blip3o','dalle3','sharegpt4o')
SOURCE_WEIGHTS={'blip3o':.06,'dalle3':.016,'sharegpt4o':.04}
def require(value,message):
    if not value: raise ValueError(message)
def pairwise_distance(x: jax.Array, y: jax.Array, eps: float=1e-08) -> jax.Array:
    """Batched Euclidean distance matching the authoritative implementation."""
    x = jnp.asarray(x, dtype=jnp.float32)
    y = jnp.asarray(y, dtype=jnp.float32)
    xydot = jnp.einsum('bnd,bmd->bnm', x, y)
    xnorms = jnp.einsum('bnd,bnd->bn', x, x)
    ynorms = jnp.einsum('bmd,bmd->bm', y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2.0 * xydot
    return jnp.sqrt(jnp.maximum(sq_dist, jnp.asarray(eps, jnp.float32)))

def _masked_row_softmax(logits: jax.Array, weights: jax.Array) -> jax.Array:
    """Row softmax that ignores zero-mass padded support entries.

    Existing v1 calls use strictly positive weights, so this is algebraically
    identical to ``jax.nn.softmax(logits, axis=-1)`` there.  The mask is needed
    by the class-conditional queue while individual class rings are filling.
    """
    valid = jnp.asarray(weights, jnp.float32)[:, None, :] > 0.0
    masked = jnp.where(valid, logits, jnp.asarray(-1e+30, jnp.float32))
    return jax.nn.softmax(masked, axis=-1)

def compute_drift_field(old_gen: jax.Array, fixed_neg: jax.Array, fixed_pos: jax.Array, weight_gen: jax.Array, weight_neg: jax.Array, weight_pos: jax.Array, radii: Iterable[float]=DEFAULT_RADII) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Construct the stopped Drifting target in scaled feature coordinates.

    Shapes are ``old_gen=[B,N,D]``, ``fixed_neg=[B,Nn,D]`` and
    ``fixed_pos=[B,Np,D]``.  The generated particles are also the first
    negative support block; their exact self interactions are masked.  Fixed
    negative supports must therefore *exclude* duplicate copies of the local
    generated query particles.
    """
    old_gen = jax.lax.stop_gradient(jnp.asarray(old_gen, jnp.float32))
    fixed_neg = jax.lax.stop_gradient(jnp.asarray(fixed_neg, jnp.float32))
    fixed_pos = jax.lax.stop_gradient(jnp.asarray(fixed_pos, jnp.float32))
    weight_gen = jax.lax.stop_gradient(jnp.asarray(weight_gen, jnp.float32))
    weight_neg = jax.lax.stop_gradient(jnp.asarray(weight_neg, jnp.float32))
    weight_pos = jax.lax.stop_gradient(jnp.asarray(weight_pos, jnp.float32))
    radii = tuple((float(radius) for radius in radii))
    if not radii or any((not math.isfinite(radius) or radius <= 0.0 for radius in radii)):
        raise ValueError('radii must contain finite positive values')
    num_gen = old_gen.shape[1]
    feature_dim = old_gen.shape[2]
    targets = jnp.concatenate([old_gen, fixed_neg, fixed_pos], axis=1)
    target_weights = jnp.concatenate([weight_gen, weight_neg, weight_pos], axis=1)
    distances = pairwise_distance(old_gen, targets)
    weighted_distances = distances * target_weights[:, None, :]
    scale = weighted_distances.mean() / (target_weights.mean() + 1e-08)
    scale_inputs = jnp.maximum(scale / jnp.sqrt(float(feature_dim)), 0.001)
    old_gen_scaled = old_gen / scale_inputs
    targets_scaled = targets / scale_inputs
    distances_normalized = distances / jnp.maximum(scale, 0.001)
    total_targets = targets.shape[1]
    diagonal = jnp.eye(num_gen, dtype=jnp.float32)
    diagonal = jnp.pad(diagonal, ((0, 0), (0, total_targets - num_gen)))
    distances_normalized = distances_normalized + diagonal[None, ...] * 100.0
    split_index = num_gen + fixed_neg.shape[1]
    force_across_radii = jnp.zeros_like(old_gen_scaled)
    force_norms = []
    for radius in radii:
        logits = -distances_normalized / float(radius)
        affinity_rows = _masked_row_softmax(logits, target_weights)
        affinity_columns = jax.nn.softmax(logits, axis=-2)
        affinity = jnp.sqrt(jnp.maximum(affinity_rows * affinity_columns, 1e-06))
        affinity = affinity * target_weights[:, None, :]
        affinity_neg = affinity[:, :, :split_index]
        affinity_pos = affinity[:, :, split_index:]
        sum_pos = affinity_pos.sum(axis=-1, keepdims=True)
        sum_neg = affinity_neg.sum(axis=-1, keepdims=True)
        coeff_neg = -affinity_neg * sum_pos
        coeff_pos = affinity_pos * sum_neg
        coefficients = jnp.concatenate([coeff_neg, coeff_pos], axis=-1)
        force = jnp.einsum('biy,byx->bix', coefficients, targets_scaled)
        force = force - coefficients.sum(axis=-1)[..., None] * old_gen_scaled
        force_norm = jnp.mean(force ** 2)
        force_norms.append(force_norm)
        force_across_radii = force_across_radii + force / jnp.sqrt(jnp.maximum(force_norm, 1e-08))
    goal_scaled = old_gen_scaled + force_across_radii
    return (jax.lax.stop_gradient(goal_scaled), jax.lax.stop_gradient(scale_inputs), jax.lax.stop_gradient(scale), jax.lax.stop_gradient(jnp.stack(force_norms)))
def partitioned_objective(fake, real_current, bank, bank_weights, *, radii=RADII):
    """Mean over branches/groups of the stopped-target squared residual.

    ``fake`` [G,R,5,D] is the only differentiable input. ``real_current`` [G,R,5,D]
    pairs row-for-row with ``fake``; ``bank`` [G,B,5,D] with binary ``bank_weights``
    [G,B] are fixed same-cluster reals. Each group is normalized independently,
    exactly like the pinned per-class field.
    """
    import jax
    import jax.numpy as jnp
    fake = jnp.asarray(fake, jnp.float32)
    real_current = jax.lax.stop_gradient(jnp.asarray(real_current, jnp.float32))
    bank = jax.lax.stop_gradient(jnp.asarray(bank, jnp.float32))
    bank_weights = jax.lax.stop_gradient(jnp.asarray(bank_weights, jnp.float32))
    (groups, rows, branches, dim) = fake.shape
    ones_rows = jnp.ones((groups, rows), jnp.float32)
    pos_weights = jnp.concatenate([ones_rows, bank_weights], axis=1)

    def one_group(gen, pos, w_gen, w_pos):
        (goal, scale_inputs, scale, norms) = compute_drift_field(gen[None], jnp.zeros((1, 0, dim), jnp.float32), pos[None], w_gen[None], jnp.zeros((1, 0), jnp.float32), w_pos[None], radii=radii)
        return (goal[0], scale_inputs, scale, norms)
    (losses, scales, force_norms) = ([], [], [])
    for branch in range(branches):
        query = fake[:, :, branch, :]
        fixed_pos = jnp.concatenate([real_current[:, :, branch, :], bank[:, :, branch, :]], axis=1)
        (goal, scale_inputs, _, norms) = jax.vmap(one_group)(jax.lax.stop_gradient(query), fixed_pos, ones_rows, pos_weights)
        losses.append(jnp.mean(jnp.square(query / scale_inputs[:, None, None] - goal)))
        scales.append(jnp.mean(scale_inputs))
        force_norms.append(jnp.mean(norms, axis=0))
    return (jnp.mean(jnp.stack(losses)), dict(branch_losses=jnp.stack(losses), input_scales=jnp.stack(scales), force_mean_squares=jnp.stack(force_norms), bank_fill=jnp.mean(bank_weights)))
def caption_probabilities(rows):
    """Exact base recipe per-caption probabilities, rather than rounded display values."""
    counts = {name: sum((row['source'] == name for row in rows)) for name in SOURCES}
    require(all(counts.values()), 'Missing curated source')
    total = sum(SOURCE_WEIGHTS.values())
    return np.asarray([SOURCE_WEIGHTS[row['source']] / total / counts[row['source']] for row in rows], np.float64)

def cluster_probabilities(partition, probabilities=None):
    pi = np.asarray(partition.probabilities if probabilities is None else probabilities, np.float64)
    require(pi.shape == (partition.population_rows,) and np.isfinite(pi).all() and np.all(pi > 0), 'Invalid per-caption probabilities')
    mass = np.bincount(partition.cluster_ids, weights=pi, minlength=partition.clusters)
    return mass / mass.sum()

def weighted_grouped_indices(partition, seed=SEED, step=0, count=1024, rows_per_group=16, stream=21000, probabilities=None):
    """64 distinct clusters x 16 distinct rows; cluster and row probabilities use pi."""
    require(type(count) is int and count >= 1, 'Invalid row count')
    rows = rows_per_group if count >= rows_per_group else count
    require(rows >= 1 and count % rows == 0, 'Row count must tile whole groups')
    groups = count // rows
    pi = np.asarray(partition.probabilities if probabilities is None else probabilities, np.float64)
    mass = cluster_probabilities(partition, pi)
    require(groups <= partition.clusters and np.min(partition.sizes) >= rows, 'Insufficient clusters or rows')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), int(stream)]))
    clusters = rng.choice(partition.clusters, groups, replace=False, p=mass)
    ids = []
    for cluster in clusters:
        members = partition.members[cluster]
        weights = pi[members]
        ids.append(rng.choice(members, rows, replace=False, p=weights / weights.sum()))
    return (np.concatenate(ids).astype(np.int64), clusters.astype(np.int64), rows)
def bank_rows(partition, seed, step, clusters, current_ids, rows_per_group, bank_rows_per_group, stream=22000):
    """Fixed same-cluster bank reals per group, excluding that group's current rows; zero-weight padding."""
    groups = len(clusters)
    current = np.asarray(current_ids).reshape(groups, rows_per_group)
    bank = np.full((groups, bank_rows_per_group), -1, np.int64)
    weights = np.zeros((groups, bank_rows_per_group), np.float32)
    for (g, c) in enumerate(clusters):
        candidates = np.setdiff1d(partition.members[c], current[g], assume_unique=True)
        require(len(candidates) >= 1, 'Cluster has no bank candidates beyond the current rows')
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(step), int(stream), int(g)]))
        take = min(bank_rows_per_group, len(candidates))
        chosen = rng.choice(candidates, take, replace=False)
        bank[g, :take] = chosen
        weights[g, :take] = 1.0
    return (bank, weights)
def digest_array(value):
    return __import__('hashlib').sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def field_keys(step, seed=SEED):
    import jax
    base = jax.random.fold_in(jax.random.PRNGKey(seed), int(step))
    return np.asarray([jax.random.fold_in(base, 17000 + i) for i in range(1024)], np.uint32)
