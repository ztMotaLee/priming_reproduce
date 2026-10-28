"""Priming T2I stage 2, paper Eq. (8), (12), (13).

The Sinkhorn loop is the fixed-iteration log-domain primitive inherited from
the inherited transport primitive, with the official Priming dual update order.  The
the detached RMS includes both live-real and live-reference distances, with
equal mass, before averaging over caption groups.  Plans and scales are not
differentiated.  Residuals are diagnostics, never convergence gates.
"""
from __future__ import annotations

from pathlib import Path
import yaml
_CONFIG=yaml.safe_load((Path(__file__).resolve().parent/'configs/bigfield_tf.yaml').read_text())['objective']
FEATURES=tuple(_CONFIG['branches'])
EPSILON=_CONFIG['epsilon']
SINKHORN_ITERATIONS=_CONFIG['sinkhorn_iterations']
RMS_FLOOR=_CONFIG['rms_floor']


def half_squared_distance(a, b):
    """Batched [caption, row, dim] x [caption, column, dim] costs."""
    import jax.numpy as jnp
    return .5 * jnp.maximum(jnp.sum(a*a, -1)[..., :, None]
        + jnp.sum(b*b, -1)[..., None, :] - 2*jnp.einsum('...id,...jd->...ij', a, b), 0.)


def sinkhorn_plan(cost):
    """Uniform balanced plan, exactly ten updates, no adaptive stopping."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp
    if cost.ndim != 3 or min(cost.shape) < 1:
        raise ValueError('Expected nonempty [caption,row,column] cost')
    kernel = -jax.lax.stop_gradient(cost) / EPSILON
    rows, columns = cost.shape[-2:]
    def iteration(_, state):
        u, v = state
        u = -jnp.log(jnp.asarray(rows, cost.dtype))-logsumexp(kernel+v[...,None,:], -1)
        v = -jnp.log(jnp.asarray(columns, cost.dtype))-logsumexp(kernel+u[...,:,None], -2)
        return u, v
    u, v = jax.lax.fori_loop(0, SINKHORN_ITERATIONS, iteration,
        (jnp.zeros(cost.shape[:-1], cost.dtype),
         jnp.zeros((*cost.shape[:-2], columns), cost.dtype)))
    return jax.lax.stop_gradient(jnp.exp(kernel+u[...,:,None]+v[...,None,:]))


def _pair_squared_mean(a, b):
    import jax.numpy as jnp
    # Exact all-pairs mean without materializing [C,B,N,D].
    return jnp.maximum(jnp.mean(jnp.sum(a*a, -1), -1)
        + jnp.mean(jnp.sum(b*b, -1), -1)
        - 2*jnp.sum(jnp.mean(a, -2)*jnp.mean(b, -2), -1), 0.)


def branch_scale(live, reference, real, *, axis_name=None):
    """One minibatch branch scale, including real and self terms (Eq. 13)."""
    import jax
    import jax.numpy as jnp
    g, ref, target = (jax.lax.stop_gradient(x) for x in (live, reference, real))
    cross = jnp.mean(_pair_squared_mean(g, target))
    self_pair = jnp.mean(_pair_squared_mean(g, ref))
    if axis_name is not None:
        cross, self_pair = (jax.lax.pmean(x, axis_name) for x in (cross, self_pair))
    sigma = jax.lax.stop_gradient(jnp.sqrt(jnp.maximum(.5*(cross+self_pair), RMS_FLOOR**2)))
    return sigma, dict(pair_rms=sigma, live_real_squared=cross,
                       live_reference_squared=self_pair)


def grouped_stage2_loss(live, reference, real, *, axis_name=None):
    """Return the caption mean and diagnostics for the supplied feature batch.

    Inputs are name-keyed pooled [C,B,D], [C,B,D], [C,1,D] arrays. Production
    uses C=64 globally and B=16; smaller arrays are accepted for CPU tests.
    With pmap pass axis_name so all 64 captions share Eq. (13)'s single scale.
    Production supplies all 64 captions, then replays disjoint image gradients
    across devices and sums parameter gradients with psum; no extra mean is due.
    Only live receives gradients. The reference noise is sampled independently
    by the host. No entropy term is added to this cost-only surrogate.
    """
    import jax
    import jax.numpy as jnp
    if any(set(x) != set(FEATURES) for x in (live, reference, real)):
        raise ValueError('Exactly siglip/mae/inception pooled branches required')
    total = jnp.asarray(0., jnp.float32)
    metrics = {}
    for name in FEATURES:
        g, ref, target = (jnp.asarray(x[name], jnp.float32) for x in (live, reference, real))
        if g.ndim != 3 or ref.shape != g.shape or target.shape != (g.shape[0], 1, g.shape[-1]):
            raise ValueError('Invalid caption-grouped features: '+name)
        sigma, scale_metrics = branch_scale(g, ref, target, axis_name=axis_name)
        a = g/sigma
        b, r = (jax.lax.stop_gradient(x/sigma) for x in (ref, target))
        # For N=1 the exact uniform cross plan is 1/B: no numerical OT solve.
        cross = .5*jnp.mean(jnp.sum((a-r)**2, -1))
        cost = half_squared_distance(a, b)
        plan = sinkhorn_plan(cost)
        repulsion = jnp.mean(jnp.sum(plan*cost, (-2, -1)))
        value = cross-repulsion
        total = total + value/len(FEATURES)
        row_residual = jnp.max(jnp.abs(jnp.sum(plan, -1)-1/g.shape[1]), -1)
        column_residual = jnp.max(jnp.abs(jnp.sum(plan, -2)-1/g.shape[1]), -1)
        branch_metrics = dict(scale_metrics, loss=value, cross=cross, self=repulsion,
            row_residual=row_residual, column_residual=column_residual,
            row_residual_max=jnp.max(row_residual), column_residual_max=jnp.max(column_residual),
            plan_mass_mean=jnp.mean(jnp.sum(plan, (-2, -1))),
            sinkhorn_iterations=jnp.asarray(SINKHORN_ITERATIONS, jnp.int32))
        metrics[name] = jax.tree_util.tree_map(jax.lax.stop_gradient, branch_metrics)
    return total, metrics


stage2_loss = grouped_stage2_loss
