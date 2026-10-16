"""Verified Stage-1 backbone initialization for downstream MiniT2I training.

This transfers weights only. It does not restore downstream optimizers, turn
MeanFlow auxiliary tokens into Stage-1-trained tokens, or assert that the
resulting auxiliary-conditioned network has the Stage-1 input/output function.
"""
from collections.abc import Mapping
from pathlib import Path
import math

from flax import serialization, traverse_util
from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import sha256, verify_checkpoint


# Exactly the extra top-level modules in eeee606 models/mean_flow_mmjit.py.
# Unknown modules cannot be silently kept under the opt-in.
MEANFLOW_AUXILIARY_MODULES = frozenset({
    "t_embedder", "pooled_embedder", "aux_pooled_embedder", "h_embedder",
    "omega_embedder", "t_min_embedder", "t_max_embedder", "context_tokens",
    "time_tokens", "omega_tokens", "t_min_tokens", "t_max_tokens",
})


def _flat(tree):
    if not isinstance(tree, Mapping):
        raise ValueError("Expected a parameter mapping")
    values = traverse_util.flatten_dict(unfreeze(tree) if isinstance(tree, FrozenDict) else tree)
    if not values or any(not hasattr(value, "shape") for value in values.values()):
        raise ValueError("Expected nonempty array-valued parameter leaves")
    return values


def _names(paths):
    return ["/".join(map(str, path)) for path in sorted(paths)]


def initialize_posttrain_params(initialized_params, checkpoint_dir, branch="net",
                                weights="online", allow_meanflow_aux=False):
    """Return (new parameter tree, provenance receipt), without mutating inputs.

    ``branch='net'`` selects that branch of an already initialized downstream
    wrapper. ``branch=None`` selects an initialized bare MMJiT backbone.
    Every trained backbone key and shape must match exactly. Only an explicit
    ``allow_meanflow_aux=True`` permits the listed upstream MeanFlow auxiliary
    modules to retain their initialized values in the selected branch.

    Unselected wrapper branches are retained and listed in the receipt. The
    caller should atomically save this receipt before starting post-training.
    Stage-1 optimizer/EMA state is intentionally not a downstream optimizer.
    """
    if weights not in {"online", "ema"}:
        raise ValueError("weights must be online or ema")
    if branch is not None and (not isinstance(branch, str) or not branch):
        raise ValueError("branch must be a nonempty key or None")
    parent = Path(checkpoint_dir).resolve()
    parent_receipt = verify_checkpoint(parent)
    step = int(parent_receipt["step"])
    export_dir = "minit2i" if weights == "online" else "minit2i_ema"
    export_path = parent / export_dir / f"checkpoint_{step}"
    payload = serialization.msgpack_restore(export_path.read_bytes())
    if not isinstance(payload, Mapping) or set(payload) != {"params"}:
        raise ValueError("Expected the native params-only Stage-1 export")
    if not isinstance(payload["params"], Mapping) or set(payload["params"]) != {"net"}:
        raise ValueError("Stage-1 export must contain exactly params.net")
    source = _flat(payload["params"]["net"])
    if not isinstance(initialized_params, Mapping):
        raise ValueError("initialized_params must be a mapping")
    if branch is None:
        selected = initialized_params
    else:
        if branch not in initialized_params:
            raise ValueError(f"Missing selected downstream branch: {branch}")
        selected = initialized_params[branch]
    target = _flat(selected)
    missing = set(source) - set(target)
    extra = set(target) - set(source)
    if missing:
        raise ValueError("Downstream backbone is missing Stage-1 keys: " + ", ".join(_names(missing)[:8]))
    unsupported = {path for path in extra
                   if not allow_meanflow_aux or path[0] not in MEANFLOW_AUXILIARY_MODULES}
    if unsupported:
        raise ValueError("Unexpected downstream parameter keys: " + ", ".join(_names(unsupported)[:8]))
    for path, value in source.items():
        destination = target[path]
        if value.shape != destination.shape:
            raise ValueError("Backbone shape differs: " + "/".join(map(str, path)))
        if not np.issubdtype(np.dtype(value.dtype), np.floating):
            raise ValueError("Source backbone parameters must be floating point")
        if not jnp.issubdtype(destination.dtype, jnp.floating):
            raise ValueError("Initialized backbone parameters must be floating point")
    transferred = dict(target)
    for path, value in source.items():
        # Preserve the downstream parameter dtype; report any casts explicitly.
        transferred[path] = jnp.asarray(value, dtype=target[path].dtype)
    selected_result = traverse_util.unflatten_dict(transferred)
    if isinstance(selected, FrozenDict):
        selected_result = freeze(selected_result)
    if branch is None:
        result = selected_result
        untouched = []
    else:
        result = unfreeze(initialized_params) if isinstance(initialized_params, FrozenDict) else dict(initialized_params)
        result[branch] = selected_result
        untouched = sorted(str(key) for key in initialized_params if key != branch)
        if isinstance(initialized_params, FrozenDict):
            result = freeze(result)
    prefix = () if branch is None else (branch,)
    kept = [prefix + path for path in extra]
    for key in untouched:
        kept.extend((key,) + path for path in _flat(initialized_params[key]))
    receipt = {
        "format": "priming-stage1-backbone-transfer-v1",
        "parent_checkpoint": str(parent), "parent_step": step,
        "parent_receipt_sha256": sha256(parent / "receipt.json"),
        "parent_manifest_sha256": sha256(parent / "manifest.json"),
        "export_sha256": sha256(export_path), "weights": weights, "branch": branch,
        "source_branch": "net", "allow_meanflow_aux": bool(allow_meanflow_aux),
        "transferred_leaves": len(source),
        "transferred_numel": sum(math.prod(value.shape) for value in source.values()),
        "kept_initialized_keys": _names(kept), "untouched_wrapper_branches": untouched,
        "dtype_cast_keys": _names(path for path in source if source[path].dtype != target[path].dtype),
        "operation": "weights_only_warmstart",
        # Parameter compatibility cannot prove a caller-defined forward is
        # functionally equivalent; MeanFlow additionally inserts new tokens.
        "stage1_function_equivalent": False,
        "function_equivalence": "not_asserted_for_downstream_forward",
        "optimizer_restored": False,
    }
    return result, receipt
