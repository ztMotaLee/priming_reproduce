"""Priming x0 prediction with the unmodified, timestep-free MiniT2I backbone.

MiniT2I-JAX commit eeee606 already removes timestep/adaLN conditioning from
MMJiT. Reusing its module directly preserves every checkpoint parameter key;
there are no removed or newly introduced generator keys. MeanFlow auxiliary
token modules are a different architecture and are not silently imported.
"""
import math

import jax

from models.mmjit import MMJiT, MMJiT_B_16_txt2


REMOVED_PARAMETER_KEYS = ()
MODEL_NAME = "MMJiT_B_16_txt2"


def build_model(image_size=512, *, tiny=False):
    """Return a Flax module accepting only (NHWC image, T5 context, mask).

    Production uses FLAN-T5-Large's 1024-dimensional token representations.
    ``tiny`` is exclusively for explicit smoke tests and uses 16-D context.
    """
    if tiny:
        return MMJiT(
            img_input_size=image_size, txt_input_size=16, txt_input_length=4,
            patch_size=4, hidden_size=32, txt_hidden_size=32,
            depth_double=1, txt_preamble_depth=1, num_heads=2,
            head_dim=16, mlp_ratio=2.0, pca_channels=8,
        )
    return MMJiT_B_16_txt2(
        img_input_size=image_size, txt_input_size=1024, txt_input_length=256,
    )


def parameter_count(params):
    return sum(math.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params))


def abstract_parameter_count():
    """Count production parameters without allocating or loading its weights."""
    model = build_model(32)
    shapes = jax.eval_shape(
        model.init, jax.random.PRNGKey(0),
        jax.ShapeDtypeStruct((1, 32, 32, 3), "float32"),
        jax.ShapeDtypeStruct((1, 4, 1024), "float32"),
        jax.ShapeDtypeStruct((1, 4), "int32"),
    )
    return parameter_count(shapes["params"])
