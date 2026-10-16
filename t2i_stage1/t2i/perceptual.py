"""Frozen, differentiable JAX LPIPS-VGG16 and ConvNeXt-V2-Base features.

Independent JAX implementation of the reference C2I reproduction equations.
Architectures/weights: pytorch/vision VGG16 (BSD-3-Clause), richzhang/
PerceptualSimilarity v0.1 calibration (BSD-2-Clause), and Hugging Face
facebook/convnextv2-base-22k-224 (model card and Transformers implementation
Apache-2.0). Weight licenses are separate from this MIT code.
No random feature weights or fallbacks are permitted in the production loader.

Paper authors: https://github.com/JaeyeonKim01/priming
Reference C2I reproduction:
https://github.com/ztMotaLee/priming_reproduce
https://github.com/richzhang/PerceptualSimilarity
https://huggingface.co/facebook/convnextv2-base-22k-224
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np


CONVNEXT_ID = "facebook/convnextv2-base-22k-224"
CONVNEXT_REVISION = "758ff0922dc09136abb55774e7f8b1e1bd0dc344"
LPIPS_URL = "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth"
LPIPS_MD5 = "d507d7349b931f0638a25a48a722f98a"
VGG_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
VGG_LAYERS = ((0, 2), (5, 7), (10, 12, 14), (17, 19, 21), (24, 26, 28))
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _conv(x, weights, bias, *, stride=1, padding="VALID", groups=1):
    out = jax.lax.conv_general_dilated(
        x, weights, (stride, stride), padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=groups,
    )
    return out + bias


def _norm(x, weight, bias, epsilon=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + epsilon) * weight + bias


def vgg_features(params, images):
    """Return relu1_2, relu2_2, relu3_3, relu4_3, relu5_3 (NHWC)."""
    features = []
    x = images
    for stage, layers in enumerate(VGG_LAYERS):
        if stage:
            x = jax.lax.reduce_window(
                x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID",
            )
        for index in layers:
            x = jax.nn.relu(_conv(x, params[f"vgg.{index}.weight"],
                                 params[f"vgg.{index}.bias"], padding=((1, 1), (1, 1))))
        features.append(x)
    return tuple(features)


def lpips_distance(params, prediction, target):
    lhs, rhs = vgg_features(params, prediction), vgg_features(params, target)
    result = jnp.zeros((prediction.shape[0],), dtype=jnp.float32)
    for index, (left, right) in enumerate(zip(lhs, rhs)):
        # Official normalize_tensor divides by sqrt(sum(x^2)) + eps.
        left = left / (jnp.sqrt(jnp.sum(left**2, axis=-1, keepdims=True)) + 1e-10)
        right = right / (jnp.sqrt(jnp.sum(right**2, axis=-1, keepdims=True)) + 1e-10)
        result += jnp.mean(jnp.sum((left - right) ** 2 * params[f"lpips.{index}"], axis=-1), axis=(1, 2))
    return result


def convnext_features(params, images, depths=(3, 3, 27, 3), layer_norm_eps=1e-6):
    """HF ConvNeXtV2 pooled output, excluding its classification head."""
    def affine(x, prefix, eps=1e-6):
        return _norm(x, params[prefix + ".weight"], params[prefix + ".bias"], eps)

    def conv(x, prefix, **kwargs):
        return _conv(x, params[prefix + ".weight"], params[prefix + ".bias"], **kwargs)

    def linear(x, prefix):
        return jnp.matmul(x, params[prefix + ".weight"]) + params[prefix + ".bias"]

    x = conv(images, "cn.embeddings.patch_embeddings", stride=4)
    x = affine(x, "cn.embeddings.layernorm")
    for stage, depth in enumerate(depths):
        prefix = f"cn.encoder.stages.{stage}"
        if stage:
            x = affine(x, prefix + ".downsampling_layer.0")
            x = conv(x, prefix + ".downsampling_layer.1", stride=2)
        for index in range(depth):
            block = prefix + f".layers.{index}"
            residual = x
            x = conv(x, block + ".dwconv", padding=((3, 3), (3, 3)), groups=x.shape[-1])
            x = affine(x, block + ".layernorm")
            x = jax.nn.gelu(linear(x, block + ".pwconv1"), approximate=False)
            # HF uses no epsilon inside the spatial L2 norm (unlike some pMF ports).
            gx = jnp.sqrt(jnp.sum(x**2, axis=(1, 2), keepdims=True))
            nx = gx / (jnp.mean(gx, axis=-1, keepdims=True) + 1e-6)
            x = params[block + ".grn.weight"] * x * nx + params[block + ".grn.bias"] + x
            x = residual + linear(x, block + ".pwconv2")
    return affine(jnp.mean(x, axis=(1, 2)), "cn.layernorm", layer_norm_eps)


def _cubic(distance):
    """PyTorch grid_sample's Keys cubic kernel (a=-0.75)."""
    x = jnp.abs(distance)
    a = -0.75
    return jnp.where(x <= 1, (a + 2) * x**3 - (a + 3) * x**2 + 1,
                     jnp.where(x < 2, a * x**3 - 5 * a * x**2 + 8 * a * x - 4 * a, 0))


def bicubic_crop(images, tops, lefts, heights, widths, out_size=224):
    """Equivalent to border/align_corners=True bicubic grid_sample.

    The crop coordinates are integer pixel centers. Both members of a pair
    must receive the same coordinates; there is no antialias filter here.
    """
    def one(image, top, left, height, width):
        ys = top + jnp.linspace(0.0, 1.0, out_size) * (height - 1)
        xs = left + jnp.linspace(0.0, 1.0, out_size) * (width - 1)
        yi, xi = jnp.floor(ys).astype(jnp.int32), jnp.floor(xs).astype(jnp.int32)
        yidx = yi[:, None] + jnp.arange(-1, 3)[None, :]
        xidx = xi[:, None] + jnp.arange(-1, 3)[None, :]
        wy = _cubic(ys[:, None] - yidx)
        wx = _cubic(xs[:, None] - xidx)
        yidx = jnp.clip(yidx, 0, image.shape[0] - 1)
        xidx = jnp.clip(xidx, 0, image.shape[1] - 1)
        # Separable interpolation avoids a large (S,S,4,4,C) intermediate.
        vertical = jnp.sum(image[yidx, :, :] * wy[:, :, None, None], axis=1)
        return jnp.sum(vertical[:, xidx, :] * wx[None, :, :, None], axis=2)
    return jax.vmap(one)(images, tops, lefts, heights, widths)


def paired_random_crop(key, prediction, target, out_size=224):
    if key is None:
        raise ValueError("paired random perceptual crops require an explicit PRNG key")
    b, h, w, _ = prediction.shape
    area_key, ratio_key, top_key, left_key = jax.random.split(key, 4)
    area = h * w * jax.random.uniform(area_key, (b,), minval=0.08, maxval=1.0)
    ratio = jnp.exp(jax.random.uniform(ratio_key, (b,), minval=np.log(0.75), maxval=np.log(4 / 3)))
    widths = jnp.clip(jnp.round(jnp.sqrt(area * ratio)).astype(jnp.int32), 1, w)
    heights = jnp.clip(jnp.round(jnp.sqrt(area / ratio)).astype(jnp.int32), 1, h)
    tops = jnp.floor(jax.random.uniform(top_key, (b,)) * (h - heights + 1)).astype(jnp.int32)
    lefts = jnp.floor(jax.random.uniform(left_key, (b,)) * (w - widths + 1)).astype(jnp.int32)
    return (bicubic_crop(prediction, tops, lefts, heights, widths, out_size),
            bicubic_crop(target, tops, lefts, heights, widths, out_size))


@dataclass
class FrozenPerceptual:
    params: dict
    depths: tuple = (3, 3, 27, 3)
    layer_norm_eps: float = 1e-6
    random_crop: bool = True

    def apply(self, params, prediction, target, rng=None):
        """Feature parameters are explicit to avoid embedding large JIT constants."""
        prediction = jnp.clip(prediction.astype(jnp.float32) * 0.5 + 0.5, 0, 1)
        target = jax.lax.stop_gradient(jnp.clip(target.astype(jnp.float32) * 0.5 + 0.5, 0, 1))
        prediction = (prediction - jnp.asarray(MEAN)) / jnp.asarray(STD)
        target = (target - jnp.asarray(MEAN)) / jnp.asarray(STD)
        if self.random_crop:
            prediction, target = paired_random_crop(rng, prediction, target)
        else:
            shape = (prediction.shape[0], 224, 224, 3)
            prediction = jax.image.resize(prediction, shape, "linear", antialias=True)
            target = jax.image.resize(target, shape, "linear", antialias=True)
        lpips = lpips_distance(params, prediction, target)
        left = convnext_features(params, prediction, self.depths, self.layer_norm_eps)
        right = convnext_features(params, target, self.depths, self.layer_norm_eps)
        return lpips, jnp.sum((left - right) ** 2, axis=-1)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_perceptual(directory, *, random_crop=True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "weights.npz"
    if manifest.get("format") != "priming-perceptual-v1" or _sha(path) != manifest["sha256"]:
        raise ValueError("perceptual weight manifest/hash mismatch")
    if (manifest["sources"]["lpips_md5"] != LPIPS_MD5
            or manifest["sources"]["convnext_id"] != CONVNEXT_ID
            or manifest["sources"].get("convnext_revision") != CONVNEXT_REVISION):
        raise ValueError("perceptual source differs from the specified Priming features")
    if manifest["depths"] != [3, 3, 27, 3] or manifest["hidden_sizes"] != [128, 256, 512, 1024]:
        raise ValueError("expected ConvNeXt-V2-Base architecture")
    with np.load(path, allow_pickle=False) as archive:
        params = {key: jnp.asarray(archive[key], dtype=jnp.float32) for key in archive.files}
    for index, layers in enumerate(VGG_LAYERS):
        for layer in layers:
            if f"vgg.{layer}.weight" not in params or f"vgg.{layer}.bias" not in params:
                raise ValueError("incomplete pretrained VGG feature parameters")
        if f"lpips.{index}" not in params:
            raise ValueError("missing pretrained LPIPS calibration")
    if len(params) != manifest["tensor_count"]:
        raise ValueError("perceptual tensor count mismatch")
    return FrozenPerceptual(params, tuple(manifest["depths"]), manifest["layer_norm_eps"], random_crop)


def _publish_once(path, writer):
    """Create an artifact atomically, refusing to overwrite an existing one."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".prepare-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def prepare(directory, *, raw_directory=None):
    """Convert pinned local raw weights offline, without any download fallback.

    The separate stdlib download phase runs on an internet-enabled node.
    Conversion belongs on compute; partial outputs fail closed. All PyTorch
    work happens in setup, never inside the JAX training graph.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        load_perceptual(directory)
        print("Perceptual assets verified")
        return
    if (directory / "weights.npz").exists():
        raise FileExistsError("partial perceptual asset set: use a new output directory")
    from .asset_downloads import validate_raw_perceptual
    raw = Path(raw_directory) if raw_directory is not None else directory.parent / 'perceptual-sources'
    validate_raw_perceptual(raw)
    import torch
    from transformers import ConvNextV2ForImageClassification
    torch.set_num_threads(1)
    revision = CONVNEXT_REVISION
    vgg_file = raw / 'vgg16-397923af.pth'
    calibration_file = raw / 'lpips-vgg.pth'
    vgg = torch.load(vgg_file, map_location="cpu", weights_only=True)
    lpips = torch.load(calibration_file, map_location="cpu", weights_only=True)
    if hashlib.md5(calibration_file.read_bytes()).hexdigest() != LPIPS_MD5:
        raise ValueError("LPIPS calibration MD5 differs from the released checkpoint")
    convnext = ConvNextV2ForImageClassification.from_pretrained(
        str(raw / 'convnext'), local_files_only=True, use_safetensors=True).eval().convnextv2
    config = convnext.config
    arrays = {}
    for layers in VGG_LAYERS:
        for index in layers:
            for kind in ("weight", "bias"):
                value = vgg[f"features.{index}.{kind}"].detach().cpu().numpy()
                arrays[f"vgg.{index}.{kind}"] = value.transpose(2, 3, 1, 0) if value.ndim == 4 else value
    for index in range(5):
        arrays[f"lpips.{index}"] = lpips[f"lin{index}.model.1.weight"].detach().cpu().numpy().reshape(-1)
    for key, tensor in convnext.state_dict().items():
        value = tensor.detach().cpu().numpy()
        if value.ndim == 4 and ".grn." not in key:
            value = value.transpose(2, 3, 1, 0)
        elif value.ndim == 2:
            value = value.T
        arrays["cn." + key] = np.asarray(value, dtype=np.float32)
    _publish_once(directory / "weights.npz", lambda stream: np.savez(stream, **arrays))
    manifest = {
        "format": "priming-perceptual-v1", "sha256": _sha(directory / "weights.npz"),
        "tensor_count": len(arrays), "depths": list(config.depths),
        "hidden_sizes": list(config.hidden_sizes), "layer_norm_eps": config.layer_norm_eps,
        "sources": {"vgg_url": VGG_URL, "vgg_sha256": _sha(vgg_file),
                    "lpips_url": LPIPS_URL, "lpips_md5": LPIPS_MD5,
                    "lpips_sha256": _sha(calibration_file), "convnext_id": CONVNEXT_ID,
                    "convnext_revision": revision},
    }
    _publish_once(directory / "manifest.json", lambda stream: stream.write((json.dumps(manifest, indent=2) + "\n").encode()))
    binding = {"files": {name: _sha(directory / name) for name in ("weights.npz", "manifest.json")}}
    _publish_once(directory / "assets.sha256.json", lambda stream: stream.write((json.dumps(binding, indent=2) + "\n").encode()))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    from t2i.asset_downloads import offline_only
    with offline_only():
        prepare(args.output_dir)
