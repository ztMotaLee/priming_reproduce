"""Exact differentiable frozen training feature functions (lazy weight loading)."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional
import numpy as np
import jax
import jax.numpy as jnp
def _precision_context(precision):
    import jax
    if precision == 'highest':
        return jax.default_matmul_precision('highest')
    if precision == 'default':
        return jax.default_matmul_precision('default')
    raise ValueError('precision must be default or highest')

def _frozen(tree):
    import jax
    import jax.numpy as jnp
    return jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(jnp.asarray(x, jnp.float32)), tree)

def _linear(x, p, name):
    import jax.numpy as jnp
    value = jnp.matmul(x, p[name + '.weight'].T)
    return value + p[name + '.bias'] if name + '.bias' in p else value

def _norm(x, p, name, eps):
    import jax.numpy as jnp
    import jax
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(variance + eps) * p[name + '.weight'] + p[name + '.bias']

def _attention(q, k, v, heads):
    import jax
    import jax.numpy as jnp

    def split(x):
        return x.reshape(x.shape[0], x.shape[1], heads, -1).transpose(0, 2, 1, 3)
    (q, k, v) = map(split, (q, k, v))
    logits = jnp.matmul(q * q.shape[-1] ** (-0.5), k.swapaxes(-1, -2))
    out = jnp.matmul(jax.nn.softmax(logits, axis=-1), v)
    return out.transpose(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)

def _patchify_gemm(images, kernel, bias=None):
    """Non-overlapping VALID NHWC/HWIO convolution as patchify + GEMM.

    Flattening order is (patch_h, patch_w, channel), matching the native
    ``_unfold_patch_embed_nhwc`` implementation. VALID trailing pixels are
    discarded exactly as in the original convolution. The input VJP is the
    inverse patch permutation after a matrix product, avoiding the dilated
    patch-convolution pullback. Dtype and precision follow the caller unchanged.
    """
    import jax.numpy as jnp
    if images.ndim != 4 or kernel.ndim != 4:
        raise ValueError('Patch GEMM requires NHWC images and HWIO kernel')
    (batch, height, width, channels) = images.shape
    (patch_h, patch_w, kernel_channels, outputs) = kernel.shape
    if channels != kernel_channels or min(patch_h, patch_w) <= 0:
        raise ValueError('Patch kernel/input channel or spatial mismatch')
    (grid_h, grid_w) = (height // patch_h, width // patch_w)
    if min(grid_h, grid_w) <= 0:
        raise ValueError('Patch kernel exceeds input spatial shape')
    value = images[:, :grid_h * patch_h, :grid_w * patch_w, :]
    value = value.reshape(batch, grid_h, patch_h, grid_w, patch_w, channels)
    value = value.transpose(0, 1, 3, 2, 4, 5)
    value = value.reshape(batch, grid_h, grid_w, patch_h * patch_w * channels)
    value = jnp.matmul(value, kernel.reshape(patch_h * patch_w * channels, outputs))
    return value + bias if bias is not None else value

def _patch_embed_conv(images, p, name, patch):
    """Frozen predecessor's convolution path, retained for same-chunk parity."""
    value = _conv(images, p, name, stride=patch, patch_implementation='conv')
    return value.reshape(value.shape[0], -1, value.shape[-1])

def _patch_embed(images, p, name, patch, *, patch_implementation='gemm'):
    value = _conv(images, p, name, stride=patch, patch_implementation=patch_implementation)
    return value.reshape(value.shape[0], -1, value.shape[-1])

def resize_continuous(images, size, *, method='cubic', antialias=True):
    """Half-pixel continuous resize; no PIL, uint8 conversion or gradient break."""
    import jax
    return jax.image.resize(images, (images.shape[0], size, size, images.shape[-1]), method=method, antialias=antialias)

def preprocess(images, encoder):
    import jax.numpy as jnp
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('Expected NHWC RGB pixels')
    value = (images.astype(jnp.float32) + 1.0) / 2.0
    value = jnp.where(value < 0.0, 0.0, jnp.where(value > 1.0, 1.0, value))
    if encoder == 'siglip':
        return (resize_continuous(value, 224) - 0.5) / 0.5
    if encoder == 'mae':
        value = resize_continuous(value, 224, method='linear')
        return (value - jnp.asarray([0.485, 0.456, 0.406])) / jnp.asarray([0.229, 0.224, 0.225])
    if encoder == 'convnext':
        value = resize_continuous(value, 224)
        return (value - jnp.asarray([0.485, 0.456, 0.406])) / jnp.asarray([0.229, 0.224, 0.225])
    raise ValueError('Unknown encoder: ' + str(encoder))

def siglip_forward(images, params, *, depth=12, heads=12, patch_implementation='gemm'):
    """Original SigLIP Base-16 trunk and native global attention pooling."""
    import jax
    import jax.numpy as jnp
    p = params
    x = _patch_embed(images, p, 'patch_embed.proj', 16, patch_implementation=patch_implementation) + p['pos_embed']
    for index in range(depth):
        base = 'blocks.' + str(index)
        (q, k, v) = jnp.split(_linear(_norm(x, p, base + '.norm1', 1e-06), p, base + '.attn.qkv'), 3, -1)
        x = x + _linear(_attention(q, k, v, heads), p, base + '.attn.proj')
        y = _linear(_norm(x, p, base + '.norm2', 1e-06), p, base + '.mlp.fc1')
        x = x + _linear(jax.nn.gelu(y, approximate=False), p, base + '.mlp.fc2')
    x = _norm(x, p, 'norm', 1e-06)
    q = _linear(jnp.broadcast_to(p['attn_pool.latent'], (len(x), *p['attn_pool.latent'].shape[1:])), p, 'attn_pool.q')
    (k, v) = jnp.split(_linear(x, p, 'attn_pool.kv'), 2, -1)
    x = _linear(_attention(q, k, v, heads), p, 'attn_pool.proj')
    y = _linear(_norm(x, p, 'attn_pool.norm', 1e-06), p, 'attn_pool.mlp.fc1')
    return (x + _linear(jax.nn.gelu(y, approximate=False), p, 'attn_pool.mlp.fc2'))[:, 0]

def mae_forward(images, params, *, depth=12, heads=12, patch_implementation='gemm'):
    """Unmasked deterministic ViT-MAE; mean of final normalized patch tokens."""
    import jax
    import jax.numpy as jnp
    p = params
    position = p['embeddings.position_embeddings']
    x = _patch_embed(images, p, 'embeddings.patch_embeddings.projection', 16, patch_implementation=patch_implementation) + position[:, 1:]
    cls = jnp.broadcast_to(p['embeddings.cls_token'] + position[:, :1], (len(x), 1, x.shape[-1]))
    x = jnp.concatenate((cls, x), axis=1)
    for index in range(depth):
        base = 'encoder.layer.' + str(index)
        y = _norm(x, p, base + '.layernorm_before', 1e-12)
        (q, k, v) = [_linear(y, p, base + '.attention.attention.' + kind) for kind in ('query', 'key', 'value')]
        x = x + _linear(_attention(q, k, v, heads), p, base + '.attention.output.dense')
        y = _linear(_norm(x, p, base + '.layernorm_after', 1e-12), p, base + '.intermediate.dense')
        x = x + _linear(jax.nn.gelu(y, approximate=False), p, base + '.output.dense')
    return jnp.mean(_norm(x, p, 'layernorm', 1e-12)[:, 1:], axis=1)

def _conv(images, params, name, stride=1, padding='VALID', groups=1, *, patch_implementation='gemm'):
    import jax
    if patch_implementation not in ('gemm', 'conv'):
        raise ValueError('patch_implementation must be gemm or conv')
    kernel = params[name + '.weight'].transpose(2, 3, 1, 0)
    if patch_implementation == 'gemm' and groups == 1 and (padding == 'VALID') and (kernel.shape[:2] == (stride, stride)):
        return _patchify_gemm(images, kernel, params[name + '.bias'])
    value = jax.lax.conv_general_dilated(images, kernel, (stride, stride), padding, dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=groups)
    return value + params[name + '.bias']

def _load_safetensors(path, prefix):
    from safetensors import safe_open
    import jax.numpy as jnp
    result = {}
    with safe_open(path, framework='np') as source:
        for key in source.keys():
            if key.startswith(prefix):
                result[key[len(prefix):]] = jnp.asarray(source.get_tensor(key), jnp.float32)
    return result

"""Frozen JAX-FID Inception, adapted from the locally pinned source.

The original network topology is retained. The loader supplies local weights;
conv and batch norm read immutable constants and never download or initialize.
The upstream topology is preserved.
"""
import functools
import pickle
from typing import Any, Callable, Iterable, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.module import merge_param
from jax import lax
from jax.nn import initializers

class utils:

    @staticmethod
    def get(dictionary, key):
        return None if dictionary is None else dictionary.get(key)
PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any

class InceptionV3(nn.Module):
    """
    InceptionV3 network with extended functionality for evaluation metrics.

    Reference: https://arxiv.org/abs/1512.00567
    Ported mostly from: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py

    Extended to support:
    - Spatial feature extraction for sFID
    - Logits extraction for Inception Score
    - Multiple output modes for different evaluation metrics

    Attributes:
        include_head (bool): If True, include classifier head.
        num_classes (int): Number of classes.
        pretrained (bool): If True, use pretrained weights.
        transform_input (bool): If True, preprocesses the input according to the method with which it
                                was trained on ImageNet.
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
        return_spatial (bool): If True, return spatial features from Mixed_6d layer.
        return_logits (bool): If True, return raw logits for Inception Score.
        dtype (str): Data type.
    """
    weights: Any = None
    include_head: bool = False
    num_classes: int = 1000
    pretrained: bool = False
    transform_input: bool = False
    aux_logits: bool = False
    return_spatial: bool = True
    return_logits: bool = False
    ckpt_path: str = 'https://www.dropbox.com/s/xt6zvlvt22dcwck/inception_v3_weights_fid.pickle?dl=1'
    dtype: str = 'float32'

    def setup(self):
        if self.weights is None:
            raise ValueError('Local pinned Inception weights are required')
        self.params_dict = self.weights
        self.num_classes_ = 1008

    @nn.compact
    def __call__(self, x, train=True, rng=None):
        """
        Args:
            x (tensor): Input image, shape [B, H, W, C].
            train (bool): If True, training mode.
            rng (jax.random.PRNGKey): Unused compatibility argument.

        Returns:
            Always returns (pooled_features, spatial_features, logits) where:
            - pooled_features: shape [B, 2048]
            - spatial_features: shape [B, H, W, C] from Mixed_6e layer
            - logits: shape [B, num_classes] or None if include_head=False
        """
        del rng
        if self.include_head or self.aux_logits:
            raise ValueError('This vendor exposes pooled features only')
        x = self._transform_input(x)
        x = BasicConv2d(out_channels=32, kernel_size=(3, 3), strides=(2, 2), params_dict=utils.get(self.params_dict, 'Conv2d_1a_3x3'), dtype=self.dtype)(x, train)
        x = BasicConv2d(out_channels=32, kernel_size=(3, 3), params_dict=utils.get(self.params_dict, 'Conv2d_2a_3x3'), dtype=self.dtype)(x, train)
        x = BasicConv2d(out_channels=64, kernel_size=(3, 3), padding=((1, 1), (1, 1)), params_dict=utils.get(self.params_dict, 'Conv2d_2b_3x3'), dtype=self.dtype)(x, train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = BasicConv2d(out_channels=80, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'Conv2d_3b_1x1'), dtype=self.dtype)(x, train)
        x = BasicConv2d(out_channels=192, kernel_size=(3, 3), params_dict=utils.get(self.params_dict, 'Conv2d_4a_3x3'), dtype=self.dtype)(x, train)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = InceptionA(pool_features=32, params_dict=utils.get(self.params_dict, 'Mixed_5b'), dtype=self.dtype)(x, train)
        x = InceptionA(pool_features=64, params_dict=utils.get(self.params_dict, 'Mixed_5c'), dtype=self.dtype)(x, train)
        x = InceptionA(pool_features=64, params_dict=utils.get(self.params_dict, 'Mixed_5d'), dtype=self.dtype)(x, train)
        x = InceptionB(params_dict=utils.get(self.params_dict, 'Mixed_6a'), dtype=self.dtype)(x, train)
        x = InceptionC(channels_7x7=128, params_dict=utils.get(self.params_dict, 'Mixed_6b'), dtype=self.dtype)(x, train)
        x = InceptionC(channels_7x7=160, params_dict=utils.get(self.params_dict, 'Mixed_6c'), dtype=self.dtype)(x, train)
        (x, spatial_features) = InceptionC(channels_7x7=160, params_dict=utils.get(self.params_dict, 'Mixed_6d'), dtype=self.dtype)(x, train, features=True)
        x = InceptionC(channels_7x7=192, params_dict=utils.get(self.params_dict, 'Mixed_6e'), dtype=self.dtype)(x, train)
        aux = None
        if self.aux_logits and train:
            aux = InceptionAux(num_classes=self.num_classes_, params_dict=utils.get(self.params_dict, 'AuxLogits'), dtype=self.dtype)(x, train)
        x = InceptionD(params_dict=utils.get(self.params_dict, 'Mixed_7a'), dtype=self.dtype)(x, train)
        x = InceptionE(avg_pool, params_dict=utils.get(self.params_dict, 'Mixed_7b'), dtype=self.dtype)(x, train)
        x = InceptionE(nn.max_pool, params_dict=utils.get(self.params_dict, 'Mixed_7c'), dtype=self.dtype)(x, train)
        x = jnp.mean(x, axis=(1, 2), keepdims=True)
        pooled_features = jnp.reshape(x, shape=(x.shape[0], -1))
        logits = None
        if self.include_head:
            x_for_logits = pooled_features
            logits = Dense(features=self.num_classes_, params_dict={'kernel': utils.get(self.params_dict, 'fc')['kernel'], 'bias': utils.get(self.params_dict, 'fc')['bias'] * 0.0}, dtype=self.dtype)(x_for_logits)
        return (pooled_features, spatial_features, logits)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = jnp.expand_dims(x[..., 0], axis=-1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = jnp.expand_dims(x[..., 1], axis=-1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = jnp.expand_dims(x[..., 2], axis=-1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = jnp.concatenate((x_ch0, x_ch1, x_ch2), axis=-1)
        return x

class BasicConv2d(nn.Module):
    """Frozen conv/BN using local constants; no Flax mutable collections."""
    out_channels: int
    kernel_size: Any = (3, 3)
    strides: Any = (1, 1)
    padding: Any = 'VALID'
    params_dict: Any = None
    dtype: str = 'float32'

    def __call__(self, x, train=False):
        if train or self.params_dict is None:
            raise ValueError('Frozen local inference only')
        p = self.params_dict
        padding = self.padding.upper() if isinstance(self.padding, str) else self.padding
        x = lax.conv_general_dilated(x, p['conv']['kernel'], self.strides, padding, dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        bn = p['bn']
        return jax.nn.relu((x - bn['mean']) * lax.rsqrt(bn['var'] + 0.001) * bn['scale'] + bn['bias'])

def avg_pool(inputs, window_shape, strides=None, padding='VALID'):
    """FID pool excludes padded pixels, matching the pinned JAX-FID source."""
    strides = (1, *(strides or (1, 1)), 1)
    window = (1, *window_shape, 1)
    if not isinstance(padding, str):
        padding = ((0, 0), *padding, (0, 0))
    total = lax.reduce_window(inputs, 0.0, lax.add, window, strides, padding)
    counts = lax.reduce_window(jnp.ones_like(inputs[..., :1]), 0.0, lax.add, window, strides, padding)
    return total / counts

class InceptionA(nn.Module):
    pool_features: int
    params_dict: dict = None
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch1x1 = BasicConv2d(out_channels=64, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch1x1'), dtype=self.dtype)(x, train)
        branch5x5 = BasicConv2d(out_channels=48, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch5x5_1'), dtype=self.dtype)(x, train)
        branch5x5 = BasicConv2d(out_channels=64, kernel_size=(5, 5), padding=((2, 2), (2, 2)), params_dict=utils.get(self.params_dict, 'branch5x5_2'), dtype=self.dtype)(branch5x5, train)
        branch3x3dbl = BasicConv2d(out_channels=64, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch3x3dbl_1'), dtype=self.dtype)(x, train)
        branch3x3dbl = BasicConv2d(out_channels=96, kernel_size=(3, 3), padding=((1, 1), (1, 1)), params_dict=utils.get(self.params_dict, 'branch3x3dbl_2'), dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl = BasicConv2d(out_channels=96, kernel_size=(3, 3), padding=((1, 1), (1, 1)), params_dict=utils.get(self.params_dict, 'branch3x3dbl_3'), dtype=self.dtype)(branch3x3dbl, train)
        branch_pool = avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        branch_pool = BasicConv2d(out_channels=self.pool_features, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch_pool'), dtype=self.dtype)(branch_pool, train)
        output = jnp.concatenate((branch1x1, branch5x5, branch3x3dbl, branch_pool), axis=-1)
        return output

class InceptionB(nn.Module):
    params_dict: dict = None
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch3x3 = BasicConv2d(out_channels=384, kernel_size=(3, 3), strides=(2, 2), params_dict=utils.get(self.params_dict, 'branch3x3'), dtype=self.dtype)(x, train)
        branch3x3dbl = BasicConv2d(out_channels=64, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch3x3dbl_1'), dtype=self.dtype)(x, train)
        branch3x3dbl = BasicConv2d(out_channels=96, kernel_size=(3, 3), padding=((1, 1), (1, 1)), params_dict=utils.get(self.params_dict, 'branch3x3dbl_2'), dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl = BasicConv2d(out_channels=96, kernel_size=(3, 3), strides=(2, 2), params_dict=utils.get(self.params_dict, 'branch3x3dbl_3'), dtype=self.dtype)(branch3x3dbl, train)
        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        output = jnp.concatenate((branch3x3, branch3x3dbl, branch_pool), axis=-1)
        return output

class InceptionC(nn.Module):
    channels_7x7: int
    params_dict: dict = None
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True, features=False):
        branch1x1 = BasicConv2d(out_channels=192, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch1x1'), dtype=self.dtype)(x, train)
        spatial_features = branch1x1[..., :7]
        branch7x7 = BasicConv2d(out_channels=self.channels_7x7, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch7x7_1'), dtype=self.dtype)(x, train)
        branch7x7 = BasicConv2d(out_channels=self.channels_7x7, kernel_size=(1, 7), padding=((0, 0), (3, 3)), params_dict=utils.get(self.params_dict, 'branch7x7_2'), dtype=self.dtype)(branch7x7, train)
        branch7x7 = BasicConv2d(out_channels=192, kernel_size=(7, 1), padding=((3, 3), (0, 0)), params_dict=utils.get(self.params_dict, 'branch7x7_3'), dtype=self.dtype)(branch7x7, train)
        branch7x7dbl = BasicConv2d(out_channels=self.channels_7x7, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch7x7dbl_1'), dtype=self.dtype)(x, train)
        branch7x7dbl = BasicConv2d(out_channels=self.channels_7x7, kernel_size=(7, 1), padding=((3, 3), (0, 0)), params_dict=utils.get(self.params_dict, 'branch7x7dbl_2'), dtype=self.dtype)(branch7x7dbl, train)
        branch7x7dbl = BasicConv2d(out_channels=self.channels_7x7, kernel_size=(1, 7), padding=((0, 0), (3, 3)), params_dict=utils.get(self.params_dict, 'branch7x7dbl_3'), dtype=self.dtype)(branch7x7dbl, train)
        branch7x7dbl = BasicConv2d(out_channels=self.channels_7x7, kernel_size=(7, 1), padding=((3, 3), (0, 0)), params_dict=utils.get(self.params_dict, 'branch7x7dbl_4'), dtype=self.dtype)(branch7x7dbl, train)
        branch7x7dbl = BasicConv2d(out_channels=192, kernel_size=(1, 7), padding=((0, 0), (3, 3)), params_dict=utils.get(self.params_dict, 'branch7x7dbl_5'), dtype=self.dtype)(branch7x7dbl, train)
        branch_pool = avg_pool(x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        branch_pool = BasicConv2d(out_channels=192, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch_pool'), dtype=self.dtype)(branch_pool, train)
        output = jnp.concatenate((branch1x1, branch7x7, branch7x7dbl, branch_pool), axis=-1)
        if features:
            return (output, spatial_features)
        else:
            return output

class InceptionD(nn.Module):
    params_dict: dict = None
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch3x3 = BasicConv2d(out_channels=192, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch3x3_1'), dtype=self.dtype)(x, train)
        branch3x3 = BasicConv2d(out_channels=320, kernel_size=(3, 3), strides=(2, 2), params_dict=utils.get(self.params_dict, 'branch3x3_2'), dtype=self.dtype)(branch3x3, train)
        branch7x7x3 = BasicConv2d(out_channels=192, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch7x7x3_1'), dtype=self.dtype)(x, train)
        branch7x7x3 = BasicConv2d(out_channels=192, kernel_size=(1, 7), padding=((0, 0), (3, 3)), params_dict=utils.get(self.params_dict, 'branch7x7x3_2'), dtype=self.dtype)(branch7x7x3, train)
        branch7x7x3 = BasicConv2d(out_channels=192, kernel_size=(7, 1), padding=((3, 3), (0, 0)), params_dict=utils.get(self.params_dict, 'branch7x7x3_3'), dtype=self.dtype)(branch7x7x3, train)
        branch7x7x3 = BasicConv2d(out_channels=192, kernel_size=(3, 3), strides=(2, 2), params_dict=utils.get(self.params_dict, 'branch7x7x3_4'), dtype=self.dtype)(branch7x7x3, train)
        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        output = jnp.concatenate((branch3x3, branch7x7x3, branch_pool), axis=-1)
        return output

class InceptionE(nn.Module):
    pooling: Callable
    params_dict: dict = None
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, train=True):
        branch1x1 = BasicConv2d(out_channels=320, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch1x1'), dtype=self.dtype)(x, train)
        branch3x3 = BasicConv2d(out_channels=384, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch3x3_1'), dtype=self.dtype)(x, train)
        branch3x3_a = BasicConv2d(out_channels=384, kernel_size=(1, 3), padding=((0, 0), (1, 1)), params_dict=utils.get(self.params_dict, 'branch3x3_2a'), dtype=self.dtype)(branch3x3, train)
        branch3x3_b = BasicConv2d(out_channels=384, kernel_size=(3, 1), padding=((1, 1), (0, 0)), params_dict=utils.get(self.params_dict, 'branch3x3_2b'), dtype=self.dtype)(branch3x3, train)
        branch3x3 = jnp.concatenate((branch3x3_a, branch3x3_b), axis=-1)
        branch3x3dbl = BasicConv2d(out_channels=448, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch3x3dbl_1'), dtype=self.dtype)(x, train)
        branch3x3dbl = BasicConv2d(out_channels=384, kernel_size=(3, 3), padding=((1, 1), (1, 1)), params_dict=utils.get(self.params_dict, 'branch3x3dbl_2'), dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl_a = BasicConv2d(out_channels=384, kernel_size=(1, 3), padding=((0, 0), (1, 1)), params_dict=utils.get(self.params_dict, 'branch3x3dbl_3a'), dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl_b = BasicConv2d(out_channels=384, kernel_size=(3, 1), padding=((1, 1), (0, 0)), params_dict=utils.get(self.params_dict, 'branch3x3dbl_3b'), dtype=self.dtype)(branch3x3dbl, train)
        branch3x3dbl = jnp.concatenate((branch3x3dbl_a, branch3x3dbl_b), axis=-1)
        branch_pool = self.pooling(x, window_shape=(3, 3), strides=(1, 1), padding=((1, 1), (1, 1)))
        branch_pool = BasicConv2d(out_channels=192, kernel_size=(1, 1), params_dict=utils.get(self.params_dict, 'branch_pool'), dtype=self.dtype)(branch_pool, train)
        output = jnp.concatenate((branch1x1, branch3x3, branch3x3dbl, branch_pool), axis=-1)
        return output
def inception_preprocess(images):
    import jax
    import jax.numpy as jnp
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('Expected continuous NHWC RGB input in [-1,1]')
    x = jnp.asarray(images, jnp.float32)
    x = jnp.where(x < -1.0, -1.0, jnp.where(x > 1.0, 1.0, x))
    return jax.image.resize(x, (x.shape[0], 299, 299, 3), method='linear', antialias=False)

def inception_forward(images, weights, *, preprocessed=False):
    x = images if preprocessed else inception_preprocess(images)
    model = InceptionV3(weights=weights, include_head=False, transform_input=False, aux_logits=False, return_spatial=False)
    return model.apply({}, x, train=False)[0]
def normalize_patch_tokens(last_hidden_state, eps=1e-06):
    """Final-layer-normalized DINO outputs: discard CLS, L2 each spatial token."""
    import jax.numpy as jnp
    x = jnp.asarray(last_hidden_state, jnp.float32)
    if x.ndim != 3 or x.shape[1:] != (257, 768) or x.shape[0] < 1:
        raise ValueError('DINO-base at 224 must return [B,257,768] final hidden states')
    if eps != 1e-06:
        raise ValueError('fixed per-token L2 epsilon is 1e-6')
    patches = x[:, 1:, :]
    scale = jnp.sqrt(jnp.maximum(jnp.sum(patches * patches, axis=-1, keepdims=True), eps * eps))
    return patches / scale
def pool_patch_tokens(tokens):
    """Pool already normalized DINO patches, preserving spatial quadrant order."""
    import jax.numpy as jnp
    tokens = jnp.asarray(tokens, jnp.float32)
    if tokens.ndim != 3 or tokens.shape[1:] != (256, 768):
        raise ValueError('Expected already normalized DINO patch tokens [B,256,768]')
    grid = tokens.reshape(tokens.shape[0], 16, 16, 768)
    pools = [tokens.mean(axis=1)]
    pools.extend((grid[:, row:row + 8, col:col + 8].mean(axis=(1, 2)) for (row, col) in ((0, 0), (0, 8), (8, 0), (8, 8))))
    return jnp.stack(pools, axis=1)

_DINOV2_PATCH_EMBED_PATCHED=False
def _unfold_patch_embed_nhwc(pixel_values: jax.Array, kernel: jax.Array, bias: Optional[jax.Array], patch_size: Any) -> jax.Array:
    pixel_values = jnp.asarray(pixel_values)
    kernel = jnp.asarray(kernel)
    if isinstance(patch_size, (tuple, list)):
        (patch_h, patch_w) = (int(patch_size[0]), int(patch_size[1]))
    else:
        patch_h = patch_w = int(patch_size)
    if pixel_values.ndim != 4:
        raise ValueError(f'pixel_values must be NHWC (4-D), got shape {pixel_values.shape}')
    (batch, height, width, channels) = pixel_values.shape
    if height % patch_h != 0 or width % patch_w != 0:
        raise ValueError(f'image size ({height}, {width}) is not divisible by patch size ({patch_h}, {patch_w})')
    if kernel.ndim != 4 or tuple(kernel.shape[:3]) != (patch_h, patch_w, channels):
        raise ValueError(f'patch kernel shape {kernel.shape} incompatible with patch size ({patch_h}, {patch_w}) and {channels} input channels')
    (grid_h, grid_w) = (height // patch_h, width // patch_w)
    patches = pixel_values.reshape(batch, grid_h, patch_h, grid_w, patch_w, channels)
    patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
    patches = patches.reshape(batch, grid_h * grid_w, patch_h * patch_w * channels)
    kernel_flat = kernel.reshape(patch_h * patch_w * channels, kernel.shape[-1])
    embeddings = patches @ kernel_flat
    if bias is not None:
        embeddings = embeddings + jnp.asarray(bias)
    return embeddings

def patch_flax_dinov2_unfold_patch_embed() -> bool:
    global _DINOV2_PATCH_EMBED_PATCHED
    if _DINOV2_PATCH_EMBED_PATCHED:
        return False
    from transformers.models.dinov2 import modeling_flax_dinov2 as flax_dinov2
    original_call = flax_dinov2.FlaxDinov2PatchEmbeddings.__call__

    def unfold_call(self, pixel_values):
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if self.is_initializing():
            return original_call(self, pixel_values)
        variables = self.projection.variables
        if 'params' not in variables:
            raise ValueError("FlaxDinov2PatchEmbeddings.projection has no 'params' collection outside Flax initialization -- the unfold+matmul replacement (patch_flax_dinov2_unfold_patch_embed) requires an already-initialized projection; call Module.init(...) (or otherwise bind the projection's kernel/bias params) before apply()")
        params = variables['params']
        kernel = params['kernel']
        if 'bias' not in params:
            raise ValueError(f"FlaxDinov2PatchEmbeddings.projection has no 'bias' param -- the unfold+matmul replacement (patch_flax_dinov2_unfold_patch_embed) requires a projection with both 'kernel' and 'bias' (use_bias=True); available params: {sorted(params)}")
        bias = params['bias']
        return _unfold_patch_embed_nhwc(pixel_values, kernel, bias, self.projection.kernel_size)
    flax_dinov2.FlaxDinov2PatchEmbeddings.__call__ = unfold_call
    _DINOV2_PATCH_EMBED_PATCHED = True
    return True

def verify_file(path, expected):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError('Weight SHA differs: ' + str(path))
    return h.hexdigest()


class FrozenEncoders:
    """Frozen SigLIP, MAE and Inception branches, with default production precision."""
    def __init__(self, weights_root):
        import pickle
        from flax.core import freeze
        root = Path(weights_root)
        expected = {
            'siglip': 'b07f8a4bad267232a87888b2527d054d130b1fba18f75e6e26c17766dc26bdc2',
            'mae': '479dcef4bd5df06259399027b789f21e9d9a1b79f37155a64176d55bc26fdae8',
            'inception': '4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77',
        }
        paths = {'siglip': root/'siglip/model.safetensors',
                 'mae': root/'mae/model.safetensors',
                 'inception': root/'inception/weights.pickle'}
        for name, path in paths.items():
            verify_file(path, expected[name])
        self._params = freeze(_frozen(dict(
            siglip=_load_safetensors(paths['siglip'], ''),
            mae=_load_safetensors(paths['mae'], 'vit.'),
            inception=pickle.loads(paths['inception'].read_bytes()))))

    def branch_features(self, images, *, precision='default'):
        with _precision_context(precision):
            p = _frozen(self._params)
            return dict(siglip=jnp.asarray(siglip_forward(preprocess(images, 'siglip'), p['siglip']), jnp.float32),
                        mae=jnp.asarray(mae_forward(preprocess(images, 'mae'), p['mae']), jnp.float32),
                        inception=jnp.asarray(inception_forward(images, p['inception']), jnp.float32))


def preprocess_512(images):
    """Whole image cubic antialias 512→224; no clipping before normalization."""
    value = (jnp.asarray(images, jnp.float32) + 1.) / 2.
    value = jax.image.resize(value, (value.shape[0], 224, 224, 3), method='cubic', antialias=True)
    return (value-jnp.asarray([.485, .456, .406], jnp.float32))/jnp.asarray([.229, .224, .225], jnp.float32)


class FrozenDINO:
    """Native DINOv2-B conversion, baked 224 position table, five patch pools."""
    EXPECTED_PARAMETER_SHA = 'd8ec3660b1d942936addda8c3613b0d533d2acb41727ae6d4f5bf8afa367a96d'

    def __init__(self, weights_root):
        import copy
        import torch
        from flax.traverse_util import flatten_dict, unflatten_dict
        from transformers import Dinov2Config, Dinov2Model, FlaxDinov2Model
        from transformers.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax
        snapshot = Path(weights_root)/'dino'
        verify_file(snapshot/'config.json', 'f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70')
        verify_file(snapshot/'model.safetensors', 'd73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841')
        patch_flax_dinov2_unfold_patch_embed()
        config = Dinov2Config.from_pretrained(snapshot, local_files_only=True)
        torch_model = Dinov2Model.from_pretrained(snapshot, local_files_only=True,
                         use_safetensors=True, torch_dtype=torch.float32).eval()
        with torch.no_grad():
            placeholder = torch.zeros((1, 257, 768), dtype=torch.float32)
            pos_224 = torch_model.embeddings.interpolate_pos_encoding(placeholder, 224, 224)
            native_pos = torch_model.embeddings.position_embeddings.detach()
        pos_224 = np.ascontiguousarray(pos_224.detach().cpu().numpy(), dtype=np.float32)
        native_pos = np.ascontiguousarray(native_pos.cpu().numpy(), dtype=np.float32)
        if pos_224.shape != (1, 257, 768) or native_pos.shape != (1, 1370, 768):
            raise ValueError('DINO position table shape differs')
        if not np.array_equal(pos_224[0, 0], native_pos[0, 0]) or not np.isfinite(pos_224).all():
            raise ValueError('DINO interpolated position table differs')
        state = {k: v.detach().cpu().clone() for k,v in torch_model.state_dict().items()}
        flax_native = FlaxDinov2Model(config, input_shape=(1,224,224,3), dtype=jnp.float32, seed=0)
        converted = convert_pytorch_state_dict_to_flax(dict(state), flax_native)
        flat_init, flat = flatten_dict(dict(flax_native.params)), flatten_dict(dict(converted))
        if set(flat_init) != set(flat) or any(tuple(flat[k].shape) != tuple(v.shape) for k,v in flat_init.items()):
            raise ValueError('DINO converted parameter schema differs')
        key = ('embeddings', 'position_embeddings')
        if not np.array_equal(np.asarray(flat[key], np.float32), native_pos):
            raise ValueError('DINO native position conversion differs')
        flat[key] = jnp.asarray(pos_224, jnp.float32)
        self.params = jax.device_get(jax.tree_util.tree_map(
            lambda v: jax.lax.stop_gradient(jnp.asarray(v,jnp.float32)), unflatten_dict(flat)))
        config_224 = copy.deepcopy(config)
        config_224.image_size = 224
        self.model = FlaxDinov2Model(config_224, input_shape=(1,224,224,3), dtype=jnp.float32, _do_init=False)
        del torch_model, state, converted, flat_init, flat, flax_native
        actual = self.parameter_digest()
        if actual != self.EXPECTED_PARAMETER_SHA:
            raise ValueError('DINO actual parameter SHA differs: '+actual)

    def parameter_digest(self):
        h=hashlib.sha256()
        for path,value in jax.tree_util.tree_flatten_with_path(self.params)[0]:
            a=np.asarray(jax.device_get(value))
            h.update(str(path).encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
        return h.hexdigest()

    def encode_with_params(self, params, images):
        with jax.default_matmul_precision('float32'):
            outputs=self.model(jnp.transpose(preprocess_512(images),(0,3,1,2)),params=params,train=False,return_dict=False)
        return normalize_patch_tokens(outputs[0])

    def encode(self, images):
        return self.encode_with_params(self.params, images)

    def pooled_with_params(self, params, images):
        return pool_patch_tokens(self.encode_with_params(params, images))

    def pooled(self, images):
        return self.pooled_with_params(self.params, images)
