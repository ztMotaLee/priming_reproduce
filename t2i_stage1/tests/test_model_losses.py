"""Small CPU checks; no public model weights or dataset downloads.

Set PRIMING_C2I_ROOT to an existing official C2I source checkout to compare
directly with its reconstruction.py, LPIPS, and PerceptualLoss definitions.
"""
import importlib.util
import inspect
import math
import os
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from t2i.model import build_model, abstract_parameter_count
from t2i.noise import sample_noise_levels, mixture_parameters, corrupt_inputs
from t2i.losses import reconstruction_loss, adaptive_weight
from t2i.perceptual import (
    VGG_LAYERS, bicubic_crop, convnext_features, lpips_distance, paired_random_crop,
)

jax.config.update("jax_default_matmul_precision", "highest")


def source_module(name, relative):
    root = os.environ.get("PRIMING_C2I_ROOT")
    if not root:
        return None
    path = Path(root) / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoiseTests(unittest.TestCase):
    def test_exact_density_cdf_and_logit_moments(self):
        # One seed, one bounded sample; analytic mixture CDF is independent of sampler.
        s = np.asarray(sample_noise_levels(jax.random.PRNGKey(7), 32768))
        self.assertTrue(np.all((s > 0) & (s < 1)))
        z = np.log(s) - np.log1p(-s)
        k = np.arange(3)
        weights = np.array([1, 2, 1]) * np.exp(-k * 0.8 + 0.5 * k**2 * 0.8**2)
        weights /= weights.sum()
        means = 0.8 - k * 0.8**2
        expected_mean = (weights * means).sum()
        expected_second = (weights * (0.8**2 + means**2)).sum()
        self.assertLess(abs(z.mean() - expected_mean), 0.018)
        self.assertLess(abs(np.mean(z**2) - expected_second), 0.028)
        for point in (0.2, 0.4, 0.6, 0.8):
            logit = math.log(point / (1 - point))
            expected = sum(w * 0.5 * (1 + math.erf((logit - mean) / (0.8 * math.sqrt(2))))
                           for w, mean in zip(weights, means))
            self.assertLess(abs(np.mean(s <= point) - expected), 0.012)
        # Mixture density matches explicitly s^-2 tilted base normal density.
        for point in (-2.0, 0.0, 1.7):
            mixture = sum(w * np.exp(-0.5 * ((point - mean) / 0.8)**2)
                          for w, mean in zip(weights, means))
            normalizer = sum(np.array([1, 2, 1]) * np.exp(-k * 0.8 + 0.5 * k**2 * 0.8**2))
            tilted = (1 + np.exp(-point))**2 * np.exp(-0.5 * ((point - 0.8) / 0.8)**2) / normalizer
            self.assertAlmostEqual(mixture, tilted, places=12)

    def test_fixed_and_pure_noise(self):
        key = jax.random.PRNGKey(1)
        s = sample_noise_levels(key, 2, mode="fixed", fixed_s=1.0)
        np.testing.assert_array_equal(s, np.ones(2))
        images = jnp.ones((2, 4, 4, 3))
        np.testing.assert_allclose(corrupt_inputs(key, images, s), jax.random.normal(key, images.shape) * 2)
        np.testing.assert_allclose(sample_noise_levels(key, 2, mode="fixed", fixed_s=.8), .8)
        with self.assertRaises(ValueError):
            mixture_parameters(sigma=0)


class ModelTests(unittest.TestCase):
    def test_tiny_forward_no_time_argument_and_text_gradient(self):
        model = build_model(8, tiny=True)
        self.assertEqual(tuple(inspect.signature(type(model).__call__).parameters),
                         ("self", "img", "context", "attn_mask"))
        images = jnp.ones((1, 8, 8, 3))
        context = jnp.ones((1, 4, 16))
        mask = jnp.ones((1, 4), jnp.int32)
        params = model.init(jax.random.PRNGKey(1), images, context, mask)["params"]
        forbidden = {"t_embedder", "time_tokens", "timestep_embedder", "noise_embedder"}
        self.assertFalse(any(any(getattr(part, "key", None) in forbidden for part in path)
                             for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]))
        output = model.apply({"params": params}, images, context, mask)
        self.assertEqual(output.shape, images.shape)
        # Upstream intentionally zero-initializes final projection. Make that
        # projection nonzero to test the actual caption-to-output gradient path.
        final = params["final_layer"]["linear"]["_flax_linear"]["kernel"]
        params["final_layer"]["linear"]["_flax_linear"]["kernel"] = jnp.linspace(-.01, .02, final.size).reshape(final.shape)
        gradients = jax.grad(lambda ctx: jnp.sum(model.apply({"params": params}, images, ctx, mask)))(context)
        self.assertTrue(np.isfinite(np.asarray(gradients)).all())
        self.assertGreater(float(jnp.linalg.norm(gradients)), 0)

    def test_production_abstract_count(self):
        count = abstract_parameter_count()
        self.assertGreater(count, 250_000_000)
        self.assertLess(count, 265_000_000)
        print("MM-JiT-B/16 abstract parameter count:", count)


class LossTests(unittest.TestCase):
    def test_pixel_value_and_gradient_against_official_torch(self):
        import torch
        torch.set_num_threads(1)
        official = source_module("priming_official_reconstruction", "losses/reconstruction.py")
        if official is None:
            self.skipTest("PRIMING_C2I_ROOT unset: direct official source parity not run")
        prediction = np.linspace(-1.2, 1.3, 2 * 4 * 4 * 3, dtype=np.float32).reshape(2, 4, 4, 3)
        target = np.linspace(.9, -.7, prediction.size, dtype=np.float32).reshape(prediction.shape)
        pred_t = torch.tensor(prediction.transpose(0, 3, 1, 2), requires_grad=True)
        value_t, metrics_t = official.reconstruction_loss(pred_t, torch.tensor(target.transpose(0, 3, 1, 2)))
        value_t.backward()
        loss_fn = lambda pred: reconstruction_loss(pred, jnp.asarray(target), jnp.array([.8, 1.]))[0]
        value_j, gradient_j = jax.value_and_grad(loss_fn)(jnp.asarray(prediction))
        np.testing.assert_allclose(value_j, value_t.detach().numpy(), rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(gradient_j, pred_t.grad.numpy().transpose(0, 2, 3, 1), rtol=3e-6, atol=1e-7)
        self.assertAlmostEqual(float(metrics_t["loss_mse"]), float(np.mean((prediction - target)**2)), places=6)

    def test_per_sample_adaptive_perceptual_threshold(self):
        prediction = jnp.ones((2, 2, 2, 3))
        target = jnp.zeros_like(prediction)
        fn = lambda pred, target, rng: (jnp.array([.2, .6]), jnp.array([2., 8.]))
        loss, metrics = reconstruction_loss(prediction, target, jnp.array([.8, 1.]),
                                             perceptual_fn=fn, perceptual_threshold=1.)
        expected = 1 / 1.0001 + .4 * np.mean(np.array([.2, .6]) / np.array([.21, .61])) + .1 * np.mean(np.array([2., 8.]) / np.array([2.01, 8.01]))
        self.assertAlmostEqual(float(loss), expected, places=6)
        self.assertEqual(float(metrics["perceptual_active_fraction"]), 1.)
        _, masked = reconstruction_loss(prediction, target, jnp.array([.8, 1.]),
                                        perceptual_fn=fn, perceptual_threshold=.8)
        self.assertAlmostEqual(float(masked["aux_loss_lpips"]), .1, places=6)
        gradient = jax.grad(lambda x: jnp.sum(adaptive_weight(x)))(jnp.array([.2, 2.]))
        np.testing.assert_allclose(gradient, 1 / (np.array([.2, 2.]) + 1e-4), rtol=1e-6)


class PerceptualTests(unittest.TestCase):
    def test_frozen_loader_rejects_changed_revision(self):
        import hashlib
        import json
        import tempfile
        from t2i.perceptual import load_perceptual, CONVNEXT_ID, LPIPS_MD5
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = b"revision-binding-fixture"
            (directory / "weights.npz").write_bytes(data)
            manifest = {"format": "priming-perceptual-v1", "sha256": hashlib.sha256(data).hexdigest(),
                        "sources": {"convnext_id": CONVNEXT_ID, "lpips_md5": LPIPS_MD5,
                                    "convnext_revision": "changed-revision"}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "perceptual source differs"):
                load_perceptual(directory)

    def test_bicubic_crop_torch_value_and_gradient(self):
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
        values = np.linspace(-.8, 1.1, 1 * 7 * 9 * 3, dtype=np.float32).reshape(1, 7, 9, 3)
        image_t = torch.tensor(values.transpose(0, 3, 1, 2), requires_grad=True)
        y = torch.linspace(1, 5, 6) * 2 / 6 - 1
        x = torch.linspace(2, 8, 6) * 2 / 8 - 1
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)[None]
        expected = F.grid_sample(image_t, grid, mode="bicubic", padding_mode="border", align_corners=True)
        expected.square().mean().backward()
        fn = lambda image: bicubic_crop(image, jnp.array([1]), jnp.array([2]), jnp.array([5]), jnp.array([7]), 6)
        actual = fn(jnp.asarray(values))
        np.testing.assert_allclose(actual, expected.detach().numpy().transpose(0, 2, 3, 1), atol=2e-6, rtol=4e-6)
        actual_grad = jax.grad(lambda image: jnp.mean(fn(image)**2))(jnp.asarray(values))
        np.testing.assert_allclose(actual_grad, image_t.grad.numpy().transpose(0, 2, 3, 1), atol=2e-6, rtol=5e-6)
        left, right = paired_random_crop(jax.random.PRNGKey(1), jnp.asarray(values), jnp.asarray(values), 6)
        np.testing.assert_array_equal(left, right)

    def test_tiny_convnext_against_huggingface_pooled_features(self):
        # Random tiny weights test operators only, never substitute production assets.
        import torch
        from transformers import ConvNextV2Config, ConvNextV2Model
        torch.set_num_threads(1)
        torch.manual_seed(0)
        config = ConvNextV2Config(depths=[1, 1], hidden_sizes=[4, 8], num_stages=2,
                                 patch_size=4, image_size=32, drop_path_rate=0.)
        model = ConvNextV2Model(config).eval()
        params = {}
        for key, tensor in model.state_dict().items():
            value = tensor.detach().numpy()
            if value.ndim == 4 and ".grn." not in key:
                value = value.transpose(2, 3, 1, 0)
            elif value.ndim == 2:
                value = value.T
            params["cn." + key] = jnp.asarray(value)
        values = np.linspace(-1, 1, 32 * 32 * 3, dtype=np.float32).reshape(1, 32, 32, 3)
        inputs = torch.tensor(values.transpose(0, 3, 1, 2), requires_grad=True)
        expected = model(inputs).pooler_output
        expected.square().mean().backward()
        fn = lambda x: convnext_features(params, x, (1, 1), config.layer_norm_eps)
        actual = fn(jnp.asarray(values))
        np.testing.assert_allclose(actual, expected.detach().numpy(), atol=2e-5, rtol=2e-5)
        gradient = jax.grad(lambda x: jnp.mean(fn(x)**2))(jnp.asarray(values))
        np.testing.assert_allclose(gradient, inputs.grad.numpy().transpose(0, 2, 3, 1), atol=2e-5, rtol=2e-3)

    def test_tiny_lpips_against_official_feature_definition(self):
        import torch
        from torch import nn
        torch.set_num_threads(1)
        official = source_module("priming_official_perceptual", "models/encoders/perceptual.py")
        if official is None:
            self.skipTest("PRIMING_C2I_ROOT unset: direct official LPIPS source parity not run")
        torch.manual_seed(0)
        params = {}
        stages = nn.ModuleList()
        in_channels = 3
        for stage, indices in enumerate(VGG_LAYERS):
            layers = [nn.MaxPool2d(2)] if stage else []
            for index in indices:
                conv = nn.Conv2d(in_channels, 2, 3, padding=1)
                params[f"vgg.{index}.weight"] = jnp.asarray(conv.weight.detach().numpy().transpose(2, 3, 1, 0))
                params[f"vgg.{index}.bias"] = jnp.asarray(conv.bias.detach().numpy())
                layers.extend([conv, nn.ReLU()])
                in_channels = 2
            stages.append(nn.Sequential(*layers))
        class TinyVGG(nn.Module):
            def __init__(self):
                super().__init__()
                self.stages = stages
            def forward(self, x):
                outputs = []
                for stage in self.stages:
                    x = stage(x)
                    outputs.append(x)
                return outputs
        model = official.LPIPS.__new__(official.LPIPS)
        nn.Module.__init__(model)
        model.net = TinyVGG()
        model.lins = nn.ModuleList([official.NetLinLayer(2, use_dropout=False) for _ in range(5)])
        for index, layer in enumerate(model.lins):
            params[f"lpips.{index}"] = jnp.asarray(layer.model[0].weight.detach().numpy().reshape(-1))
        values = np.linspace(-1, 1, 32 * 32 * 3, dtype=np.float32).reshape(1, 32, 32, 3)
        targets = np.flip(values, axis=2).copy()
        inputs_t = torch.tensor(values.transpose(0, 3, 1, 2), requires_grad=True)
        target_t = torch.tensor(targets.transpose(0, 3, 1, 2))
        expected = model(inputs_t, target_t)
        expected.sum().backward()
        fn = lambda image: lpips_distance(params, image, jnp.asarray(targets))
        actual = fn(jnp.asarray(values))
        np.testing.assert_allclose(actual, expected.detach().numpy(), atol=2e-6, rtol=3e-5)
        gradient = jax.grad(lambda x: jnp.sum(fn(x)))(jnp.asarray(values))
        np.testing.assert_allclose(gradient, inputs_t.grad.numpy().transpose(0, 2, 3, 1), atol=3e-6, rtol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
