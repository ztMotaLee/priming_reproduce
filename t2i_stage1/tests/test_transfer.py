"""Tiny CPU tests for strict Stage-1 -> downstream backbone initialization."""
import copy
import tempfile
import unittest

from flax.core import freeze, FrozenDict
import jax.numpy as jnp
import numpy as np

from t2i.checkpoint import claim_run, save_checkpoint, sha256
from t2i.config import Recipe
from t2i.train import create_state
from t2i.transfer import initialize_posttrain_params


class TransferTests(unittest.TestCase):
    def source(self, root):
        params = {"img_embedder": {"kernel": jnp.arange(6, dtype=jnp.float32).reshape(2, 3)},
                  "final_layer": {"bias": jnp.array([.1, .2, .3])}}
        state = create_state(params, Recipe())
        state = state.replace(ema_params={"img_embedder": {"kernel": params["img_embedder"]["kernel"] + 10},
                                          "final_layer": {"bias": params["final_layer"]["bias"] + 10}})
        binding = {"test": "transfer"}
        run, _ = claim_run(root, binding)
        return params, save_checkpoint(root, state, {**binding, "run_identity": run})

    def target(self, params):
        return {key: {name: jnp.zeros_like(value) for name, value in sub.items()}
                for key, sub in params.items()}

    def test_exact_bare_and_wrapper_online_ema(self):
        with tempfile.TemporaryDirectory() as root:
            params, path = self.source(root)
            bare, receipt = initialize_posttrain_params(self.target(params), path, branch=None)
            np.testing.assert_array_equal(bare["img_embedder"]["kernel"], params["img_embedder"]["kernel"])
            self.assertEqual(receipt["transferred_numel"], 9)
            self.assertEqual(receipt["parent_receipt_sha256"], sha256(path / "receipt.json"))
            self.assertEqual(receipt["parent_manifest_sha256"], sha256(path / "manifest.json"))
            self.assertEqual(receipt["kept_initialized_keys"], [])
            target = freeze({"net": self.target(params)})
            result, ema_receipt = initialize_posttrain_params(target, path, weights="ema")
            self.assertIsInstance(result, FrozenDict)
            np.testing.assert_array_equal(result["net"]["final_layer"]["bias"], params["final_layer"]["bias"] + 10)
            np.testing.assert_array_equal(target["net"]["final_layer"]["bias"], np.zeros(3))
            self.assertEqual(ema_receipt["weights"], "ema")
            self.assertFalse(ema_receipt["optimizer_restored"])

    def test_shape_missing_and_unknown_extra_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            params, path = self.source(root)
            wrong = self.target(params)
            wrong["img_embedder"]["kernel"] = jnp.zeros((3, 2))
            with self.assertRaisesRegex(ValueError, "shape"):
                initialize_posttrain_params(wrong, path, branch=None)
            missing = self.target(params)
            del missing["final_layer"]
            with self.assertRaisesRegex(ValueError, "missing"):
                initialize_posttrain_params(missing, path, branch=None)
            extra = self.target(params)
            extra["unrecognized_module"] = {"kernel": jnp.ones(1)}
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                initialize_posttrain_params(extra, path, branch=None, allow_meanflow_aux=True)
            with self.assertRaisesRegex(ValueError, "branch"):
                initialize_posttrain_params(self.target(params), path)

    def test_meanflow_auxiliary_retention_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as root:
            params, path = self.source(root)
            target = {"net": self.target(params), "pt_net": {"frozen": jnp.array([7.])}}
            target["net"]["time_tokens"] = {"param_tensor": jnp.array([8.])}
            target["net"]["h_embedder"] = {"kernel": jnp.array([9.])}
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                initialize_posttrain_params(target, path)
            result, receipt = initialize_posttrain_params(target, path, allow_meanflow_aux=True)
            np.testing.assert_array_equal(result["net"]["time_tokens"]["param_tensor"], [8.])
            np.testing.assert_array_equal(result["pt_net"]["frozen"], [7.])
            self.assertEqual(receipt["untouched_wrapper_branches"], ["pt_net"])
            self.assertEqual(set(receipt["kept_initialized_keys"]),
                             {"net/time_tokens/param_tensor", "net/h_embedder/kernel", "pt_net/frozen"})
            self.assertFalse(receipt["stage1_function_equivalent"])

    def test_payload_tampering_rejected_before_transfer(self):
        with tempfile.TemporaryDirectory() as root:
            params, path = self.source(root)
            payload = path / "minit2i" / "checkpoint_0"
            payload.write_bytes(payload.read_bytes() + b"broken")
            with self.assertRaisesRegex(ValueError, "payload differs"):
                initialize_posttrain_params(self.target(params), path, branch=None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
