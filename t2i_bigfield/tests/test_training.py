"""Small CPU checks; run from this directory's parent without loading weights."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import json
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from checkpoint import save_checkpoint,restore_checkpoint,load_params,segment_report_path
from distributed import Geometry
from field import partitioned_objective,field_keys
from objective import grouped_stage2_loss
from training import optimizer,image_keys,load_config

class TrainingTests(unittest.TestCase):
    def test_checkpoint_restores_exact_next_update(self):
        tx=optimizer();params={'pt_net':{'w':jnp.asarray([1.,-2.],jnp.float32)}}
        gradient=jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*.25,params)
        state=tx.init(params)
        updates,state=tx.update(gradient,state,params);params=optax.apply_updates(params,updates)
        rng=np.asarray(jax.random.fold_in(jax.random.PRNGKey(20260906),1),np.uint32)
        with tempfile.TemporaryDirectory(prefix='checkpoint_',dir=os.environ.get('TMPDIR')) as root:
            first=segment_report_path(root,1,1000)
            first.write_text('{"status":"FAILED"}')
            second=segment_report_path(root,1,1000)
            self.assertNotEqual(first,second)
            self.assertEqual(first.read_text(),'{"status":"FAILED"}')
            self.assertFalse(second.exists())
            path=Path(root)/'step_000001'
            save_checkpoint(path,params=params,opt_state=state,step=1,rng=rng,training_seed=20260906,binding={'test':1})
            payload,_=restore_checkpoint(path,tx=tx,binding={'test':1},expected_step=1)
            for a,b in zip(jax.tree_util.tree_leaves(params),jax.tree_util.tree_leaves(load_params(path))):
                np.testing.assert_array_equal(a,b)
            expected,_=tx.update(gradient,state,params)
            actual,_=tx.update(gradient,payload['opt_state'],payload['params'])
            for a,b in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual)):
                np.testing.assert_array_equal(a,b)
            with self.assertRaises(ValueError):restore_checkpoint(path,tx=tx,binding={'test':2})
            raw=bytearray((path/'state.msgpack').read_bytes());raw[-1]^=1;(path/'state.msgpack').write_bytes(raw)
            with self.assertRaises(ValueError):restore_checkpoint(path,tx=tx,binding={'test':1})

    def test_fixed_geometry_stream_reassembly(self):
        stream=image_keys(3)
        for nodes,gpus in ((1,1),(1,4),(2,4),(4,4),(8,8)):
            parts=[Geometry(nodes,rank,gpus).mapped_images(stream).reshape(-1,2) for rank in range(nodes)]
            np.testing.assert_array_equal(np.concatenate(parts),stream)
        with self.assertRaises(ValueError):Geometry(3,0,4)
        self.assertFalse(np.array_equal(stream,image_keys(3,reference=True)))
        self.assertFalse(np.array_equal(stream,field_keys(3)))

    def test_stopped_support_gradients(self):
        rng=np.random.default_rng(9)
        fake=jnp.asarray(rng.normal(size=(2,3,5,4)),jnp.float32)
        real=fake+.2;bank=jnp.zeros((2,2,5,4));weights=jnp.asarray([[1,0],[1,1]],jnp.float32)
        (loss,_),grad=jax.value_and_grad(partitioned_objective,has_aux=True)(fake,real,bank,weights)
        self.assertTrue(np.isfinite(loss));self.assertTrue(np.isfinite(grad).all())
        stopped=jax.grad(lambda r:partitioned_objective(fake,r,bank,weights)[0])(real)
        np.testing.assert_array_equal(stopped,np.zeros_like(real))
        features={name:jnp.asarray(rng.normal(size=(2,3,4)),jnp.float32) for name in ('siglip','mae','inception')}
        target={name:value[:,:1] for name,value in features.items()}
        refgrad=jax.grad(lambda ref:grouped_stage2_loss(features,ref,target)[0])(features)
        for value in refgrad.values():np.testing.assert_array_equal(value,np.zeros_like(value))

    def test_config_and_native_import_namespace(self):
        cfg=load_config();self.assertEqual(cfg['field']['weight'],.07498942093324559)
        from bigfield_native.configs.load_config import get_config
        from bigfield_native.models.mean_flow_mmjit import get_mean_flow_teacher_mmjit_model_cls
        from bigfield_native.models.t5_encoder import T5Config
        native=get_config('eval_distill_b16')
        self.assertEqual(native.model.cls,'MMJiT_B_16_txt2')
        self.assertTrue(callable(get_mean_flow_teacher_mmjit_model_cls))
        self.assertEqual(T5Config(d_model=1024).d_model,1024)

if __name__=='__main__':unittest.main()
