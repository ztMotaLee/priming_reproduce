"""Small CPU tests: schedule, exact resume, export, and accumulated update."""
import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import serialization

jax.config.update('jax_default_matmul_precision', 'highest')

from t2i.checkpoint import (assert_tree_compatible, restore_checkpoint,
                            claim_run, save_checkpoint as _save_checkpoint, sha256, verify_checkpoint)
from t2i.config import Recipe, batch_layout
from t2i.train import create_state, directory_identity, make_update


def save_checkpoint(output, state, metadata):
    # The checkpoint fixture owns a real tiny run; production never saves an
    # unowned output or a checkpoint without immutable run provenance.
    run, _ = claim_run(output, metadata)
    return _save_checkpoint(output, state, {**metadata, 'run_identity': run})


class TinyPredictor(nn.Module):
    @nn.compact
    def __call__(self, images, context, masks):
        # Non-null caption gradient and genuine trainable pixel output.
        scale = self.param('scale', nn.initializers.ones, ())
        text_scale = self.param('text_scale', nn.initializers.ones, ())
        return images * scale + context.mean((1,2))[:,None,None,None] * text_scale


class TinyPerceptual:
    params = {'scale': jnp.asarray(1.)}
    def apply(self, params, pred, target, rng):
        delta = jnp.mean((pred-target)**2, axis=(1,2,3))
        return delta * params['scale'], delta * 2


class TrainTests(unittest.TestCase):
    def test_schedule(self):
        r = Recipe()
        self.assertEqual(r.stage(249999),'A'); self.assertEqual(r.stage(250000),'B')
        self.assertAlmostEqual(float(r.learning_rate(249999)),4e-4,places=9)
        self.assertAlmostEqual(float(r.learning_rate(250000)),4e-5,places=10)
        self.assertEqual(r.stage_a_steps+r.stage_b_steps,290000)

    def test_batch_layout(self):
        self.assertEqual(batch_layout(1024,16,8,1)['per_device'],8)
        self.assertEqual(batch_layout(1024,16,8,2)['accumulation_steps'],4)
        self.assertEqual(batch_layout(1024,1,4,1)['per_device'],256)
        for values in [(1024,3,8,1),(1024,16,8,3),(1024,0,8,1)]:
            with self.assertRaises(ValueError): batch_layout(*values)

    def test_exact_resume_and_export(self):
        r = Recipe(); params = {'weight': jnp.arange(6,dtype=jnp.float32).reshape(2,3)}
        original = create_state(params,r)
        grads = jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*.2,params)
        trained = original.apply_gradients(grads=grads)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(tmp,trained,{'binding':'unit'})
            restored, metadata = restore_checkpoint(path,create_state(params,r),{'binding':'unit'})
            self.assertEqual(int(restored.step),1)
            self.assertEqual(verify_checkpoint(path)['step'],1)
            for a,b in zip(jax.tree_util.tree_leaves(trained),jax.tree_util.tree_leaves(restored)):
                np.testing.assert_array_equal(a,b)
            left=trained.apply_gradients(grads=grads)
            right=restored.apply_gradients(grads=grads)
            for a,b in zip(jax.tree_util.tree_leaves(left),jax.tree_util.tree_leaves(right)):
                np.testing.assert_array_equal(a,b)
            exported=serialization.msgpack_restore((path/'minit2i'/'checkpoint_1').read_bytes())
            self.assertEqual(set(exported), {'params'})
            np.testing.assert_array_equal(exported['params']['net']['weight'],trained.params['weight'])
            from flax.training import checkpoints
            raw=checkpoints.restore_checkpoint(str(path/'minit2i'),target=None)
            np.testing.assert_array_equal(raw['params']['net']['weight'],trained.params['weight'])
            ema_raw=checkpoints.restore_checkpoint(str(path/'minit2i_ema'),target=None)
            np.testing.assert_array_equal(ema_raw['params']['net']['weight'],trained.ema_params['weight'])
            with self.assertRaises(ValueError): restore_checkpoint(path,original,{'binding':'wrong'})
            receipt = json.loads((path/'receipt.json').read_text()); receipt['step']=2
            (path/'receipt.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError): verify_checkpoint(path)

    def test_schedule_count_and_variable_step_dtype(self):
        params={'weight':jnp.ones((2,))}; original=create_state(params,Recipe())
        trained=original.apply_gradients(grads={'weight':jnp.ones((2,))})
        with tempfile.TemporaryDirectory() as tmp:
            variable=trained.replace(step=np.float32(1))
            path=save_checkpoint(tmp,variable,{'binding':'unit'})
            restored,_=restore_checkpoint(path,original,{'binding':'unit'})
            self.assertEqual(int(restored.step),1)
        with tempfile.TemporaryDirectory() as tmp:
            components=list(trained.opt_state)
            components[-1]=components[-1]._replace(count=jnp.int32(99))
            broken=trained.replace(opt_state=tuple(components))
            path=save_checkpoint(tmp,broken,{'binding':'unit'})
            audit=Path(tmp)/'restore-gate.json'
            with self.assertRaises(ValueError):
                restore_checkpoint(path,original,{'binding':'unit'},audit_path=audit)
            self.assertEqual(json.loads(audit.read_text())['values']['optimizer_count_2'],99)
            self.assertFalse(json.loads(audit.read_text())['passed'])

    def test_receipt_manifest_step_failures_are_audited(self):
        params={'weight':jnp.ones((2,))}; original=create_state(params,Recipe())
        trained=original.apply_gradients(grads={'weight':jnp.ones((2,))})
        cases=[('receipt',2,2,1), ('manifest',2,1,2),
               ('receipt',1.5,1.5,1), ('manifest',1.5,1,1.5),
               ('receipt',True,True,1), ('manifest','1',1,'1'),
               ('receipt','invalid','invalid',1), ('manifest',float('nan'),1,'nan'),
               ('receipt',float('inf'),'inf',1), ('manifest',None,1,None)]
        for document,step,expected,actual in cases:
            with self.subTest(document=document,step=step), tempfile.TemporaryDirectory() as tmp:
                path=save_checkpoint(tmp,trained,{'binding':'unit'})
                changed=path/(document+'.json')
                payload=json.loads(changed.read_text()); payload['step']=step
                changed.write_text(json.dumps(payload))
                if document=='manifest':
                    receipt=json.loads((path/'receipt.json').read_text())
                    receipt['manifest_sha256']=sha256(changed)
                    (path/'receipt.json').write_text(json.dumps(receipt))
                for operation in ('verify','restore'):
                    audit=Path(tmp)/(operation+'-gate.json')
                    with self.assertRaises((TypeError,ValueError,OverflowError)):
                        if operation=='verify':
                            verify_checkpoint(path,audit_path=audit)
                        else:
                            restore_checkpoint(path,original,{'binding':'unit'},audit_path=audit)
                    result=json.loads(audit.read_text(),
                        parse_constant=lambda value:self.fail('Nonfinite JSON token: '+value))
                    self.assertEqual(result['gate'],'strict_resume_step')
                    self.assertEqual(result['phase'],'receipt_manifest_step')
                    self.assertFalse(result['passed'])
                    self.assertEqual(result['values']['expected_step'],expected)
                    self.assertEqual(result['values']['actual_step'],actual)
                    self.assertEqual(result['values']['receipt_step'],expected)
                    self.assertEqual(result['values']['manifest_step'],actual)

    def test_missing_and_excess_optimizer_counters_are_audited(self):
        params={'weight':jnp.ones((2,))}; original=create_state(params,Recipe())
        trained=original.apply_gradients(grads={'weight':jnp.ones((2,))})
        for indices in ((1,), (0,1), (0,1,2,2)):
            with self.subTest(indices=indices), tempfile.TemporaryDirectory() as tmp:
                broken=trained.replace(opt_state=tuple(trained.opt_state[i] for i in indices))
                target=original.replace(opt_state=tuple(original.opt_state[i] for i in indices))
                path=save_checkpoint(tmp,broken,{'binding':'unit'})
                audit=Path(tmp)/'restore-gate.json'
                with self.assertRaisesRegex(ValueError,'Expected Adam and learning-rate schedule counters'):
                    restore_checkpoint(path,target,{'binding':'unit'},audit_path=audit)
                result=json.loads(audit.read_text())
                expected_counts={f'optimizer_count_{i}':1
                    for i,component in enumerate(broken.opt_state)
                    if 'count' in getattr(component,'_fields',())}
                self.assertEqual(result['phase'],'restored_state')
                self.assertFalse(result['passed'])
                self.assertEqual(result['values']['optimizer_count_count'],len(expected_counts))
                self.assertEqual(result['values']['expected_optimizer_count_count'],2)
                self.assertEqual(result['values']['expected_step'],1)
                self.assertEqual(result['values']['actual_step'],1)
                actual_counts={key:value for key,value in result['values'].items()
                    if key.startswith('optimizer_count_') and key.removeprefix('optimizer_count_').isdigit()}
                self.assertEqual(actual_counts,expected_counts)

    def test_fractional_or_malformed_metadata_step_is_audited(self):
        params={'weight':jnp.ones((2,))}; original=create_state(params,Recipe())
        trained=original.apply_gradients(grads={'weight':jnp.ones((2,))})
        cases=[(1.5,1.5), (True,True), ('1','1'), ([1],[1]),
               (float('nan'),'nan'), (float('inf'),'inf')]
        for step,expected in cases:
            with self.subTest(step=step), tempfile.TemporaryDirectory() as tmp:
                path=save_checkpoint(tmp,trained,{'binding':'unit'})
                metadata=json.loads((path/'metadata.json').read_text()); metadata['step']=step
                (path/'metadata.json').write_text(json.dumps(metadata))
                manifest=json.loads((path/'manifest.json').read_text())
                manifest['files']['metadata.json']=sha256(path/'metadata.json')
                (path/'manifest.json').write_text(json.dumps(manifest))
                receipt=json.loads((path/'receipt.json').read_text())
                receipt['manifest_sha256']=sha256(path/'manifest.json')
                (path/'receipt.json').write_text(json.dumps(receipt))
                audit=Path(tmp)/'restore-gate.json'
                with self.assertRaisesRegex(ValueError,'Restored step or Adam count differs from receipt'):
                    restore_checkpoint(path,original,{'binding':'unit'},audit_path=audit)
                result=json.loads(audit.read_text(),
                    parse_constant=lambda value:self.fail('Nonfinite JSON token: '+value))
                self.assertFalse(result['passed'])
                self.assertEqual(result['values']['metadata_step'],expected)
                self.assertEqual(result['values']['metadata_step_shape'],[1] if isinstance(step,list) else [])
                self.assertEqual(result['values']['expected_step'],1)
                self.assertEqual(result['values']['actual_step'],1)

    def test_optimizer_structure_mismatch_audits_raw_state_with_unchanged_target(self):
        params={'weight':jnp.ones((2,))}; original=create_state(params,Recipe())
        trained=original.apply_gradients(grads={'weight':jnp.ones((2,))})
        for indices in ((1,), (0,1), (0,1,2,2)):
            with self.subTest(indices=indices), tempfile.TemporaryDirectory() as tmp:
                broken=trained.replace(step=np.float32(1.5),
                    opt_state=tuple(trained.opt_state[i] for i in indices))
                path=save_checkpoint(tmp,broken,{'binding':'unit'})
                audit=Path(tmp)/'restore-gate.json'
                with self.assertRaisesRegex(ValueError,'size of the list and the state dict do not match'):
                    restore_checkpoint(path,original,{'binding':'unit'},audit_path=audit)
                result=json.loads(audit.read_text())
                self.assertEqual(result['phase'],'raw_state')
                self.assertFalse(result['passed'])
                self.assertEqual(result['values']['raw_step'],1.5)
                self.assertEqual(result['values']['expected_step'],1)
                counts={f'optimizer_count_{i}':1 for i,component in enumerate(broken.opt_state)
                    if 'count' in getattr(component,'_fields',())}
                self.assertEqual(result['values']['optimizer_count_count'],len(counts))
                for key,value in counts.items():
                    self.assertEqual(result['values'][key],value)
                self.assertEqual(set(result['values']['raw_optimizer_structure']),
                                 {str(i) for i in range(len(indices))})
                self.assertEqual(set(result['values']['target_optimizer_structure']),{'0','1','2'})

    def test_missing_optimizer_field_audits_raw_counters_before_restore(self):
        params={'weight':jnp.ones((2,))}; original=create_state(params,Recipe())
        trained=original.apply_gradients(grads={'weight':jnp.ones((2,))})
        with tempfile.TemporaryDirectory() as tmp:
            path=save_checkpoint(tmp,trained,{'binding':'unit'})
            raw=serialization.msgpack_restore((path/'state.msgpack').read_bytes())
            raw['opt_state']['0']['count']=np.float32(99)
            del raw['opt_state']['2']['count']
            (path/'state.msgpack').write_bytes(serialization.msgpack_serialize(raw))
            manifest=json.loads((path/'manifest.json').read_text())
            manifest['files']['state.msgpack']=sha256(path/'state.msgpack')
            (path/'manifest.json').write_text(json.dumps(manifest))
            receipt=json.loads((path/'receipt.json').read_text())
            receipt['manifest_sha256']=sha256(path/'manifest.json')
            (path/'receipt.json').write_text(json.dumps(receipt))
            audit=Path(tmp)/'restore-gate.json'
            with self.assertRaisesRegex(ValueError,'field names of the state dict and the named tuple do not match'):
                restore_checkpoint(path,original,{'binding':'unit'},audit_path=audit)
            result=json.loads(audit.read_text())
            self.assertEqual(result['phase'],'raw_state')
            self.assertFalse(result['passed'])
            self.assertEqual(result['values']['raw_step'],1)
            self.assertEqual(result['values']['optimizer_count_0'],99)
            self.assertEqual(result['values']['optimizer_count_count'],1)
            self.assertEqual(result['values']['raw_optimizer_structure']['2'],{})
            self.assertEqual(result['values']['target_optimizer_structure']['2']['count']['shape'],[])

    def test_integral_float_checkpoint_steps_remain_supported(self):
        params={'weight':jnp.ones((2,))}; original=create_state(params,Recipe())
        trained=original.apply_gradients(grads={'weight':jnp.ones((2,))}).replace(step=np.float32(1))
        with tempfile.TemporaryDirectory() as tmp:
            path=save_checkpoint(tmp,trained,{'binding':'unit'})
            metadata=json.loads((path/'metadata.json').read_text()); metadata['step']=1.0
            (path/'metadata.json').write_text(json.dumps(metadata))
            manifest=json.loads((path/'manifest.json').read_text()); manifest['step']=1.0
            manifest['files']['metadata.json']=sha256(path/'metadata.json')
            (path/'manifest.json').write_text(json.dumps(manifest))
            receipt=json.loads((path/'receipt.json').read_text()); receipt['step']=1.0
            receipt['manifest_sha256']=sha256(path/'manifest.json')
            (path/'receipt.json').write_text(json.dumps(receipt))
            audit=Path(tmp)/'restore-gate.json'
            restored,_=restore_checkpoint(path,original,{'binding':'unit'},audit_path=audit)
            self.assertEqual(int(restored.step),1)
            self.assertEqual(np.asarray(restored.step).dtype,np.dtype(np.int32))
            self.assertTrue(json.loads(audit.read_text())['passed'])

    def test_array_optimizer_counter_is_audited_before_shape_rejection(self):
        params={'weight':jnp.ones((2,))}; original=create_state(params,Recipe())
        trained=original.apply_gradients(grads={'weight':jnp.ones((2,))})
        for matching_target in (False,True):
            with self.subTest(matching_target=matching_target), tempfile.TemporaryDirectory() as tmp:
                components=list(trained.opt_state)
                components[-1]=components[-1]._replace(count=np.array([1,1],dtype=np.int32))
                broken=trained.replace(opt_state=tuple(components))
                target=original
                if matching_target:
                    target_components=list(original.opt_state)
                    target_components[-1]=target_components[-1]._replace(count=np.zeros(2,dtype=np.int32))
                    target=original.replace(opt_state=tuple(target_components))
                path=save_checkpoint(tmp,broken,{'binding':'unit'})
                audit=Path(tmp)/'restore-gate.json'
                message=('Restored step or Adam count differs from receipt' if matching_target
                         else 'Checkpoint state leaf shape differs')
                with self.assertRaisesRegex(ValueError,message):
                    restore_checkpoint(path,target,{'binding':'unit'},audit_path=audit)
                result=json.loads(audit.read_text())
                self.assertFalse(result['passed'])
                self.assertEqual(result['values']['optimizer_count_2'],[1,1])
                self.assertEqual(result['values']['optimizer_count_2_shape'],[2])
                self.assertEqual(result['values']['optimizer_count_count'],2)
                self.assertEqual(result['values']['expected_optimizer_count_count'],2)

    def test_nonfinite_optimizer_counter_is_audited_as_strict_json(self):
        params={'weight':jnp.ones((2,))}; original=create_state(params,Recipe())
        trained=original.apply_gradients(grads={'weight':jnp.ones((2,))})
        for count,label in ((float('nan'),'nan'), (float('inf'),'inf'), (-float('inf'),'-inf')):
            with self.subTest(count=label), tempfile.TemporaryDirectory() as tmp:
                components=list(trained.opt_state)
                components[-1]=components[-1]._replace(count=np.float32(count))
                broken=trained.replace(opt_state=tuple(components))
                target_components=list(original.opt_state)
                target_components[-1]=target_components[-1]._replace(count=np.float32(0))
                target=original.replace(opt_state=tuple(target_components))
                path=save_checkpoint(tmp,broken,{'binding':'unit'})
                audit=Path(tmp)/'restore-gate.json'
                with self.assertRaisesRegex(ValueError,'Restored step or Adam count differs from receipt'):
                    restore_checkpoint(path,target,{'binding':'unit'},audit_path=audit)
                result=json.loads(audit.read_text(),
                    parse_constant=lambda value:self.fail('Nonfinite JSON token: '+value))
                self.assertFalse(result['passed'])
                self.assertEqual(result['values']['optimizer_count_2'],label)
                self.assertEqual(result['values']['optimizer_count_count'],2)
                self.assertEqual(result['values']['expected_optimizer_count_count'],2)

    def test_wrong_shape_and_dtype(self):
        with self.assertRaises(ValueError):
            assert_tree_compatible({'x':np.zeros(2)},{'x':np.zeros(3)})
        with self.assertRaises(ValueError):
            assert_tree_compatible({'x':np.zeros(2,dtype=np.int32)},{'x':np.zeros(2)})

    def test_device_independent_directory_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(set(directory_identity(tmp)),{'path','inode','uid'})

    def test_accumulated_update_caption_and_resume(self):
        model=TinyPredictor(); perceptual=TinyPerceptual(); r=Recipe()
        images=jnp.zeros((1,4,4,3)); context=jnp.ones((1,4,2)); mask=jnp.ones((1,4))
        params=model.init(jax.random.PRNGKey(1),images,context,mask)['params']
        def encode(params,ids,masks): return ids[...,None]*jnp.ones((2,))*params['scale']
        update=make_update(model,encode,perceptual,r,'logit-normal',1)
        from flax import jax_utils
        state=jax_utils.replicate(create_state(params,r))
        batch=jnp.zeros((1,2,1,4,4,3)); ids=jnp.ones((1,2,1,4),jnp.int32)
        frozen=jax_utils.replicate({'scale':jnp.asarray(1.)})
        result,metrics=update(state,batch,ids,jnp.ones_like(ids),frozen,frozen)
        self.assertEqual(int(np.asarray(result.step)[0]),1)
        self.assertTrue(np.isfinite(float(np.asarray(metrics['loss'])[0])))
        self.assertNotEqual(float(np.asarray(result.params['text_scale'])[0]),float(params['text_scale']))

if __name__=='__main__': unittest.main()
