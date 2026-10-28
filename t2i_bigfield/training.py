"""Memory-bounded Stage-2 adaptation plus a same-parameter pseudo-class field."""
from __future__ import annotations
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from checkpoint import (atomic_json,numeric,sha256_file,tree_digest,save_checkpoint,
                        restore_checkpoint)
from distributed import gather_groups,barrier,sum_replay_gradients
from diagnostics import (gradient_metric_function,update_diagnostics,load_dose_history,
                         append_training_record,dose_status)

ROOT=Path(__file__).resolve().parent
SEED=20260906
FIELD_WEIGHT=0.07498942093324559


def load_config(path=None):
    import yaml
    path=Path(path or ROOT/'configs/bigfield_tf.yaml')
    config=yaml.safe_load(path.read_text())
    reference=yaml.safe_load((ROOT/'configs/bigfield_tf.yaml').read_text())
    if sha256_file(ROOT/'configs/bigfield_tf.yaml')!='32957ee5370318dacb7faee78799dba9a931e8cfec8a50fe8b62fd896062bd22' or config!=reference:

        raise ValueError('Scientific configuration changes require a separate experiment')
    return config


def optimizer():
    import optax
    cfg=load_config()['optimizer']
    return optax.chain(optax.clip_by_global_norm(cfg['clip_global_norm']),optax.adamw(
        cfg['learning_rate'],b1=cfg['b1'],b2=cfg['b2'],eps=cfg['eps'],weight_decay=cfg['weight_decay']))


def image_keys(step,*,reference=False,seed=SEED):
    import jax
    base=jax.random.fold_in(jax.random.PRNGKey(seed),int(step))
    return np.asarray([jax.random.fold_in(base,i+1024*int(reference)) for i in range(1024)],np.uint32)


def make_models(native_root=None):
    """One official x0 teacher forward: split-key Gaussian, tau zero, no CFG."""
    import jax
    import jax.numpy as jnp
    from bigfield_native.configs.load_config import get_config
    from bigfield_native.models.mean_flow_mmjit import get_mean_flow_teacher_mmjit_model_cls
    if native_root and Path(native_root).resolve()!=ROOT/'bigfield_native':
        raise ValueError('Use the bundled audited native model source')
    cfg=get_config('eval_distill_b16');cfg.dataset.image_size=512;cfg.eval.num_steps=1
    mc=cfg.model.to_dict();name=mc.pop('cls');mc.pop('gradient_checkpointing',None)
    cls=get_mean_flow_teacher_mmjit_model_cls(name,cfg.dataset.llm,img_size=512)
    model=cls(**mc)
    def sample(params,text,mask,key):
        _,noise_key=jax.random.split(key)
        z=2.*jax.random.normal(noise_key,(1,512,512,3),dtype=jnp.float32)
        return model.apply({'params':params['pt_net']},z,jnp.zeros((1,),jnp.float32),text[None],mask[None])[0]
    return sample


def text_encoder(weights_root,native_root=None):
    import jax
    from jax.sharding import Mesh,PartitionSpec
    from jax.experimental.pjit import pjit
    from transformers import AutoTokenizer
    from bigfield_native.models.t5_encoder import create_t5_encode_fn,T5Config
    path=Path(weights_root)/'t5'
    verify_weights(weights_root, ('t5',))
    mesh=Mesh(np.asarray([jax.local_devices()[0]],object).reshape(1,1),('data','model'))
    specs=lambda tree,param_mode:jax.tree_util.tree_map(lambda _:PartitionSpec(),tree)
    def compiler(fn,in_shardings,out_shardings):
        compiled=pjit(fn,in_shardings=in_shardings,out_shardings=out_shardings)
        def run(*args):
            with mesh:return compiled(*args)
        return run
    native_config=dict(d_ff=2816,d_kv=64,d_model=1024,dropout_rate=.1,is_gated_act=True,
                       layer_norm_epsilon=1e-6,num_heads=16,num_layers=24,vocab_size=32128)
    fn,model,p=create_t5_encode_fn(str(path),256,(mesh,specs,lambda x:x,lambda x:x,compiler),T5Config(**native_config))
    return AutoTokenizer.from_pretrained(str(path),local_files_only=True),fn,p


def verify_weights(weights_root,names):
    from data import public_pins
    pins=public_pins()
    root=Path(weights_root)
    for name in names:
        for item in pins['weights'][name]['files']:
            path=root/name/item['name']
            if path.stat().st_size!=item['size'] or sha256_file(path)!=item['sha256']:
                raise ValueError('Weight inventory SHA differs: '+str(path))
    return {name:pins['weights'][name] for name in names}


def load_initial(weights_root,expected_sha):
    from flax.training import checkpoints
    import jax
    verify_weights(weights_root,('initial',))
    raw=checkpoints.restore_checkpoint(str(Path(weights_root)/'initial/checkpoint'),target=None)
    params={'pt_net':raw['params']['pt_net']};del raw;gc.collect()
    if len(jax.tree_util.tree_leaves(params['pt_net']))!=375:
        raise ValueError('Official generator branch leaf count differs')
    actual=tree_digest(params)
    if actual!=expected_sha:raise ValueError('Official initial parameter SHA differs: '+actual)
    return params


def release_tree(jax,tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf,'delete'):leaf.delete()


class Runtime:
    def __init__(self,config,data_root,weights_root,out,geometry,*,resume=None,native_root=None):
        import jax
        import jax.numpy as jnp
        import optax
        from data import load_dataset,load_real,Partition,sample_caption_indices
        from encoders import FrozenEncoders,FrozenDINO
        from objective import grouped_stage2_loss
        from field import partitioned_objective
        self.jax,self.jnp=jax,jnp
        self.config,self.geometry,self.out=config,geometry,Path(out)
        self.rank=geometry.rank;self.devices=jax.local_devices()
        self.data_root,self.weights_root=Path(data_root),Path(weights_root)
        self.out.mkdir(parents=True,exist_ok=True)
        self.rows,self.data_binding=load_dataset(data_root)
        self.partition=Partition(data_root,self.rows,verify_bank=True)
        self.load_real=lambda ids:load_real(self.rows,ids,data_root)
        self.sample_caption_indices=sample_caption_indices
        self.seed=SEED;self.chunk=4;self.weight=FIELD_WEIGHT
        self.diagnostics_config=config['dose_diagnostics']
        self.tx=optimizer()
        weight_binding=verify_weights(weights_root,('initial','t5','siglip','mae','inception','dino'))
        self.binding=dict(science=config,data=self.data_binding,field_assets=self.partition.identity,weights=weight_binding,
                          initial_parameter_sha256=config['initial_parameter_sha256'])
        if resume:
            payload,_=restore_checkpoint(resume,tx=self.tx,binding=self.binding)
            params,state=payload['params'],payload['opt_state']
            self.step=int(payload['step']);self.rng=payload['rng']
        else:
            params=load_initial(weights_root,config['initial_parameter_sha256'])
            state=self.tx.init(params);self.step=0;self.rng=np.asarray(jax.random.PRNGKey(self.seed),np.uint32)
        self.dose_history=load_dose_history(self.out,self.step)
        self.rep=lambda x:jax.device_put_replicated(jax.device_get(x),self.devices)
        self.params,self.opt_state=self.rep(params),self.rep(state)
        del params,state;gc.collect()
        self.sample_one=make_models(native_root)
        self.tokenizer,self.encode,self.text_params=text_encoder(weights_root,native_root)
        self.encoders=FrozenEncoders(weights_root)
        self.dino=FrozenDINO(weights_root)
        self.dino_params=None
        self.sample=jax.pmap(jax.vmap(self.sample_one,in_axes=(None,0,0,0)))
        def features(p,t,m,k):
            images=jax.vmap(self.sample_one,in_axes=(None,0,0,0))(p,t,m,k)
            return self.encoders.branch_features(images,precision='default')
        self.features=jax.pmap(features)
        self.real_features=jax.pmap(self.encoders.branch_features)
        self.feature_cotangents=jax.jit(jax.value_and_grad(grouped_stage2_loss,has_aux=True))
        self.field_cotangents=jax.jit(jax.value_and_grad(partitioned_objective,has_aux=True))
        def replay(acc,p,t,m,k,q):
            _,pull=jax.vjp(lambda weights:features(weights,t,m,k),p)
            return jax.tree_util.tree_map(jnp.add,acc,pull(q)[0])
        def field_replay(acc,p,t,m,k,image_q):
            _,pull=jax.vjp(lambda weights:jax.vmap(self.sample_one,in_axes=(None,0,0,0))(weights,t,m,k),p)
            return jax.tree_util.tree_map(jnp.add,acc,pull(image_q)[0])
        self.replay=jax.pmap(replay,donate_argnums=(0,))
        self.field_replay=jax.pmap(field_replay,donate_argnums=(0,))
        self.dino_encode=jax.pmap(self.dino.pooled_with_params)
        self.dino_image_q=jax.pmap(jax.grad(lambda p,x,q:jnp.sum(self.dino.pooled_with_params(p,x)*jax.lax.stop_gradient(q)),argnums=1))
        self.zeros=jax.pmap(lambda p:jax.tree_util.tree_map(jnp.zeros_like,p))
        self.finite=jax.pmap(lambda tree:jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)])))
        self.reduce=sum_replay_gradients(jax)
        self.norm=jax.pmap(lambda tree:jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in jax.tree_util.tree_leaves(tree))))
        self.gradient_metrics=gradient_metric_function(jax,jnp)
        # No donation, parameter application or state assignment in these probes.
        self.counterfactual_updates=jax.pmap(lambda p,s,g:self.tx.update(g,s,p)[0])
        def apply(p,s,g):
            updates,new_state=self.tx.update(g,s,p)
            return optax.apply_updates(p,updates),new_state
        self.apply=jax.pmap(apply,donate_argnums=(0,1))
        barrier('runtime_ready')

    def text(self,captions):
        results=[];masks=[]
        for start in range(0,len(captions),4):
            tokens=self.tokenizer(captions[start:start+4],max_length=256,padding='max_length',truncation=True,return_tensors='np')
            mask=np.asarray(tokens['attention_mask'],np.int32)
            result=self.encode(self.text_params,self.jnp.asarray(tokens['input_ids'],self.jnp.int32),self.jnp.asarray(mask))
            results.append(np.asarray(result,np.float32));masks.append(mask)
        return np.concatenate(results),np.concatenate(masks)

    def host(self,tree):
        return self.jax.device_get(self.jax.tree_util.tree_map(lambda x:x[0],tree))

    def collect_batch(self):
        geometry,jax=self.geometry,self.jax
        ids=self.sample_caption_indices(self.rows,self.step,self.seed)
        local_ids=ids[geometry.group_slice]
        text,mask=self.text([self.rows[int(i)]['caption'] for i in local_ids])
        text=np.repeat(text,16,axis=0).reshape(geometry.local_devices,geometry.device_images,256,-1)
        mask=np.repeat(mask,16,axis=0).reshape(geometry.local_devices,geometry.device_images,256)
        live_keys=geometry.mapped_images(image_keys(self.step,seed=self.seed))
        ref_keys=geometry.mapped_images(image_keys(self.step,reference=True,seed=self.seed))
        def collect(keys):
            parts={}
            for start in range(0,geometry.device_images,self.chunk):
                result=jax.device_get(self.features(self.params,text[:,start:start+self.chunk],mask[:,start:start+self.chunk],keys[:,start:start+self.chunk]))
                for name,value in result.items():parts.setdefault(name,[]).append(value)
            return {name:gather_groups(np.concatenate(values,axis=1).reshape(geometry.process_groups,16,-1),geometry) for name,values in parts.items()}
        live,reference=collect(live_keys),collect(ref_keys)
        real=self.load_real(local_ids).reshape(geometry.local_devices,geometry.process_groups//geometry.local_devices,512,512,3)
        parts={}
        for start in range(0,real.shape[1],self.chunk):
            result=jax.device_get(self.real_features(real[:,start:start+self.chunk]))
            for name,value in result.items():parts.setdefault(name,[]).append(value)
        real_features={name:gather_groups(np.concatenate(values,axis=1).reshape(geometry.process_groups,1,-1),geometry) for name,values in parts.items()}
        return text,mask,live_keys,live,reference,real_features,ids

    def field_gradient(self):
        from field import weighted_grouped_indices,bank_rows,field_keys,digest_array
        jax,g=self.jax,self.geometry
        ids,clusters,_=weighted_grouped_indices(self.partition,self.seed,self.step,1024,16,stream=21000)
        bank_ids,weights=bank_rows(self.partition,self.seed,self.step,clusters,ids,16,64,stream=22000)
        text,mask=self.text([self.rows[int(i)]['caption'] for i in ids[g.image_slice]])
        text=text.reshape(g.local_devices,g.device_images,256,-1);mask=mask.reshape(g.local_devices,g.device_images,256)
        keys=g.mapped_images(field_keys(self.step))
        self.dino_params=self.rep(self.dino.params)
        try:
            fake_parts=[];hashes=[]
            for start in range(0,g.device_images,self.chunk):
                images=np.asarray(jax.device_get(self.sample(self.params,text[:,start:start+self.chunk],mask[:,start:start+self.chunk],keys[:,start:start+self.chunk])),np.float32)
                hashes.append(digest_array(images))
                encoded=self.dino_encode(self.dino_params,images)
                fake_parts.append(np.asarray(jax.device_get(encoded)));release_tree(jax,encoded)
            fake=gather_groups(np.concatenate(fake_parts,axis=1).reshape(g.process_groups,16,5,768),g)
            current=self.partition.real_features[ids].reshape(64,16,5,768)
            bank=self.partition.real_features[np.maximum(bank_ids,0)]
            bank=np.where(weights[:,:,None,None]>0,bank,np.float32(0))
            (loss,metrics),q=self.field_cotangents(fake,current,bank,weights)
            q=g.local_groups(np.asarray(q,np.float32)).reshape(g.local_devices,g.device_images,5,768)
            gradient=self.zeros(self.params)
            for block,start in enumerate(range(0,g.device_images,self.chunk)):
                t,m,k=text[:,start:start+self.chunk],mask[:,start:start+self.chunk],keys[:,start:start+self.chunk]
                images=np.asarray(jax.device_get(self.sample(self.params,t,m,k)),np.float32)
                if digest_array(images)!=hashes[block]:raise ValueError('Field collection/replay image SHA differs')
                image_q=self.dino_image_q(self.dino_params,images,q[:,start:start+self.chunk])
                try:
                    gradient=self.field_replay(gradient,self.params,t,m,k,image_q)
                    jax.block_until_ready(gradient)
                finally:release_tree(jax,image_q)
            gradient=self.reduce(gradient);jax.block_until_ready(gradient)
            return gradient,dict(loss=float(loss),metrics=numeric(metrics),sampled_ids_sha256=digest_array(ids),
                                 bank_ids_sha256=digest_array(bank_ids),replay_images_match=True)
        finally:
            release_tree(jax,self.dino_params);self.dino_params=None

    def update(self):
        started=time.monotonic();jax,g=self.jax,self.geometry
        text,mask,keys,live,reference,real,ids=self.collect_batch()
        (loss,metrics),q=self.feature_cotangents(live,reference,real)
        q={name:g.local_groups(np.asarray(value)).reshape(g.local_devices,g.device_images,-1) for name,value in q.items()}
        gradient=self.zeros(self.params)
        for start in range(0,g.device_images,self.chunk):
            cotangent={name:value[:,start:start+self.chunk] for name,value in q.items()}
            gradient=self.replay(gradient,self.params,text[:,start:start+self.chunk],mask[:,start:start+self.chunk],keys[:,start:start+self.chunk],cotangent)
        gradient=self.reduce(gradient)
        field,field_metrics=self.field_gradient()
        combined=jax.tree_util.tree_map(lambda a,b:a+self.weight*b,gradient,field)
        diagnostics=self.host(self.gradient_metrics(gradient,field,self.weight))
        native_norm=float(diagnostics['all']['native_norm'])
        norm=float(diagnostics['all']['combined_norm'])
        finite=all(bool(np.asarray(self.finite(tree)).all()) for tree in (gradient,field,combined)) and all(math.isfinite(x) for x in (float(loss),field_metrics['loss'],norm,native_norm))
        record=dict(step=self.step+1,loss=float(loss),native_gradient_norm=native_norm,gradient_norm=norm,
                    field_weight=self.weight,field=field_metrics,metrics=numeric(metrics),
                    gradient=numeric(diagnostics),rho_t=numeric(diagnostics['all']['rho']),
                    batch_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),gradient_finite=finite,
                    status='READY' if finite else 'NONFINITE')
        if self.rank==0:atomic_json(self.out/'update_status.json',record)
        if not finite:raise ValueError('Nonfinite loss/native/field/combined gradient')
        if (self.step+1)%self.diagnostics_config['counterfactual_interval']==0:
            base_update=joint_update=None
            try:
                base_update=self.counterfactual_updates(self.params,self.opt_state,gradient)
                joint_update=self.counterfactual_updates(self.params,self.opt_state,combined)
                record['counterfactual_actual_update']=update_diagnostics(jax,self.host(base_update),self.host(joint_update))
                if self.rank==0:atomic_json(self.out/'update_status.json',record)
            finally:
                if base_update is not None:release_tree(jax,base_update)
                if joint_update is not None:release_tree(jax,joint_update)
        jax.block_until_ready(combined);release_tree(jax,gradient);release_tree(jax,field)
        self.params,self.opt_state=self.apply(self.params,self.opt_state,combined)
        jax.block_until_ready(self.params)
        self.step+=1;self.rng=np.asarray(jax.random.fold_in(jax.random.PRNGKey(self.seed),self.step),np.uint32)
        self.dose_history[self.step]=record['rho_t']
        record.update(status='APPLIED',seconds=time.monotonic()-started)
        if self.rank==0:
            append_training_record(self.out/'training.jsonl',numeric(record))
            atomic_json(self.out/'progress.json',record)
        return record

    @property
    def dose_status(self):
        cfg=self.diagnostics_config
        return dose_status(self.dose_history,self.step,rho0=cfg['rho0'],
                           calibration_receipt_sha256=cfg['calibration_receipt_sha256'])

    def save(self):
        barrier('checkpoint_before_'+str(self.step))
        if self.rank==0:
            directory=self.out/'checkpoints'/('step_%06d'%self.step)
            if directory.exists():
                _,meta=restore_checkpoint(directory,tx=self.tx,binding=self.binding,expected_step=self.step)
                if meta['parameter_sha256']!=tree_digest(self.host(self.params)):
                    raise ValueError('Existing checkpoint differs from current state')
            else:
                meta=save_checkpoint(directory,params=self.host(self.params),opt_state=self.host(self.opt_state),
                       step=self.step,rng=self.rng,training_seed=self.seed,binding=self.binding)
            atomic_json(self.out/'latest.json',dict(step=self.step,checkpoint=directory.name,
                       checkpoint_sha256=sha256_file(directory/'checkpoint.json')))
            summary=dict(self.dose_status,checkpoint_sha256=sha256_file(directory/'checkpoint.json'))
            atomic_json(self.out/'dose_summary.json',summary)
            if self.step in self.config['permanent_checkpoint_steps']:
                # Derived reports follow a verified committed checkpoint. Replay
                # can refresh diagnostics without turning dose into a new gate.
                atomic_json(self.out/'readouts'/('step_%06d.dose.json'%self.step),summary)
        barrier('checkpoint_after_'+str(self.step))
