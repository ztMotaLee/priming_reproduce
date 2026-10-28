"""Small CPU D5 checks, including the real Runtime.update integration path."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax
from checkpoint import tree_digest
from diagnostics import (gradient_metric_function,update_diagnostics,dose_status,
                         load_dose_history,append_training_record,role_for_path)
from training import Runtime,optimizer,load_config,FIELD_WEIGHT


def toy_tree(scale=1.):
    return {'pt_net':{'img':jnp.asarray([.5,-.2,1.])*scale,
                      'txt_proj':jnp.asarray([-.3,.4])*scale,
                      'h_embedder':jnp.asarray([.1])*scale,
                      't_embedder':jnp.asarray([0.])*scale}}


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config()['dose_diagnostics']

    def summary(self,rows,step=2000):
        return dose_status(rows,step,rho0=self.cfg['rho0'],
                           calibration_receipt_sha256=self.cfg['calibration_receipt_sha256'])

    def test_role_precedence_and_all_gradient_metrics(self):
        self.assertEqual(role_for_path(('pt_net','t_embedder','txt_x')),'inactive')
        self.assertEqual(role_for_path(('pt_net','txt_h_embedder')),'text_interface')
        self.assertEqual(role_for_path(('pt_net','h_embedder')),'controls')
        self.assertEqual(role_for_path(('pt_net','img')),'synthesis')
        native,field=toy_tree(),toy_tree(-2.)
        mapped=lambda tree:jax.tree_util.tree_map(lambda v:v[None],tree)
        result=gradient_metric_function(jax,jnp)(mapped(native),mapped(field),FIELD_WEIGHT)
        result=jax.tree_util.tree_map(lambda v:np.asarray(v)[0],result)
        for role in ('all','synthesis','text_interface','controls'):
            self.assertAlmostEqual(float(result[role]['rho']),2*FIELD_WEIGHT,places=6)
            self.assertAlmostEqual(float(result[role]['gradient_cosine']),-1.,places=6)
            self.assertGreater(float(result[role]['native_norm']),0.)
            self.assertGreater(float(result[role]['field_norm']),0.)
        self.assertTrue(np.isnan(result['inactive']['rho']))
        self.assertTrue(result['clipped'])
        self.assertAlmostEqual(float(result['clip_factor']),1/float(result['all']['combined_norm']),places=6)

    def test_counterfactual_same_state_no_mutation_and_same_next_update(self):
        tx=optimizer();params=toy_tree();native=toy_tree(.3);field=toy_tree(-8.)
        state=tx.init(params)
        for _ in range(3):
            updates,state=tx.update(native,state,params);params=optax.apply_updates(params,updates)
        combined=jax.tree_util.tree_map(lambda a,b:a+FIELD_WEIGHT*b,native,field)
        before=(tree_digest(params),tree_digest(state))
        expected_updates,expected_state=tx.update(combined,state,params)
        expected_params=optax.apply_updates(params,expected_updates)
        probe=jax.jit(lambda p,s,g:tx.update(g,s,p)[0])
        metrics=update_diagnostics(jax,jax.device_get(probe(params,state,native)),
                                   jax.device_get(probe(params,state,combined)))
        self.assertGreater(metrics['relative_update_difference'],0.)
        self.assertEqual(before,(tree_digest(params),tree_digest(state)))
        actual_updates,actual_state=tx.update(combined,state,params)
        self.assertEqual(tree_digest(expected_state),tree_digest(actual_state))
        self.assertEqual(tree_digest(expected_params),tree_digest(optax.apply_updates(params,actual_updates)))

    def test_registered_window_strict_boundaries_missing_and_flags(self):
        rho0=self.cfg['rho0']
        rows={i:rho0 for i in range(1,2001)}
        for i in range(901,1001):rows[i]=rho0/10
        for i in range(1901,2001):rows[i]=10*rho0
        report=self.summary(rows)
        self.assertEqual(report['status'],'MAINTAINED')
        self.assertFalse(report['dose_unmaintained'])
        for i in range(901,1001):rows[i]=np.nextafter(rho0/10,-np.inf)
        self.assertEqual(self.summary(rows)['windows'][0]['flag'],'剂量塌缩')
        for i in range(1901,2001):rows[i]=np.nextafter(10*rho0,np.inf)
        self.assertEqual(self.summary(rows)['windows'][1]['flag'],'剂量主导')
        self.assertTrue(self.summary(rows)['dose_unmaintained'])
        complete={i:rho0 for i in range(1,1001)}
        first=self.summary(complete,1000)
        self.assertEqual(first['windows'][0]['status'],'COMPLETE')
        self.assertEqual(first['windows'][1]['status'],'PENDING')
        self.assertIsNone(first['dose_unmaintained'])
        self.assertEqual(self.summary(complete,2000)['status'],'INCOMPLETE')
        bad={i:rho0 for i in range(1,2001)};bad[901]='nan'
        self.assertEqual(self.summary(bad)['windows'][0]['observed_steps'],99)
        self.assertEqual(self.summary(bad)['status'],'INCOMPLETE')

    def test_resume_history_retains_replayed_attempt_and_ignores_unsaved_tail(self):
        with tempfile.TemporaryDirectory(prefix='dose-history-',dir=os.environ.get('TMPDIR')) as temp:
            path=Path(temp)/'training.jsonl'
            records=[dict(step=1,status='APPLIED',rho_t=.2),dict(step=1001,status='APPLIED',rho_t=.4),
                     dict(step=1001,status='APPLIED',rho_t=.3),dict(step=1002,status='READY',rho_t=.5)]
            path.write_text(''.join(json.dumps(row)+'\n' for row in records))
            self.assertEqual(load_dose_history(temp,1000),{1:.2})
            self.assertEqual(load_dose_history(temp,1001),{1:.2,1001:.3})
            self.assertEqual(len(path.read_text().splitlines()),4)

    def test_interrupted_log_tail_preserved_without_blocking_resume_or_next_append(self):
        with tempfile.TemporaryDirectory(prefix='dose-truncated-',dir=os.environ.get('TMPDIR')) as temp:
            path=Path(temp)/'training.jsonl'
            original=b'{"step":1,"status":"APPLIED","rho_t":0.2}\n{"step":901,"rho_t":'
            path.write_bytes(original)
            self.assertEqual(load_dose_history(temp,1000),{1:.2})
            self.assertEqual(self.summary(load_dose_history(temp,1000),1000)['windows'][0]['status'],'INCOMPLETE')
            append_training_record(path,dict(step=902,status='APPLIED',rho_t=.3))
            self.assertTrue(path.read_bytes().startswith(original))
            self.assertEqual(load_dose_history(temp,1000),{1:.2,902:.3})
            self.assertEqual(path.read_bytes()[len(original):len(original)+1],b'\n')

    def test_runtime_records_before_single_apply_and_only_step50_probes(self):
        tx=optimizer();params=toy_tree();state=tx.init(params)
        mapped=lambda tree:jax.tree_util.tree_map(lambda x:x[None],tree)
        with tempfile.TemporaryDirectory(prefix='dose-runtime-',dir=os.environ.get('TMPDIR')) as temp:
            rt=Runtime.__new__(Runtime)
            rt.jax,rt.jnp=jax,jnp;rt.rank=0;rt.out=Path(temp);rt.step=49;rt.seed=20260906
            rt.weight=FIELD_WEIGHT;rt.tx=tx;rt.geometry=SimpleNamespace(device_images=0)
            rt.chunk=4;rt.params=mapped(params);rt.opt_state=mapped(state)
            rt.diagnostics_config=self.cfg;rt.dose_history={};rt.config=load_config()
            rt.collect_batch=lambda:(None,None,None,None,None,None,np.arange(64))
            rt.feature_cotangents=lambda *a:((jnp.asarray(.5),{}),{})
            rt.zeros=lambda p:mapped(toy_tree(.3))
            rt.reduce=lambda v:v
            rt.field_gradient=lambda:(mapped(toy_tree(-8.)),dict(loss=.25))
            rt.gradient_metrics=gradient_metric_function(jax,jnp)
            rt.finite=jax.pmap(lambda tree:jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(tree)])))
            probes=[];applications=[]
            probe=jax.pmap(lambda p,s,g:tx.update(g,s,p)[0])
            def apply_core(p,s,g):
                updates,next_state=tx.update(g,s,p)
                return optax.apply_updates(p,updates),next_state
            plain_apply=jax.pmap(apply_core)
            def counterfactual(p,s,g):
                probes.append((tree_digest(rt.host(p)),tree_digest(rt.host(s))))
                return probe(p,s,g)
            rt.counterfactual_updates=counterfactual
            def apply(p,s,g):
                pending=json.loads((rt.out/'update_status.json').read_text())
                self.assertEqual(pending['status'],'READY')
                self.assertIn('gradient',pending)
                if rt.step==49:self.assertIn('counterfactual_actual_update',pending)
                applications.append(rt.step+1)
                return plain_apply(p,s,g)
            rt.apply=apply
            expected_params,expected_state=mapped(params),mapped(state)
            for step in (50,51):
                combined=jax.tree_util.tree_map(lambda a,b:a+FIELD_WEIGHT*b,toy_tree(.3),toy_tree(-8.))
                expected_params,expected_state=plain_apply(expected_params,expected_state,mapped(combined))
                record=rt.update()
                self.assertEqual('counterfactual_actual_update' in record,step==50)
                self.assertEqual(record['status'],'APPLIED')
                self.assertEqual(tree_digest(rt.host(rt.params)),tree_digest(rt.host(expected_params)))
                self.assertEqual(tree_digest(rt.host(rt.opt_state)),tree_digest(rt.host(expected_state)))
            self.assertEqual(applications,[50,51]);self.assertEqual(len(probes),2)
            self.assertEqual(probes[0],probes[1])
            self.assertEqual(len((Path(temp)/'training.jsonl').read_text().splitlines()),2)

    def test_save_commits_checkpoint_before_refreshing_derived_dose_report(self):
        tx=optimizer();params=toy_tree();state=tx.init(params)
        updates,state=tx.update(toy_tree(.2),state,params);params=optax.apply_updates(params,updates)
        with tempfile.TemporaryDirectory(prefix='dose-save-',dir=os.environ.get('TMPDIR')) as temp:
            rt=Runtime.__new__(Runtime)
            rt.jax=jax;rt.rank=0;rt.out=Path(temp);rt.step=1;rt.seed=20260906;rt.tx=tx
            rt.params=jax.tree_util.tree_map(lambda x:x[None],params)
            rt.opt_state=jax.tree_util.tree_map(lambda x:x[None],state)
            rt.rng=np.asarray(jax.random.fold_in(jax.random.PRNGKey(rt.seed),1),np.uint32)
            rt.binding={'toy':'save-diagnostics'};rt.diagnostics_config=self.cfg
            rt.config={'permanent_checkpoint_steps':[1]};rt.dose_history={1:.2}
            rt.save()
            report_path=rt.out/'readouts/step_000001.dose.json'
            first=json.loads(report_path.read_text())
            self.assertEqual(first['status'],'PENDING')
            self.assertEqual(len(first['checkpoint_sha256']),64)
            # A stale derived report must not prevent verification/reuse of the
            # already committed checkpoint, including manual recovery.
            report_path.write_text('{"status":"stale derived report"}')
            rt.save()
            self.assertEqual(first,json.loads(report_path.read_text()))
            self.assertTrue((rt.out/'checkpoints/step_000001/checkpoint.json').is_file())


if __name__=='__main__':unittest.main()
