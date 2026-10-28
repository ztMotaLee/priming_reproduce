"""Whole-caption process slices; globally normalized replay uses psum once."""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Geometry:
    nodes:int
    rank:int
    local_devices:int
    groups:int=64
    rows:int=16
    def __post_init__(self):
        if (self.nodes<1 or self.local_devices<1 or not 0<=self.rank<self.nodes
                or self.groups!=64 or self.rows!=16
                or 64%(self.nodes*self.local_devices)):
            raise ValueError('Total GPU count must divide 64; whole caption groups stay intact')
    @property
    def process_groups(self):return self.groups//self.nodes
    @property
    def process_images(self):return self.process_groups*self.rows
    @property
    def device_images(self):return self.process_images//self.local_devices
    @property
    def group_slice(self):return slice(self.rank*self.process_groups,(self.rank+1)*self.process_groups)
    @property
    def image_slice(self):return slice(self.rank*self.process_images,(self.rank+1)*self.process_images)
    def local_groups(self,value):
        value=np.asarray(value)
        if value.shape[0]!=64:raise ValueError('Global group axis must be 64')
        return value[self.group_slice]
    def mapped_images(self,value):
        value=np.asarray(value)
        if value.shape[0]!=1024:raise ValueError('Global image axis must be 1024')
        return value[self.image_slice].reshape((self.local_devices,self.device_images)+value.shape[1:])

def initialize(nodes,gpus_per_node,coordinator=None):
    import jax
    rank=int(os.environ.get('SLURM_PROCID',os.environ.get('JAX_PROCESS_INDEX','0')))
    geometry=Geometry(nodes,rank,gpus_per_node)
    if int(os.environ.get('SLURM_NTASKS',str(nodes)))!=nodes:
        raise ValueError('Use one process per node exposing all requested local GPUs')
    if nodes>1:
        address=coordinator or os.environ.get('JAX_COORDINATOR_ADDRESS')
        if not address or ':' not in address:raise ValueError('Set coordinator node-0 host:port')
        jax.distributed.initialize(coordinator_address=address,num_processes=nodes,
                    process_id=rank,local_device_ids=list(range(gpus_per_node)))
    if jax.process_count()!=nodes or jax.process_index()!=rank:
        raise ValueError('JAX process geometry differs')
    devices=jax.local_devices()
    if len(devices)!=gpus_per_node or any(d.platform!='gpu' or 'H100' not in d.device_kind for d in devices):
        raise ValueError('Real training requires the requested allocated H100 devices')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Launch real training inside an allocated Slurm job')
    return geometry

def gather_groups(local,geometry):
    local=np.asarray(local)
    if local.shape[0]!=geometry.process_groups:raise ValueError('Local group axis differs')
    if geometry.nodes==1:return local
    from jax.experimental import multihost_utils
    value=np.asarray(multihost_utils.process_allgather(local,tiled=True))
    if value.shape!=(64,)+local.shape[1:]:raise ValueError('Gathered group shape differs')
    return value

def barrier(label):
    import jax
    if jax.process_count()>1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(label)

def sum_replay_gradients(jax):
    return jax.pmap(lambda g:jax.lax.psum(g,'global_devices'),axis_name='global_devices')

def world_any(value):
    import jax
    value=np.asarray([bool(value)],np.int32)
    if jax.process_count()>1:
        from jax.experimental import multihost_utils
        value=np.asarray(multihost_utils.process_allgather(value,tiled=True))
    return bool(value.max())
