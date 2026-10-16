"""CPU-only Slurm topology regressions; never start distributed services."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from jax._src.clusters import ClusterEnv

from t2i.train import (distributed_configuration, initialize_distributed, main,
                       runtime_topology_gate)


def slurm_environment(gpus=8, processes=16, rank=7):
    return dict(JAX_PROCESS_COUNT=str(processes), JAX_PROCESS_INDEX=str(rank),
                JAX_COORDINATOR_ADDRESS='node0:12355', SLURM_JOB_ID='12345',
                SLURM_NNODES=str(processes), SLURM_NTASKS=str(processes),
                SLURM_PROCID=str(rank), SLURM_NODEID=str(rank), SLURM_LOCALID='0',
                STAGE1_GPUS_PER_NODE=str(gpus), SLURM_GPUS_ON_NODE=str(gpus),
                CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(gpus)))


def devices(processes=16, local=8, platform='gpu'):
    return [SimpleNamespace(id=rank * local + index, process_index=rank,
                            platform=platform, device_kind='NVIDIA H100')
            for rank in range(processes) for index in range(local)]


class DistributedTests(unittest.TestCase):
    def test_jax_0431_slurm_parser_and_explicit_all_devices(self):
        env = slurm_environment()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, env, clear=True):
            # This is the installed, pinned JAX 0.4.31 argument parser, with no
            # distributed service or backend initialization.
            legacy = ClusterEnv.auto_detect_unset_distributed_params(
                'node0:12355', 16, 7, None, None, 300)
            self.assertEqual(legacy, ('node0:12355', 16, 7, [0]))
            with mock.patch('t2i.train.jax.distributed.initialize') as initialize, \
                 mock.patch('t2i.train.jax.local_devices', side_effect=AssertionError('early device discovery')):
                configuration = initialize_distributed(Path(directory) / 'init.json')
            kwargs = initialize.call_args.kwargs
            self.assertEqual(kwargs, dict(coordinator_address='node0:12355',
                num_processes=16, process_id=7, local_device_ids=list(range(8))))
            explicit = ClusterEnv.auto_detect_unset_distributed_params(
                kwargs['coordinator_address'], kwargs['num_processes'], kwargs['process_id'],
                kwargs['local_device_ids'], None, 300)
            self.assertEqual(explicit, ('node0:12355', 16, 7, list(range(8))))
            self.assertEqual(configuration['expected_global_devices'], 128)
            self.assertTrue(json.loads((Path(directory) / 'init.json').read_text())['passed'])

    def test_visibility_mismatch_persists_and_never_initializes(self):
        env = slurm_environment()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, env, clear=True):
            target = Path(directory) / 'init.json'
            with mock.patch('t2i.train.jax.distributed.initialize') as initialize:
                with self.assertRaisesRegex(ValueError, 'visibility'):
                    initialize_distributed(target)
                initialize.assert_not_called()
            receipt = json.loads(target.read_text())
            self.assertFalse(receipt['passed'])
            self.assertEqual(receipt['expected_local_devices'], 8)
            self.assertEqual(receipt['visible_gpu_count'], 1)
            self.assertEqual(receipt['expected_processes'], 16)
            self.assertEqual(receipt['expected_global_devices'], 128)
            self.assertEqual(receipt['raw']['SLURM_LOCALID'], '0')

    def test_initialization_exception_persists(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.dict(os.environ, slurm_environment(), clear=True), \
             mock.patch('t2i.train.jax.distributed.initialize', side_effect=RuntimeError('communication')):
            target = Path(directory) / 'init.json'
            with self.assertRaisesRegex(RuntimeError, 'communication'):
                initialize_distributed(target)
            receipt = json.loads(target.read_text())
            self.assertEqual(receipt['phase'], 'initialize_failed')
            self.assertFalse(receipt['passed'])
            self.assertEqual(receipt['local_device_ids'], list(range(8)))
            self.assertEqual(receipt['expected_global_devices'], 128)

    def test_single_node_auto_and_four_gpu_slurm_smoke(self):
        auto = distributed_configuration({'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b'})
        self.assertTrue(auto['passed'], auto)
        self.assertEqual(auto['expected_local_devices'], 2)
        self.assertEqual(auto['local_device_ids'], [0, 1])
        env = slurm_environment(gpus=4, processes=1, rank=0)
        del env['STAGE1_GPUS_PER_NODE']
        smoke = distributed_configuration(env)
        self.assertTrue(smoke['passed'], smoke)
        self.assertEqual(smoke['expected_local_devices'], 4)
        self.assertEqual(smoke['expected_global_devices'], 4)

    def test_standalone_probe_without_jax_backend(self):
        with mock.patch('t2i.train.subprocess.run', return_value=SimpleNamespace(stdout='GPU-a\nGPU-b\n')) as query:
            configuration = distributed_configuration({})
        query.assert_called_once()
        self.assertTrue(configuration['passed'], configuration)
        self.assertEqual(configuration['local_device_ids'], [0, 1])

    def test_conflicting_or_missing_request_and_process_contract(self):
        for patch in ({'SLURM_GPUS_ON_NODE': '4'}, {'SLURM_NTASKS': '128'},
                      {'SLURM_LOCALID': '1'}, {'SLURM_NODEID': '8'},
                      {'CUDA_VISIBLE_DEVICES': '0,0'}, {'JAX_PROCESS_COUNT': 'bad'}):
            with self.subTest(patch=patch):
                env = slurm_environment(); env.update(patch)
                self.assertFalse(distributed_configuration(env)['passed'])
        env = slurm_environment(); del env['STAGE1_GPUS_PER_NODE']; del env['SLURM_GPUS_ON_NODE']
        self.assertFalse(distributed_configuration(env)['passed'])

    def test_actual_topology_matches_16_by_8(self):
        expected = distributed_configuration(slurm_environment())
        inventory = devices()
        with mock.patch.multiple('t2i.train.jax', process_count=mock.Mock(return_value=16),
                process_index=mock.Mock(return_value=7), local_device_count=mock.Mock(return_value=8),
                device_count=mock.Mock(return_value=128), devices=mock.Mock(return_value=inventory),
                local_devices=mock.Mock(return_value=inventory[56:64])):
            record = runtime_topology_gate(expected)
        self.assertTrue(record['passed'], record)
        self.assertEqual(record['actual_devices_per_process'], {rank: 8 for rank in range(16)})

    def test_local_global_process_and_remote_platform_failures_persist(self):
        expected = distributed_configuration(slurm_environment())
        for case, rank in (('local', 0), ('local', 7), ('global', 7),
                           ('processes', 7), ('remote-platform', 7)):
            with self.subTest(case=case, rank=rank), tempfile.TemporaryDirectory() as directory:
                expected = distributed_configuration(slurm_environment(rank=rank))
                inventory = devices()
                local = inventory[rank * 8:(rank + 1) * 8]
                if case == 'local': local = local[:1]
                if case == 'global': inventory = inventory[:-1]
                if case == 'remote-platform': inventory[0].platform = 'cpu'
                with mock.patch.multiple('t2i.train.jax',
                        process_count=mock.Mock(return_value=8 if case == 'processes' else 16),
                        process_index=mock.Mock(return_value=rank),
                        local_device_count=mock.Mock(return_value=len(local)),
                        device_count=mock.Mock(return_value=len(inventory)),
                        local_devices=mock.Mock(return_value=local), devices=mock.Mock(return_value=inventory)), \
                     mock.patch('t2i.train.initialize_distributed', return_value=expected), \
                     mock.patch('t2i.train.source_digest', return_value='source'), \
                     mock.patch('t2i.train.multihost_utils.sync_global_devices'), \
                     mock.patch('t2i.train.tempfile.mkdtemp', return_value=directory):
                    out = Path(directory) / 'run'
                    # Nonzero ranks normally observe rank zero's directory.
                    if rank != 0:
                        out.mkdir()
                    with self.assertRaisesRegex(ValueError, 'production configuration'):
                        main(['--output-dir', str(out), '--steps', '20', '--smoke'])
                record = json.loads((Path(directory) / f'preflight-rank{rank}.json').read_text())
                self.assertEqual(list(out.iterdir()), [])
                self.assertFalse(record['passed'])
                self.assertFalse(record['topology']['passed'])
                self.assertEqual(record['topology']['expected']['expected_local_devices'], 8)
                self.assertEqual(record['topology']['expected']['expected_global_devices'], 128)
                self.assertEqual(record['topology']['actual']['local_devices'], len(local))
                self.assertEqual(record['topology']['actual']['global_devices'], len(inventory))


if __name__ == '__main__':
    unittest.main()
