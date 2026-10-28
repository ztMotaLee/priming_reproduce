"""Small CPU checks; run from the experiment directory."""
import hashlib
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import numpy as np
import data
import prepare_data
TMP = Path(os.environ.get('TMPDIR', tempfile.gettempdir()))
class TestData(unittest.TestCase):
    def setUp(self):
        TMP.mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix='data-test-', dir=TMP))
    def test_published_population_partition_and_hash(self):
        pins = data.public_pins(); arrays = data.published_rows()
        self.assertEqual(arrays['cluster_ids'].shape, (110230,))
        self.assertEqual(arrays['cluster_ids'].dtype, np.int32)
        self.assertEqual(int(arrays['cluster_ids'].max()) + 1, 956)
        self.assertEqual(int(np.bincount(arrays['cluster_ids']).min()), 32)
        self.assertEqual({s: int(np.sum(arrays['source'] == i)) for i, s in enumerate(pins['sources'])}, {'blip3o':57486,'dalle3':11398,'sharegpt4o':41346})
        self.assertEqual(set(arrays), {'source','raw_file','raw_key','caption_sha256','image_sha256','t5_tokens','geneval_train','cluster_ids'})
    def test_caption_sampler_unique_and_stateless(self):
        rows = data.DatasetRows([dict(caption=f'{s}-{i}', source=s) for s in data.SOURCES for i in range(40)])
        result = data.sample_caption_indices(rows, 17)
        self.assertEqual(len(set(result.tolist())),64)
        np.testing.assert_array_equal(result,data.sample_caption_indices(rows,17))
        self.assertFalse(np.array_equal(result,data.sample_caption_indices(rows,18)))
    def test_writer_offsets_and_bound_payload(self):
        from PIL import Image
        raw = io.BytesIO(); Image.new('RGB',(17,23),(10,70,140)).save(raw,format='PNG')
        jpeg = prepare_data.jpeg512(raw.getvalue()); writer = prepare_data.ImageWriter(self.work)
        row = writer.add(jpeg); writer.close()
        with (self.work / row['shard']).open('rb') as stream:
            stream.seek(row['image_offset']); actual = stream.read(row['image_bytes'])
        self.assertEqual(actual,jpeg)
        self.assertEqual(hashlib.sha256(actual).hexdigest(),row['image_sha256'])
        loaded = data.load_real([row],[0],self.work)
        self.assertEqual(loaded.shape,(1,512,512,3)); self.assertEqual(loaded.dtype,np.float32)
        self.assertTrue(np.isfinite(loaded).all())
        row['image_sha256']='0'*64
        with self.assertRaises(ValueError): data.load_real([row],[0],self.work)
    def test_source_pair_key_and_caption_bytes(self):
        path=self.work/'pairs.tar'
        with tarfile.open(path,'w') as stream:
            for name,payload in [('37.jpg',b'image'),('37.txt','exact text\n'.encode())]:
                info=tarfile.TarInfo(name);info.size=len(payload);stream.addfile(info,io.BytesIO(payload))
        self.assertEqual(list(prepare_data.source_pairs('blip3o',path,self.work)),[('37',b'image','exact text\n')])
    def test_download_resume_and_wrong_sha(self):
        payload=b'1234567890';item=dict(size=len(payload),sha256=hashlib.sha256(payload).hexdigest())
        path=self.work/'weights.bin';path.with_name('weights.bin.part').write_bytes(payload[:3])
        class Response(io.BytesIO):
            status=206
            headers={'Content-Range':'bytes 3-9/10'}
        with mock.patch.object(prepare_data.urllib.request,'urlopen',return_value=Response(payload[3:])):
            prepare_data.download_file(path,item,'https://example.invalid/file')
        self.assertEqual(path.read_bytes(),payload);path.write_bytes(b'bad')
        with self.assertRaises(ValueError):prepare_data.download_file(path,item,'https://example.invalid/file')
    def test_data_path_escape_rejected(self):
        with self.assertRaises(ValueError):data._relative(self.work,'../escape')
    def test_bank_probe_reports_row_bytes_and_statistics(self):
        values = np.asarray([[1., -2., 3.], [4., 0., -6.]], dtype=np.float32)
        result = prepare_data.bank_probe(values, 57486)
        self.assertEqual(result['row'], 57486)
        self.assertEqual(result['sha256'], hashlib.sha256(values.tobytes()).hexdigest())
        self.assertEqual(result['mean'], 0.)
        self.assertEqual(result['minimum'], -6.)
        self.assertEqual(result['maximum'], 4.)
        self.assertEqual(result['rms'], np.sqrt(11.))
if __name__=='__main__':unittest.main()
