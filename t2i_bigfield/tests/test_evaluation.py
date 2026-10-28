"""Small CPU checks; no model downloads, checkpoint loads, or device initialization."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from eval_sample import atomic_json, canonical_digest, quantize, rows_from_prompts
from score import aggregate_answers, aggregate_geneval, gate_scores, questions_from_rows, validate_answers
from evaluation.prepare import download, sha256


class EvaluationTests(unittest.TestCase):
    def test_prompt_order_and_paths(self):
        prompts = [dict(key='z', prompt='first'), dict(key='a', prompt='second')]
        rows = rows_from_prompts('dpg', prompts)
        self.assertEqual([(r['key'], r['seed']) for r in rows], [('z', s) for s in range(4)] + [('a', s) for s in range(4)])
        self.assertEqual(rows[0]['relative_path'], 'samples/z/0000.png')
        self.assertNotEqual(canonical_digest(rows), canonical_digest(list(reversed(rows))))
        generated = rows_from_prompts('geneval', [dict(prompt='first', tag='single_object')])
        self.assertEqual(generated[3]['relative_path'], 'geneval/00000/samples/0003.png')
        self.assertEqual(generated[0]['metadata'], dict(prompt='first', tag='single_object'))

    def test_quantization_matches_integer_pixels(self):
        np.testing.assert_array_equal(quantize([-2., -1., 0., 1., 2.]), [0, 0, 128, 255, 255])
        for invalid in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                quantize([invalid])

    def test_geneval_category_equal_weight(self):
        result = aggregate_geneval([dict(tag='a', correct=True), dict(tag='a', correct=True), dict(tag='b', correct=False)])
        self.assertEqual(result['overall_category_balanced'], .5)

    def test_dpg_forward_dependency_remains_one_pass(self):
        self.assertEqual(gate_scores({1: 1., 2: 1., 3: 0.}, {1: [2, 2], 2: [3], 3: [0]}), {1: 1., 2: 0., 3: 0.})
        self.assertEqual(gate_scores({1: 1.}, {1: [1]}), {1: 1.})

    def test_dpg_fullrow_exact_yes(self):
        question_rows = [dict(item_id='sample', proposition_id='1', dependency='0', tuple='entity(cat)', question_natural_language='cat?')]
        questions = questions_from_rows(question_rows)
        answers = {('sample', seed, 1): answer for seed, answer in enumerate(['yes', 'Yes', 'yes ', 'no'])}
        result = aggregate_answers(answers, questions, ['sample'])
        self.assertEqual(result['fullrow_corrected_overall'], 25.)
        records = [dict(key=k, seed=s, qid=q, question='cat?', answer=a) for (k, s, q), a in answers.items()]
        self.assertEqual(validate_answers(records, questions, ['sample'], True), answers)
        with self.assertRaises(ValueError):
            validate_answers(records[:-1], questions, ['sample'], True)
        with self.assertRaises(ValueError):
            validate_answers(records + [records[0]], questions, ['sample'], True)

    def test_download_verifies_existing_file(self):
        with tempfile.TemporaryDirectory(prefix='eval-', dir=os.environ.get('TMPDIR')) as temp:
            root = Path(temp)
            source = root / 'source'
            source.write_bytes(b'public test asset')
            row = dict(path='assets/weight', url=source.as_uri(), sha256=sha256(source), bytes=source.stat().st_size)
            partial = root / 'assets/weight.partial'
            partial.parent.mkdir()
            partial.write_bytes(b'')
            target = download(row, root)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(download(row, root), target)
            target.write_bytes(b'different')
            with self.assertRaises(ValueError):
                download(row, root)
            self.assertEqual(target.read_bytes(), b'different')

    def test_atomic_numeric_report(self):
        with tempfile.TemporaryDirectory(prefix='eval-', dir=os.environ.get('TMPDIR')) as temp:
            path = Path(temp) / 'report.json'
            atomic_json(path, dict(nonfinite_count=1, batch_start=4))
            self.assertEqual(json.loads(path.read_text()), dict(nonfinite_count=1, batch_start=4))


if __name__ == '__main__':
    unittest.main()
