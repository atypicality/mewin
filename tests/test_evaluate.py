import json
from pathlib import Path
import tempfile
import unittest

from evaluate import score, summarize, validate_manifest


class EvaluationTests(unittest.TestCase):
    def test_duplicate_count_is_failure(self):
        result = score({'1p': 2}, {'1p': 1})
        self.assertFalse(result['passed'])
        self.assertEqual(result['missing'], {'1p': 1})

    def test_wrong_identity_has_missing_and_extra(self):
        result = score({'5s': 1}, {'1p': 1})
        self.assertEqual(result['failure_type'], 'missing_and_extra')
        self.assertEqual(result['extra'], {'1p': 1})

    def test_negative_and_exact_match(self):
        self.assertTrue(score({}, {})['passed'])
        self.assertTrue(score({'1p': 2, '5s': 1}, {'5s': 1, '1p': 2})['passed'])
        self.assertEqual(score({}, {'1p': 1})['failure_type'], 'false_detection')

    def test_empty_group_is_not_zero_accuracy(self):
        self.assertIsNone(summarize([])['exact_hand_accuracy'])
        self.assertEqual(summarize([{'passed': True}, {'passed': False}])['exact_hand_accuracy'], 0.5)

    def test_unlabeled_ready_case_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cases.json'
            path.write_text(json.dumps({'cases': [dict(id='T01', status='ready', split='test', expected=None)]}))
            with self.assertRaises(ValueError):
                validate_manifest(path)

    def test_pending_cases_need_no_images(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cases.json'
            path.write_text(json.dumps({'cases': [dict(id='T01', status='pending', split='test', expected=None)]}))
            self.assertEqual(len(validate_manifest(path)['cases']), 1)


if __name__ == '__main__':
    unittest.main()
