"""Protocol guardrails, independent of the MNIST dataset."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import wsp_lenet as lab


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.base = {'fc3.weight': torch.arange(840.).reshape(10, 84),
                     'fc3.bias': torch.zeros(10), 'conv1.weight': torch.ones(1)}

    def test_candidate_does_not_mutate_base_or_other_layers(self):
        snapshot = {k: v.clone() for k, v in self.base.items()}
        edited = lab.patch_state(self.base, {'op': 'scale_row', 'index': 7, 'value': -1})
        self.assertTrue(all(torch.equal(self.base[k], snapshot[k]) for k in self.base))
        self.assertTrue(torch.equal(edited['conv1.weight'], self.base['conv1.weight']))
        self.assertEqual(int((edited['fc3.weight'] != self.base['fc3.weight']).sum()), 84)
        again = lab.patch_state(self.base, {'op': 'scale_row', 'index': 7, 'value': -1})
        self.assertTrue(torch.equal(edited['fc3.weight'], again['fc3.weight']))

    def test_rejects_malformed_and_unbounded_edits(self):
        cases = [
            {'op': 'scale_row', 'index': True, 'value': 1},
            {'op': 'scale_row', 'index': 10, 'value': 1},
            {'op': 'scale_feature', 'index': 84, 'value': 1},
            {'op': 'scale_row', 'index': 0, 'value': 2.01},
            {'op': 'set_bias', 'index': 0, 'value': 20.01},
            {'op': 'set_bias', 'index': 0, 'value': float('nan')},
            {'op': 'set_bias', 'index': 0, 'value': float('inf')},
            {'op': 'train', 'index': 0, 'value': 1},
            {'op': 'scale_row', 'index': 0, 'value': 1, 'extra': 1},
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                lab.patch_state(self.base, value)

    def test_budgets_reject_without_writing(self):
        session = (self.base, {}, {'inspections': 6, 'proposals': 3})
        with patch.object(lab, 'session', return_value=session):
            with self.assertRaisesRegex(SystemExit, 'Inspection budget'):
                lab.inspect(None)
            with self.assertRaisesRegex(SystemExit, 'Proposal budget'):
                lab.propose(None)

    def test_final_test_locks_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'final_results.json').write_text('{}')
            with patch.object(lab, 'P', root):
                with self.assertRaisesRegex(SystemExit, 'Final test already used'):
                    lab.session()


if __name__ == '__main__':
    unittest.main()
