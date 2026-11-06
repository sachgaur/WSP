"""Tests for the boundary between model responses and executable tools."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ollama_study as adapter


class AdapterTests(unittest.TestCase):
    def test_shell_or_file_requests_cannot_be_actions(self):
        for action in [
            {'action': 'shell', 'command': 'cat private/base.pt', 'reason': 'test'},
            {'action': 'inspect', 'query': '../../private/base.pt', 'reason': 'test'},
            {'action': 'inspect', 'query': 'row', 'index': '0; cat private/base.pt', 'reason': 'test'},
            {'action': 'inspect', 'query': 'row', 'index': True, 'reason': 'test'},
            {'action': 'finish', 'candidate': True, 'reason': 'test'},
        ]:
            with self.subTest(action=action), self.assertRaises(ValueError):
                adapter.validate_action(action)

    def test_hypothesis_is_required_before_a_proposal(self):
        action = {'action': 'propose', 'patch': {'op': 'scale_row', 'index': 0, 'value': 1}, 'reason': ''}
        with self.assertRaises(ValueError):
            adapter.validate_action(action)
        action['reason'] = 'Hypothesis and predicted effect'
        self.assertEqual(adapter.validate_action(action), action)

    def test_patch_bounds_reject_nonfinite_or_arbitrary_edits(self):
        for op, index, value in [('train', 0, 1), ('scale_row', 10, 1),
                                 ('scale_feature', 84, 1), ('scale_row', 0, 2.1),
                                 ('set_bias', 0, float('nan')), ('set_bias', 0, 21)]:
            action = {'action': 'propose', 'reason': 'Hypothesis',
                      'patch': {'op': op, 'index': index, 'value': value}}
            with self.subTest(action=action), self.assertRaises(ValueError):
                adapter.validate_action(action)

    def test_clean_tolerance_does_not_allow_marked_regression(self):
        def candidate(clean, marked):
            return {'base': {'clean': {'accuracy': .95}, 'marked': {'accuracy': .85}},
                    'edited': {'clean': {'accuracy': clean}, 'marked': {'accuracy': marked}}}
        self.assertTrue(adapter.acceptable(candidate(.94, .86)))
        self.assertFalse(adapter.acceptable(candidate(.939, .90)))
        self.assertFalse(adapter.acceptable(candidate(.96, .85)))

    def test_budgets_block_subprocess_dispatch(self):
        usage = {'inspections': 6, 'proposals': 3}
        with patch.object(adapter, 'read', return_value=usage), patch.object(adapter, 'invoke') as invoke:
            for action in [{'action': 'inspect', 'query': 'summary'},
                           {'action': 'propose', 'patch': {}}]:
                with self.assertRaises(ValueError):
                    adapter.execute_action(Path('/unused'), action)
            invoke.assert_not_called()

    def test_finalization_requires_all_six_decisions(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(adapter, 'RUNS', Path(directory)):
            with self.assertRaisesRegex(SystemExit, 'Freeze all six'):
                adapter.finalize_all()


if __name__ == '__main__':
    unittest.main()
