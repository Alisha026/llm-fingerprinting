import unittest
import numpy as np
import pandas as pd
from src.two_stage_lora import triplets, centered, inferred_groups


class TwoStageTests(unittest.TestCase):
    def test_triplets_cover_rows_without_mixing_prompts_or_authors(self):
        frame = pd.DataFrame({'prompt_id': ['a'] * 5 + ['b'] * 4,
                              'model': list('abcde') + list('abcd')})
        batches = triplets(frame, 42)
        self.assertEqual(batches, triplets(frame, 42))
        self.assertEqual(set(sum(batches, [])), set(frame.index))
        for batch in batches:
            self.assertEqual(len(set(batch)), 3)
            self.assertEqual(frame.loc[batch].prompt_id.nunique(), 1)
            self.assertEqual(frame.loc[batch].model.nunique(), 3)

    def test_invalid_prompt_groups_fail_instead_of_silently_dropping(self):
        for labels in [list('ab'), list('aab')]:
            with self.assertRaises(ValueError):
                triplets(pd.DataFrame({'prompt_id': ['a'] * len(labels), 'model': labels}), 42)

    def test_repeated_author_responses_are_covered_in_distinct_author_triplets(self):
        frame = pd.DataFrame({'prompt_id': ['a'] * 11, 'model': list('aaaaaaabbcd')})
        batches = triplets(frame, 42)
        self.assertEqual(set(sum(batches, [])), set(frame.index))
        for batch in batches:
            self.assertEqual(frame.loc[batch].model.nunique(), 3)
        self.assertEqual(batches, triplets(frame, 42))

    def test_centering_removes_shared_offset_and_preserves_singletons(self):
        x = np.array([[10., 20.], [14., 26.], [99., 100.]])
        result, eligible = centered(x, np.array([1, 1, 2]))
        np.testing.assert_allclose(result[:2].mean(0), [0, 0])
        np.testing.assert_array_equal(result[2], x[2])
        np.testing.assert_array_equal(eligible, [True, True, False])
        np.testing.assert_array_equal(x[0], [10, 20])

    def test_grouping_uses_text_and_handles_single_row(self):
        np.testing.assert_array_equal(inferred_groups(['a passage']), [1])
        groups = inferred_groups(['quantum electron physics', 'quantum electron physics', 'cooking pasta tomato'])
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])


if __name__ == '__main__':
    unittest.main()
