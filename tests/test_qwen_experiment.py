"""Tensor/schema checks only; no model downloads, inference, or training."""
import unittest
import pandas as pd
import torch
from src.qwen_experiment import last_token_pool, validate_frames


class QwenChecks(unittest.TestCase):
    def test_last_token_with_left_and_right_padding(self):
        hidden = torch.arange(12).reshape(3, 4, 1)
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(last_token_pool(hidden, mask).flatten().tolist(), [3, 5, 11])
        with self.assertRaises(ValueError):
            last_token_pool(hidden, torch.zeros_like(mask))

    def test_prompt_overlap_rejected(self):
        def frame(prefix):
            return pd.DataFrame([{'id': f'{prefix}-{i}', 'prompt_id': prefix,
                                  'text': f'{prefix} text {i}', 'model': str(i)} for i in range(18)])
        train, valid = frame('a'), frame('b')
        validate_frames(train, valid)
        valid.loc[0, 'prompt_id'] = 'a'
        with self.assertRaisesRegex(ValueError, 'prompt_id'):
            validate_frames(train, valid)


if __name__ == '__main__':
    unittest.main()
