"""Offline integration check: train, reload, and predict with a tiny encoder."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from transformers import AutoModel, AutoTokenizer, DistilBertConfig

from src.train_full import train_full
from src.kaggle_inference import submit


class FullTrainingTest(unittest.TestCase):
    def test_full_refit_and_offline_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / 'encoder'
            tokenizer = AutoTokenizer.from_pretrained('artifacts/encoder_v1/encoder', local_files_only=True)
            tokenizer.save_pretrained(encoder)
            AutoModel.from_config(DistilBertConfig(
                vocab_size=len(tokenizer), dim=16, hidden_dim=32,
                n_layers=1, n_heads=2, max_position_embeddings=512,
            )).save_pretrained(encoder)
            frame = pd.DataFrame([
                {'id': f'{i:04d}', 'prompt_id': str(i // 18),
                 'text': f'This is a sample answer about science and model {i % 18}.',
                 'model': f'label-{i % 18:02d}'} for i in range(36)
            ])
            train = root / 'train.csv'
            frame.to_csv(train, index=False)
            bundle = train_full(train, encoder, root / 'bundle', epochs=1, device_name='cpu')
            test = root / 'test.csv'
            frame[['id', 'text']].iloc[:3].to_csv(test, index=False)
            output = root / 'submission.parquet'
            result = submit(bundle, test, output)
            self.assertEqual(result.id.tolist(), ['0000', '0001', '0002'])
            self.assertTrue(result.model.isin(frame.model).all())
            pd.testing.assert_frame_equal(result, pd.read_parquet(output))
            with self.assertRaisesRegex(ValueError, 'new or empty'):
                train_full(train, encoder, bundle, epochs=1, device_name='cpu')


if __name__ == '__main__':
    unittest.main()
