"""Synthetic checks only; no competition data or performance claims."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import LABELS, build_model, load_data, make_folds, make_groups


def fixture():
    return pd.DataFrame([
        {"id": f"{prompt}-{i}", "prompt_id": str(prompt), "model": label,
         "text": f"Shared discussion prompt {prompt}. Signature token{i} " * 3}
        for prompt in range(10) for i, label in enumerate(LABELS)
    ])


class BaselineTests(unittest.TestCase):
    def test_prompt_and_duplicate_transitive_groups(self):
        df = pd.DataFrame({"prompt_id": ["a", "a", "b", "b", "c"],
                           "text": ["one", " SAME text ", "same TEXT", "four", "five"]})
        groups = make_groups(df, "text")
        self.assertEqual(len(set(groups[:4])), 1)
        self.assertNotEqual(groups[0], groups[4])

    def test_splits_reproducible_and_disjoint(self):
        df = fixture()
        groups = make_groups(df, "text")
        folds = make_folds(df, groups, "model", 5, 42)
        np.testing.assert_array_equal(folds, make_folds(df, groups, "model", 5, 42))
        for fold in range(5):
            self.assertFalse(set(groups[folds == fold]) & set(groups[folds != fold]))
            self.assertEqual(set(df.loc[folds == fold, "model"]), set(LABELS))

    def test_no_validation_vocabulary_leakage(self):
        model = build_model(42, None)
        model.fit(["alpha shared", "alpha common", "beta shared", "beta common"],
                  ["a", "a", "b", "b"])
        prediction = model.predict(["validationonlytoken"])
        vocab = model.named_steps["features"].transformer_list[0][1].vocabulary_
        self.assertNotIn("validationonlytoken", vocab)
        self.assertEqual(len(prediction), 1)

    def test_missing_class_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            df = fixture()
            df.to_csv(path, index=False)
            self.assertEqual(len(load_data(path, "text", "model")), len(df))
            df.iloc[1, df.columns.get_loc("id")] = df.iloc[0].id
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_data(path, "text", "model")
            fixture().query("model != @LABELS[0]").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Label mismatch"):
                load_data(path, "text", "model")


if __name__ == "__main__":
    unittest.main()
