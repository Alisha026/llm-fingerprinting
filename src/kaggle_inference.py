"""Offline inference using the existing fold-0 50/50 ensemble."""
from pathlib import Path
import gc
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def submit(root, test_path, output='/kaggle/working/submission.parquet'):
    root = Path(root)
    frame = pd.read_csv(test_path, dtype={'id': str}, keep_default_na=False)
    assert len(frame) and {'id', 'text'}.issubset(frame.columns)
    assert frame.id.is_unique and frame.id.ne('').all()
    texts = frame.text.tolist()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    print('Device:', device, '| passages:', len(texts), flush=True)
    bundle = joblib.load(root / 'classifier.joblib')
    labels = bundle['classifier'].classes_.tolist()

    def run(folder, classification=False):
        tokenizer = AutoTokenizer.from_pretrained(root / folder, local_files_only=True)
        cls = AutoModelForSequenceClassification if classification else AutoModel
        model = cls.from_pretrained(root / folder, local_files_only=True).to(device).eval()
        order = None
        if classification:
            model_labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            assert len(labels) == 18 and set(labels) == set(model_labels)
            order = [model_labels.index(label) for label in labels]
        parts = []
        with torch.inference_mode():
            for start in range(0, len(texts), 8):
                tokens = tokenizer(texts[start:start+8], padding=True, truncation=True,
                                   max_length=512, return_tensors='pt')
                tokens = {k: v.to(device) for k, v in tokens.items()}
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=classification and device.type == 'cuda'):
                    result = model(**tokens)
                if classification:
                    values = result.logits.float().softmax(-1).cpu().numpy()[:, order]
                else:
                    hidden = result.last_hidden_state
                    mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                    values = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
                parts.append(values)
                if start % 256 == 0:
                    print(folder, min(start+8, len(texts)), '/', len(texts), flush=True)
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return np.concatenate(parts)

    embeddings = run('frozen_encoder')
    embeddings = normalize((embeddings - bundle['embedding_mean']) / bundle['embedding_scale'])
    features = hstack([normalize(bundle['tfidf'].transform(texts)),
                       bundle['embedding_weight'] * csr_matrix(embeddings)], format='csr')
    hybrid = bundle['classifier'].predict_proba(features)
    fine = run('finetuned_model', classification=True)
    probabilities = 0.5 * hybrid + 0.5 * fine
    submission = pd.DataFrame({'id': frame.id, 'model': np.asarray(labels)[probabilities.argmax(1)]})
    assert len(submission) == len(frame) and submission.model.isin(labels).all()
    submission.to_parquet(output, index=False)
    print('Saved', output, '| rows:', len(submission))
    return submission
