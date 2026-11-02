"""Refit the fixed DistilBERT/hybrid ensemble on all official labeled rows.

No validation or test data is used for selection. Nine fine-tuning epochs were
selected before this run from the existing fold-0 and fold-1 best epochs.
"""
import argparse
import gc
import hashlib
import json
import math
import random
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)


class Passages(Dataset):
    def __init__(self, frame, tokenizer, labels):
        self.texts = frame.text.tolist()
        self.targets = frame.model.map({label: i for i, label in enumerate(labels)}).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        item = self.tokenizer(self.texts[index], truncation=True, max_length=512)
        item['labels'] = self.targets[index]
        return item

# Train the full-data hybrid and fine-tuned encoder, saving all artifacts to the output directory.
def train_full(train_path, encoder_path, output, epochs=9, device_name='cuda'):
    output, train_path, encoder_path = Path(output), Path(train_path), Path(encoder_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new or empty output directory; existing models are preserved.')
    if epochs < 1:
        raise ValueError('epochs must be positive')
    frame = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    required = ['id', 'prompt_id', 'text', 'model']
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError('Expected labeled competition train.csv')
    if frame.id.duplicated().any() or any(frame[c].str.strip().eq('').any() for c in required):
        raise ValueError('Duplicate IDs or empty training values')
    labels = sorted(frame.model.unique())
    if len(labels) != 18:
        raise ValueError('Expected all 18 model labels')
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select GPU T4 x2 in Kaggle, or run in a GPU allocation.')
    device = torch.device(device_name)
    use_amp = device.type == 'cuda'
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if use_amp:
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output.mkdir(parents=True, exist_ok=True)
    config = dict(train_rows=len(frame), train_prompts=int(frame.prompt_id.nunique()),
                  train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
                  labels=labels, epochs=epochs, max_length=512, learning_rate=2e-5,
                  batch_size=8, gradient_accumulation=4, seed=42, first_weight=0.5,
                  selection='Fixed epochs from prior grouped validation; no validation on full-data refit',
                  status='training', device=str(device),
                  packages={name: version(name) for name in
                            ['torch', 'transformers', 'numpy', 'pandas', 'scikit-learn', 'joblib', 'scipy']})
    (output / 'training_config.json').write_text(json.dumps(config, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    model = AutoModel.from_pretrained(encoder_path, local_files_only=True).to(device).eval()
    vectors = []
    print(f'Encoding all {len(frame)} labeled passages', flush=True)
    with torch.inference_mode():
        for start in range(0, len(frame), 16):
            batch = tokenizer(frame.text.iloc[start:start+16].tolist(), padding=True,
                              truncation=True, max_length=512, return_tensors='pt')
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            vectors.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu())
    model.save_pretrained(output / 'frozen_encoder')
    tokenizer.save_pretrained(output / 'frozen_encoder')
    del model
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    vectors = torch.cat(vectors)
    mean = vectors.mean(0)
    scale = vectors.std(0, unbiased=False).clamp(min=1e-6)
    embeddings = normalize(((vectors - mean) / scale).numpy())
    vectorizer = FeatureUnion([('word', TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        max_features=100000, dtype=np.float32))])
    features = hstack([normalize(vectorizer.fit_transform(frame.text)), csr_matrix(embeddings)], format='csr')
    classifier = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    classifier.fit(features, frame.model)
    joblib.dump(dict(classifier=classifier, tfidf=vectorizer,
                     embedding_mean=mean.numpy(), embedding_scale=scale.numpy(),
                     embedding_weight=1.0, max_length=512, labels=classifier.classes_.tolist(),
                     head='hybrid', pooling='attention-mask mean', text_features='word'),
                output / 'classifier.joblib')
    del vectors, embeddings, features
    print('Full-data hybrid saved. Starting fresh encoder fine-tuning.', flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        encoder_path, num_labels=18, id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)}, local_files_only=True,
        ignore_mismatched_sizes=True).to(device)
    loader = DataLoader(Passages(frame, tokenizer, labels), batch_size=8, shuffle=True,
                        generator=torch.Generator().manual_seed(42),
                        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_amp else None))
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith('bias') or 'LayerNorm.weight' in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01},
                                   {'params': no_decay, 'weight_decay': 0.0}], lr=2e-5)
    updates = math.ceil(len(loader) / 4) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * 0.1), updates)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        # Normalize the last accumulation group by its actual number of examples.
        window_examples = 0
        for step, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            count = len(batch['labels'])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss * count).backward()
            total_loss += loss.detach().item() * count
            window_examples += count
            if (step + 1) % 4 == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_examples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                window_examples = 0
        history.append(dict(epoch=epoch, train_loss=total_loss / len(frame)))
        (output / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'Epoch {epoch}/{epochs}: training loss {history[-1]["train_loss"]:.4f}', flush=True)
    model.save_pretrained(output / 'finetuned_model')
    tokenizer.save_pretrained(output / 'finetuned_model')
    config['status'] = 'complete'
    (output / 'training_config.json').write_text(json.dumps(config, indent=2))
    print('Full-data ensemble saved:', output, flush=True)
    del model
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=9)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    train_full(args.train, args.encoder, args.output, args.epochs, args.device)
