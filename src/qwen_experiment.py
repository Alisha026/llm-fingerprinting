"""Download a pinned Qwen encoder, or evaluate two fixed heads on grouped splits.

Training/inference only occurs when the user invokes `run`.
"""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


def validate_frames(train, valid):
    required = {'id', 'prompt_id', 'text', 'model'}
    for frame in (train, valid):
        if frame.empty or not required.issubset(frame.columns):
            raise ValueError('Expected prepared labeled CSVs')
        if frame.id.duplicated().any() or any(frame[c].str.strip().eq('').any() for c in required):
            raise ValueError('Duplicate IDs or empty fields')
    for column in ('id', 'prompt_id'):
        if set(train[column]) & set(valid[column]):
            raise ValueError(f'Train/validation overlap in {column}')
    normal = lambda s: ' '.join(s.casefold().split())
    if set(train.text.map(normal)) & set(valid.text.map(normal)):
        raise ValueError('Duplicate text crosses split')
    if len(set(train.model)) != 18 or set(train.model) != set(valid.model):
        raise ValueError('Both splits must contain the same 18 labels')


def last_token_pool(hidden, mask):
    """Select the last nonpadding token for either padding direction."""
    import torch
    if not torch.all(mask.sum(1) > 0):
        raise ValueError('Empty token sequence')
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(mask == 0, -1).max(1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


def download(destination):
    from huggingface_hub import HfApi, snapshot_download
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Download destination must be new or empty')
    repo = 'Qwen/Qwen3-Embedding-0.6B'
    sha = HfApi().model_info(repo).sha
    snapshot_download(repo, revision=sha, local_dir=destination,
                      allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', '*.model'])
    (destination / 'download_provenance.json').write_text(json.dumps({'repo_id': repo, 'revision': sha}, indent=2))
    print(f'Downloaded {repo} revision {sha} to {destination}')


def run(args):
    import joblib
    import torch
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, recall_score
    from sklearn.preprocessing import normalize
    from transformers import AutoModel, AutoTokenizer

    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError('Batch size and max length must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new or empty output directory')
    train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
    valid = pd.read_csv(args.validation, dtype=str, keep_default_na=False)
    validate_frames(train, valid)
    if not torch.cuda.is_available():
        raise RuntimeError('Run inside a GPU allocation; CUDA is unavailable')
    # CUDA visibility does not guarantee this PyTorch wheel supports the GPU.
    # Exercise a kernel before loading weights or creating output artifacts.
    try:
        probe = torch.ones(1, device='cuda') + 1
        torch.cuda.synchronize()
        del probe
    except RuntimeError as exc:
        raise RuntimeError(
            f'GPU/PyTorch compatibility check failed: {torch.cuda.get_device_name(0)}, '
            f'compute capability {torch.cuda.get_device_capability(0)}, torch {torch.__version__}. '
            'Use a supported GPU allocation or install a PyTorch CUDA build supporting this GPU. '
            'No model inference or training has started.'
        ) from exc
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    # Native BF16 requires Ampere or newer; do not select software emulation.
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True, padding_side='left')
    model = AutoModel.from_pretrained(args.encoder, local_files_only=True,
                                     dtype=dtype, trust_remote_code=False).to('cuda').eval()
    model.requires_grad_(False)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError('Token limit exceeds model capacity')
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = args.encoder / 'download_provenance.json'
    config = {'encoder': str(args.encoder.resolve()),
              'encoder_provenance': json.loads(provenance.read_text()) if provenance.exists() else None,
              'encoder_config_sha256': hashlib.sha256((args.encoder / 'config.json').read_bytes()).hexdigest(),
              'train_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(),
              'validation_sha256': hashlib.sha256(args.validation.read_bytes()).hexdigest(),
              'batch_size': args.batch_size, 'max_length': args.max_length,
              'pooling': 'last nonpadding token', 'normalization': 'row L2 only',
              'instruction': None, 'dtype': str(dtype), 'C': 1.0, 'seed': 42,
              'heads': ['embedding', 'hybrid_word'], 'status': 'running',
              'packages': {p: version(p) for p in ['torch', 'transformers', 'scikit-learn', 'numpy', 'pandas']}}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    lengths = {}

    def encode(frame, name):
        chunks, token_lengths = [], []
        with torch.inference_mode():
            for start in range(0, len(frame), args.batch_size):
                texts = frame.text.iloc[start:start+args.batch_size].tolist()
                token_lengths.extend(map(len, tokenizer(texts, truncation=False)['input_ids']))
                tokens = tokenizer(texts, padding=True, truncation=True,
                                   max_length=args.max_length, return_tensors='pt').to('cuda')
                hidden = model(**tokens, use_cache=False).last_hidden_state
                chunks.append(last_token_pool(hidden, tokens.attention_mask).float().cpu().numpy())
                if start % (args.batch_size * 25) == 0:
                    print(f'{name}: {min(start+args.batch_size, len(frame))}/{len(frame)}', flush=True)
        lengths[name] = {'rows': len(frame), 'max_tokens': max(token_lengths),
                         'truncated': sum(n > args.max_length for n in token_lengths)}
        return normalize(np.concatenate(chunks))

    x_train, x_valid = encode(train, 'train'), encode(valid, 'validation')
    del model
    torch.cuda.empty_cache()
    # These frozen representations have no fit statistics; both heads reuse them.
    np.savez_compressed(args.output / 'embeddings.npz', train=x_train, validation=x_valid,
                        train_ids=train.id.to_numpy(dtype=str), validation_ids=valid.id.to_numpy(dtype=str))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 max_features=100000, dtype=np.float32)
    text_train = normalize(vectorizer.fit_transform(train.text))
    text_valid = normalize(vectorizer.transform(valid.text))
    summary = {}
    for name in ('embedding', 'hybrid_word'):
        tr, va = csr_matrix(x_train), csr_matrix(x_valid)
        if name == 'hybrid_word':
            tr, va = hstack([text_train, tr], format='csr'), hstack([text_valid, va], format='csr')
        classifier = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=42)
        classifier.fit(tr, train.model)
        probabilities = classifier.predict_proba(va)
        labels = classifier.classes_
        predicted = labels[probabilities.argmax(1)]
        folder = args.output / name
        folder.mkdir()
        metrics = {'balanced_accuracy': balanced_accuracy_score(valid.model, predicted),
                   'train_balanced_accuracy': balanced_accuracy_score(train.model, classifier.predict(tr)),
                   'per_class_recall': dict(zip(labels, recall_score(valid.model, predicted, labels=labels, average=None).tolist())),
                   'validation_rows': len(valid), 'scope': 'local prompt-grouped validation only'}
        (folder / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        (folder / 'config.json').write_text(json.dumps({**config, 'head': name, 'status': 'complete'}, indent=2))
        pd.DataFrame({'id': valid.id, 'true_model': valid.model, 'predicted_model': predicted}).to_csv(folder / 'validation_predictions.csv', index=False)
        pd.DataFrame(probabilities, columns=labels, index=pd.Index(valid.id, name='id')).to_csv(folder / 'validation_probabilities.csv')
        joblib.dump({'classifier': classifier, 'tfidf': vectorizer if name == 'hybrid_word' else None,
                     'labels': labels.tolist(), 'encoder': str(args.encoder.resolve()),
                     'pooling': 'last nonpadding token', 'normalization': 'row L2 only',
                     'max_length': args.max_length}, folder / 'classifier.joblib')
        summary[name] = metrics['balanced_accuracy']
        print(f'{name}: validation balanced accuracy {summary[name]:.2%}', flush=True)
    (args.output / 'token_lengths.json').write_text(json.dumps(lengths, indent=2))
    config.update(status='complete', results=summary)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('download')
    fetch.add_argument('--destination', type=Path, default=Path('artifacts/pretrained/qwen3_embedding_06b'))
    experiment = commands.add_parser('run')
    experiment.add_argument('--encoder', type=Path, default=Path('artifacts/pretrained/qwen3_embedding_06b'))
    experiment.add_argument('--train', type=Path, required=True)
    experiment.add_argument('--validation', type=Path, required=True)
    experiment.add_argument('--output', type=Path, required=True)
    experiment.add_argument('--batch-size', type=int, default=8)
    experiment.add_argument('--max-length', type=int, default=512)
    args = parser.parse_args()
    if args.command == 'download':
        download(args.destination)
    else:
        run(args)
