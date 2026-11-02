"""CPU-only classifier diagnostic on cached Qwen vectors; no encoder inference."""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from src.qwen_experiment import validate_frames


def transform_embeddings(train, valid, standardize):
    if standardize:
        mean = train.mean(axis=0)
        scale = np.maximum(train.std(axis=0), 1e-6)
        train, valid = (train - mean) / scale, (valid - mean) / scale
    return normalize(train), normalize(valid)


def load_cache(run, train_path, valid_path):
    config = json.loads((run / 'config.json').read_text())
    for key, path in [('train_sha256', train_path), ('validation_sha256', valid_path)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != config[key]:
            raise ValueError(f'{key}: CSV differs from original embedding run')
    train = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    valid = pd.read_csv(valid_path, dtype=str, keep_default_na=False)
    validate_frames(train, valid)
    with np.load(run / 'embeddings.npz', allow_pickle=False) as cache:
        x, y = cache['train'].copy(), cache['validation'].copy()
        for ids, frame in [(cache['train_ids'], train), (cache['validation_ids'], valid)]:
            if not np.array_equal(ids, frame.id.to_numpy()):
                raise ValueError('Cached embedding IDs/order do not match CSV')
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]
            or len(x) != len(train) or len(y) != len(valid)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Invalid cached vectors')
    return train, valid, x, y


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new or empty output directory')
    train, valid, x, y = load_cache(args.run, args.train, args.validation)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'source_run': str(args.run), 'C_values': [0.1, 1.0, 10.0],
              'seed': 42, 'status': 'running',
              'cache_sha256': hashlib.sha256((args.run / 'embeddings.npz').read_bytes()).hexdigest(),
              'scope': 'Exploratory validation sweep; confirm selected settings on another fold.',
              'caveat': 'Cache contains L2-normalized vectors. Standardization is applied AFTER original L2 normalization, then L2 again; not raw-vector standardization.'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 max_features=100000, dtype=np.float32)
    text_train = normalize(vectorizer.fit_transform(train.text))
    text_valid = normalize(vectorizer.transform(valid.text))
    rows = []

    def fit(name, tr, va, c, preprocessing, head):
        clf = LogisticRegression(C=c, solver='lbfgs', max_iter=2000, random_state=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            clf.fit(tr, train.model)
        probs = clf.predict_proba(va)
        predicted = clf.classes_[probs.argmax(1)]
        folder = args.output / name
        folder.mkdir()
        row = {'experiment': name, 'preprocessing': preprocessing, 'head': head, 'C': c,
               'train_balanced_accuracy': balanced_accuracy_score(train.model, clf.predict(tr)),
               'validation_balanced_accuracy': balanced_accuracy_score(valid.model, predicted),
               'converged': not any(issubclass(w.category, ConvergenceWarning) for w in caught),
               'iterations': int(clf.n_iter_.max())}
        # Reproduce original head at C=1 before interpreting preprocessing changes.
        if preprocessing == 'l2' and c == 1.0:
            reference = pd.read_csv(args.run / head / 'validation_probabilities.csv', dtype={'id': str}).set_index('id')
            expected = reference.loc[valid.id.tolist(), clf.classes_.tolist()].to_numpy()
            delta = float(np.max(np.abs(probs - expected)))
            row['reference_max_probability_delta'] = delta
            if not np.allclose(probs, expected, atol=1e-5, rtol=1e-4):
                raise ValueError(f'Original {head} probabilities did not reproduce (max delta {delta}); stop and investigate')
        pd.DataFrame({'id': valid.id, 'true_model': valid.model, 'predicted_model': predicted}).to_csv(folder / 'validation_predictions.csv', index=False)
        pd.DataFrame(probs, columns=clf.classes_, index=pd.Index(valid.id, name='id')).to_csv(folder / 'validation_probabilities.csv')
        (folder / 'metrics.json').write_text(json.dumps(row, indent=2))
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output / 'summary.csv', index=False)
        print(f'{name}: train={row["train_balanced_accuracy"]:.2%}, validation={row["validation_balanced_accuracy"]:.2%}, converged={row["converged"]}', flush=True)

    with threadpool_limits(limits=4):
        # C=1 first so the original results are checked before trying other values.
        for standardized in (False, True):
            preprocessing = 'standardize_l2' if standardized else 'l2'
            tr, va = transform_embeddings(x, y, standardized)
            for head in ('embedding', 'hybrid_word'):
                a, b = csr_matrix(tr), csr_matrix(va)
                if head == 'hybrid_word':
                    a, b = hstack([text_train, a], format='csr'), hstack([text_valid, b], format='csr')
                for c in (1.0, 0.1, 10.0):
                    fit(f'{preprocessing}_{head}_c{c:g}', a, b, c, preprocessing, head)
        fit('tfidf_only_c1', text_train, text_valid, 1.0, 'none', 'tfidf_only')
    config['status'] = 'complete'
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    print('\nExploratory ranking (not an independent validation estimate):')
    print(pd.DataFrame(rows).sort_values('validation_balanced_accuracy', ascending=False)[
        ['experiment', 'train_balanced_accuracy', 'validation_balanced_accuracy', 'converged']].to_string(index=False))
    print(f'Full results: {args.output / "summary.csv"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/qwen_embedding_fold0_retry'))
    parser.add_argument('--train', type=Path, default=Path('data/processed/train.csv'))
    parser.add_argument('--validation', type=Path, default=Path('data/processed/validation.csv'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/qwen_diagnostic_fold0'))
    run(parser.parse_args())
