"""Leakage-aware CPU baseline. Run `python baseline.py --help`."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import csr_matrix, hstack

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash",
    "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6",
    "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_data(path, text_col, label_col):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["id", text_col, label_col]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}; found {list(df.columns)}")
    if df[required].isna().any().any():
        raise ValueError(f"Null required values: {df[required].isna().sum().to_dict()}")
    if df[required].apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Required columns contain blank values.")
    if df.id.duplicated().any():
        raise ValueError("Training IDs must be unique.")
    observed = set(df[label_col])
    if observed != set(LABELS):
        raise ValueError(f"Label mismatch: missing={sorted(set(LABELS)-observed)}, "
                         f"unknown={sorted(observed-set(LABELS))}")
    if "prompt_id" in df and (
        df.prompt_id.isna().any() or df.prompt_id.str.strip().eq("").any()
    ):
        raise ValueError("prompt_id contains missing/blank values; resolve before splitting.")
    return df


def make_groups(df, text_col):
    """Connected components of shared prompts OR normalized duplicate texts."""
    parent = list(range(len(df)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys = [df[text_col].map(normalized)]
    if "prompt_id" in df:
        keys.append(df.prompt_id)
    for column in keys:
        seen = {}
        for i, value in enumerate(column):
            if value in seen:
                parent[root(i)] = root(seen[value])
            else:
                seen[value] = i
    return np.array([root(i) for i in range(len(df))])


def make_folds(df, groups, label_col, n_splits, seed):
    if n_splits < 2:
        raise ValueError("At least two folds are required.")
    for label in LABELS:
        count = len(set(groups[df[label_col].eq(label)]))
        if count < n_splits:
            raise ValueError(f"{label} has only {count} independent groups; "
                             f"cannot make {n_splits} folds.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(df), -1, dtype=int)
    for fold, (train, valid) in enumerate(splitter.split(df, df[label_col], groups)):
        if set(groups[train]) & set(groups[valid]):
            raise AssertionError("Group leakage.")
        for indices in (train, valid):
            if set(df.iloc[indices][label_col]) != set(LABELS):
                raise ValueError("A fold lacks a class; reduce --folds or inspect group sizes.")
        folds[valid] = fold
    return folds


def explore(df, text_col, label_col, groups, out):
    texts = df[text_col]
    norm = texts.map(normalized)
    lengths = pd.DataFrame({"characters": texts.str.len(), "words": texts.str.split().str.len()})
    conflicts = pd.DataFrame({"text": norm, "label": df[label_col]}).groupby("text").label.nunique()
    report = {
        "rows": len(df), "columns": list(df.columns),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_texts": int(texts.duplicated().sum()),
        "normalized_duplicate_texts": int(norm.duplicated().sum()),
        "duplicate_texts_with_conflicting_labels": int((conflicts > 1).sum()),
        "class_counts": df[label_col].value_counts().to_dict(),
        "length_summary": lengths.describe(percentiles=[.05, .5, .95]).to_dict(),
        "prompt_ids_available": "prompt_id" in df,
        "independent_groups": len(set(groups)),
        "largest_group": int(pd.Series(groups).value_counts().max()),
        "literal_label_mentions": {label: int(texts.str.contains(label, regex=False).sum()) for label in LABELS},
    }
    (out / "eda.json").write_text(json.dumps(report, indent=2))
    # Text examples remain in ignored artifacts, never in a public report.
    df.groupby(label_col, sort=True).head(2).to_csv(out / "examples.csv", index=False)
    lengths.assign(model=df[label_col]).groupby("model").agg(
        ["count", "mean", "median", "min", "max"]
    ).to_csv(out / "lengths_by_class.csv")
    grouping = "prompt IDs plus duplicate text" if "prompt_id" in df else "duplicate text only"
    (out / "findings.md").write_text(
        f"# Data findings\n\nRows: {len(df)}; labels: {df[label_col].nunique()}.\n\n"
        f"Grouping: {grouping}; {len(set(groups))} independent groups.\n\n"
        f"Normalized duplicate texts: {report['normalized_duplicate_texts']}; "
        f"conflicting-label duplicate groups: {report['duplicate_texts_with_conflicting_labels']}.\n\n"
        "See eda.json for missingness, class balance, length distributions and label mentions; "
        "examples.csv contains two examples per class for manual review.\n\n"
        "Only passage text is used as a feature. ID, prompt and topic columns are excluded. "
        "Shared prompts and normalized duplicate passages cannot cross folds. Near-duplicates "
        "and unidentified related prompts remain a limitation. Without prompt_id, unseen-prompt "
        "generalization cannot be established. Literal model-name mentions need manual inspection.\n"
    )


def build_model(seed, class_weight, text_features="both"):
    if text_features not in {"word", "char", "both"}:
        raise ValueError("text_features must be word, char or both.")
    transformers = []
    if text_features in {"word", "both"}:
        transformers.append(("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True,
            max_features=100_000, dtype=np.float32)))
    if text_features in {"char", "both"}:
        transformers.append(("char", TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, max_features=150_000, dtype=np.float32)))
    return Pipeline([
        ("features", FeatureUnion(transformers)),
        ("classifier", LinearSVC(C=1.0, class_weight=class_weight, random_state=seed,
                                 dual="auto", max_iter=10_000)),
    ])


def fit_frozen_classifier(x_train, x_valid, mean, scale, train, valid, args, config):
    """Fit a classical classifier to frozen embeddings, optionally with TF-IDF."""
    # Give feature blocks comparable scales so embedding dimensionality does not
    # dominate the word/character TF-IDF blocks. Statistics above use training only.
    train_features = csr_matrix(normalize(x_train.numpy()))
    valid_features = csr_matrix(normalize(x_valid.numpy()))
    vectorizer, text_train, text_valid = None, None, None
    if args.head in {"hybrid", "ensemble"}:
        vectorizer = build_model(args.seed, None, args.text_features).named_steps["features"]
        text_train = normalize(vectorizer.fit_transform(train[args.text_column]))
        text_valid = normalize(vectorizer.transform(valid[args.text_column]))
        if args.head == "hybrid":
            train_features = hstack([text_train, args.embedding_weight * train_features], format="csr")
            valid_features = hstack([text_valid, args.embedding_weight * valid_features], format="csr")
    class_weight = None if args.class_weight == "none" else "balanced"
    caught, models = [], {}
    if args.head in {"logistic", "hybrid"}:
        classifier = LogisticRegression(C=args.c, solver="lbfgs", max_iter=args.max_iter,
                                        class_weight=class_weight, random_state=args.seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(train_features, train[args.label_column])
        train_probabilities = classifier.predict_proba(train_features)
        valid_probabilities = classifier.predict_proba(valid_features)
        models["classifier"] = classifier
    elif args.head == "centroid":
        classifier = NearestCentroid(metric="euclidean", shrink_threshold=args.centroid_shrinkage)
        classifier.fit(train_features.toarray(), train[args.label_column])
        train_distances = classifier.decision_function(train_features.toarray())
        valid_distances = classifier.decision_function(valid_features.toarray())
        train_probabilities = np.exp(train_distances - train_distances.max(1, keepdims=True))
        valid_probabilities = np.exp(valid_distances - valid_distances.max(1, keepdims=True))
        train_probabilities /= train_probabilities.sum(1, keepdims=True)
        valid_probabilities /= valid_probabilities.sum(1, keepdims=True)
        models["classifier"] = classifier
    elif args.head == "rbf":
        classifier = SVC(C=args.c, gamma=args.gamma, kernel="rbf", probability=True,
                         class_weight=class_weight, random_state=args.seed,
                         cache_size=args.svc_cache_mb)
        classifier.fit(train_features.toarray(), train[args.label_column])
        train_probabilities = classifier.predict_proba(train_features.toarray())
        valid_probabilities = classifier.predict_proba(valid_features.toarray())
        models["classifier"] = classifier
    elif args.head == "ensemble":
        embedding_model = LogisticRegression(C=args.c, solver="lbfgs", max_iter=args.max_iter,
                                             class_weight=class_weight, random_state=args.seed)
        text_model = LogisticRegression(C=args.c, solver="lbfgs", max_iter=args.max_iter,
                                        class_weight=class_weight, random_state=args.seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            embedding_model.fit(train_features, train[args.label_column])
            text_model.fit(text_train, train[args.label_column])
        if not np.array_equal(embedding_model.classes_, text_model.classes_):
            raise ValueError("Ensemble classifiers learned inconsistent class orders.")
        classifier = embedding_model
        alpha = args.ensemble_embedding_weight
        train_probabilities = (alpha * embedding_model.predict_proba(train_features) +
                               (1 - alpha) * text_model.predict_proba(text_train))
        valid_probabilities = (alpha * embedding_model.predict_proba(valid_features) +
                               (1 - alpha) * text_model.predict_proba(text_valid))
        models.update({"embedding_classifier": embedding_model, "text_classifier": text_model})
    else:
        raise ValueError(f"Unsupported classical head: {args.head}")
    for warning in caught:
        print(f"Solver warning: {warning.message}", flush=True)
    classes = classifier.classes_
    train_predictions = classes[train_probabilities.argmax(1)]
    predictions = classes[valid_probabilities.argmax(1)]
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    print(f"Fitted {args.head} classifier on CPU with {train_features.shape[1]:,} embedding/hybrid features.", flush=True)
    summary = {
        "head": args.head, "C": args.c,
        "text_features": args.text_features if vectorizer is not None else "none",
        "feature_count": train_features.shape[1],
        "converged": converged,
        "train_balanced_accuracy": balanced_accuracy_score(train[args.label_column], train_predictions),
        "validation_balanced_accuracy": balanced_accuracy_score(valid[args.label_column], predictions),
        "gamma": args.gamma if args.head == "rbf" else None,
        "centroid_shrinkage": args.centroid_shrinkage if args.head == "centroid" else None,
        "ensemble_embedding_weight": args.ensemble_embedding_weight if args.head == "ensemble" else None,
    }
    print(json.dumps(summary, indent=2), flush=True)
    (args.output / "classifier_summary.json").write_text(json.dumps(summary, indent=2))
    # Pipeline components and explicit transform order allow later offline inference.
    joblib.dump({**models, "tfidf": vectorizer,
                 "embedding_mean": mean.numpy(), "embedding_scale": scale.numpy(),
                 "embedding_normalization": "standardize then row L2 normalize",
                 "tfidf_normalization": "row L2 normalize selected text feature block",
                 "text_features": args.text_features if vectorizer is not None else "none",
                 "feature_order": ("separate tfidf and embedding probability models" if args.head == "ensemble"
                                   else "tfidf then weighted embeddings" if vectorizer is not None
                                   else "embeddings"),
                 "embedding_weight": args.embedding_weight if vectorizer is not None else 1.0,
                 "pooling": "attention-mask mean", "max_length": args.max_length,
                 "head": args.head, "labels": classes.tolist()}, args.output / "classifier.joblib")
    pd.DataFrame(valid_probabilities, columns=classes,
                 index=pd.Index(valid["id"].to_numpy(), name="id")).to_csv(
                     args.output / "validation_probabilities.csv")
    config.update({"classifier_device": "cpu", "classifier_settings": summary,
                   "embedding_normalization": "standardize then row L2 normalize",
                   "selection": "fixed settings for this run; no validation-based hyperparameter selection"})
    return predictions


def train_encoder_classifier(train, valid, args, config):
    """Encode once with frozen weights; train only a linear PyTorch head."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Encoder mode requires torch and transformers; install them first.") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Request a GPU job or use --device cpu.")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f"Device: {device}. Loading frozen encoder {args.encoder}...", flush=True)
    options = dict(revision=args.revision, local_files_only=args.local_files_only,
                   cache_dir=str(Path("artifacts/hf_cache")), trust_remote_code=False)
    encoder = AutoModel.from_pretrained(args.encoder, **options).to(device)
    # Pin the tokenizer to the resolved model revision when downloading from the Hub.
    revision = getattr(encoder.config, "_commit_hash", None)
    if revision:
        options["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, **options)
    limit = min(tokenizer.model_max_length,
                getattr(encoder.config, "max_position_embeddings", args.max_length))
    if args.max_length > limit:
        raise ValueError(f"--max-length exceeds encoder/tokenizer limit {limit}.")
    encoder.requires_grad_(False)
    encoder.eval()

    def token_lengths(frame):
        lengths = []
        texts = frame[args.text_column].tolist()
        for offset in range(0, len(texts), args.batch_size):
            encoded = tokenizer(texts[offset:offset + args.batch_size], truncation=False,
                                padding=False, add_special_tokens=True)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        return np.asarray(lengths)

    train_lengths, valid_lengths = token_lengths(train), token_lengths(valid)
    token_report = {}
    for name, lengths in [("train", train_lengths), ("validation", valid_lengths)]:
        token_report[name] = {
            "rows": len(lengths), "max_tokens": int(lengths.max()),
            "median_tokens": float(np.median(lengths)),
            "p95_tokens": float(np.percentile(lengths, 95)),
            "above_256": int((lengths > 256).sum()),
            "above_512": int((lengths > 512).sum()),
            "truncated_rows": int((lengths > args.max_length).sum()),
            "truncated_fraction": float((lengths > args.max_length).mean()),
        }
        print(f"{name}: {token_report[name]['truncated_rows']}/{len(lengths)} passages "
              f"exceed {args.max_length} tokens; longest={lengths.max()}.", flush=True)
    (args.output / "token_lengths.json").write_text(json.dumps(token_report, indent=2))

    def encode(texts, name):
        chunks = []
        texts = texts.tolist()
        with torch.no_grad():
            for offset in range(0, len(texts), args.batch_size):
                batch = tokenizer(texts[offset:offset + args.batch_size], padding=True,
                                  truncation=True, max_length=args.max_length, return_tensors="pt")
                batch = {key: value.to(device) for key, value in batch.items()}
                hidden = encoder(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                chunks.append(pooled.cpu())
                if offset == 0 or (offset // args.batch_size + 1) % 25 == 0:
                    print(f"Encoding {name}: {min(offset + args.batch_size, len(texts))}/{len(texts)}", flush=True)
        return torch.cat(chunks)

    x_train = encode(train[args.text_column], "train")
    x_valid = encode(valid[args.text_column], "validation")
    # Save the frozen backbone and tokenizer so the classifier can be reused offline.
    encoder.save_pretrained(args.output / "encoder")
    tokenizer.save_pretrained(args.output / "encoder")
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Fit feature normalization on training embeddings only.
    mean = x_train.mean(dim=0)
    scale = x_train.std(dim=0, unbiased=False).clamp(min=1e-6)
    x_train = (x_train - mean) / scale
    x_valid = (x_valid - mean) / scale
    config.update({"resolved_device": str(device), "encoder_revision": revision,
                   "encoder_frozen": True, "pooling": "attention-mask mean"})
    config["packages"].update({name: importlib.metadata.version(name)
                               for name in ["torch", "transformers", "scipy"]})
    if args.head not in {"linear", "mlp"}:
        predictions = fit_frozen_classifier(x_train, x_valid, mean, scale,
                                             train, valid, args, config)
        save_error_analysis(valid, predictions, valid_lengths, args)
        return predictions
    mapping = {label: i for i, label in enumerate(LABELS)}
    y_train = torch.tensor(train[args.label_column].map(mapping).tolist(), dtype=torch.long)
    y_valid = valid[args.label_column].map(mapping).to_numpy()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    if args.head == "mlp":
        head = torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], args.hidden_size), torch.nn.ReLU(),
            torch.nn.Dropout(args.dropout), torch.nn.Linear(args.hidden_size, len(LABELS))).to(device)
    else:
        head = torch.nn.Linear(x_train.shape[1], len(LABELS)).to(device)
    weights = None
    if args.class_weight == "balanced":
        counts = torch.bincount(y_train, minlength=len(LABELS)).float()
        weights = (len(y_train) / (len(LABELS) * counts)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_score, stale, best_epoch = -1.0, 0, 0
    history, best_state = [], None

    def predict(features=x_valid):
        head.eval()
        with torch.no_grad():
            return torch.cat([head(batch.to(device)).argmax(dim=1).cpu()
                              for batch in features.split(args.batch_size)]).numpy()

    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        score = balanced_accuracy_score(y_valid, predict())
        train_score = balanced_accuracy_score(y_train.numpy(), predict(x_train))
        history.append({"epoch": epoch, "train_loss": total_loss / len(y_train),
                        "train_balanced_accuracy": train_score,
                        "validation_balanced_accuracy": score})
        print(f"Epoch {epoch}/{args.epochs}: loss={total_loss / len(y_train):.4f}, "
              f"train balanced accuracy={train_score:.2%}, "
              f"validation balanced accuracy={score:.2%}", flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping. Restoring epoch {best_epoch}.", flush=True)
            break
    head.load_state_dict(best_state)
    torch.save({"head_state_dict": best_state, "mean": mean, "scale": scale,
                "labels": LABELS, "input_dim": x_train.shape[1], "best_epoch": best_epoch,
                "pooling": "attention-mask mean", "max_length": args.max_length,
                "head": args.head, "hidden_size": args.hidden_size if args.head == "mlp" else None,
                "dropout": args.dropout if args.head == "mlp" else None},
               args.output / "classifier.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    config.update({"resolved_device": str(device), "encoder_revision": revision,
                   "encoder_frozen": True, "best_epoch": best_epoch,
                   "epochs_completed": len(history), "pooling": "attention-mask mean",
                   "head": args.head,
                   "weight_decay": args.weight_decay,
                   "selection": "best validation balanced accuracy; same validation used for early stopping"})
    config["packages"].update({name: importlib.metadata.version(name) for name in ["torch", "transformers"]})
    predictions = np.array(LABELS)[predict()]
    save_error_analysis(valid, predictions, valid_lengths, args)
    return predictions


def save_error_analysis(valid, predictions, valid_lengths, args):
    diagnostics = pd.DataFrame({
        "id": valid["id"].to_numpy(),
        "true_model": valid[args.label_column].to_numpy(), "predicted_model": predictions,
        "tokens": valid_lengths, "truncated": valid_lengths > args.max_length,
        "correct": predictions == valid[args.label_column].to_numpy(),
    })
    if "prompt_id" in valid:
        diagnostics["prompt_id"] = valid["prompt_id"].to_numpy()
    diagnostics.to_csv(args.output / "error_analysis.csv", index=False)
    # Accuracy by length is descriptive: class composition may differ between buckets.
    diagnostics["length_bucket"] = pd.cut(valid_lengths, [0, 128, 256, 512, np.inf],
                                          labels=["1-128", "129-256", "257-512", "513+"])
    diagnostics.groupby("length_bucket", observed=True)["correct"].agg(
        rows="size", accuracy="mean").to_csv(args.output / "accuracy_by_length.csv")


def diagnose_hybrid(run_dir, output_dir, device_name):
    """Audit an existing hybrid checkpoint using inference only; never fit models."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Diagnostic output directory must be new or empty.")
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("head") != "hybrid":
        raise ValueError("This diagnostic mode expects a saved hybrid run.")
    text_col, label_col = config["text_column"], config["label_column"]
    train_path, valid_path = Path(config["train"]), Path(config["validation"])
    train = load_data(train_path, text_col, label_col)
    valid = load_data(valid_path, text_col, label_col)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = {}

    def record(name, passed):
        checks[name] = bool(passed)
        (output_dir / "integrity.json").write_text(json.dumps(checks, indent=2))

    record("training_file_hash_matches", hashlib.sha256(train_path.read_bytes()).hexdigest() == config["data_sha256"])
    record("validation_file_hash_matches", hashlib.sha256(valid_path.read_bytes()).hexdigest() == config["validation_sha256"])
    for column in ["id", "prompt_id"]:
        record(f"disjoint_{column}", column in train and column in valid and
               not (set(train[column]) & set(valid[column])))
    record("disjoint_normalized_text", not (set(train[text_col].map(normalized)) &
                                             set(valid[text_col].map(normalized))))
    saved = pd.read_csv(run_dir / "validation_predictions.csv", dtype=str, keep_default_na=False)
    record("prediction_ids_match", saved.id.is_unique and set(saved.id) == set(valid.id))
    if not all(checks.values()):
        raise ValueError(f"Integrity check failed; inspect {output_dir / 'integrity.json'}")
    saved = saved.set_index("id").loc[valid.id]
    record("prediction_labels_match", np.array_equal(saved.true_model, valid[label_col]))
    bundle = joblib.load(run_dir / "classifier.joblib")
    clf, vectorizer = bundle["classifier"], bundle["tfidf"]
    record("classifier_labels_match", set(clf.classes_) == set(LABELS) and
           bundle["labels"] == clf.classes_.tolist())
    if not all(checks.values()):
        raise ValueError("Saved labels disagree with data; see integrity.json.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu or allocate a GPU.")
    device = torch.device("cuda" if device_name == "cuda" or
                          (device_name == "auto" and torch.cuda.is_available()) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "encoder", local_files_only=True)
    encoder = AutoModel.from_pretrained(run_dir / "encoder", local_files_only=True).to(device)
    encoder.eval().requires_grad_(False)
    print(f"Diagnostics: frozen encoder inference on {device}; no training.", flush=True)

    def embed(frame):
        arrays, lengths = [], []
        for start in range(0, len(frame), config["batch_size"]):
            texts = frame[text_col].iloc[start:start + config["batch_size"]].tolist()
            lengths.extend(map(len, tokenizer(texts, truncation=False)["input_ids"]))
            tokens = tokenizer(texts, padding=True, truncation=True,
                               max_length=bundle["max_length"], return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.no_grad():
                hidden = encoder(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                arrays.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
            if start % (25 * config["batch_size"]) == 0:
                print(f"Encoded {min(start + config['batch_size'], len(frame))}/{len(frame)}", flush=True)
        return np.concatenate(arrays), np.asarray(lengths)

    train_embeddings, train_lengths = embed(train)
    valid_embeddings, valid_lengths = embed(valid)
    record("embedding_mean_matches_training", np.allclose(
        train_embeddings.mean(0), bundle["embedding_mean"], atol=2e-5, rtol=2e-4))
    record("embedding_scale_matches_training", np.allclose(
        np.maximum(train_embeddings.std(0), 1e-6), bundle["embedding_scale"], atol=2e-5, rtol=2e-4))
    # Reconstruct training document frequencies without fitting a vectorizer.
    for name, transformer in vectorizer.transformer_list:
        matrix = transformer.transform(train[text_col])
        df = np.asarray((matrix > 0).sum(axis=0)).ravel()
        expected_idf = np.log((1 + len(train)) / (1 + df)) + 1
        record(f"{name}_vocabulary_seen_in_training", np.all(df > 0))
        record(f"{name}_idf_matches_training", np.allclose(transformer.idf_, expected_idf))
    text_train = normalize(vectorizer.transform(train[text_col]))
    text_valid = normalize(vectorizer.transform(valid[text_col]))

    def features(embeddings, text):
        emb = normalize((embeddings - bundle["embedding_mean"]) / bundle["embedding_scale"])
        return hstack([text, bundle["embedding_weight"] * csr_matrix(emb)], format="csr")

    train_pred = clf.predict(features(train_embeddings, text_train))
    valid_pred = clf.predict(features(valid_embeddings, text_valid))
    record("recomputed_predictions_match", np.array_equal(valid_pred, saved.predicted_model))
    summary = json.loads((run_dir / "classifier_summary.json").read_text())
    train_score = balanced_accuracy_score(train[label_col], train_pred)
    valid_score = balanced_accuracy_score(valid[label_col], valid_pred)
    record("training_score_matches", np.isclose(train_score, summary["train_balanced_accuracy"]))
    record("validation_score_matches", np.isclose(valid_score, summary["validation_balanced_accuracy"]))
    recalls = pd.DataFrame({"model": clf.classes_,
        "train_recall": recall_score(train[label_col], train_pred, labels=clf.classes_, average=None),
        "validation_recall": recall_score(valid[label_col], valid_pred, labels=clf.classes_, average=None)})
    recalls["gap"] = recalls.train_recall - recalls.validation_recall
    recalls["train_support"] = recalls.model.map(train[label_col].value_counts())
    recalls["validation_support"] = recalls.model.map(valid[label_col].value_counts())
    recalls.sort_values("gap", ascending=False).to_csv(output_dir / "per_class_gaps.csv", index=False)
    frames = []
    for split, frame, predictions, lengths in [("train", train, train_pred, train_lengths),
                                               ("validation", valid, valid_pred, valid_lengths)]:
        detail = frame[["id", "prompt_id", text_col, label_col]].copy()
        detail["split"], detail["prediction"], detail["tokens"] = split, predictions, lengths
        detail["correct"] = predictions == frame[label_col].to_numpy()
        detail["truncated"] = lengths > bundle["max_length"]
        detail["length_bucket"] = pd.cut(lengths, [0, 128, 256, 512, np.inf],
                                          labels=["1-128", "129-256", "257-512", "513+"])
        frames.append(detail)
    details = pd.concat(frames, ignore_index=True)
    details.to_csv(output_dir / "passage_predictions.csv", index=False)
    details.groupby(["split", "prompt_id"])["correct"].agg(rows="size", accuracy="mean").to_csv(
        output_dir / "by_prompt.csv")
    details.groupby(["split", "length_bucket"], observed=True)["correct"].agg(
        rows="size", accuracy="mean").to_csv(output_dir / "by_length.csv")
    details.groupby(["split", "length_bucket", label_col], observed=True)["correct"].agg(
        rows="size", recall="mean").to_csv(output_dir / "by_length_and_class.csv")
    # Inspect high positive coefficients separately for word and character features.
    names = vectorizer.get_feature_names_out()
    feature_rows = []
    for class_index, label in enumerate(clf.classes_):
        for prefix in ["word__", "char__"]:
            indices = np.flatnonzero(np.char.startswith(names.astype(str), prefix))
            order = indices[np.argsort(clf.coef_[class_index, indices])[-20:][::-1]]
            for index in order:
                present = text_train[:, index].toarray().ravel() > 0
                valid_present = text_valid[:, index].toarray().ravel() > 0
                prompts = train.loc[present, "prompt_id"].value_counts()
                feature_rows.append({"model": label, "feature": names[index],
                    "coefficient": float(clf.coef_[class_index, index]),
                    "training_documents": int(present.sum()),
                    "validation_documents": int(valid_present.sum()),
                    "training_prompt_count": len(prompts),
                    "largest_prompt_fraction": float(prompts.max() / present.sum()) if present.any() else 0,
                    "mean_train_score_contribution": float(text_train[:, index].mean() * clf.coef_[class_index, index])})
    pd.DataFrame(feature_rows).to_csv(output_dir / "top_text_features.csv", index=False)
    status = "PASS" if all(checks.values()) else "CHECK FAILED: inspect integrity.json before interpreting scores"
    (output_dir / "findings.md").write_text(
        f"# Hybrid diagnostics\n\nIntegrity: {status}.\n\n"
        f"Train balanced accuracy: {train_score:.2%}; validation: {valid_score:.2%}.\n\n"
        "See per_class_gaps.csv, by_prompt.csv and by_length_and_class.csv for error concentration. "
        "Prompt/length aggregate accuracy is descriptive; class composition can differ. "
        "passage_predictions.csv contains original text for manual error review and must stay private.\n\n"
        "top_text_features.csv lists positive word/character coefficients and prompt coverage. "
        "High concentration in few training prompts is a candidate topic shortcut, not proof. "
        "Coefficients alone are not causal explanations or full feature importance.\n\n"
        "IDF and embedding-statistic checks compare saved preprocessing with training-only quantities; "
        "they cannot prove all historical data handling was leakage-free. Near-duplicate leakage is not tested. "
        "A large gap supports overfitting but can also reflect unseen-prompt distribution differences. "
        "Confirm conclusions on other prompt folds. No model was fitted or changed.\n")
    print(f"Integrity: {status}\nTrain: {train_score:.2%}; validation: {valid_score:.2%}\nSaved to {output_dir}")


def analyze_providers(run_dir, output_dir):
    """Collapse saved top-1 model predictions to providers; no model inference."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Provider report output directory must be new or empty.")
    prefixes = {"gpt-": "OpenAI", "claude-": "Anthropic", "gemini-": "Google",
                "qwen": "Qwen", "glm-": "ZAI", "kimi-": "Moonshot AI"}
    mapping = {label: next(provider for prefix, provider in prefixes.items()
                           if label.startswith(prefix)) for label in LABELS}
    providers = list(prefixes.values())
    predictions_path = run_dir / "validation_predictions.csv"
    data = pd.read_csv(predictions_path, dtype=str, keep_default_na=False)
    required = {"id", "true_model", "predicted_model"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError("Expected nonempty predictions with id, true_model, predicted_model.")
    if data.id.duplicated().any() or data.id.str.strip().eq("").any():
        raise ValueError("Prediction IDs must be unique and nonblank.")
    for col in ["true_model", "predicted_model"]:
        unknown = set(data[col]) - set(mapping)
        if unknown:
            raise ValueError(f"Unknown model labels in {col}: {sorted(unknown)}")
    data["true_provider"] = data.true_model.map(mapping)
    data["predicted_provider"] = data.predicted_model.map(mapping)
    exact = data.true_model == data.predicted_model
    same = data.true_provider == data.predicted_provider
    data["error_type"] = np.where(exact, "exact_model_correct",
                                   np.where(same, "wrong_model_same_provider", "wrong_provider"))
    rows = []
    for provider in providers:
        subset = data[data.true_provider == provider]
        if subset.empty:
            continue
        counts = subset.error_type.value_counts()
        n_exact = int(counts.get("exact_model_correct", 0))
        n_within = int(counts.get("wrong_model_same_provider", 0))
        provider_correct = n_exact + n_within
        rows.append({"provider": provider, "rows": len(subset),
                     "candidate_models": sum(p == provider for p in mapping.values()),
                     "provider_recall": provider_correct / len(subset),
                     "exact_model_accuracy": n_exact / len(subset),
                     "exact_correct": n_exact, "wrong_model_same_provider": n_within,
                     "wrong_provider": int(counts.get("wrong_provider", 0)),
                     "exact_accuracy_given_correct_provider":
                         n_exact / provider_correct if provider_correct else None})
    summary = {
        "rows": len(data),
        "exact_model_balanced_accuracy": balanced_accuracy_score(data.true_model, data.predicted_model),
        "provider_accuracy": float(same.mean()),
        "provider_balanced_accuracy": balanced_accuracy_score(data.true_provider, data.predicted_provider),
        "providers_present": sorted(data.true_provider.unique().tolist()),
        "exact_correct": int(exact.sum()),
        "wrong_model_same_provider": int((same & ~exact).sum()),
        "wrong_provider": int((~same).sum()),
        "fraction_of_model_errors_within_provider": float((same & ~exact).sum() / (~exact).sum())
            if (~exact).sum() else None,
        "prediction_file_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "method": "Map saved top-1 model prediction to provider; no probability aggregation or retraining.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "per_provider.csv", index=False)
    data.to_csv(output_dir / "provider_predictions.csv", index=False)
    matrix = confusion_matrix(data.true_provider, data.predicted_provider, labels=providers)
    pd.DataFrame(matrix, index=providers, columns=providers).to_csv(output_dir / "confusion_counts.csv")
    normalized_matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                                  out=np.zeros_like(matrix, dtype=float),
                                  where=matrix.sum(axis=1, keepdims=True) != 0)
    pd.DataFrame(normalized_matrix, index=providers, columns=providers).to_csv(
        output_dir / "confusion_normalized.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(normalized_matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(providers)), yticks=range(len(providers)),
           xticklabels=providers, yticklabels=providers,
           xlabel="Predicted provider", ylabel="True provider", title="Provider recall / confusion")
    for i in range(len(providers)):
        for j in range(len(providers)):
            ax.text(j, i, f"{normalized_matrix[i, j]:.0%}", ha="center", va="center",
                    color="white" if normalized_matrix[i, j] > .5 else "black")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion.png", dpi=150)
    plt.close(fig)
    (output_dir / "README.md").write_text(
        "Provider scores collapse the existing 18-label predictions; this is not a separately trained "
        "provider classifier or the competition metric. Provider balanced accuracy weights each observed "
        "provider equally, while ordinary accuracy gives more weight to providers with more passages. "
        "Qwen and Moonshot each have one candidate model, so their within-provider model error is "
        "necessarily zero. Conditional accuracy uses only rows with the correct predicted provider "
        "and is not the performance of a classifier given the true provider. Compare identical "
        "validation IDs across runs. No predictions or models were modified.\n")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"Saved provider analysis to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", type=Path, help="Audit an existing hybrid run without training.")
    parser.add_argument("--provider-analysis", type=Path,
                        help="Analyze provider errors from any run's saved validation predictions.")
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--validation", type=Path,
                        help="Prepared validation CSV. Use the supplied split without resplitting.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="model")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--explore-only", action="store_true")
    parser.add_argument("--model", choices=["tfidf", "encoder"], default="tfidf")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--encoder", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--head",
                        choices=["linear", "logistic", "hybrid", "centroid", "rbf", "mlp", "ensemble"],
                        default="linear", help="Classifier trained on frozen encoder features.")
    parser.add_argument("--c", type=float, default=1.0, help="Logistic inverse L2 regularization strength.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Logistic solver iteration limit.")
    parser.add_argument("--embedding-weight", type=float, default=1.0,
                        help="Relative embedding block weight in hybrid mode.")
    parser.add_argument("--text-features", choices=["both", "word", "char"], default="both",
                        help="TF-IDF feature types for the TF-IDF baseline or hybrid head.")
    parser.add_argument("--gamma", default="scale",
                        help="RBF SVM gamma: scale, auto, or a positive number.")
    parser.add_argument("--svc-cache-mb", type=float, default=1024,
                        help="RBF SVM kernel cache size in MB.")
    parser.add_argument("--centroid-shrinkage", type=float,
                        help="Optional nonnegative nearest-centroid shrink threshold.")
    parser.add_argument("--hidden-size", type=int, default=256, help="MLP hidden units.")
    parser.add_argument("--dropout", type=float, default=0.5, help="MLP dropout probability.")
    parser.add_argument("--ensemble-embedding-weight", type=float, default=0.5,
                        help="Embedding probability weight; TF-IDF receives 1 minus this value.")
    args = parser.parse_args()
    if args.provider_analysis is not None:
        if args.diagnose is not None:
            parser.error("Choose either --provider-analysis or --diagnose.")
        if args.output == Path("artifacts/baseline"):
            args.output = args.provider_analysis / "providers"
        analyze_providers(args.provider_analysis, args.output)
        return
    if args.diagnose is not None:
        if args.output == Path("artifacts/baseline"):
            args.output = args.diagnose / "diagnostics"
        diagnose_hybrid(args.diagnose, args.output, args.device)
        return
    if not np.isfinite(args.c) or args.c <= 0 or args.max_iter < 1:
        parser.error("--c must be positive and finite; --max-iter must be positive.")
    if not np.isfinite(args.embedding_weight) or args.embedding_weight <= 0:
        parser.error("--embedding-weight must be positive and finite.")
    if args.model != "encoder" and args.head != "linear":
        parser.error("Nonlinear/logistic/ensemble heads require --model encoder.")
    try:
        args.gamma = float(args.gamma) if args.gamma not in {"scale", "auto"} else args.gamma
    except ValueError:
        parser.error("--gamma must be scale, auto, or a positive number.")
    if isinstance(args.gamma, float) and (not np.isfinite(args.gamma) or args.gamma <= 0):
        parser.error("Numeric --gamma must be positive and finite.")
    if args.svc_cache_mb <= 0:
        parser.error("--svc-cache-mb must be positive.")
    if args.centroid_shrinkage is not None and args.centroid_shrinkage < 0:
        parser.error("--centroid-shrinkage must be nonnegative.")
    if args.hidden_size < 1 or not 0 <= args.dropout < 1:
        parser.error("--hidden-size must be positive and --dropout must be in [0, 1).")
    if not 0 <= args.ensemble_embedding_weight <= 1:
        parser.error("--ensemble-embedding-weight must be in [0, 1].")
    if min(args.epochs, args.patience, args.batch_size, args.max_length) < 1 or args.learning_rate <= 0:
        parser.error("Epochs, patience, batch size, max length and learning rate must be positive.")
    if args.weight_decay < 0:
        parser.error("Weight decay must be nonnegative.")
    if args.model == "tfidf" and args.device == "cuda":
        parser.error("TF-IDF/LinearSVC supports CPU only. Use --model encoder for CUDA.")
    if not args.train.is_file():
        parser.error(f"Missing labeled training file: {args.train}")
    if args.validation is not None and not args.validation.is_file():
        parser.error(f"Missing validation file: {args.validation}")
    if args.validation is not None and args.explore_only:
        parser.error("Use --validation for training, not --explore-only.")
    if not 0 <= args.validation_fold < args.folds:
        parser.error("--validation-fold must be in [0, --folds).")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is nonempty; choose a new run directory.")
    df = load_data(args.train, args.text_column, args.label_column)
    valid = None
    if args.validation is not None:
        valid = load_data(args.validation, args.text_column, args.label_column)
        for column in ["id", "prompt_id"]:
            if column not in df or column not in valid:
                parser.error(f"Prepared training and validation files must contain {column}.")
            if set(df[column]) & set(valid[column]):
                parser.error(f"Training/validation overlap in {column}; recreate the split.")
        if set(df[args.text_column].map(normalized)) & set(valid[args.text_column].map(normalized)):
            parser.error("Normalized duplicate passages cross the training/validation split.")
    groups = make_groups(df, args.text_column)
    args.output.mkdir(parents=True, exist_ok=True)
    explore(df, args.text_column, args.label_column, groups, args.output)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "data_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]},
        "labels": LABELS,
        "feature_settings": {"word_ngrams": [1, 2], "char_ngrams": [3, 5],
                             "word_max_features": 100_000, "char_max_features": 150_000,
                             "min_df": 2, "sublinear_tf": True, "C": 1.0},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split_mode": "supplied CSVs" if args.validation is not None else "generated folds",
        "validation_sha256": (hashlib.sha256(args.validation.read_bytes()).hexdigest()
                              if args.validation is not None else None),
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    if args.explore_only:
        print(f"Exploration saved to {args.output}")
        return
    if valid is None:
        folds = make_folds(df, groups, args.label_column, args.folds, args.seed)
        pd.DataFrame({"id": df.id, "group": groups, "fold": folds}).to_csv(
            args.output / "splits.csv", index=False)
        train, valid = df.loc[folds != args.validation_fold], df.loc[folds == args.validation_fold]
    else:
        train = df
        # Record exactly which supplied rows were used, retaining fold metadata if present.
        columns = [c for c in ["id", "prompt_id", "group_id", "fold"]
                   if c in train and c in valid]
        pd.concat([train[columns].assign(partition="train"),
                   valid[columns].assign(partition="validation")], ignore_index=True).to_csv(
                       args.output / "splits.csv", index=False)
    print(f"Training on {len(train):,} passages; validating on {len(valid):,}.", flush=True)
    start = time.perf_counter()
    if args.model == "encoder":
        predictions = train_encoder_classifier(train, valid, args, config)
    else:
        print("Device: cpu. Fitting TF-IDF and LinearSVC on training data only...", flush=True)
        model = build_model(args.seed, None if args.class_weight == "none" else "balanced", args.text_features)
        model.fit(train[args.text_column], train[args.label_column])
        predictions = model.predict(valid[args.text_column])
        joblib.dump(model, args.output / "model.joblib")
        config["resolved_device"] = "cpu"
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    recalls = recall_score(valid[args.label_column], predictions, labels=LABELS, average=None)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(valid[args.label_column], predictions),
        "per_class_recall": dict(zip(LABELS, recalls.tolist())),
        "train_rows": len(train), "validation_rows": len(valid),
        "fit_and_predict_seconds": time.perf_counter() - start,
        "scope": ("local validation from supplied CSV; not a leaderboard score"
                  if args.validation is not None else
                  f"local validation fold {args.validation_fold}; not a leaderboard score"),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"id": valid.id, "true_model": valid[args.label_column],
                  "predicted_model": predictions}).to_csv(args.output / "validation_predictions.csv", index=False)
    matrix = confusion_matrix(valid[args.label_column], predictions, labels=LABELS)
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(args.output / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix / matrix.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(18), yticks=range(18), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted model", ylabel="True model", title="Validation confusion matrix (row normalized)")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.2%}")
    print(f"Saved model, metrics, predictions and confusion matrix to {args.output}")


if __name__ == "__main__":
    main()
