"""Load competition data and prepare reproducible training/validation splits."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd


# Resolve defaults relative to the project, regardless of the working directory.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _load_csv(path: str | Path, required_columns: list[str]) -> pd.DataFrame:
    """Read a CSV and check that required fields and unique IDs are present."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Preserve identifiers as strings and literal text such as 'NA'.
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError(f"{path.name} contains no rows.")

    for column in required_columns:
        if data[column].str.strip().eq("").any():
            raise ValueError(f"{path.name}: '{column}' contains empty values.")
    if data["id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate IDs.")

    return data


def load_train(path: str | Path = DATA_DIR / "train.csv") -> pd.DataFrame:
    """Load labeled passages, retaining prompt IDs for grouped validation."""
    return _load_csv(path, ["id", "prompt_id", "text", "model"])


def load_test(path: str | Path = DATA_DIR / "test.csv") -> pd.DataFrame:
    """Load unlabeled passages for inference."""
    return _load_csv(path, ["id", "text"])


def prepare_data(
    train_path: Path,
    output_dir: Path,
    n_folds: int = 5,
    validation_fold: int = 0,
    seed: int = 42,
) -> dict:
    """Keep shared prompts and duplicate texts together, then save assignments."""
    if n_folds < 2 or not 0 <= validation_fold < n_folds:
        raise ValueError("Use at least two folds and a validation fold in [0, n_folds).")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output path must be a new or empty directory: {output_dir}")

    from baseline import LABELS, make_folds, make_groups

    data = load_train(train_path)
    observed = set(data["model"])
    if observed != set(LABELS):
        raise ValueError(
            f"Unexpected labels: {sorted(observed - set(LABELS))}; "
            f"missing labels: {sorted(set(LABELS) - observed)}"
        )
    reserved = {"label_id", "group_id", "fold"} & set(data.columns)
    if reserved:
        raise ValueError(f"Input already contains preparation columns: {sorted(reserved)}")

    # Reuse the original baseline's split algorithm and row order for consistency.
    groups = make_groups(data, "text")
    folds = make_folds(data, groups, "model", n_folds, seed)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    prepared = data.assign(
        label_id=data["model"].map(label_to_id), group_id=groups, fold=folds
    )
    train = prepared.loc[prepared["fold"] != validation_fold].copy()
    valid = prepared.loc[prepared["fold"] == validation_fold].copy()
    if set(train["prompt_id"]) & set(valid["prompt_id"]):
        raise ValueError("A prompt appears in both training and validation.")
    if set(train["group_id"]) & set(valid["group_id"]):
        raise ValueError("A duplicate-text/prompt group crosses the split.")

    summary = {
        "source": str(train_path.resolve()),
        "source_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "seed": seed,
        "n_folds": n_folds,
        "validation_fold": validation_fold,
        "rows": len(prepared),
        "train_rows": len(train),
        "validation_rows": len(valid),
        "train_prompts": int(train["prompt_id"].nunique()),
        "validation_prompts": int(valid["prompt_id"].nunique()),
        "fold_sizes": prepared["fold"].value_counts().sort_index().to_dict(),
        "packages": {name: version(name) for name in ["pandas", "numpy", "scikit-learn"]},
        "text_processing": "Original text retained; normalization used only for grouping.",
    }
    counts = pd.DataFrame({
        "train": train["model"].value_counts(),
        "validation": valid["model"].value_counts(),
    }).reindex(LABELS).fillna(0).astype(int)
    counts.index.name = "model"

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    valid.to_csv(output_dir / "validation.csv", index=False)
    prepared[["id", "prompt_id", "group_id", "fold"]].to_csv(
        output_dir / "splits.csv", index=False
    )
    counts.to_csv(output_dir / "class_counts.csv")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_to_id, indent=2), encoding="utf-8"
    )
    (output_dir / "preparation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Save train/validation splits.")
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR.parent / "processed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.prepare:
        train = load_train(args.train)
        print(f"Training rows: {len(train):,}")
        print(f"Model labels: {train['model'].nunique()}")
        print(f"Prompt IDs: {train['prompt_id'].nunique()}")
        print(f"Training columns: {list(train.columns)}")
        return
    summary = prepare_data(args.train, args.output, args.folds, args.validation_fold, args.seed)
    print(f"Training: {summary['train_rows']:,} passages / {summary['train_prompts']} prompts")
    print(f"Validation: {summary['validation_rows']:,} passages / {summary['validation_prompts']} prompts")
    print(f"Saved prepared data and split assignments to {args.output}")
    print("No model training or TF-IDF fitting performed.")


if __name__ == "__main__":
    main()
