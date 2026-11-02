"""Fine-tune a pretrained encoder for 18-class model attribution."""

import argparse
import hashlib
import json
import math
import random
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.data_loaders import load_train


class PassageDataset(Dataset):
    """Tokenize passages lazily so the CSV remains the source of truth."""

    def __init__(self, frame, tokenizer, label_to_id, max_length):
        self.ids = frame["id"].tolist()
        self.texts = frame["text"].tolist()
        self.labels = frame["model"].map(label_to_id).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index], truncation=True, max_length=self.max_length
        )
        encoded["labels"] = self.labels[index]
        return encoded


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this job.")
    return torch.device(
        "cuda" if requested == "cuda" or
        (requested == "auto" and torch.cuda.is_available()) else "cpu"
    )


def validate_split(train, valid):
    """Reject obvious leakage and schema errors before loading a model."""
    if set(train["model"]) != set(valid["model"]):
        raise ValueError("Training and validation must contain the same model labels.")
    for column in ["id", "prompt_id"]:
        overlap = set(train[column]) & set(valid[column])
        if overlap:
            raise ValueError(f"Training and validation overlap in {column}: {len(overlap)} values.")
    normalize_text = lambda value: " ".join(value.casefold().split())
    if set(train.text.map(normalize_text)) & set(valid.text.map(normalize_text)):
        raise ValueError("A normalized duplicate passage crosses the split.")


def evaluate(model, loader, device, use_amp):
    model.eval()
    losses, labels, predictions, probabilities = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                output = model(**batch)
            probs = output.logits.float().softmax(dim=-1)
            losses.append(output.loss.item() * len(batch["labels"]))
            labels.append(batch["labels"].cpu().numpy())
            predictions.append(probs.argmax(dim=-1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
    labels = np.concatenate(labels)
    predictions = np.concatenate(predictions)
    probabilities = np.concatenate(probabilities)
    return {
        "loss": sum(losses) / len(labels),
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
    }


def save_evaluation(result, valid, labels, output_dir, elapsed_seconds, best_epoch):
    true_names = np.asarray(labels)[result["labels"]]
    predicted_names = np.asarray(labels)[result["predictions"]]
    recalls = recall_score(true_names, predicted_names, labels=labels, average=None)
    metrics = {
        "balanced_accuracy": result["balanced_accuracy"],
        "validation_loss": result["loss"],
        "per_class_recall": dict(zip(labels, recalls.tolist())),
        "validation_rows": len(valid),
        "best_epoch": best_epoch,
        "training_seconds": elapsed_seconds,
        "scope": "local prompt-grouped validation; not a leaderboard score",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({
        "id": valid.id.to_numpy(),
        "prompt_id": valid.prompt_id.to_numpy(),
        "true_model": true_names,
        "predicted_model": predicted_names,
    }).to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(result["probabilities"], columns=labels,
                 index=pd.Index(valid.id, name="id")).to_csv(
                     output_dir / "validation_probabilities.csv")
    matrix = confusion_matrix(true_names, predicted_names, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv")
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float),
                           where=row_totals != 0)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted model", ylabel="True model",
           title="Fine-tuned encoder: row-normalized validation confusion")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics


def blend_saved_probabilities(first_run, second_run, first_weight, output_dir):
    """Blend two runs on identical validation IDs without model inference."""
    if not 0 <= first_weight <= 1:
        raise ValueError("--first-weight must be in [0, 1].")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Blend output must be a new or empty directory.")

    def load_run(run_dir):
        prediction_path = run_dir / "validation_predictions.csv"
        probability_path = run_dir / "validation_probabilities.csv"
        if not prediction_path.is_file() or not probability_path.is_file():
            raise FileNotFoundError(f"Missing validation predictions/probabilities in {run_dir}")
        predictions = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)
        probabilities = pd.read_csv(probability_path, dtype={"id": str})
        required = {"id", "true_model", "predicted_model"}
        if not required.issubset(predictions.columns):
            raise ValueError(f"Invalid validation_predictions.csv in {run_dir}")
        if predictions.id.duplicated().any() or probabilities.id.duplicated().any():
            raise ValueError(f"Duplicate validation IDs in {run_dir}")
        if set(predictions.id) != set(probabilities.id):
            raise ValueError(f"Prediction/probability IDs disagree in {run_dir}")
        return (predictions.set_index("id").sort_index(),
                probabilities.set_index("id").sort_index(),
                prediction_path, probability_path)

    first_pred, first_prob, first_pred_path, first_prob_path = load_run(first_run)
    second_pred, second_prob, second_pred_path, second_prob_path = load_run(second_run)
    if not first_pred.index.equals(second_pred.index):
        raise ValueError("The two runs do not contain identical validation IDs.")
    if not np.array_equal(first_pred.true_model, second_pred.true_model):
        raise ValueError("The two runs disagree on true labels.")
    if set(first_prob.columns) != set(second_prob.columns):
        raise ValueError("The two runs have different probability-label columns.")
    labels = sorted(first_prob.columns.tolist())
    for name, probabilities in [("first", first_prob), ("second", second_prob)]:
        values = probabilities[labels].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} probabilities contain invalid values.")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError(f"{name} probability rows do not sum to one.")
    blended = (first_weight * first_prob[labels].to_numpy() +
               (1 - first_weight) * second_prob[labels].to_numpy())
    predicted = np.asarray(labels)[blended.argmax(axis=1)]
    truth = first_pred.true_model.to_numpy()
    recalls = recall_score(truth, predicted, labels=labels, average=None)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "per_class_recall": dict(zip(labels, recalls.tolist())),
        "rows": len(truth), "first_run": str(first_run), "second_run": str(second_run),
        "first_weight": first_weight, "second_weight": 1 - first_weight,
        "first_balanced_accuracy": balanced_accuracy_score(truth, first_pred.predicted_model),
        "second_balanced_accuracy": balanced_accuracy_score(truth, second_pred.predicted_model),
        "scope": "fixed-weight blend on local validation; not a leaderboard score",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"id": first_pred.index, "true_model": truth,
                  "predicted_model": predicted}).to_csv(
                      output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(blended, columns=labels, index=first_pred.index).to_csv(
        output_dir / "validation_probabilities.csv", index_label="id")
    matrix = confusion_matrix(truth, predicted, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / "confusion_matrix.csv")
    config = {
        **metrics,
        "first_predictions_sha256": hashlib.sha256(first_pred_path.read_bytes()).hexdigest(),
        "first_probabilities_sha256": hashlib.sha256(first_prob_path.read_bytes()).hexdigest(),
        "second_predictions_sha256": hashlib.sha256(second_pred_path.read_bytes()).hexdigest(),
        "second_probabilities_sha256": hashlib.sha256(second_prob_path.read_bytes()).hexdigest(),
        "selection": "weight fixed before comparing folds; no fitting or model inference",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved blended predictions to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--encoder", default="artifacts/encoder_v1/encoder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--blend-runs", nargs=2, type=Path, metavar=("FIRST", "SECOND"),
                        help="Blend two saved validation probability files; do not train.")
    parser.add_argument("--first-weight", type=float, default=0.5,
                        help="Probability weight for the first run in --blend-runs.")
    args = parser.parse_args()

    if args.blend_runs is not None:
        blend_saved_probabilities(*args.blend_runs, args.first_weight, args.output)
        return
    if args.train is None or args.validation is None:
        parser.error("Fine-tuning requires --train and --validation.")

    positive = [args.max_length, args.epochs, args.patience, args.batch_size,
                args.gradient_accumulation, args.learning_rate, args.max_grad_norm]
    if any(value <= 0 for value in positive):
        parser.error("Lengths, epochs, patience, batch sizes, learning rate and grad norm must be positive.")
    if args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1 or args.min_delta < 0:
        parser.error("Use nonnegative weight decay/min-delta and warmup ratio in [0, 1).")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative.")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Output must be a new or empty directory.")

    train, valid = load_train(args.train), load_train(args.validation)
    validate_split(train, valid)
    labels = sorted(train.model.unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    tokenizer = AutoTokenizer.from_pretrained(
        args.encoder, local_files_only=args.local_files_only, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.encoder, num_labels=len(labels), label2id=label_to_id, id2label=id_to_label,
        local_files_only=args.local_files_only, trust_remote_code=False,
        ignore_mismatched_sizes=True,
    ).to(device)
    model_limit = min(tokenizer.model_max_length,
                      getattr(model.config, "max_position_embeddings", args.max_length))
    if args.max_length > model_limit:
        raise ValueError(f"--max-length {args.max_length} exceeds model limit {model_limit}.")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Some model parameters are frozen; full fine-tuning was requested.")

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if use_amp else None)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        PassageDataset(train, tokenizer, label_to_id, args.max_length),
        batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, collate_fn=collator, pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        PassageDataset(valid, tokenizer, label_to_id, args.max_length),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collator, pin_memory=device.type == "cuda",
    )
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * args.warmup_ratio), total_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {**vars(args), "train": str(args.train), "validation": str(args.validation),
              "output": str(args.output), "encoder": str(args.encoder),
              "resolved_device": str(device), "mixed_precision": use_amp,
              "effective_batch_size": args.batch_size * args.gradient_accumulation,
              "train_rows": len(train), "validation_rows": len(valid),
              "train_prompts": train.prompt_id.nunique(),
              "validation_prompts": valid.prompt_id.nunique(),
              "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
              "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
              "labels": labels,
              "packages": {name: version(name) for name in
                           ["torch", "transformers", "numpy", "pandas", "scikit-learn"]}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"Full fine-tuning on {device}; AMP={use_amp}; effective batch={config['effective_batch_size']}.")
    print(f"Training rows={len(train):,}; validation rows={len(valid):,}; total updates={total_updates}.")

    best_score, best_epoch, stale = -1.0, 0, 0
    history, started = [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            running_loss += output.loss.detach().item() * len(batch["labels"])
            should_update = ((step + 1) % args.gradient_accumulation == 0 or
                             step + 1 == len(train_loader))
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        result = evaluate(model, valid_loader, device, use_amp)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train),
            "validation_loss": result["loss"],
            "validation_balanced_accuracy": result["balanced_accuracy"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch}/{args.epochs}: train loss={epoch_record['train_loss']:.4f}, "
              f"validation loss={result['loss']:.4f}, "
              f"balanced accuracy={result['balanced_accuracy']:.2%}", flush=True)
        if result["balanced_accuracy"] > best_score + args.min_delta:
            best_score, best_epoch, stale = result["balanced_accuracy"], epoch, 0
            model.save_pretrained(args.output / "best_model", safe_serialization=True)
            tokenizer.save_pretrained(args.output / "best_model")
        else:
            stale += 1
        if stale >= args.patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
            break

    best_model = AutoModelForSequenceClassification.from_pretrained(
        args.output / "best_model", local_files_only=True, trust_remote_code=False
    ).to(device)
    final_result = evaluate(best_model, valid_loader, device, use_amp)
    metrics = save_evaluation(final_result, valid, labels, args.output,
                              time.perf_counter() - started, best_epoch)
    print(json.dumps(metrics, indent=2))
    print(f"Best checkpoint: epoch {best_epoch}; balanced accuracy={metrics['balanced_accuracy']:.2%}")


if __name__ == "__main__":
    main()
