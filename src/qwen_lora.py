"""Supervised Qwen LoRA attribution experiment; user launches training explicitly."""
import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', type=Path, default=Path('artifacts/pretrained/qwen3_embedding_06b'))
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--gradient-accumulation', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.gradient_accumulation) < 1:
        parser.error('Epochs, patience and batch settings must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('Learning rate must be finite and positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new or empty output directory')

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import balanced_accuracy_score, recall_score
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, get_linear_schedule_with_warmup, set_seed)
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError('Install PEFT first: python -m pip install "peft==0.21.0"') from exc
    from src.qwen_experiment import validate_frames

    train = pd.read_csv(args.train, dtype=str, keep_default_na=False)
    valid = pd.read_csv(args.validation, dtype=str, keep_default_na=False)
    validate_frames(train, valid)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8:
        raise RuntimeError('This experiment requires an Ampere-or-newer GPU, such as A100, with native BF16')
    probe = torch.ones(1, device='cuda') + 1
    torch.cuda.synchronize()
    del probe
    set_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    labels = sorted(train.model.unique())
    mapping = {label: i for i, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.encoder, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=mapping,
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        attn_implementation='sdpa', pad_token_id=tokenizer.pad_token_id,
    )
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj'], modules_to_save=['score'], bias='none',
    )).to('cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.print_trainable_parameters()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not any('lora_' in name for name, _ in trainable) or not any('score' in name for name, _ in trainable):
        raise RuntimeError('Expected trainable adapters and classification head')
    unexpected = [name for name, _ in trainable if 'lora_' not in name and 'score' not in name]
    if unexpected:
        raise RuntimeError(f'Unexpected unfrozen base parameters: {unexpected}')

    lengths = {}
    class Passages(Dataset):
        def __init__(self, frame, name):
            self.items = []
            token_lengths = []
            for text, label in zip(frame.text, frame.model):
                item = tokenizer(text, truncation=False)
                token_lengths.append(len(item['input_ids']))
                item = {key: value[:512] for key, value in item.items()}
                item['labels'] = mapping[label]
                self.items.append(item)
            lengths[name] = {'max_tokens': max(token_lengths), 'truncated': sum(n > 512 for n in token_lengths)}
        def __len__(self):
            return len(self.items)
        def __getitem__(self, index):
            return dict(self.items[index])

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(Passages(train, 'train'), batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(42), collate_fn=collator)
    valid_loader = DataLoader(Passages(valid, 'validation'), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collator)
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=args.learning_rate, weight_decay=0.01)
    steps = math.ceil(len(train_loader) / args.gradient_accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = args.encoder / 'download_provenance.json'
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'seed': 42, 'rank': 8, 'alpha': 16, 'dropout': 0.1, 'max_length': 512,
              'target_modules': ['q_proj', 'v_proj'], 'labels': labels,
              'effective_batch_size': args.batch_size * args.gradient_accumulation,
              'trainable_parameters': sum(p.numel() for _, p in trainable),
              'train_sha256': hashlib.sha256(args.train.read_bytes()).hexdigest(),
              'validation_sha256': hashlib.sha256(args.validation.read_bytes()).hexdigest(),
              'encoder_provenance': json.loads(provenance.read_text()) if provenance.exists() else None,
              'packages': {p: version(p) for p in ['torch', 'transformers', 'peft', 'numpy', 'scikit-learn']},
              'status': 'running', 'scope': 'prompt-grouped local validation; not a Kaggle score'}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    (args.output / 'token_lengths.json').write_text(json.dumps(lengths, indent=2))
    best, stale, history = -1, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, count_window = 0.0, 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            count = len(batch['labels'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss; stop rather than save a broken model')
            (loss * count).backward()
            total_loss += loss.detach().item() * count
            count_window += count
            if (step + 1) % args.gradient_accumulation == 0 or step + 1 == len(train_loader):
                for _, p in trainable:
                    if p.grad is not None:
                        p.grad.div_(count_window)
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                count_window = 0
        model.eval()
        chunks, valid_loss = [], 0.0
        with torch.inference_mode():
            for batch in valid_loader:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model(**batch)
                chunks.append(result.logits.float().softmax(-1).cpu().numpy())
                valid_loss += result.loss.item() * len(batch['labels'])
        probabilities = np.concatenate(chunks)
        if not np.isfinite(probabilities).all():
            raise RuntimeError('Nonfinite validation probabilities')
        predicted = np.asarray(labels)[probabilities.argmax(1)]
        score = balanced_accuracy_score(valid.model, predicted)
        record = {'epoch': epoch, 'train_loss': total_loss / len(train),
                  'validation_loss': valid_loss / len(valid), 'validation_balanced_accuracy': score,
                  'learning_rate': scheduler.get_last_lr()[0]}
        history.append(record)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        print(f'Epoch {epoch}: train loss={record["train_loss"]:.4f}; validation loss={record["validation_loss"]:.4f}; balanced accuracy={score:.2%}', flush=True)
        if score > best:
            best, stale = score, 0
            model.save_pretrained(args.output / 'best_adapter', safe_serialization=True)
            tokenizer.save_pretrained(args.output / 'best_adapter')
            model.config.save_pretrained(args.output / 'best_adapter')
            pd.DataFrame({'id': valid.id, 'true_model': valid.model, 'predicted_model': predicted}).to_csv(args.output / 'validation_predictions.csv', index=False)
            pd.DataFrame(probabilities, columns=labels, index=pd.Index(valid.id, name='id')).to_csv(args.output / 'validation_probabilities.csv')
            metrics = {'balanced_accuracy': score, 'best_epoch': epoch,
                       'validation_loss': record['validation_loss'], 'validation_rows': len(valid),
                       'per_class_recall': dict(zip(labels, recall_score(valid.model, predicted, labels=labels, average=None).tolist())),
                       'scope': config['scope']}
            (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        else:
            stale += 1
        if stale >= args.patience:
            print('Early stopping; best adapter retained.', flush=True)
            break
    config['status'] = 'complete'
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'Best validation balanced accuracy: {best:.2%}. Results: {args.output}')
    print('best_adapter requires the original Qwen encoder; it is not a standalone full model.')


if __name__ == '__main__':
    main()
