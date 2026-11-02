"""User-run approximation of the public two-stage, prompt-centered Qwen recipe.

`check` validates inputs without model loading or fitting. `train` is explicit.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.qwen_experiment import validate_frames


def triplets(frame, seed):
    """Cover every row, filling each triplet with three distinct authors.

    Groups with fewer than three authors are rejected, never silently discarded.
    """
    rng = np.random.default_rng(seed)
    result = []
    for _, group in frame.groupby('prompt_id', sort=True):
        if group.model.nunique() < 3:
            raise ValueError('Each pool prompt must have at least 3 distinct authors')
        by_author = {label: list(rng.permutation(rows.index.to_numpy()))
                     for label, rows in group.groupby('model', sort=True)}
        original = {label: rows[:] for label, rows in by_author.items()}
        while any(by_author.values()):
            active = [label for label, rows in by_author.items() if rows]
            selected = list(rng.choice(active, min(3, len(active)), replace=False))
            chosen = [by_author[label].pop() for label in selected]
            if len(selected) < 3:
                fillers = [label for label in original if label not in selected]
                for label in rng.choice(fillers, 3 - len(selected), replace=False):
                    chosen.append(int(rng.choice(original[label])))
            result.append(chosen)
    rng.shuffle(result)
    return result


def inferred_groups(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    x = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts)
    d = np.clip(1 - (x @ x.T).toarray(), 0, None)
    np.fill_diagonal(d, 0)
    return fcluster(linkage(squareform(d, checks=False), method='average'), .85, criterion='distance')


def centered(features, groups):
    result = features.copy()
    eligible = np.zeros(len(groups), dtype=bool)
    for group in np.unique(groups):
        rows = groups == group
        if 2 <= rows.sum() <= 18:
            result[rows] -= result[rows].mean(0)
            eligible |= rows
    return result, eligible


def inputs(args):
    frames = []
    for path in (args.pool, args.train, args.validation):
        if not path.is_file():
            raise ValueError(f'Missing input: {path}')
        frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype=str, keep_default_na=False)
        required = ['id', 'prompt_id', 'text', 'model']
        if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
            raise ValueError(f'{path}: need non-null id, prompt_id, text, model')
        frames.append(frame[required].astype(str).reset_index(drop=True))
    pool, train, valid = frames
    validate_frames(train, valid)
    validate_frames(pool, valid)
    normal = lambda s: ' '.join(s.casefold().split())
    if set(pool.text.map(normal)) & set(train.text.map(normal)):
        raise ValueError('Public pool duplicates official training text; deduplicate before training')
    for frame in (pool, train):
        triplets(frame, args.seed)
    cfg_path = args.encoder / 'config.json'
    if not cfg_path.is_file():
        raise ValueError(f'Missing Qwen3-8B-Base model: {args.encoder}')
    cfg = json.loads(cfg_path.read_text())
    if cfg.get('model_type') != 'qwen3' or cfg.get('hidden_size') != 4096 or cfg.get('num_hidden_layers') != 36:
        raise ValueError('Expected Qwen3 8B architecture, not the existing 0.6B embedding model')
    provenance = args.encoder / 'download_provenance.json'
    if not provenance.is_file() or json.loads(provenance.read_text()).get('repo_id') != 'Qwen/Qwen3-8B-Base':
        raise ValueError('Missing Qwen3-8B-Base download provenance')
    index_path = args.encoder / 'model.safetensors.index.json'
    if not index_path.is_file():
        raise ValueError('Missing weight shard index')
    index = json.loads(index_path.read_text())
    for name in set(index['weight_map'].values()):
        shard = args.encoder / name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f'Missing or empty weight shard: {name}')
    for name in ['tokenizer.json', 'tokenizer_config.json']:
        if not (args.encoder / name).is_file():
            raise ValueError(f'Missing {name}')
    report = args.pool.parent / 'preparation.json'
    if not report.is_file():
        raise ValueError('Missing pool preparation.json; use src.prepare_public_pool first')
    expected = json.loads(report.read_text()).get('pool_sha256')
    if expected != hashlib.sha256(args.pool.read_bytes()).hexdigest():
        raise ValueError('Pool checksum does not match preparation.json')
    return frames


def validate_full_refit(full, train, valid, checkpoint, encoder):
    """Require the exact official union and a matching saved public checkpoint."""
    columns = ['id', 'prompt_id', 'text', 'model']
    expected = pd.concat([train, valid], ignore_index=True)[columns].sort_values('id').reset_index(drop=True)
    if not set(columns).issubset(full) or full[columns].isna().any().any():
        raise ValueError('Full-data file is missing required values')
    actual = full[columns].astype(str).sort_values('id').reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError('Full data must exactly equal the original official train/validation union')
    if checkpoint.name != 'public':
        raise ValueError('Start from the public stage, not the official-stage checkpoint')
    for name in ['meta.json', 'heads.safetensors', 'adapter/adapter_config.json',
                 'adapter/adapter_model.safetensors', 'tokenizer/tokenizer.json']:
        if not (checkpoint / name).is_file():
            raise ValueError(f'Missing public checkpoint file: {name}')
    meta = json.loads((checkpoint / 'meta.json').read_text())
    provenance = json.loads((encoder / 'download_provenance.json').read_text())
    if meta['labels'] != sorted(full.model.unique()) or meta['cfg']['encoder_provenance'] != provenance:
        raise ValueError('Public checkpoint labels or base revision do not match')
    if meta['cfg']['keep'] != 24 or meta['cfg']['layers'] != [2,4,6,8,12,16,20,24]:
        raise ValueError('Unexpected public checkpoint architecture')
    return full[columns].astype(str).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', 'train'])
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--encoder', type=Path, default=Path('artifacts/pretrained/qwen3_8b_base'))
    p.add_argument('--train', type=Path, default=Path('data/processed/train.csv'))
    p.add_argument('--validation', type=Path, default=Path('data/processed/validation.csv'))
    p.add_argument('--output', type=Path, default=Path('artifacts/two_stage_8b_fold0'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--full-data', type=Path, help='Official union for a refit without held-out validation')
    p.add_argument('--from-public', type=Path, help='Saved public-stage checkpoint; requires --full-data')
    args = p.parse_args()
    if bool(args.full_data) != bool(args.from_public):
        p.error('--full-data and --from-public must be supplied together')
    try:
        pool, train, valid = inputs(args)
        if args.full_data:
            full = pd.read_csv(args.full_data, dtype=str, keep_default_na=False)
            train = validate_full_refit(full, train, valid, args.from_public, args.encoder)
            triplets(train, args.seed)
            args.train, args.validation, valid = args.full_data, None, None
    except ValueError as exc:
        p.error(str(exc))
    print(f'Inputs checked: pool={len(pool)}, train={len(train)}, validation={len(valid) if valid is not None else 0}', flush=True)
    if args.full_data:
        print('Full-data refit: reuse public checkpoint; 2 official epochs; no held-out score.', flush=True)
    if not args.full_data:
        print('Validation is exploratory: the upstream public pool used all-official-data style filters.', flush=True)
    if args.action == 'check':
        return
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output must be new or empty')
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers import AutoConfig, AutoModel, AutoTokenizer, set_seed, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, PeftModel
    from sklearn.metrics import balanced_accuracy_score
    from importlib.metadata import version
    from safetensors.torch import save_file, load_file
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Training requires a CUDA GPU with BF16 support and sufficient memory')
    set_seed(args.seed)
    torch.set_num_threads(4)
    print(f'GPU: {torch.cuda.get_device_name(0)}; loading the 24-layer trunk', flush=True)
    labels = sorted(train.model.unique())
    mapping = {label: i for i, label in enumerate(labels)}
    layers = [2, 4, 6, 8, 12, 16, 20, 24]
    tok = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True, padding_side='right')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(args.encoder, local_files_only=True)
    cfg.num_hidden_layers, cfg.use_cache = 24, False
    if getattr(cfg, 'layer_types', None) is not None:
        cfg.layer_types = cfg.layer_types[:24]
    base = AutoModel.from_pretrained(args.encoder, config=cfg, local_files_only=True,
                                    dtype=torch.bfloat16, attn_implementation='sdpa')
    base.norm = nn.Identity()
    if args.from_public:
        trunk = PeftModel.from_pretrained(base, args.from_public / 'adapter', is_trainable=True,
                                          local_files_only=True).to('cuda')
    else:
        trunk = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                             target_modules='all-linear', bias='none')).to('cuda')
    trunk.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    norms = nn.ModuleList([nn.LayerNorm(4096) for _ in layers]).to('cuda')
    plain, twin = nn.Linear(4096 * 8, 18).to('cuda'), nn.Linear(4096 * 8, 18).to('cuda')
    if args.from_public:
        state = load_file(str(args.from_public / 'heads.safetensors'))
        for prefix, module in [('norms.', norms), ('plain.', plain), ('centered.', twin)]:
            module.load_state_dict({k.removeprefix(prefix): v for k,v in state.items() if k.startswith(prefix)})
        print('Loaded public-stage adapters, norms, plain head and centered head.', flush=True)
    adapters = [v for v in trunk.parameters() if v.requires_grad]
    heads = list(norms.parameters()) + list(plain.parameters()) + list(twin.parameters())
    print(f'Trainable parameters: {sum(v.numel() for v in adapters + heads):,}', flush=True)

    def tokens(frame):
        ids = tok(frame.text.tolist(), add_special_tokens=False, truncation=True, max_length=447)['input_ids']
        return [x + [tok.eos_token_id] for x in ids]

    def features(ids):
        enc = tok.pad({'input_ids': ids}, return_tensors='pt').to('cuda')
        hs = trunk(**enc, output_hidden_states=True, use_cache=False).hidden_states
        mask = enc.attention_mask.unsqueeze(-1).float()
        mask[:, 0] = 0
        z = torch.cat([nm((hs[layer].float() * mask).sum(1) / mask.sum(1).clamp(min=1))
                       for layer, nm in zip(layers, norms)], -1)
        if not torch.isfinite(z).all():
            raise RuntimeError('Nonfinite features')
        return z

    args.output.mkdir(parents=True, exist_ok=True)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'status': 'running', 'epochs': 3, 'stage_epochs': [1, 2], 'keep': 24, 'layers': layers,
              'max_len': 447, 'rank': 16, 'alpha': 32, 'dropout': .05,
              'microbatch_triplets': 1, 'triplets_per_update': 6, 'weight_decay': .01,
              'learning_rates': [[1e-4, 1e-3], [5e-5, 5e-4]], 'warmup_fraction': .1,
              'centered_log_weight': .35, 'labels': labels,
              'packages': {name: version(name) for name in ['torch', 'transformers', 'peft']},
              'scope': 'exploratory prompt-grouped local validation; upstream pool used official-wide style filters; not a Kaggle score',
              'encoder_provenance': json.loads((args.encoder / 'download_provenance.json').read_text()),
              'pool_preparation': json.loads((args.pool.parent / 'preparation.json').read_text()),
              'training_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': 'Upstream filtering used all official rows. External prompt IDs do not establish semantic independence.',
              **{f'{key}_sha256': hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
                 for key in ['pool', 'train', 'validation'] if getattr(args, key) is not None}}
    if args.full_data:
        config.update(epochs=2, stage_epochs=[2], learning_rates=[[5e-5, 5e-4]],
                      scope='full official-data refit; no held-out validation; not a Kaggle score',
                      public_checkpoint_sha256={str(path.relative_to(args.from_public)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                for path in sorted(args.from_public.rglob('*')) if path.is_file()},
                      optimizer_restart=True, rng_restart_seed=args.seed)
    def write(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + '\n')
    write('config.json', config)
    history = []
    valid_ids = tokens(valid) if valid is not None else None
    groups = inferred_groups(valid.text.tolist()) if valid is not None else None

    def evaluate(stage):
        trunk.eval(); norms.eval(); plain.eval(); twin.eval()
        with torch.inference_mode():
            z = torch.cat([features(valid_ids[i:i+3]).cpu() for i in range(0, len(valid_ids), 3)]).numpy()
            zc, eligible = centered(z, groups)
            lp = plain(torch.from_numpy(z).to('cuda')).log_softmax(-1)
            tlp = twin(torch.from_numpy(zc).to('cuda')).log_softmax(-1)
            mixed = lp.clone()
            mixed[torch.from_numpy(eligible).to('cuda')] += .35 * tlp[torch.from_numpy(eligible).to('cuda')]
            probabilities = mixed.softmax(-1).cpu().numpy()
            plain_prob = lp.exp().cpu().numpy()
        pred = np.asarray(labels)[probabilities.argmax(1)]
        for name, values in [('plain', plain_prob), ('centered_blend', probabilities)]:
            pd.DataFrame(values, columns=labels, index=pd.Index(valid.id, name='id')).to_csv(args.output / f'{stage}_{name}_probabilities.csv')
        pd.DataFrame({'id': valid.id, 'true_model': valid.model, 'predicted_model': pred}).to_csv(args.output / f'{stage}_predictions.csv', index=False)
        return {'balanced_accuracy': balanced_accuracy_score(valid.model, pred),
                'plain_balanced_accuracy': balanced_accuracy_score(valid.model, np.asarray(labels)[plain_prob.argmax(1)]),
                'eligible_centered_rows': int(eligible.sum()), 'scope': config['scope']}

    stages = [('official', train, 2, 5e-5)] if args.full_data else [('public', pool, 1, 1e-4), ('official', train, 2, 5e-5)]
    for stage, frame, epochs, lr in stages:
        ids = tokens(frame)
        y = torch.tensor(frame.model.map(mapping).to_numpy(), device='cuda')
        counts = np.bincount(frame.model.map(mapping), minlength=18)
        weights = torch.tensor(len(frame) / (18 * counts), dtype=torch.float32, device='cuda') if stage == 'public' else None
        schedules = [triplets(frame, args.seed + (0 if stage == 'public' else 100) + e) for e in range(epochs)]
        steps = sum(math.ceil(len(t) / 6) for t in schedules)
        print(f'Stage {stage}: {len(frame)} rows, {epochs} epochs, {steps} optimizer updates', flush=True)
        optimizer = torch.optim.AdamW([{'params': adapters, 'lr': lr}, {'params': heads, 'lr': lr * 10}], weight_decay=.01)
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(.1 * steps), steps)
        for epoch, batches in enumerate(schedules, 1):
            started = time.monotonic()
            trunk.train(); norms.train(); plain.train(); twin.train()
            total, seen = 0., 0
            for start in range(0, len(batches), 6):
                window = batches[start:start+6]
                denominator = sum(float(weights[y[t]].sum()) if weights is not None else 3 for t in window)
                optimizer.zero_grad(set_to_none=True)
                for t in window:
                    z = features([ids[i] for i in t])
                    loss = F.cross_entropy(plain(z), y[t], weight=weights, reduction='sum')
                    loss = loss + F.cross_entropy(twin(z - z.mean(0, keepdim=True)), y[t], weight=weights, reduction='sum')
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    (loss / denominator).backward()
                    total += loss.detach().item(); seen += 3
                nn.utils.clip_grad_norm_(adapters + heads, 1., error_if_nonfinite=True)
                optimizer.step(); scheduler.step()
                update = start // 6 + 1
                if update == 1 or update % 25 == 0 or start + 6 >= len(batches):
                    print(f'{stage} epoch {epoch}/{epochs}: update {update}/{math.ceil(len(batches)/6)}, '
                          f'loss/sample={total / seen:.4f}, elapsed={time.monotonic() - started:.0f}s', flush=True)
            history.append({'stage': stage, 'epoch': epoch, 'train_loss_sum_per_sample': total / seen,
                            'sample_presentations': seen, 'optimizer_steps': math.ceil(len(batches)/6),
                            'elapsed_seconds': time.monotonic() - started})
            write('history.json', history)
            print(history[-1], flush=True)
        folder = args.output / stage
        trunk.save_pretrained(folder / 'adapter')
        tok.save_pretrained(folder / 'tokenizer')
        state = {f'norms.{k}': v for k, v in norms.state_dict().items()}
        state.update({f'plain.{k}': v for k, v in plain.state_dict().items()})
        state.update({f'centered.{k}': v for k, v in twin.state_dict().items()})
        save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(folder / 'heads.safetensors'))
        (folder / 'meta.json').write_text(json.dumps({'labels': labels, 'cfg': config}, indent=2))
        metrics = evaluate(stage) if valid is not None else {'status': 'complete', 'scope': config['scope'],
                                                           'training_rows': len(train), 'epochs': 2}
        write(f'{stage}_metrics.json', metrics)
        print(stage, metrics, flush=True)
    write('metrics.json', metrics)  # Fixed final epoch; no validation checkpoint selection.
    config['status'] = 'complete'
    write('config.json', config)
    from src.record_experiments import record
    record()


if __name__ == '__main__':
    main()
