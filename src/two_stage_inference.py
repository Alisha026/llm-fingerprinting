"""Offline inference for our saved two-stage checkpoint; no fitting of classifiers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class AdapterLinear(nn.Module):
    """Unmerged standard LoRA, preserving the training-time FP32 adapter arithmetic."""
    def __init__(self, base, a, b, scale):
        super().__init__()
        self.base = base
        self.register_buffer('a', a.to(device=base.weight.device, dtype=torch.float32))
        self.register_buffer('b', b.to(device=base.weight.device, dtype=torch.float32))
        self.scale = scale

    def forward(self, x):
        result = self.base(x)
        delta = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return (result + delta).to(result.dtype)


def attach_adapters(model, state, config):
    if any(config.get(k) for k in ['use_dora', 'use_rslora', 'rank_pattern', 'alpha_pattern', 'fan_in_fan_out']):
        raise ValueError('Only our standard, fixed-rank LoRA is supported')
    consumed = set()
    for key, a in state.items():
        if not key.endswith('.lora_A.weight'):
            continue
        name = key.removeprefix('base_model.model.').removesuffix('.lora_A.weight')
        b_key = key.replace('.lora_A.weight', '.lora_B.weight')
        base = model.get_submodule(name)
        parent, _, child = name.rpartition('.')
        wrapped = AdapterLinear(base, a, state[b_key], config['lora_alpha'] / config['r'])
        setattr(model.get_submodule(parent), child, wrapped)
        consumed.update([key, b_key])
    if consumed != set(state) or not consumed:
        raise ValueError('Adapter keys missing or unsupported')


def infer_groups(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    if len(texts) < 2:
        return np.ones(len(texts), dtype=int)
    try:
        x = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit_transform(texts)
    except ValueError as exc:
        if 'empty vocabulary' not in str(exc):
            raise
        return np.arange(len(texts))
    d = np.clip(1 - (x @ x.T).toarray(), 0, None)
    np.fill_diagonal(d, 0)
    return fcluster(linkage(squareform(d, checks=False), method='average'), .85, criterion='distance')


def centered_features(z, groups):
    result = z.copy()
    eligible = np.zeros(len(z), dtype=bool)
    for group in np.unique(groups):
        rows = groups == group
        if 2 <= rows.sum() <= 18:
            result[rows] -= result[rows].mean(0)
            eligible |= rows
    return result, eligible


class Predictor:
    def __init__(self, checkpoint, base_path):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('GPU required for the 8B submission')
        torch.set_num_threads(4)
        checkpoint = Path(checkpoint)
        meta = json.loads((checkpoint / 'meta.json').read_text())
        self.cfg, self.labels = meta['cfg'], meta['labels']
        assert len(self.labels) == len(set(self.labels)) == 18
        self.layers = self.cfg['layers']
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint / 'tokenizer', local_files_only=True, padding_side='right')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cfg = AutoConfig.from_pretrained(base_path, local_files_only=True)
        assert cfg.model_type == 'qwen3' and cfg.hidden_size == 4096 and cfg.num_hidden_layers == 36
        cfg.num_hidden_layers, cfg.use_cache = self.cfg['keep'], False
        if getattr(cfg, 'layer_types', None) is not None:
            cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        self.model = AutoModel.from_pretrained(base_path, config=cfg, local_files_only=True,
                                               torch_dtype=dtype, attn_implementation='sdpa').to('cuda').eval()
        self.model.norm = nn.Identity()
        adapter = checkpoint / 'adapter'
        attach_adapters(self.model, load_file(str(adapter / 'adapter_model.safetensors')),
                        json.loads((adapter / 'adapter_config.json').read_text()))
        heads = load_file(str(checkpoint / 'heads.safetensors'))
        self.norms = nn.ModuleList([nn.LayerNorm(4096) for _ in self.layers]).to('cuda').eval()
        self.norms.load_state_dict({k.removeprefix('norms.'): v for k,v in heads.items() if k.startswith('norms.')})
        self.plain = nn.Linear(4096 * len(self.layers), 18).to('cuda').eval()
        self.twin = nn.Linear(4096 * len(self.layers), 18).to('cuda').eval()
        self.plain.load_state_dict({k.removeprefix('plain.'): v for k,v in heads.items() if k.startswith('plain.')})
        self.twin.load_state_dict({k.removeprefix('centered.'): v for k,v in heads.items() if k.startswith('centered.')})
        print('Loaded checkpoint; device:', torch.cuda.get_device_name(0), 'dtype:', dtype, flush=True)

    @torch.inference_mode()
    def predict(self, texts, batch_size=3):
        if not len(texts):
            raise ValueError('Empty input')
        ids = self.tokenizer(list(texts), add_special_tokens=False, truncation=True,
                             max_length=self.cfg['max_len'])['input_ids']
        ids = [x + [self.tokenizer.eos_token_id] for x in ids]
        chunks = []
        for start in range(0, len(ids), batch_size):
            enc = self.tokenizer.pad({'input_ids': ids[start:start+batch_size]}, return_tensors='pt').to('cuda')
            hs = self.model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            if not all(torch.isfinite(hs[layer]).all() for layer in self.layers):
                self.model.to(torch.bfloat16)
                for module in self.model.modules():
                    if isinstance(module, AdapterLinear):
                        module.a = module.a.float(); module.b = module.b.float()
                hs = self.model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            mask = enc.attention_mask.unsqueeze(-1).float()
            mask[:, 0] = 0
            z = torch.cat([norm((hs[layer].float() * mask).sum(1) / mask.sum(1).clamp(min=1))
                           for layer, norm in zip(self.layers, self.norms)], -1)
            if not torch.isfinite(z).all():
                raise RuntimeError('Nonfinite features')
            chunks.append(z.cpu().numpy())
            if start % 90 == 0:
                print(f'Encoded {min(start+batch_size,len(ids))}/{len(ids)}', flush=True)
        z = np.concatenate(chunks)
        zc, eligible = centered_features(z, infer_groups(list(texts)))
        lp = self.plain(torch.from_numpy(z).to('cuda')).log_softmax(-1)
        tlp = self.twin(torch.from_numpy(zc).to('cuda')).log_softmax(-1)
        mixed = lp.clone()
        rows = torch.from_numpy(eligible).to('cuda')
        mixed[rows] += self.cfg['centered_log_weight'] * tlp[rows]
        probabilities = mixed.softmax(-1).cpu().numpy()
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
            raise RuntimeError('Invalid probabilities')
        return probabilities


def run(bundle, base_path, test_path, output='/kaggle/working/submission.parquet'):
    bundle = Path(bundle)
    model = Predictor(bundle / 'official', base_path)
    valid = pd.read_csv(bundle / 'validation.csv', dtype={'id': str}, keep_default_na=False)
    reference = pd.read_csv(bundle / 'reference_probabilities.csv', dtype={'id': str}).set_index('id')
    assert reference.index.tolist() == valid.id.tolist() and list(reference.columns) == model.labels
    prob = model.predict(valid.text.tolist())
    saved = reference.to_numpy()
    error = float(np.abs(prob - saved).max())
    agreement = float((prob.argmax(1) == saved.argmax(1)).mean())
    accuracy = float((np.asarray(model.labels)[prob.argmax(1)] == valid.model).mean())
    reference_accuracy = float((np.asarray(model.labels)[saved.argmax(1)] == valid.model).mean())
    report = {'maximum_probability_difference': error, 'prediction_agreement': agreement,
              'local_validation_accuracy': accuracy, 'reference_accuracy': reference_accuracy}
    print('Checkpoint parity:', report, flush=True)
    if error > .08 or agreement < .97 or abs(accuracy - report['reference_accuracy']) > .015:
        raise RuntimeError('Checkpoint parity failed; do not submit this runtime')
    Path('/kaggle/working/parity.json').write_text(json.dumps(report, indent=2))
    frame = pd.read_csv(test_path, dtype={'id': str}, keep_default_na=False)
    if not {'id','text'}.issubset(frame) or frame.empty or frame.id.duplicated().any() or frame.id.eq('').any():
        raise ValueError('Invalid test rows')
    pred = model.predict(frame.text.astype(str).tolist())
    sub = pd.DataFrame({'id':frame.id, 'model':np.asarray(model.labels)[pred.argmax(1)]})
    sub.to_parquet(output, index=False)
    print('Saved',output,'rows:',len(sub), flush=True)
