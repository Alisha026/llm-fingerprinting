"""Prepare the downloaded public pool for grouped training; never fits a model."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from src.prepare_external_pilot import normalized, overlap_groups

SOURCE = 'https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora'


def prepare(pool, official):
    required = {'id', 'prompt_id', 'text', 'model', 'source', 'prompt_key'}
    if not required.issubset(pool.columns) or pool[list(required)].isna().any().any():
        raise ValueError('Missing public-pool columns or values')
    pool = pool.copy()
    for column in required:
        pool[column] = pool[column].astype(str)
        if pool[column].str.strip().eq('').any():
            raise ValueError(f'Empty {column}')
    if pool.id.duplicated().any() or not set(pool.model).issubset(set(official.model)):
        raise ValueError('Duplicate IDs or unexpected model labels')
    if pool.groupby('prompt_id').prompt_key.nunique().max() != 1:
        raise ValueError('A prompt ID maps to multiple upstream prompt keys')
    reasons = {}
    # Remove entire groups when an answer matches any official answer. Validation
    # labels do not select data or thresholds; text is used only for exclusion.
    probe = pool.assign(prompt_group_id=pool.prompt_id)
    bad_groups, matches = overlap_groups(probe, official)
    bad_groups |= set(pool.prompt_id) & set(official.prompt_id)
    bad_groups |= set(pool.loc[pool.id.isin(official.id), 'prompt_id'])
    reasons.update({row_id: 'official_answer_or_id_overlap_group'
                    for row_id in pool.loc[pool.prompt_id.isin(bad_groups), 'id']})
    pool = pool.loc[~pool.prompt_id.isin(bad_groups)].copy()
    pool['_normalized'] = pool.text.map(normalized)
    conflicts = pool.groupby('_normalized').model.nunique()
    bad = pool._normalized.isin(conflicts[conflicts > 1].index)
    reasons.update({i: 'conflicting_label_duplicate' for i in pool.loc[bad, 'id']})
    pool = pool.loc[~bad].sort_values('id').copy()
    duplicate = pool.duplicated('_normalized')
    reasons.update({i: 'normalized_duplicate' for i in pool.loc[duplicate, 'id']})
    pool = pool.loc[~duplicate].copy()
    authors = pool.groupby('prompt_id').model.nunique()
    small = pool.prompt_id.isin(authors[authors < 3].index)
    reasons.update({i: 'fewer_than_three_authors' for i in pool.loc[small, 'id']})
    pool = pool.loc[~small].drop(columns='_normalized').reset_index(drop=True)
    if set(pool.model) != set(official.model):
        raise ValueError('Preparation removed a model class')
    return pool, reasons, {'official_answer_matches': matches, 'excluded_overlap_groups': len(bad_groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/external/two_stage_source/pool_v1.parquet'))
    parser.add_argument('--official', type=Path, default=Path('data/raw/train.csv'))
    parser.add_argument('--output', type=Path, default=Path('data/public_pool_v1'))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new or empty output directory; existing preparation is immutable')
    raw = pd.read_parquet(args.source)
    official = pd.read_csv(args.official, dtype=str, keep_default_na=False)
    pool, reasons, audit = prepare(raw, official)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / 'pool.parquet'
    pool.to_parquet(output, index=False)
    pd.DataFrame(sorted(reasons.items()), columns=['id', 'reason']).to_csv(args.output / 'excluded_rows.csv', index=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        'source_url': SOURCE, 'license_as_reported_by_kaggle': 'CC-BY-SA-4.0',
        'source_file': str(args.source), 'source_sha256': sha(args.source),
        'official_file': str(args.official), 'official_sha256': sha(args.official),
        'pool_sha256': sha(output), 'preparation_script_sha256': sha(Path(__file__)),
        'input_rows': len(raw), 'output_rows': len(pool), 'prompt_groups': pool.prompt_id.nunique(),
        'class_counts': pool.model.value_counts().sort_index().to_dict(),
        'source_counts': pool.source.value_counts().sort_index().to_dict(),
        'excluded_reasons': pd.Series(list(reasons.values()), dtype=str).value_counts().to_dict(),
        'duplicate_prompt_model_rows_retained': int(pool.duplicated(['prompt_id', 'model']).sum()),
        **audit,
        'limitations': [
            'Upstream pool was reportedly filtered using class-specific punctuation statistics from all official labeled rows. Local evaluation is exploratory, not strictly untouched validation.',
            'Official prompt text is unavailable; exact and 5-word shingle exclusions do not prove semantic prompt independence.',
            'Labels and shared-prompt IDs are publisher supplied; model identities and grouping have not been independently authenticated.',
            'The supplied pool is already filtered; its upstream filtering cannot be reversed locally.',
            'Rows from prompts with fewer than three authors are excluded; other repeated prompt/author responses are retained and sampled into distinct-author triplets.',
        ],
    }
    (args.output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
