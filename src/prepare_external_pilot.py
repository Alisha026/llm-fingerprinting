"""Prepare an auditable, source-controlled augmentation pilot. Never trains."""
import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    import unicodedata
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def shingles(text):
    words = normalized(text).split()
    return {tuple(words[i:i+5]) for i in range(max(0, len(words)-4))}


def overlap_groups(external, official):
    """Exact/punctuation-normalized and word-shingle similarity, not semantics."""
    exact = set(official.text.map(normalized))
    refs = [shingles(text) for text in official.text]
    index = defaultdict(set)
    for i, grams in enumerate(refs):
        for gram in grams:
            index[gram].add(i)
    groups = set()
    matched = 0
    for row in external.itertuples():
        norm, grams = normalized(row.text), shingles(row.text)
        hit = norm in exact
        possible = set()
        for gram in grams:
            possible.update(index.get(gram, ()))
        for i in possible:
            intersection = len(grams & refs[i])
            union = len(grams | refs[i])
            containment = intersection / max(1, min(len(grams), len(refs[i])))
            if intersection / max(1, union) >= 0.8 or containment >= 0.9:
                hit = True
                break
        if hit:
            matched += 1
            groups.add(row.prompt_group_id)
    return groups, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=Path('data/external/response_corpus'))
    parser.add_argument('--output', type=Path, default=Path('data/external_pilot_v1'))
    parser.add_argument('--per-model', type=int, default=40)
    args = parser.parse_args()
    if args.per_model < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error('Use a positive per-model cap and a new or empty output folder')
    root = args.corpus
    files = ['external_short_wide.parquet', 'external_cell_metadata.parquet', 'external_cell_provenance.parquet']
    manifest = json.loads((root / 'manifest.json').read_text())
    for name in files:
        if digest(root / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
    official = pd.read_csv('data/raw/train.csv', dtype=str, keep_default_na=False)
    labels = sorted(official.model.unique())
    wide = pd.read_parquet(root / files[0], use_threads=False)
    long = wide.melt(id_vars=['id', 'prompt'], value_vars=labels,
                     var_name='model', value_name='text').dropna(subset=['text'])
    metadata = pd.read_parquet(root / files[1], use_threads=False)
    metadata = metadata.loc[metadata['view'].eq('short')]
    long = long.merge(metadata, on=['id', 'model'], validate='one_to_one')
    cols = ['cell_id_short','source','source_url','raw_model','returned_model','license',
            'license_unknown','output_terms_review','model_evidence_class','no_train',
            'possible_truncation','truncation_metadata_missing','possible_reasoning_in_answer',
            'text_only_suitable','mechanically_eligible','prompt_exact','submitted_context_sha256','source_row_id']
    prov = pd.read_parquet(root / files[2], columns=cols, use_threads=False)
    prov = prov.loc[prov.source.eq('speechmap') & prov.cell_id_short.notna()].copy()
    for flag in ['no_train','possible_truncation','truncation_metadata_missing','possible_reasoning_in_answer','license_unknown']:
        prov = prov.loc[prov[flag].eq(False)]
    for flag in ['text_only_suitable','mechanically_eligible','prompt_exact']:
        prov = prov.loc[prov[flag].eq(True)]
    prov = prov.loc[prov.model_evidence_class.eq('request_and_returned_model') &
                    prov.returned_model.fillna('').str.strip().ne('') &
                    prov.raw_model.fillna('').str.strip().ne('')]
    pool = long.loc[long.cell_id.isin(prov.cell_id_short)].copy()
    for flag in ['no_train','possible_truncation','possible_reasoning_in_answer','text_shared_across_labels','license_unknown']:
        pool = pool.loc[pool[flag].eq(False)]
    pool = pool.loc[pool.text.str.split().str.len().between(40,250) & pool.prompt_exact.eq(True)]
    pool['normalized_text'] = pool.text.map(normalized)
    # Exclude cross-label identical text even if source flags missed it.
    conflicts = pool.groupby('normalized_text').model.nunique()
    pool = pool.loc[~pool.normalized_text.isin(conflicts[conflicts > 1].index)]
    pool = pool.sort_values('cell_id').drop_duplicates(['model','normalized_text'])
    groups, matched = overlap_groups(pool, official)
    pool = pool.loc[~pool.prompt_group_id.isin(groups)].copy()
    # One answer per model and prompt; prioritize prompts shared across labels.
    pool = pool.drop_duplicates(['model','prompt_group_id'])
    coverage = pool.groupby('prompt_group_id').model.nunique()
    pool['coverage'] = pool.prompt_group_id.map(coverage)
    pool['order'] = pool.prompt_group_id.map(lambda x: hashlib.sha256(('42:'+x).encode()).hexdigest())
    pool = pool.sort_values(['coverage','order','cell_id'], ascending=[False,True,True])
    counts = pool.model.value_counts().reindex(labels,fill_value=0)
    cap = min(args.per_model, int(counts.min()))
    if cap < 1:
        raise ValueError(f'No balanced all-label pilot possible: {counts.to_dict()}')
    selected = pool.groupby('model',sort=True).head(cap).copy()
    selected['original_cell_id'] = selected.cell_id
    selected['id'] = 'external:speechmap:' + selected.cell_id
    selected['prompt_id'] = 'external:' + selected.prompt_group_id
    selected['source'] = 'speechmap'
    columns = ['id','prompt_id','text','model']
    if not selected.id.is_unique:
        raise ValueError('Nonunique external IDs')
    if len(selected) != cap * len(labels):
        raise ValueError('Unbalanced selection')
    args.output.mkdir(parents=True,exist_ok=True)
    selected[columns+['source']].to_csv(args.output/'external.csv',index=False)
    # Keep all provenance occurrences for selected cells, including review flags.
    prov.loc[prov.cell_id_short.isin(selected.cell_id)].to_parquet(args.output/'provenance.parquet',index=False)
    selected.drop(columns=['text','normalized_text']).to_parquet(args.output/'selection_metadata.parquet',index=False)
    split_checks = {}
    for fold, directory in [(0,Path('data/processed')),(1,Path('data/processed_fold1'))]:
        train = pd.read_csv(directory/'train.csv',dtype=str,keep_default_na=False)
        valid = pd.read_csv(directory/'validation.csv',dtype=str,keep_default_na=False)
        combined = pd.concat([train[columns].assign(source='official'), selected[columns+['source']]],ignore_index=True)
        if not combined.id.is_unique or set(combined.id)&set(valid.id) or set(combined.prompt_id)&set(valid.prompt_id):
            raise ValueError('Split ID/group collision')
        if set(combined.text.map(normalized)) & set(valid.text.map(normalized)):
            raise ValueError('Normalized text leakage across split')
        dest = args.output/f'fold{fold}'
        dest.mkdir()
        combined.to_csv(dest/'train.csv',index=False)
        shutil.copyfile(directory/'validation.csv',dest/'validation.csv')
        assert digest(dest/'validation.csv') == digest(directory/'validation.csv')
        split_checks[f'fold{fold}'] = {'official_train_rows':len(train),'external_rows':len(selected),
                                     'combined_train_rows':len(combined),'validation_rows':len(valid),
                                     'validation_sha256':digest(dest/'validation.csv'),
                                     'combined_train_sha256':digest(dest/'train.csv')}
    report = {'source':'speechmap via olegzholobov/name-that-llm-response-corpus',
              'input_hashes':{f:digest(root/f) for f in files},'official_train_sha256':digest('data/raw/train.csv'),
              'requested_per_model':args.per_model,'selected_per_model':cap,'selected_total':len(selected),
              'available_per_model_after_filter':counts.to_dict(),
              'removed_overlap_prompt_groups':len(groups),'overlap_response_matches':matched,
              'selected_prompt_groups':selected.prompt_group_id.nunique(),
              'filter':'Short, text-only, exact prompt, request/returned model evidence; no unknown licenses, no_train, flagged truncation/reasoning, cross-label duplicates. One response/model/prompt.',
              'overlap_check':'NFKC/punctuation/case/whitespace normalized exact; 5-word shingle Jaccard >=0.8 or containment >=0.9 against ALL official training answers; remove entire matching external prompt groups.',
              'selection':'Coverage-first global prompt ordering, SHA256 seed 42; equal per-label cap.',
              'model_identity':'Preserves curator mapping and raw/returned IDs; not independently authenticated.',
              'limitations':['SpeechMap domain differs from competition; this is an experimental pilot.',
                             'Provider output terms review flags retained; repository Apache-2.0 is not blanket competition approval.',
                             'Prompt wording for official data is unavailable; semantic prompt leakage cannot be ruled out.',
                             'Not all selected prompts have all 18 models; coverage prioritization is not full matching.',
                             'Shared text filtering does not detect all paraphrases.'],
              'splits':split_checks,'training_started':False}
    (args.output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
