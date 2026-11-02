"""Read-only corpus audit; writes reports, never merges data or trains models."""
import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def main():
    root = Path('data/external/response_corpus')
    manifest = json.loads((root / 'manifest.json').read_text())
    verified = []
    for name, meta in manifest['files'].items():
        path = root / name
        if path.exists():
            if path.stat().st_size != meta['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
                raise ValueError(f'Integrity mismatch: {name}')
            verified.append(name)
    official = pd.read_csv('data/raw/train.csv', dtype=str, keep_default_na=False)
    wide = pd.read_parquet(root / 'external_short_wide.parquet')
    labels = sorted(set(official.model))
    if set(wide.columns) != set(labels + ['id', 'prompt']):
        raise ValueError('Unexpected labels or columns')
    long = wide.melt(id_vars=['id', 'prompt'], value_vars=labels, var_name='model', value_name='text').dropna(subset=['text'])
    metadata = pd.read_parquet(root / 'external_cell_metadata.parquet')
    metadata = metadata.loc[metadata['view'] == 'short']
    long = long.merge(metadata, on=['id', 'model'], how='left', validate='one_to_one', indicator=True)
    if not long['_merge'].eq('both').all():
        raise ValueError('Missing cell metadata')
    long = long.drop(columns='_merge')
    long['normalized_text'] = long.text.map(normalized)
    official_texts = set(official.text.map(normalized))
    long['official_overlap'] = long.normalized_text.isin(official_texts)
    long['actual_words'] = long.text.str.split().str.len()
    provenance = pd.read_parquet(root / 'external_cell_provenance.parquet',
                                 columns=['cell_id_short', 'model', 'source', 'license_status'],
                                 use_threads=False)
    occurrences = provenance.loc[provenance.cell_id_short.isin(long.cell_id)].copy()
    flags = ['no_train', 'possible_truncation', 'possible_reasoning_in_answer',
             'truncation_metadata_missing', 'text_shared_across_labels', 'license_unknown', 'output_terms_review']
    # Screening only: source declarations are not independently authenticated.
    candidate = long.prompt_exact.eq(True) & long.text_only_suitable.eq(True) & long.mechanically_eligible.eq(True)
    for flag in flags:
        candidate &= long[flag].eq(False)
    candidate &= ~long.official_overlap & long.actual_words.between(40, 250)
    screened = long.loc[candidate].drop_duplicates(['model', 'normalized_text'])
    counts = pd.DataFrame({'official': official.model.value_counts(),
                           'short_cells': long.model.value_counts(),
                           'unique_short_texts': long.drop_duplicates(['model', 'normalized_text']).model.value_counts(),
                           'strict_metadata_candidates': screened.model.value_counts()}).reindex(labels).fillna(0).astype(int)
    out = Path('documentation')
    counts.to_csv(out / 'response_corpus_counts.csv', index_label='model')
    report = {'source': 'https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus',
              'audit_date': '2026-10-26', 'snapshot_date': manifest['created_date'],
              'verified_files': verified, 'wide_rows': len(wide), 'short_cells': len(long),
              'unique_response_ids': long.response_id.nunique(),
              'unique_model_normalized_text': len(long.drop_duplicates(['model', 'normalized_text'])),
              'prompt_groups': long.prompt_group_id.nunique(),
              'missing_prompt_text_cells': int(long.prompt.isna().sum()),
              'official_normalized_overlap_cells': int(long.official_overlap.sum()),
              'out_of_range_word_count_cells': int((~long.actual_words.between(40, 250)).sum()),
              'flags_on_short_cells': {f: long[f].astype(str).value_counts().to_dict() for f in flags},
              'model_evidence_on_short_cells': long.model_evidence_class.value_counts().to_dict(),
              'strict_metadata_candidates': len(screened),
              'provenance_occurrences_for_short_cells': len(occurrences),
              'source_occurrence_counts': occurrences.source.value_counts().to_dict(),
              'license_status_occurrence_counts': occurrences.license_status.value_counts().to_dict(),
              'limitations': ['Metadata/source declarations are not independent authentication.',
                             'Strict candidates are not approved training data; upstream terms/model IDs still require review.',
                             'No semantic paraphrase or exhaustive substring audit performed.',
                             'Official prompt wording is absent, so prompt-level independence is unresolved.',
                             'No hidden test data was inspected; quarantine Luna collection excluded.'],
              'training_started': False}
    (out / 'response_corpus_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    pd.crosstab(occurrences.model, occurrences.source).to_csv(out / 'response_corpus_source_counts.csv')
    print(counts.to_string())
    print(json.dumps({k:v for k,v in report.items() if k not in ['flags_on_short_cells','source_occurrence_counts','model_evidence_on_short_cells','license_status_occurrence_counts']},indent=2))
    print('FLAGS',json.dumps(report['flags_on_short_cells']))


if __name__ == '__main__':
    main()
