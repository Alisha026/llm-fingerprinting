"""Refresh the experiment ledger and archive small evidence files; no training."""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def record():
    root = Path('artifacts')
    archive = Path('documentation/experiment_archive')
    archive.mkdir(parents=True, exist_ok=True)
    # Retain archived evidence when incomplete directories are later removed.
    for file in root.rglob('*.json'):
        rel = file.relative_to(root)
        if len(rel.parts) > 3 or any(p in {'pretrained','encoder','frozen_encoder','finetuned_model','best_model','best_adapter'} for p in rel.parts):
            continue
        target = archive / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    rows = []
    for run in sorted({p.parent for p in archive.rglob('metrics.json')} | {p.parent for p in archive.glob('*/config.json')} | {p.parent for p in archive.glob('*/training_config.json')}):
        def read(name):
            p = run / name
            return json.loads(p.read_text()) if p.exists() else {}
        cfg = read('config.json') or read('training_config.json')
        metrics = read('metrics.json')
        history = read('history.json')
        name = str(run.relative_to(archive))
        validation = metrics.get('balanced_accuracy', metrics.get('validation_balanced_accuracy',''))
        row = {'run': name, 'status': 'completed' if metrics else cfg.get('status','incomplete_or_unscored'),
               'train_balanced_accuracy': metrics.get('train_balanced_accuracy',''),
               'validation_balanced_accuracy': validation,
               'best_epoch': metrics.get('best_epoch',''), 'epoch_budget': cfg.get('epochs',''),
               'epochs_recorded': len(history) if isinstance(history,list) else '',
               'last_train_loss': history[-1].get('train_loss','') if isinstance(history,list) and history else '',
               'train_file': cfg.get('train',''), 'validation_file': cfg.get('validation',''),
               'train_sha256': cfg.get('train_sha256', cfg.get('data_sha256','')),
               'validation_sha256': cfg.get('validation_sha256',''),
               'scope': metrics.get('scope',cfg.get('scope','local run; see archived config')),
               'evidence': str(run)}
        if cfg.get('status') == 'running' and not metrics:
            row['status'] = 'incomplete_at_recording'
        if validation != '' and not row['validation_file'] and '/'+name.split('/')[-1] in ('/embedding','/hybrid_word'):
            row['scope'] = 'local prompt-grouped validation; parent config identifies split'
        rows.append(row)
    path = Path('documentation/experiment_registry.csv')
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    inventory={str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(archive.rglob('*.json')) if p.name != 'archive_index.json'}
    (archive/'archive_index.json').write_text(json.dumps({'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
                                                        'sha256':inventory},indent=2)+'\n')
    print(f'Recorded {len(rows)} runs; archived {len(inventory)} JSON evidence files.')


if __name__ == '__main__':
    record()
