# Experiment workflow

- The user runs training. Do not launch training, model fitting, or paid generation
  unless the user explicitly changes that instruction. Read-only analysis and
  data preparation are allowed when requested.
- Record every completed experiment before cleanup or reporting conclusions.
  Run `python3 -m src.record_experiments` to refresh the registry and archive
  configs, histories and metrics. Preserve prediction/probability files needed
  for comparisons. Update `documentation/latest_findings.md` with important
  conclusions and limitations; retain older results.
- Keep local validation scores separate from user-reported Kaggle scores.
  Record seeds, data hashes, training budgets and configuration changes. Do not
  claim a controlled data-only improvement when schedules or compute differ.
- Before removing experiment artifacts, archive evidence and record the cleanup
  in documentation. Do not delete unique trained models, source data, credentials,
  or environments as routine cleanup. Verify identical file hashes before
  deduplicating weights. Weight symlinks refer to the retained original encoder;
  treat that base checkpoint as immutable.
