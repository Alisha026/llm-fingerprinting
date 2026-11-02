# Full-data 8B refit — prepared, not trained

Submit from the project directory:

```bash
sbatch scripts/train_two_stage_8b_full.sbatch
```

Requests one A100 on `lrz-hgx-a100-80x4`, four CPUs, 64 GB host memory and one
hour. Based on the measured prior stage speed, allow roughly 15–25 minutes
of execution, excluding queue time; this is an estimate, not a guarantee.
Monitor the returned job ID:

```bash
tail -f logs/two_stage_8b_full_JOB_ID.log
```

Outputs go to `artifacts/two_stage_8b_full_job_JOB_ID/official` with new adapters,
pooled-layer norms, both heads, metadata and tokenizer. Source checkpoints and
the 0.555 submission remain unchanged. No training was launched during setup.

The trainer reloads `artifacts/two_stage_8b_job_5895767/public`, including both
heads, and performs only two official-data epochs on all 2,699 rows. It starts
a fresh AdamW/cosine schedule at adapter/head LR 5e-5/5e-4, seed 42, six triplets
per update, matching the previous official-stage settings. Public training is
not repeated, and the previous official checkpoint is not used as initialization.
The prediction recipe remains plain plus .35 centered log probabilities.

Preflight verifies the exact union of the original official split, label order,
base revision, architecture and required public-checkpoint files. Input check,
Python compilation, batch syntax and six focused tests passed. GPU checkpoint
loading and training for this new path have not been exercised by the assistant.

There is no held-out validation: `metrics.json` records completion and row counts,
not accuracy. Compare the next Kaggle submission with the user-reported 0.555.
Actual updates increase from 240 to 300 at the same two-epoch budget. Dropout RNG
is restarted at seed 42, rather than restored from the old public-stage process.
Thus this is the practical full-data refit, not a strictly compute/RNG-matched
causal data comparison. Original training-data and upstream-filtering caveats remain.

The inference packager still targets the earlier fold-0 checkpoint. Once this
refit finishes, prepare a separate submission package and in-sample runtime
parity anchors; do not reuse the old held-out probabilities as references for
the new model or present in-sample accuracy as validation.
