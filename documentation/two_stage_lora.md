# Two-stage 8B experiment — inputs ready, training not run

## Run on the user's GPU allocation

Run directly in your existing GPU terminal:

```bash
source myenv/bin/activate
python -u -m src.two_stage_lora train \
  --pool data/public_pool_v1/pool.parquet \
  --encoder artifacts/pretrained/qwen3_8b_base \
  --train data/processed/train.csv \
  --validation data/processed/validation.csv \
  --output artifacts/two_stage_8b_fold0
```

The shell wrapper was removed at the user's request. `-u` streams output to the
terminal. Loss/progress prints every 25 updates; validation scores print after
the public stage and after the two-epoch official stage. Keep the session alive.
GPU memory use and duration have not been measured; no training was launched
by the assistant.

For a preflight without fitting or model loading:

```bash
python -m src.two_stage_lora check --pool data/public_pool_v1/pool.parquet
```

Expected: `pool=34896, train=2159, validation=540`. The output folder must be new
or empty. For a fresh retry, change `--output` to a new name; optimizer resume
is not implemented. The Python trainer refreshes the registry on successful
completion. After an interrupted run, record available evidence manually with
`python3 -m src.record_experiments` before cleanup or analysis.

## Inputs and preparation

Qwen3-8B-Base weights and tokenizer are present at
`artifacts/pretrained/qwen3_8b_base`, revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. The download provenance, index and
five shards are retained. Preflight checks architecture, provenance, expected
files and the prepared pool checksum. It does not rehash every model weight.

On 2026-11-02 the exact upstream `pool_v1.parquet` was downloaded from
[nishantdahal/ntl-two-stage-lora](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora).
Its 34,940 rows cover 18 labels and 14 sources. The downloaded source notebook,
metadata and original pool are retained in `data/external/two_stage_source`.
Kaggle reports CC-BY-SA-4.0 for the combined dataset; upstream attribution and
individual license descriptions remain in the saved source notebook/metadata.

`src.prepare_public_pool` produced **34,896 rows in 2,217 prompt groups** at
`data/public_pool_v1/pool.parquet`. It removed 44 rows from groups with fewer
than three authors. It found zero official-answer matches using normalized
exact, five-word-shingle Jaccard >= .8 or containment >= .9 comparisons.
Any detected answer overlap would exclude the entire public prompt group.
6,506 repeated prompt/author rows are retained as distinct response variants.
The audit and hashes are in `documentation/public_pool_preparation.json` and
alongside the prepared pool. Excluded IDs and reasons are saved separately.

To reproduce preparation in a fresh directory (no model fitting):

```bash
myenv/bin/python -m src.prepare_public_pool --output data/public_pool_v2
```

**Validation limitation:** the upstream author describes class-specific style
filters derived from all official labeled data. That preparation cannot be
undone using this already filtered pool, so local results are exploratory,
not strictly untouched validation. Official prompt text is unavailable;
answer-level overlap checks cannot prove semantic prompt independence.
Labels and prompt grouping remain publisher supplied, not independently
verified. We do not claim a controlled data-only gain.

## Training recipe

`src/two_stage_lora.py` implements a first 8B-only approximation of the supplied
recipe, not the full public ensemble:

- First 24 Qwen3-8B-Base layers; identity final RMSNorm; mean pools from layers
  2, 4, 6, 8, 12, 16, 20, 24, each with a LayerNorm; first token excluded;
  truncate to 447 tokens then append EOS.
- Rank-16 all-linear LoRA, alpha 32, dropout .05; plain and centered heads.
- Public stage: one epoch, class-weighted losses, adapter/head LR 1e-4/1e-3.
- Official stage: two epochs on 2,159 training rows, LR 5e-5/5e-4; 540 official
  validation rows stay out of gradient updates.
- Six same-prompt triplets per optimizer update, one triplet per microbatch.
  Every triplet has three distinct authors. Sampling covers every response;
  filler samples from other authors complete uneven author queues. This means
  sample presentations exceed unique rows; actual counts and updates are saved.
- Fresh AdamW and cosine schedule per stage, 10% warmup, weight decay .01,
  gradient clipping 1.0, seed 42, BF16 backbone and float32 heads.

The supplied notebook omitted its training loop. Sampling, optimizer reset,
weight decay, clipping and seed are explicit choices, not claims of exact
reproduction. Checkpoints are saved at stage ends with no validation selection.

## Evaluation and outputs

Validation groups are inferred from text rather than held-out prompt IDs. The
script reports plain probabilities and a fixed plain + .35 centered
log-probability blend, approximately the supplied 8B twin/plain weight ratio.
No SVM, Gaussian head, Sinkhorn, 14B, or older feature stack is included yet.

Outputs include `public/` and `official/` adapters, pooled-layer norms and
heads, stage validation probabilities/predictions, `history.json`, stage metrics
and final `metrics.json`. Config records data hashes, model revision, pool audit,
training script hash, seed, versions and optimizer/sample budgets. Base weights
are never changed; adapters still require the retained base model.

After training, inspect the final metrics and compare with the historical
33.70% fold-0 baseline, with the upstream-filtering limitation stated explicitly.
Update `documentation/latest_findings.md` with results and limitations. A clean
confirmation would require raw external data filtered without held-out official
class statistics as well as prompt-held-out official fine-tuning.

## Slurm batch job (prepared 2026-11-02)

The user requested batch submission instead of a one-hour interactive session.
`scripts/train_two_stage_8b.sbatch` uses the user-confirmed
`lrz-hgx-a100-80x4` partition, one GPU, four CPU cores, 64 GB host memory and a
four-hour requested walltime. The account is left to the scheduler default,
as in the user's successful interactive command. Batch limits/availability
have not been queried: scheduler socket access was blocked in the sandbox
and the escalation request was declined.

After confirming the interactive trainer has stopped, submit from the project:

```bash
mkdir -p logs
sbatch scripts/train_two_stage_8b.sbatch
```

The submission returns a job ID. Monitor it using that ID:

```bash
squeue -j JOB_ID
tail -f logs/two_stage_8b_JOB_ID.log
```

Output goes to `artifacts/two_stage_8b_job_JOB_ID` so the interactive run is
preserved. The job starts from the base weights, not the interactive optimizer.
The registry is refreshed on normal exit or catchable failure; a hard kill may
prevent the exit trap. Four hours is a resource request, not a measured runtime
guarantee. The trainer still checkpoints only at stage boundaries and does not
resume interrupted optimizer state. Do not submit duplicate training jobs.
