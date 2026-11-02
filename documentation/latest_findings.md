# Experiment findings — updated 2026-11-02

## Public score received — 2026-11-02

The user reports **0.555 public balanced accuracy** for the two-stage 8B
fold-0 notebook version 1. This is **23.9 percentage points** above the earlier
0.316 public submission. The score is user-reported, not independently fetched
in this update. Prior pending-status entries below preserve the timeline.

Public accuracy is 5.98 percentage points below the 61.48% exploratory local
validation result (5.80 points below Kaggle runtime validation at 61.30%).
The submission supports a substantial improvement for the combined recipe on
the public test subset; it does not isolate the effect of data, architecture,
centering or training budget. Upstream filtering and prompt-group composition
remain limitations for interpreting the local/public gap. This checkpoint was
fine-tuned on 2,159 official rows, not a full 2,699-row official refit. No new
training or submission was launched when recording this result.

## Completed two-stage 8B run — job 5895767

Job log ends with exit code 0. All three scheduled epochs completed; this was
not an interruption or time-limit failure. Public-stage training took 4,450 s;
the two official epochs took 253 s each (about 83 minutes of training, plus
loading, evaluation and saving). Both stage checkpoints and predictions remain
in `artifacts/two_stage_8b_job_5895767`; registry refreshed before reporting.

Public-stage local balanced accuracy: plain 49.63%, centered blend 54.26%.
After two official-only epochs: plain 55.37%, centered blend **61.48%** on
540 fold-0 validation rows. These are exploratory local results, not Kaggle
submission scores. Upstream all-official-data style filtering remains a
validation limitation; 522 validation rows received inferred-group evidence,
whose benefit can differ with hidden-test group composition. Data, backbone,
training objective and compute differ from the previous DistilBERT baseline;
do not attribute the improvement solely to extra data. No new submission score
or unused-fold confirmation is available.

## Two-stage inputs ready — 2026-11-02

Downloaded the source notebook's exact `pool_v1.parquet` from
`nishantdahal/ntl-two-stage-lora` (34,940 rows). Original file, source notebook,
metadata and hash manifest are retained under `data/external/two_stage_source`.
`src.prepare_public_pool` wrote `data/public_pool_v1/pool.parquet`: 34,896 rows,
2,217 prompt groups, all 18 classes. It excluded 44 rows from groups with fewer
than three authors. Normalized exact and five-word-shingle comparisons found
zero official-answer matches. This does not rule out semantic prompt overlap.
See `public_pool_preparation.json` for counts, hashes and exclusions.

The upstream author reports filtering with class-specific style statistics from
all official labeled rows. That makes local evaluation exploratory even though
our fine-tuning excludes official validation rows. The prepared data cannot undo
upstream filtering; do not describe the resulting score as strictly untouched
validation or a controlled data-only gain.

The sampler now retains response variants for repeated prompt/author pairs,
covers every retained row and uses distinct authors within each triplet.
Weight/provenance and pool-checksum preflight passed with 34,896 public rows,
2,159 official training rows and 540 validation rows. Seven focused tests passed.
The user requested direct terminal execution; the shell wrapper was removed.
Use `python -u -m src.two_stage_lora train` with the arguments in
`two_stage_lora.md`. No GPU training has been launched or tested, and no new
model scores are available. Earlier preparation-status notes below are historical.

## Current reference

The original fixed 50/50 DistilBERT fine-tuning + frozen word-hybrid ensemble
remains the best recorded two-fold local configuration: **33.70% fold 0,
30.93% fold 1, 32.32% combined**. The submitted fold-0 ensemble scored **0.316
on Kaggle's public leaderboard**, reported by the user. Other local results
must not be presented as submission scores.

## Subsequent experiments

| Experiment | Fold 0 balanced accuracy | Fold 1 balanced accuracy |
| --- | ---: | ---: |
| Original fine-tuned DistilBERT | 29.63% | 27.61% |
| DistilBERT, 20-epoch budget, best checkpoint | 28.15% | 30.93% |
| Frozen Qwen embeddings, C=1 | 15.56% | Not run |
| Frozen Qwen + word TF-IDF, C=1 | 22.59% | Not run |
| Best exploratory frozen-Qwen diagnostic hybrid | 23.52% | Not run |
| TF-IDF-only logistic diagnostic control | 27.04% | Not run |
| Qwen LoRA, 6-epoch budget | 25.56% | Not run |
| Qwen LoRA, 12-epoch budget | 27.96% | Not run |
| Qwen LoRA, official-only, 15-epoch budget | **28.33%** | Not run |
| Qwen LoRA, external pilot, 15-epoch budget | **27.96%** | Not run |

The 15-epoch official-only checkpoint selected epoch 9. The external-data
checkpoint selected epoch 5 and stopped after epoch 7. Both use patience 2.
On the balanced 540-row fold this is 153 versus 151 correct: the external pilot
is lower by **0.37 percentage points (two passages)**. This small difference is
not evidence that external data is inherently harmful, but it does not establish
an accuracy gain. Equal epoch budgets with different dataset sizes still imply
different update counts and schedules in optimizer-step units.

Full-data DistilBERT/hybrid refits completed for nine and fifteen epochs.
Neither has a held-out validation score or a recorded public submission score.
They remain available as candidate models; low training loss is not evidence of
improved generalization.

## What we learned

- The hybrids fit training data much better than unseen prompts. More DistilBERT
  epochs did not improve both validation folds consistently.
- Frozen Qwen was worse than the original representation. Its cached-vector
  diagnostic reproduced the original probabilities, and standardization did not
  rescue it. TF-IDF alone beat all tested frozen-Qwen hybrids.
- LoRA improved on frozen Qwen but has not beaten the original ensemble.
- The external pilot adds 720 SpeechMap responses (40 per label, 115 prompt
  groups), with official validation files unchanged. It is source-controlled,
  not domain-matched or fully matched by prompt across all models. Exact and
  word-shingle checks found no official-answer overlap; semantic overlap and
  prompt independence are not guaranteed. Provenance and terms-review flags
  remain alongside the data.
- No evidence yet supports a claim that a larger model, more epochs, or this
  external-data sample can reach 70%. Repeated fold-0 selection is exploratory;
  reserve other prompt folds for later confirmation.

## Plateau review — 2026-10-27

- The user now describes approximately "3.3" versus a leader's "7.1".
  Interpretation as 0.33 versus 0.71 is pending confirmation, as is the
  submitted checkpoint. Do not replace the recorded 0.316 submission with
  an inferred exact score. The live leaderboard could not be read through
  the web tool during this review.
- The Qwen LoRA configuration uses `Qwen/Qwen3-Embedding-0.6B`, rank 8,
  q/v projection adapters, seed 42, and effective batch size 32. These runs
  do not establish the performance of larger Qwen backbones or other
  objectives. Archived configs retain data hashes and training budgets.
- Similar local and reported public performance is consistent with a
  generalization limitation; it does not rule out inference defects.
  Static inspection confirms explicit class-order alignment before blending
  in `src/kaggle_inference.py`; packaged-checkpoint parity still needs a
  direct comparison on saved validation rows before declaring it verified.
- Prioritize auditing a stronger public notebook's actual source and
  evaluating methods that reduce prompt/topic dependence. The public
  "Teach the Model, Remove the Prompt" page was accessible only as a title,
  not executable source; no reproduction or algorithm claim is justified.
  Prompt-invariant training is a hypothesis to test, not a demonstrated gain.
- For the next user-run experiment, specify one changed factor and fixed
  data, seed, update budget, and selection policy. Fold 0 is exploratory;
  freeze the recipe before evaluating an unused prompt fold. Neither more
  epochs nor the current external pilot has demonstrated a path to 71%.

## Two-stage implementation — prepared, not run

`src/two_stage_lora.py` now implements a first 8B-only approximation of the
supplied two-stage recipe, including shared-prompt triplet losses, pooled
intermediate layers, and inferred-group centered validation predictions.
See [the run guide](two_stage_lora.md) for explicit differences and commands.
No training or fitting has been launched. The public pool and Qwen3-8B-Base
weights are not present; preflight correctly stops on the missing pool.
The full 85/15 public ensemble is not implemented by this new training command.

Update: Qwen3-8B-Base was downloaded on 2026-10-27 to
`artifacts/pretrained/qwen3_8b_base`, pinned to revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4` in its provenance file.
The public pool is still missing. No training was launched.

## Registry contents

- `experiment_registry.csv`: generated registry of artifact-backed runs.
- `experiment_archive/`: copied configs, histories, per-class metrics and hashes.
- `experiment_results.csv`: original historical experiment table, retained.
- `submission_results.csv`: separate public submission record.
- `results.md`: earlier detailed experimental history.

Refresh records after user-run experiments with:

```bash
python3 -m src.record_experiments
```

Cleanup preserves unique model checkpoints and prediction/probability files.
The original upload ZIP can be recreated with `python3 -m src.package_baseline`.

## Batch preparation — 2026-11-02

The user explicitly authorized preparing and submitting a Slurm training job.
An interactive-run config now exists at `artifacts/two_stage_8b_fold0` with
status `running`, but no metrics; active process status is not established by
that file. Registry refreshed before reporting status. Existing artifacts are
preserved. The user confirmed partition `lrz-hgx-a100-80x4` via their interactive
`srun` command. Batch script `scripts/train_two_stage_8b.sbatch` requests one GPU,
four CPU cores, 64 GB host memory and four hours with job-specific logs/results.
Shell syntax passed. Scheduler access was declined; job submission and duplicate
run check remain pending. No batch job has been submitted at this point.

The user confirmed interactive training is still running. A held batch
submission (`sbatch --hold --parsable scripts/train_two_stage_8b.sbatch`) was
attempted to avoid concurrent training, but the execution permission request
was declined before the command ran. No job ID exists and no batch job was
submitted. Neither the interactive allocation nor its training was stopped.

## New Kaggle submission — 2026-11-02

Submitted private notebook `alishasaifi/ntl-two-stage-8b-fold0`, version 1,
using the existing job5895767 checkpoint, without training. Its local-runtime
check yielded 61.30% accuracy versus 61.48% saved reference, with 98.70%
prediction agreement. Competition submission is accepted and pending hidden-test
evaluation; no public score yet. The earlier 0.316 submission is confirmed by
Kaggle API. See `two_stage_submission.md` for parity tolerances and assets.

## Full-data refit prepared — 2026-11-02

Added paired `--full-data` / `--from-public` options to `src.two_stage_lora`.
The new batch script `scripts/train_two_stage_8b_full.sbatch` reloads the saved
public-stage adapter, pooled-layer norms and both heads, then trains two epochs
on all 2,699 official rows. Exact-union and checkpoint preflight passed. The
previous official checkpoint is not used. No hold-out score will be produced.
Source weights/checkpoints and the 0.555 submission are preserved. This update
prepares user-run training only; no new job has been submitted by the assistant.
See `two_stage_full_refit.md` for instructions and compute/RNG limitations.

## Repository maintenance — 2026-11-02

Refreshed the experiment registry before cleanup. Removed 133,496,019 bytes of
verified duplicate Kaggle staging files and regenerable Python caches. All
staging files were verified against the retained upload ZIP; checkpoint copies
were additionally hash-matched to the retained original checkpoint. No original
weights, training inputs, environments, prediction files or logs were removed.
`cleanup_2026-11-02.json` records each path, checksum and restoration source.

README now summarizes the 0.316 to 0.555 public-score progression and links to
`progress.md`, the full evidence registry and current refit instructions.
Historical code/results are retained; logs and tool configuration are explicitly
ignored. Notebook outputs and common credential patterns were checked before
preparing the Git update. GitHub remote inspection was declined; publication
status must be confirmed separately rather than inferred from local files.
