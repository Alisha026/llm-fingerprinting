# Experiment progression

## 1. Establish prompt-held-out baselines

The official set contains 2,699 passages, 18 labels and 150 prompt groups.
Responses to the same prompt stay together during splitting. TF-IDF + LinearSVC
reached 28.33% on fold 0. Frozen DistilBERT plus word TF-IDF reached 31.67% on
fold 0 and 27.78% on fold 1 (29.72% combined).

## 2. Exploit complementary errors

End-to-end DistilBERT fine-tuning did not beat the frozen hybrid alone, but their
errors differed. A fixed 50/50 probability blend improved both folds: 33.70%
and 30.93%, or 32.32% combined. Its public Kaggle submission scored **0.316**.
See [the original report](results.md) for classifier variants and class-level findings.

## 3. Record what did not transfer

Longer DistilBERT training was inconsistent across folds. Frozen Qwen embeddings
performed worse. Qwen3-Embedding-0.6B LoRA improved over frozen Qwen, but its best
official-only fold-0 result was 28.33%. The 720-response SpeechMap pilot reached
27.96%: two fewer correct responses, not evidence of an augmentation gain.
Equal epoch counts with different row counts did not match optimizer budgets.
These runs remain in the registry rather than being discarded.

## 4. Change the data and training recipe

The supplied public notebook described much larger Qwen base models, a public-data
stage followed by short official-only tuning, and prompt-centered training heads.
We implemented an 8B-only approximation, not the entire multi-model ensemble.

The public pool contained 34,940 rows. Preparation retained 34,896 rows from 2,217
prompt groups after removing 44 responses in groups too small for distinct-author
triplets. Exact/word-shingle exclusions found no official-answer matches, but cannot
prove semantic prompt independence. Upstream official-wide style filtering limits
how cleanly we can interpret local validation.

Job 5895767 completed one public-data epoch and two official-data epochs in about
85 minutes overall. Local plain-head accuracy rose from 49.63% to 55.37%; the
plain-plus-centered result rose from 54.26% to **61.48%**. All checkpoints and
predictions are retained locally. Data, model, objective and compute all changed.

## 5. Confirm improvement on Kaggle

A separate private inference notebook reused the fold-0 checkpoint. Runtime checks
reproduced 61.30% local accuracy with 98.70% prediction agreement, allowing documented
BF16/FP16 differences. The user reported **0.555 public balanced accuracy** on
2026-11-02: **+23.9 percentage points** over the original 0.316 submission.
This confirms improvement on the public subset, not a private leaderboard result.

## 6. Current experiment

The next user-run job reuses the saved public-stage adapter and both heads, then
fine-tunes for two epochs on all 2,699 official rows. No new public-data training is
needed. There will be no independent local validation score. Its next submission
will be compared with 0.555; no refit result is recorded yet.

[Full-data run instructions](two_stage_full_refit.md) ·
[Public submission ledger](submission_results.csv) ·
[Complete experiment registry](experiment_registry.csv)
