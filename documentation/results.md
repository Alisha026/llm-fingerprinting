# Results and findings

Historical report through 2026-10-24. For subsequent runs and current status, see
[latest findings](latest_findings.md). Experiment scores use local prompt-grouped validation.
The separately labeled Kaggle result at the end is user-reported public evaluation.

## Data

- 2,699 labeled passages, 18 model labels, and 150 prompt groups.
- Seventeen labels have 150 passages; `claude-fable-5` has 149.
- No missing required values, duplicate IDs, or exact/normalized duplicate texts.
- Passage length is 36–274 whitespace-separated words (median 101).
- Validation keeps every response for a prompt together. This prevents the same
  prompt from appearing in both training and validation.

## Experiments

| Experiment | Train balanced accuracy | Validation balanced accuracy |
| --- | ---: | ---: |
| Random guess reference | — | 5.56% |
| RBF SVM on frozen embeddings | 18.84% | 17.22% |
| Nearest centroid on frozen embeddings | 23.98% | 18.15% |
| Strong hybrid regularization, C=0.01 | 30.83% | 20.19% |
| Strong hybrid regularization, C=0.1 | 46.21% | 24.44% |
| TF-IDF + LinearSVC | — | 28.33% |
| DistilBERT frozen, 256 tokens, linear head | — | 28.52% |
| Frozen embeddings + logistic regression | 44.68% | 28.70% |
| Frozen embeddings + MLP | 45.42% at selected epoch | 28.89% |
| DistilBERT frozen, 512 tokens, linear head | 54.92% at selected epoch | 29.63% |
| Late TF-IDF/embedding probability ensemble | 82.11% | 29.81% |
| Word + character TF-IDF and embeddings | 92.68% | 31.30% |
| Character TF-IDF and embeddings | 91.94% | 31.48% |
| **Word TF-IDF and embeddings** | **93.28%** | **31.67%** |

The selected configuration is frozen DistilBERT mean-pooled embeddings at 512
tokens, word unigram/bigram TF-IDF, and L2 logistic regression with C=1. It
classified 171 of 540 fold-0 passages correctly. Balanced accuracy equals
ordinary accuracy there because each validation class has 30 passages.

## Fold-1 confirmation

The unchanged selected configuration reached **27.78% balanced accuracy** on
fold 1, correctly classifying 150 of 539 passages (27.83% ordinary accuracy).
Its training balanced accuracy was 93.38%, nearly identical to fold 0's 93.28%,
while validation fell by 3.89 percentage points. Across the 1,079 held-out rows
from folds 0 and 1, combined balanced accuracy is **29.72%** (321 exact model
predictions correct). This is a partial two-fold estimate, not full five-fold OOF.

Fold 1 confirms that the large training/validation gap persists and that fold 0
was favorable. The 31.67% fold-0 score should not be used as the expected local
generalization score. Per-class performance also moves substantially: for
example, `claude-opus-5` rose from 63.3% to 76.7%, while
`gemini-3.1-pro-preview` fell from 70.0% to 60.0%. `claude-fable-5` remained at
0% recall on both folds.

## What the experiments show

- Raising the token limit from 256 to 512 removed truncation: 264 training and
  70 validation passages exceeded 256 tokens; none exceeded 512. The score rose
  by 1.11 percentage points.
- Linear, logistic, MLP, RBF, centroid, and late-ensemble classifiers did not
  produce a large improvement on frozen embeddings. RBF and centroid underfit.
- Stronger L2 regularization reduced both the training gap and validation score.
  A smaller gap was not useful when predictive performance also fell.
- Word-only, character-only, and combined hybrids differ by at most two correct
  validation passages. Their ranking may be split noise.
- The best hybrid has a large training/validation gap, consistent with poor
  transfer to unseen prompts. Pipeline checks found no ID, prompt, exact-text,
  label, TF-IDF, or embedding-normalization mismatch.
- Validation accuracy rises with passage length: 24.7% for 1–128 tokens, 37.2%
  for 129–256, and 40.0% for 257–512. Class composition differs by bucket, so
  these figures are descriptive.
- Some top TF-IDF features appear in few training prompts or no validation
  passages, which suggests possible prompt/topic shortcuts. Coefficients alone
  do not prove causation.
- Performance varies strongly by label. Google/Gemini models are relatively
  recognizable, but members of the same provider family are often confused.

## Interpretation and limitations

The evidence points to the frozen representation and limited prompt diversity as
larger constraints than classifier choice. Fold-1 confirmation lowered the
estimated performance and strengthened the case that tiny differences between
word, character, and combined hybrids may be split noise. A full five-fold OOF
estimate would measure variance more reliably, but it would mostly refine the
same conclusion. The next materially different modeling step is careful encoder
fine-tuning, evaluated with the existing prompt-grouped folds.

The observed public Kaggle score of 0.322 came from another notebook whose method
is unknown. It is context only and is not directly comparable with local fold 0.

## Fold-1 confirmation commands (completed)

Prepare a second split without overwriting fold 0:

```bash
python -m src.data_loaders --prepare \
  --validation-fold 1 \
  --output data/processed_fold1
```

Then run the fixed leading configuration:

```bash
python baseline.py \
  --model encoder --head hybrid --device auto \
  --text-features word \
  --encoder artifacts/encoder_v1/encoder --local-files-only \
  --train data/processed_fold1/train.csv \
  --validation data/processed_fold1/validation.csv \
  --max-length 512 --c 1.0 \
  --output artifacts/hybrid_word_fold1
```

The run was completed without changing settings. Its artifacts are stored in
ignored `artifacts/hybrid_word_fold1/`.

## Next experiment: full encoder fine-tuning

`src/model.py` provides a controlled end-to-end DistilBERT experiment. All
encoder and classification-head parameters are trainable. The initial settings
are 512 tokens, learning rate 2e-5, five maximum epochs, effective batch size 32,
linear warmup/decay, gradient clipping, mixed precision on CUDA, and early
stopping after two non-improving epochs. The best checkpoint is selected by
validation balanced accuracy.

Run fold 0 first:

```bash
python -m src.model \
  --encoder artifacts/encoder_v1/encoder --local-files-only \
  --train data/processed/train.csv \
  --validation data/processed/validation.csv \
  --device auto --max-length 512 \
  --batch-size 8 --gradient-accumulation 4 \
  --epochs 5 --patience 2 --learning-rate 2e-5 \
  --output artifacts/finetune_distilbert_fold0
```

If memory is insufficient, reduce `--batch-size` and increase
`--gradient-accumulation` to retain an effective batch size near 32. Do not
change settings for fold 1 after observing fold 0.

### Fine-tuning results

Fine-tuning completed with the fixed configuration on both folds:

| Model | Fold 0 | Fold 1 | Combined held-out balanced accuracy |
| --- | ---: | ---: | ---: |
| Frozen word hybrid | 31.67% | 27.78% | **29.72%** |
| Fully fine-tuned DistilBERT | 29.63% | 27.61% | **28.62%** |

Both fine-tuning runs selected epoch 9. Fold 0 took 146.7 seconds and fold 1
took 138.9 seconds on CUDA with mixed precision. End-to-end fine-tuning alone
did not improve the two-fold estimate; it trails the frozen word hybrid by 1.10
percentage points.

The prediction sets are complementary despite the lower fine-tuned score. On
fold 0, both models correctly classify 90 passages, the hybrid alone gets 81,
and fine-tuning alone gets 70. On fold 1, both get 75 correct, the hybrid alone
gets 75, and fine-tuning alone gets 74. Their exact predictions agree on only
33.0% and 30.8% of rows respectively. This supports testing a fixed probability
blend, while also showing that an oracle-style union would greatly overstate
achievable ensemble performance.

### Fixed probability blend

`src/model.py --blend-runs` averages saved class probabilities after verifying
identical IDs, true labels, and class columns. The first controlled blend assigns
50% to fine-tuned DistilBERT and 50% to the frozen word hybrid. The same weight
must be used on both folds; it is not fitted to either validation set.

```bash
python -m src.model \
  --blend-runs artifacts/finetune_distilbert_fold0_retry artifacts/hybrid_word \
  --first-weight 0.5 \
  --output artifacts/blend_fold0

python -m src.model \
  --blend-runs artifacts/finetune_distilbert_fold1 artifacts/hybrid_word_fold1 \
  --first-weight 0.5 \
  --output artifacts/blend_fold1
```

This performs no training or model inference. The blend is useful only if its
two-fold result improves on the hybrid's 29.72% combined balanced accuracy.

The fixed blend completed successfully and improved both folds:

| Model | Fold 0 | Fold 1 | Combined held-out balanced accuracy |
| --- | ---: | ---: | ---: |
| Fine-tuned DistilBERT | 29.63% | 27.61% | 28.62% |
| Frozen word hybrid | 31.67% | 27.78% | 29.72% |
| **50/50 probability blend** | **33.70%** | **30.93%** | **32.32%** |

The blend correctly classifies 349 of 1,079 held-out passages, compared with
321 for the hybrid and 309 for fine-tuning. It improves the combined balanced
accuracy by 2.59 percentage points over the previous best and succeeds on both
prompt folds with one preselected weight. This is the strongest controlled local
result so far. It remains a partial two-fold estimate, and `claude-fable-5`
recall remains near zero (0% on fold 0 and 3.45% on fold 1).


## First Kaggle submission — 2026-10-24

The user reports **0.316 public balanced accuracy** for the uploaded fold-0
50/50 ensemble. This score was not independently retrieved from their account.
The submission combines `hybrid_word/classifier.joblib`, the original encoder
from `encoder_v1/encoder`, and `finetune_distilbert_fold0_retry/best_model`.
Its local fold-0 score is 0.3370; the two-fold 0.3232 estimate evaluates two
separately trained ensembles and is not the local score of this single checkpoint.

The public score is 2.10 percentage points below its fold-0 score. The user also
reports a leader score of 0.703; that leaves a 38.7-point public-score gap.

## Full-data refit candidate — prepared, not yet trained

`src/train_full.py` and `notebooks/kaggle_full_data.ipynb` refit both components
on all 2,699 labeled rows. TF-IDF vocabulary and embedding normalization are
fitted on all official training rows. Fine-tuning starts from the original
DistilBERT encoder for nine fixed epochs (the best epoch on both earlier folds),
with the previous learning rate, effective batch size, token limit and 50/50 blend.
The final partial gradient accumulation window is normalized by its actual
example count; scheduler advancement skips AMP-overflow updates.

No validation/test labels are used for selection in this refit. Therefore it
has no held-out validation score. It is an incremental candidate, not a claim
that using more rows will close the leaderboard gap. Execution on Kaggle is pending.

## Stronger public approaches to investigate

Public notebook metadata found on 2026-10-24:

| Notebook | Reported public / best score | Evidence available |
| --- | --- | --- |
| [SVM Frozen Qwen](https://www.kaggle.com/code/hanifnoerrofiq/svm-frozen-qwen) | 0.429 / 0.452 | Title, scores, Qwen 3 input; exact implementation unavailable |
| [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering/output?scriptVersionId=359126546) | 0.549 / 0.550 | Indexed version-3 page; exact implementation unavailable |
| [Teach the Model, Remove the Prompt](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt) | 0.600 / 0.600 | Title and score; exact implementation unavailable |

These are public leaderboard scores, not results reproduced on our folds. The
web reader did not expose the notebook bodies; model size, pooling, training
settings and centering algorithm must not be inferred from the titles alone.
The next substantial experiment should examine a stronger frozen Qwen encoder
and then adapter tuning, retaining our exact prompt-grouped splits. Read the
source before claiming a reproduction. Fit preprocessing on each training fold
only and assess the unchanged candidate on both folds before a full-data refit.
