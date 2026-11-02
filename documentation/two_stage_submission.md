# Two-stage 8B Kaggle submission candidate

Uses the completed `artifacts/two_stage_8b_job_5895767/official` checkpoint;
no retraining or full-data refit. Prediction is plain log probabilities plus
0.35 times centered-head log probabilities for inferred groups of 2–18.
There is no Sinkhorn balancing or additional ensemble.

`src/two_stage_inference.py` loads a 24-layer Qwen3-8B-Base trunk and injects
unmerged rank-16 adapters with FP32 arithmetic matching PEFT. It preserves
pooling, tokenizer, norms and heads. Native BF16 is used on Ampere/newer;
older GPUs use FP16 with a BF16 retry on nonfinite hidden states. The three
unit tests passed, including exact synthetic parity with PEFT's unmerged
forward, and grouping/centering parity with the trainer. No fitting occurred.

`src/package_two_stage.py` created `artifacts/kaggle_two_stage_8b`, a private
checkpoint upload (133,428,748 bytes) and a new notebook. Base weights are
attached separately using Kaggle's Qwen3-8B-Base model input. The notebook
verifies bundle hashes and recomputes all 540 validation predictions before
predicting test rows. It aborts if maximum probability error exceeds .08,
prediction agreement is below .97, or validation accuracy differs by more than
.015. These tolerances allow BF16/FP16 runtime differences; GPU parity remains
to be established by the Kaggle run.

Dataset: `alishasaifi/ntl-two-stage-8b-fold0-checkpoint` (private).
Notebook: `alishasaifi/ntl-two-stage-8b-fold0` (private, internet off).
Source notebook: `notebooks/ntl-two-stage-8b-fold0.ipynb`.
Output: `submission.parquet`; parity diagnostics: `parity.json`.

Local exploratory accuracy remains 61.48%, with the upstream all-official-data
filtering limitation. No Kaggle score should be recorded until the competition
returns one. The old 0.316 submission is preserved as a separate reference.

## Submitted version 1 — 2026-11-02

Private dataset created and notebook version 1 completed. Downloaded outputs
passed schema/ID checks on the three example test rows. GPU parity passed:
maximum probability difference .03226, prediction agreement 98.7037%, local
accuracy 61.2963% versus 61.4815% saved reference. This is runtime reproduction,
not a public leaderboard score.

Competition submission was accepted at 13:18:25 UTC (14:18:25 Europe/Berlin),
status PENDING. The API confirms the earlier 0.316 public score. Track the new
score separately in `submission_results.csv` and `two_stage_submission_status.json`.

## Score received

On 2026-11-02 the user reported public balanced accuracy **0.555**, versus
the earlier 0.316: +23.9 percentage points. This updates the pending status
above. Score provenance is user report; local validation remains separate.
