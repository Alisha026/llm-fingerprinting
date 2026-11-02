# Name That LLM — model fingerprinting

Identify which of 18 language models wrote a short passage. This project tracks
our progression from TF-IDF and DistilBERT baselines to a two-stage Qwen3-8B LoRA
classifier, with experiment evidence and separate local/Kaggle score records.

**Latest public Kaggle score: 0.555**, up from **0.316** (+23.9 percentage points).
The new score was reported by the user on November 2, 2026; the earlier score
was also confirmed through the Kaggle API. The full-data refit is the next
comparison; no result from that refit is recorded yet.

## Progress

| Milestone | Local balanced accuracy | Kaggle public score |
| --- | --- | --- |
| TF-IDF + LinearSVC | 28.33%, fold 0 | — |
| Frozen DistilBERT + word TF-IDF | 29.72%, combined folds 0–1 | — |
| Fine-tuned DistilBERT + frozen hybrid | 32.32%, combined folds 0–1 | 0.316 |
| Qwen 0.6B LoRA, official-only | 28.33%, fold 0 | — |
| Qwen 0.6B LoRA + 720 external passages | 27.96%, fold 0 | — |
| Two-stage Qwen3-8B, plain + centered heads | 61.48%, exploratory fold 0 | **0.555** |

The two-stage model first trains on **34,896 prepared public responses**, then
fine-tunes for two epochs on **2,159 official rows**. It uses intermediate-layer
features and a second head trained to distinguish responses to the same prompt.
The current submission uses that fold-0 checkpoint; it is not a full-data refit.

Local and public scores are different evaluations. Official splits hold out
whole prompts. However, the public pool was filtered upstream using style
statistics from all official rows, so the new local result is **exploratory**.
The larger model, data, objective and compute changed together; the improvement
cannot be attributed solely to more data. No private leaderboard result is known.

## Evidence and current work

- [Progress and lessons](documentation/progress.md): what we tried and learned.
- [Latest findings](documentation/latest_findings.md): current conclusions and dated history.
- [Submission ledger](documentation/submission_results.csv): public scores and their provenance.
- [Experiment registry](documentation/experiment_registry.csv): configurations, budgets, hashes and metrics.
- [Archived evidence](documentation/experiment_archive): small configs, histories and metrics; no model weights.
- [Full-data refit](documentation/two_stage_full_refit.md): reuse the public-stage checkpoint, then train on all 2,699 official rows.
- [Kaggle submission](documentation/two_stage_submission.md): packaging and runtime reproduction checks.

## Code map

| Files | Purpose |
| --- | --- |
| `baseline.py`, `src/data_loaders.py`, `src/model.py` | Original baselines, grouped splits, fine-tuning and blends |
| `src/qwen_experiment.py`, `src/qwen_diagnostic.py`, `src/qwen_lora.py` | Earlier Qwen experiments and diagnostics |
| `src/prepare_public_pool.py`, `src/two_stage_lora.py` | Public-data preparation and two-stage / full-data training |
| `src/two_stage_inference.py`, `src/package_two_stage.py` | Offline inference and private Kaggle packaging |
| `scripts/*.sbatch` | User-launched LRZ GPU batch jobs |
| `tests/` | Data isolation, sampling, model plumbing and inference checks |
| `src/record_experiments.py` | Registry and evidence archival |

Older methods are retained because they explain the experiments, including the
approaches that did not improve results. The older full-data DistilBERT notebook
is historical; the current submission notebook is
[`ntl-two-stage-8b-fold0.ipynb`](notebooks/ntl-two-stage-8b-fold0.ipynb).

## Running locally

Data and checkpoints are not distributed in this repository. See the
[two-stage guide](documentation/two_stage_lora.md) for required inputs, provenance,
preparation and training instructions. Batch scripts contain LRZ-specific resource
requests; adjust them for your cluster before use. Kaggle packaging contains the
original account and run identifiers and should be adapted for another account.

`requirements.lock` pins the CPU baseline dependencies. GPU experiments additionally
need PyTorch, Transformers, PEFT, Safetensors and PyArrow; actual versions are saved
in each archived run configuration. Do not sync the CPU-only lock over an existing
GPU environment. Training is launched by the user, not by repository maintenance.

```bash
# Input validation only; no training or GPU model loading
python -m src.two_stage_lora check --pool data/public_pool_v1/pool.parquet

# Refresh experiment records after a run
python3 -m src.record_experiments
```

The test suite includes tiny model-fitting tests. Run it separately from production
training when desired: `python -m unittest discover -s tests -v`.

## Data and method attribution

The public pool comes from
[Nishant Dahal's two-stage dataset](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora)
and its credited upstream authors. The combined dataset is reported as CC-BY-SA-4.0;
individual source attribution remains in the downloaded metadata. The method is
informed by the supplied two-stage LoRA notebook and prior LoRA24/prompt-centering
work. See [the submission guide](documentation/two_stage_submission.md) and
[pool preparation audit](documentation/public_pool_preparation.json) for scope and
limitations. The base model is Qwen3-8B-Base (Apache-2.0).

Raw data, credentials, environments, logs, prediction files and model weights stay
local and are excluded from Git. Public evidence consists of source, documentation,
notebooks without outputs, and small experiment records.
