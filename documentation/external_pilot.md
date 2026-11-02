# External augmentation pilot v1

Prepared 720 external responses (40 per label), representing 115 external
prompt groups, in `data/external_pilot_v1/`. Source: SpeechMap rows within
[the response corpus](https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus).

The source choice keeps collection source constant across classes. The
[upstream repository](https://github.com/xlr8harder/speechmap-data) declares
Apache-2.0; the corpus flags provider output terms for review, and those flags
remain in provenance. This is an experimental local dataset, not an assertion
of organizer approval or independent authentication of every model identity.

Selection requires short text-only responses, exact prompt metadata,
request/returned model evidence, and no flagged truncation/reasoning or unknown
license. Weak-label Luna quarantine is excluded. Deduplication removes repeated
normalized answers and keeps one response per model/prompt. Equal per-label
sampling prioritizes shared prompts, without inspecting validation scores.
All official answers were checked for normalized exact and 5-word shingle
similarity/containment matches; no matches were found. This does not rule out
semantic paraphrases or shared prompts: official prompt wording is unavailable.
SpeechMap's domain differs from the competition and may help or hurt.

Files:
- `external.csv`: selected texts, labels, namespaced IDs and prompt groups.
- `selection_metadata.parquet`: original cell IDs, prompt text and quality flags.
- `provenance.parquet`: source URLs, requested/returned model IDs, context hashes,
  source row IDs, licenses and review flags for selected cells.
- `preparation.json`: hashes, filters, counts and limitations.
- `fold0/train.csv`: 2,879 rows; `fold0/validation.csv`: 540 unchanged official rows.
- `fold1/train.csv`: 2,880 rows; `fold1/validation.csv`: 539 unchanged official rows.

Reproduce preparation using a new output directory:

```bash
myenv/bin/python -m src.prepare_external_pilot --output data/external_pilot_v2
```

User-run controlled comparison, inside an A100 allocation:

```bash
myenv/bin/python -m src.qwen_lora \
  --train data/external_pilot_v1/fold0/train.csv \
  --validation data/external_pilot_v1/fold0/validation.csv \
  --epochs 6 --patience 2 \
  --output artifacts/qwen_lora_external_pilot_fold0
```

Compare against the original six-epoch LoRA run's 25.56% fold-0 score. Keep
settings fixed for fold 1 by substituting fold1 paths and output. The original
DistilBERT ensemble's 33.70% fold-0 score remains the stronger baseline.
Adding data at a fixed epoch count increases optimizer updates; this experiment
measures the practical augmentation recipe, not a compute-matched data effect.
No training was launched during preparation.
