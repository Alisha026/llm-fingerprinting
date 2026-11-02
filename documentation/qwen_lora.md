# Qwen task adaptation experiment

The cached-vector diagnostic reproduced the original probabilities (maximum
absolute difference below 1.5e-7). The best frozen-Qwen hybrid scored 23.52%
on fold 0, versus 27.04% for the TF-IDF-only logistic control. Standardization
did not rescue this setup. These are exploratory fold-0 results.

Next hypothesis: supervised adapters can make the existing embedding encoder
more useful for model attribution than its frozen representation. This is not a
reproduction of a public competition notebook, and it may still overfit.

`src/qwen_lora.py` uses the already downloaded Qwen3-Embedding-0.6B checkpoint.
It trains rank-8 LoRA adapters on q_proj/v_proj and an 18-class classification
head. Base weights remain frozen. Initial fixed settings: alpha 16, dropout 0.1,
learning rate 1e-4, effective batch size 32, 512 tokens, at most six epochs,
early stopping after two non-improving validation epochs. Best checkpoint
selection uses balanced accuracy, not training loss. The same settings should
be confirmed on fold 1 before drawing conclusions.

Setup on an internet-connected node (user runs):

```bash
source myenv/bin/activate
python -m pip install 'peft==0.21.0'
```

Run inside an A100 GPU allocation:

```bash
python -m src.qwen_lora \
  --train data/processed/train.csv \
  --validation data/processed/validation.csv \
  --output artifacts/qwen_lora_fold0
```

Then confirm without changing settings:

```bash
python -m src.qwen_lora \
  --train data/processed_fold1/train.csv \
  --validation data/processed_fold1/validation.csv \
  --output artifacts/qwen_lora_fold1
```

If memory is insufficient, use `--batch-size 2 --gradient-accumulation 16`
and a new output directory. No model download is needed. The script checks
CUDA execution, requires native BF16 hardware, validates split separation,
records provenance/versions, and refuses nonempty output directories.

Expected classification-head initialization warnings are normal because this
checkpoint did not previously contain our attribution head. The saved
`best_adapter` includes adapters and the classification head, but requires the
original base encoder to reload. No Kaggle packaging is provided for this
candidate until its validation results are assessed.

Compare against the unchanged original ensemble (33.70% fold 0, 30.93% fold 1),
and consider a fixed blend only after inspecting the standalone results.
Saved probability files use the existing blending format. Treat both folds
as development validation; repeated selection on them is not an unbiased final
estimate. Reserve other prompt folds for later confirmation.

Preparation checks: Python compilation and CLI help passed. PEFT is not yet
installed here; adapter loading and GPU training have not been runtime-tested.
No experiment training was launched by the assistant.

References:
- https://huggingface.co/docs/peft/main/en/package_reference/lora
- https://huggingface.co/docs/transformers/main/en/model_doc/qwen3
