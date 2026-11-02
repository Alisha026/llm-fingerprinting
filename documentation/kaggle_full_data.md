# Run the second submission candidate

The old fold-0 submission scored 0.316. Keep it as the reference. This notebook
trains on all official labeled rows; a higher public score is not guaranteed.

1. Download `notebooks/kaggle_full_data.ipynb` from this project to your computer.
2. Create a new Kaggle notebook for Name That LLM, preserving your first notebook.
3. In the new notebook, select **File → Import Notebook** (or **Upload Notebook**)
   and select the downloaded `.ipynb` file. The imported cells contain all code.
4. Use **Add Input** to attach the existing private `llm-fingerprint-fold0`
   dataset and the **Name That LLM** competition. You do not upload another ZIP.
5. Select **GPU T4 x2** and set **Internet Off**.
6. Choose **Run All**. The notebook encodes all labeled passages, trains the
   hybrid classifier, fine-tunes a fresh encoder for nine epochs, and predicts.
   Normal initialization messages about a new classification head are expected.
7. Confirm it prints `Saved /kaggle/working/submission.parquet`. The few visible
   test rows are only a runtime check, not the public evaluation.
8. Click **Save Version → Save & Run All** and wait for completion. This starts
   a fresh run, including training. Do not rerun the training cell in a session
   with an existing `full_data_ensemble` folder; it intentionally refuses to
   overwrite a model. Restart a clean session if needed.
9. Open the completed saved version and choose **Submit to Competition**.
   Select `submission.parquet` if prompted. Do not submit the model ZIP.
10. Check **My Submissions** and record the new public score against 0.316.

The output also contains `llm-fingerprint-full-data.zip`, a reusable trained
model bundle. Download it from the completed notebook's output to preserve it.
The notebook uses one T4 GPU. It does not need TPU support, internet access,
paid APIs, or any additional training data.

Local submission tests require PyArrow (`pyarrow==25.0.1` was installed).

Validation: all 5 tests passed, including an offline integration test that trains
a tiny encoder, reloads both ensemble components, preserves test IDs, and writes
and reads submission.parquet. The full real-data GPU run has not yet been executed here.
