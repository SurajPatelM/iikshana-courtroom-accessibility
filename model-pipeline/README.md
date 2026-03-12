# Model Pipeline (Task 1: Clarify your setup)

This folder implements the **model pipeline** for Iikshana: it runs **after** the data pipeline has produced processed, versioned datasets. The model pipeline applies configured prompts and Gemini API calls to that data for evaluation, bias analysis, and (later) registry packaging.

It is **separate** from the data pipeline: different DAG, manually triggered, and focused only on model/config application and evaluation.

---

## 1. Clarify your setup (assumptions)

### Model

- **Pre-trained model via API**: We use **Google Gemini** (e.g. `gemini-1.5-flash`, `gemini-1.5-pro`) via Vertex AI or the Generative Language API.
- We do **not** train or fine-tune any model. Our “model” is the **combination of**:
  - Model name and provider.
  - Prompt templates (in `prompts/`).
  - Generation parameters (temperature, top_p, max_output_tokens).
  - Task schema and parsing rules (translation: source → target language).

### Task

- **Courtroom accessibility – translation**: Given transcribed speech text (or courtroom-like utterances) in a **source language**, the system produces a **translation** in a **target language** of our choice.
- Input: `source_text`, `source_language`, `target_language` (and optional meta: speaker, emotion, domain).
- Output: `translated_text` from the Gemini API.
- Ground truth for evaluation: `reference_translation` from the data pipeline’s processed splits. We compute translation quality metrics (e.g. BLEU, COMET) against this reference.

### Data pipeline (upstream dependency)

- The **data pipeline** (in `data-pipeline/`, orchestrated by Airflow data DAGs) is **not modified** by the model pipeline. It produces:
  - **Existing splits**: `data/processed/dev/`, `data/processed/test/`, `data/processed/holdout/` (each with WAVs and `manifest.json`).
- The model pipeline **uses these same splits** (dev, test, holdout). For **translation**, it looks for an **optional** per-split file that you add yourself (so the data pipeline code stays untouched):
  - `data/processed/<split>/translation_inputs.csv` or `translation_inputs.parquet` with columns: `source_text`, `source_language`, `target_language` (and optionally `reference_translation`).
- If that file is missing, the model pipeline skips translation for that split and exits successfully with a clear message. No changes to the data pipeline are required.

### What we do and do not do

- **We do not**: Train or fine-tune model weights.
- **We do**:
  - **Design prompts and model configs** (see `config/models/`, `prompts/` at repo root).
  - **Evaluate performance and fairness** (validation metrics, bias by slice) on the processed data.
  - **Track experiments** (e.g. config id, metrics, fairness metrics per run).
  - **Package and version the “model configuration”** (prompt + settings + code) in a registry (e.g. GCP Artifact Registry) as a later task.

---

## 2. How the model pipeline runs

1. **Data pipeline** (separate) downloads, processes, and can push data to GCS (via `full_pipeline_dag` or scripts).
2. **Model pipeline is triggered manually** when you want to apply the model and prompts. You only need to click **Trigger** on the `model_pipeline_dag` in the Airflow UI.
3. **When you trigger the model pipeline DAG** (Airflow):
   - **Task 1 – `dvc_pull`**: If `DVC_GCS_BUCKET` is set in the Airflow environment (e.g. in `airflow/.env`), the DAG pulls the latest processed data from GCS into `/workspace/data/` so the next task has fresh data. If the bucket is not set, this task skips (and the eval task uses whatever data is already under `data/processed/`).
   - **Task 2 – `run_translation_eval`**: Runs the Gemini translation on the chosen split and writes predictions.
4. **Trigger options**:
   - **Airflow (recommended)**: Run **`model_pipeline_dag`** from the Airflow UI (manual trigger). The DAG first pulls from GCS (when configured), then runs the model. No need to run the full data pipeline first on that machine.
   - **CLI**: From repo root with the right `PYTHONPATH`, run the script (see below). You must have processed data already present (e.g. from a previous DVC pull or local run).

When the model pipeline runs, it:

- (In Airflow) Pulls the latest processed data from GCS when `DVC_GCS_BUCKET` is set.
- Uses the **same split names** as the data pipeline: **dev**, **test**, **holdout**.
- **Primary path (pulled data):** Reads `data/processed/<split>/manifest.json` (always present after DVC pull). For each manifest entry (file, dataset, speaker_id, emotion), calls the Gemini API to generate a short courtroom phrase and its translation. By default processes the first 10 entries (`--max-rows 10`). Writes `data/processed/<split>/translation_predictions_<config_id>.parquet`.
- **Optional override:** If `data/processed/<split>/translation_inputs.csv` (or `.parquet`) exists with columns `source_text`, `source_language`, `target_language`, that file is used instead and each row is sent to the translation API.
- If neither manifest nor translation_inputs exists, the script skips and exits 0 with a message.

---

## 3. Folder structure (model-pipeline)

```text
model-pipeline/
├── README.md                 # This file (Task 1: Clarify your setup + how to run)
├── scripts/
│   └── run_translation_eval.py   # Entry point: load data, run Gemini translation, write predictions
└── (optional later: config overrides, outputs/, logs/)
```

- **Model configs and prompts** live at **repo root**: `config/models/*.yaml`, `prompts/*.txt`.
- **Data** lives at **repo root**: `data/processed/dev`, `data/processed/test`, `data/processed/holdout` (written by the data pipeline). Optional translation inputs: `data/processed/<split>/translation_inputs.csv` or `.parquet` (you add these; the data pipeline is not modified).
- **Backend translation client** lives in `backend/src/services/gemini_translation.py`; the model-pipeline script reuses it.

---

## 4. Running the model pipeline

### Via Airflow (recommended after data pipeline has run)

1. Open Airflow UI at `http://localhost:8080`.
2. Find the DAG **`model_pipeline_dag`** (tag: `model`, `model-pipeline`).
3. Unpause it if needed, then click **Trigger DAG**.
4. The DAG will run the translation evaluation task(s) using the current config and data under `data/processed/`.

### Via CLI (for local testing)

From the **repository root**:

```bash
# Ensure dependencies and Google auth are set (e.g. gcloud auth application-default login)
export PYTHONPATH="${PWD}"

python model-pipeline/scripts/run_translation_eval.py \
  --split dev \
  --config-id translation_flash_v1
```

Optional arguments:

- `--split`: One of **dev**, **test**, **holdout** (same names as the data pipeline). Default: `dev`.
- `--config-id`: Model config id under `config/models/`, e.g. `translation_flash_v1`, `translation_pro_glossary_v1`.
- `--data-dir`: Override directory for processed data (default: `data/processed`).

**Using pulled data (default)**  
After DVC pull, each split contains `manifest.json` (file, dataset, speaker_id, emotion per WAV). The script uses this by default: it calls Gemini for each entry (up to `--max-rows`, default 10) to generate a courtroom phrase and its translation. No extra file is required.

**Optional: translation_inputs**  
If you add `data/processed/<split>/translation_inputs.csv` with columns `source_text`, `source_language`, `target_language`, the script uses that instead and translates each row.

---

## 5. Summary (Task 1 checklist)

| Item | Status |
|------|--------|
| Model: Pre-trained via API (Gemini) | Yes |
| Task: Courtroom accessibility (translation) | Yes |
| Data pipeline: Outputs clean, versioned train/val/test to storage | Yes (consumed from `data/processed/`) |
| No weight training | Yes |
| Design prompts and model configs | Yes (`config/models/`, `prompts/`) |
| Evaluate performance and fairness | Script in place; full metrics/bias in later tasks |
| Track experiments | To be extended (e.g. MLflow/W&B) in later tasks |
| Package/version model config in registry | To be implemented in later tasks |
| Model pipeline separate from data pipeline; manual trigger | Yes (separate DAG, manual trigger only) |

This completes **Task 1: Clarify your setup** for the model development guidelines.
