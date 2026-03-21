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
- The model pipeline **uses these same splits** (dev, test, holdout). **Pipeline-owned files** stay under `data/processed/<split>/` (manifest, WAVs). **Model-run artifacts** (e.g. `translation_inputs.*`, `translation_predictions_*`, validation/bias outputs) default to **`data/model_runs/<split>/`**, with reads falling back to `data/processed/<split>/` if you still have an old layout. Optional per-split `translation_inputs` columns: `source_text`, `source_language`, `target_language` (and `reference_translation` for eval).
- If manifest-based and file-based inputs are both missing, the model pipeline skips translation for that split and exits successfully with a clear message. No changes to the data pipeline are required.

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
   - **Task 2 – `run_model_setup`**: Runs the Gemini translation on the chosen split and writes predictions.
4. **Trigger options**:
   - **Airflow (recommended)**: Run **`model_pipeline_dag`** from the Airflow UI (manual trigger). The DAG first pulls from GCS (when configured), then runs the model. No need to run the full data pipeline first on that machine.
   - **CLI**: From repo root with the right `PYTHONPATH`, run the script (see below). You must have processed data already present (e.g. from a previous DVC pull or local run).

When the model pipeline runs, it:

- (In Airflow) Pulls the latest processed data from GCS when `DVC_GCS_BUCKET` is set.
- Uses the **same split names** as the data pipeline: **dev**, **test**, **holdout**.
- **Primary path (pulled data):** Reads **`data/processed/<split>/manifest.json`** (and WAV paths relative to that split). For each manifest entry, calls the Gemini API (or uses STT when building inputs). By default processes the first 10 entries (`--max-rows 10`). Writes predictions under **`data/model_runs/<split>/`** (e.g. `translation_predictions_<config_id>.parquet`).
- **Optional override:** If `translation_inputs.csv` (or `.parquet`) exists under **model_runs** or **processed** for that split, that table is used and each row is sent to the translation API.
- If neither manifest nor translation_inputs exists, the script skips and exits 0 with a message.

---

## 3. Folder structure (model-pipeline)

```text
model-pipeline/
├── README.md                 # This file (Task 1: Clarify your setup + how to run)
├── scripts/
│   ├── model_setup.py                        # Entry point: load data, run Gemini translation, write predictions
│   ├── build_eval_dataset.py                 # 2.1: load pipeline output, split features/labels (translation, emotion, ASR)
│   ├── build_translation_inputs_from_audio.py # Build translation_inputs from manifest + WAVs + STT + ref
│   ├── build_translation_inputs_from_court_phrases.py # Optional: court-phrase translation_inputs
│   ├── build_combined_translation_inputs.py           # Merge translation_inputs + court_translation_inputs for combined eval
│   ├── build_asr_inputs_from_audio.py        # Build asr_inputs (file, reference_transcript) for WER
│   ├── run_config_search.py                  # 2.2: prompt & config search, BLEU, glossary, best config
│   ├── run_translation_eval.py               # Run translation on manifest or translation_inputs
│   ├── run_asr_eval.py                       # Run STT on asr_inputs, compute WER (target < 10%)
│   ├── run_emotion_eval.py                   # Compute emotion F1 from emotion_predictions (target > 0.70)
│   ├── run_validation.py                     # 2.3: full validation report (translation + emotion)
│   ├── run_model_bias_detection.py           # Modeling bias: API/predictions, Fairlearn slices, report
│   ├── model_bias_detection_core.py          # Helpers for run_model_bias_detection (metrics, disparities)
│   ├── build_emotion_dataset.py              # Model dev: manifest → emotion_dataset.csv
│   ├── train_emotion_model.py                # Model dev: train emotion classifier (MFCC)
│   ├── predict_emotion.py                    # Model dev: inference → emotion_predictions.csv
│   ├── build_model_package.py                # 2.6: build API model package (manifest + config + prompts)
│   └── push_model_to_registry.py             # 2.6: push package to GCP Artifact Registry (Docker or tarball)
├── artifacts/                                # Built model packages (build_model_package.py output)
├── docs/                                     # 2.6 and other task docs (e.g. 2.6_push_model_to_registry.md)
└── (optional: config overrides, outputs/, logs/)
```

- **Model configs and prompts** live at **repo root**: `config/models/*.yaml`, `prompts/*.txt`.
- **Data** at **repo root**: `data/processed/<split>/` — manifest and audio from the data pipeline. **`data/model_runs/<split>/`** — translation inputs, predictions, and eval artifacts from model-pipeline scripts (optional env `PIPELINE_DATA_DIR`, `MODEL_OUTPUT_ROOT`; legacy `--data-dir` uses one root for both).
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

python model-pipeline/scripts/model_setup.py \
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

### Modeling bias detection (API, sliced metrics)

Per-slice translation fairness for the **model/API** (separate from data-pipeline `detect_bias.py`): Fairlearn metrics over `dataset` (corpus), `emotion`, etc., plus disparity flags and mitigation text. New scripts: `scripts/run_model_bias_detection.py`, `scripts/model_bias_detection_core.py`.

```bash
export PYTHONPATH="${PWD}"
python model-pipeline/scripts/run_model_bias_detection.py \
  --split dev --config-id translation_flash_v1 --group-cols dataset,emotion
# Or reuse predictions only (no extra API calls):
python model-pipeline/scripts/run_model_bias_detection.py \
  --split dev --config-id translation_flash_v1 --from-predictions --group-cols dataset,emotion
```

Full reference: **`docs/model_bias_detection.md`**.

---

## 5. Summary (Task 1 checklist)

| Item                                                              | Status                                            |
| ----------------------------------------------------------------- | ------------------------------------------------- |
| Model: Pre-trained via API (Gemini)                               | Yes                                               |
| Task: Courtroom accessibility (translation)                       | Yes                                               |
| Data pipeline: Outputs clean, versioned train/val/test to storage | Yes (consumed from `data/processed/`)             |
| No weight training                                                | Yes                                               |
| Design prompts and model configs                                  | Yes (`config/models/`, `prompts/`)                |
| Evaluate performance and fairness                                 | Script in place; full metrics/bias in later tasks |
| Track experiments                                                 | To be extended (e.g. MLflow/W&B) in later tasks   |
| Package/version model config in registry                          | Yes (2.6: build_model_package + push_model_to_registry) |
| Model pipeline separate from data pipeline; manual trigger        | Yes (separate DAG, manual trigger only)           |

This completes **Task 1: Clarify your setup** for the model development guidelines.

---

## 6. Project proposal alignment (2.1 & 2.2)

The project proposal (*ADA Compliant Courtroom Visual Aid for Blind Individuals*) defines data splits, evaluation targets, and “training” as config/prompt search over pretrained models. This section maps the implementation to the proposal.

### 2.1 Loading data from your data pipeline

**Proposal (Section 3.3, 6.1.3):** Public datasets are ingested, validated, and stratified into **Dev 20%**, **Test 70%**, **Holdout 10%**. Data is used only for offline evaluation (no model training). Stratification by speaker identity, language, emotion class, demographics, and audio quality.

**Implementation:**

- **Data source:** Translation eval tables live under **`data/model_runs/<split>/`** by default (predictions, validation, bias outputs use the same root). **`data/processed/<split>/`** holds pipeline-owned artifacts (manifest, WAVs, QA JSON); `load_eval_dataset` looks for `translation_inputs.*` under **model_runs first**, then processed, for backward compatibility. Legacy: `--data-dir` on scripts sets one root for both. Optional env: `PIPELINE_DATA_DIR`, `MODEL_OUTPUT_ROOT`.
  - Per-split: `translation_inputs.csv` or `.parquet` (split = `dev` | `test` | `holdout`).
  - Optional single file: any path (e.g. `data/processed/val.parquet`).
  - In-memory: `split_features_labels(df)` for a DataFrame from BigQuery or elsewhere.
- **Module:** `model-pipeline/scripts/build_eval_dataset.py`:
  - `load_eval_dataset(split)` — resolve `translation_inputs.*` from model_runs then processed; optional `model_output_root=` / legacy `data_dir=`.
  - `load_eval_dataset_from_file(path)` — load from a single file.
  - `split_features_labels(df)` — split any table into features and labels.
- **Features vs labels:**
  - **Features** (prompt input): `source_text`, `source_language`, `target_language` (and any extra columns).
  - **Labels** (expected output): `reference_translation` (gold translation for metric computation).
- **Pipeline-derived eval data:** `build_translation_inputs_from_audio.py` builds `translation_inputs.csv` from pipeline output (manifest + WAVs, STT for source text, RAVDESS script for reference). The same pattern can be used for other datasets (IEMOCAP, CREMA-D, MELD, etc.) when manifests and references are produced.

**Conclusion:** 2.1 is implemented: we load from the pipeline’s processed splits, use the same split names (dev/test/holdout), and split into features (prompt input) and labels (expected output). The proposal also references **emotion** (F1) and **ASR** (WER) evaluation; the same loader pattern applies when those splits include the corresponding label columns (e.g. `emotion`, `reference_transcript`).

### 2.1 Clarifications (FAQ)

**Is 2.1 implemented properly?**  
Yes. The eval module loads `translation_inputs` from **`data/model_runs/<split>/` first**, then `data/processed/<split>/`, splits into features (e.g. `source_text`, `source_language`, `target_language`) and labels (`reference_translation`), and supports `load_eval_dataset(split)`, `load_eval_dataset_from_file(path)`, and `split_features_labels(df)` for any DataFrame source.

**Should `translation_inputs.csv` be based on audio files or court-related content?**  
Both are valid; you can use one or both.

- **Audio-based (current):** Built from the **data pipeline output**: manifest + WAV files → STT for `source_text`, known script (e.g. RAVDESS) for `reference_translation`. Use `build_translation_inputs_from_audio.py`. Good for benchmarking translation on real speech and matching the proposal’s use of public datasets (RAVDESS, IEMOCAP, etc.) for offline evaluation.
- **Court-related:** Short courtroom phrases (e.g. “Your Honor, I object.”) with reference translations. Good for legal glossary enforcement and courtroom relevance. Use the optional `build_translation_inputs_from_court_phrases.py` and `data/court_phrases.csv`; you can merge with audio-based inputs or keep a separate `court_translation_inputs.csv` for dev.

**Eval on both RAVDESS and court phrases:** The code supports both workflows. (1) **Run once per file and compare:** run config search with default (translation_inputs) then with `--inputs-basename court_translation_inputs`; results go to `config_search_results.json` and `config_search_results_court_translation_inputs.json`. (2) **Merge and run on combined:** run `build_combined_translation_inputs.py --split dev` to create `combined_translation_inputs.csv`, then run config search with `--inputs-basename combined_translation_inputs`.

The proposal uses public datasets for benchmarking and also cares about legal terminology and courtroom-style language; having both audio-derived and court-phrase eval sets aligns with that.

**Am I using BigQuery?**  
No. BigQuery is **optional**. The code does not depend on BigQuery. If you load a DataFrame from BigQuery (or any other source), pass it to `split_features_labels(df)` and use the same features/labels schema. No change is required for 2.1 if you are not using BigQuery.

**Should I do emotion (F1) and ASR (WER) with the same pattern?**  
Yes. The proposal (Section 7.1.2, 7.2.2) expects:
- **Emotion:** F1 on a held-out set (target > 0.70); bias audit by demographics.
- **ASR:** WER on streaming transcriptions vs reference (target &lt; 10%).

The same pattern applies: an eval table per task (e.g. `emotion_inputs.csv`, `asr_inputs.csv`) with **features** (e.g. audio path or transcript) and **labels** (emotion class or reference transcript) → load → run model/API → compute metric. The repo provides:
- **Emotion:** Schema and `run_emotion_eval.py` (load emotion inputs, compute F1 from predictions; emotion model integration is optional).
- **ASR:** Schema and `run_asr_eval.py` (load manifest + reference transcripts, run STT, compute WER).

---

### 2.2 “Training” & selecting the best model (prompt & config search)

**Proposal (Section 2.3, 6.1.3, 7.2.2):** No model training or fine-tuning. Pretrained Google (or API) models are used. Evaluation runs on validation data; **translation** is evaluated with BLEU (target **BLEU > 0.40**) and **legal glossary enforcement** (target **> 95%**). “Training” is interpreted as selecting the best **prompt and model configuration** by comparing metrics across candidate configs.

**Implementation:**

- **Script:** `model-pipeline/scripts/run_config_search.py`:
  - Loops over **candidate configs** (different models, prompts, parameters) defined in `config/models/*.yaml`.
  - For each config: runs the translation API on the chosen split (using `translation_inputs`), collects predictions, and computes a **translation metric** (BLEU or chrF).
  - Optionally computes **glossary enforcement rate** when `data/legal_glossary/legal_terms.json` exists (proposal target > 95%).
  - **Ranks** configs by the chosen metric and **selects the best**; writes `data/processed/<split>/config_search_results.json` with results, best config id, and proposal target (BLEU > 0.40).
- **Configs:** Multiple configs are provided (e.g. `translation_flash_v1`, `translation_flash_glossary`, `translation_flash_short_prompt`, `translation_flash_temp03`, `translation_hf_v1`) varying model, prompt, and temperature.
- **Prompts:** Stored under `prompts/` (e.g. `translation_baseline_system.txt`, `translation_baseline_user.txt`) and referenced by config YAMLs (`system_prompt_id`, `prompt_template_id`).
- **2.6 Push model to registry:** Build a model package (provider, name, prompts, parsing, version) and push as Docker image or tarball to GCP Artifact Registry. See **`model-pipeline/docs/2.6_push_model_to_registry.md`** and scripts `build_model_package.py`, `push_model_to_registry.py`.

**Conclusion:** 2.2 is implemented: we do not train weights; we run a **prompt & config search** over the validation set, compute translation metrics (BLEU/chrF) and optionally glossary enforcement, and select the best config. The written results file records whether the best config meets the proposal’s translation target (BLEU > 0.40).
