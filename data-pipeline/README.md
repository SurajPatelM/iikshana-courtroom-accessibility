# Iikshana Data Pipeline

Data pipeline for the Iikshana ADA-compliant courtroom visual aid system: data acquisition, **inference-style preprocessing**, **API-input validation**, bias detection, and **API evaluation** (WER, BLEU, F1). Built with **Apache Airflow**, **DVC**, and **Great Expectations**-style validation.

## Pipeline principles (inference-first)

This pipeline is **inference- and API-oriented**, not training-oriented. We **do not train models**. All stages align with how we send data to **Gemini and other APIs** in production:

- **Preprocessing** = exactly what we do before calling Gemini/APIs at inference (16 kHz mono, normalize, trim). Same spec for pipeline batches and for live backend input.
- **Validation** = API-input quality: format, sample rate, duration. Same checks for evaluation data and for live inference.
- **Splits (dev / test / holdout)** = **evaluation sets** only. We run Gemini/APIs on these sets to measure quality (WER, BLEU, F1). No train split.
- **Evaluation** = run APIs (STT, translation, emotion) on the evaluation set and compare to targets.

One preprocessing path, one validation contract, evaluation = API runs on evaluation sets.

## Project Overview

- **Data acquisition**: Download RAVDESS, IEMOCAP, CREMA-D, MELD, TESS, SAVEE, EMO-DB, Common Voice, etc.
- **Preprocessing (inference-style)**: 16 kHz mono WAV, loudness normalization, silence trimming — same as live input to APIs. Then stratified **evaluation sets** (dev 20%, test 70%, holdout 10%) with no speaker overlap.
- **Validation (API-input)**: Schema checks (sample rate, duration, format), emotion labels; ensures audio is valid for API consumption.
- **Gemini verification (optional)**: Smoke test that pipeline output is accepted by the Gemini API (one or a few WAVs). Skippable via `RUN_GEMINI_VERIFICATION=false`; when enabled, proves end-to-end compatibility.
- **Bias detection**: Slicing by demographics, emotion, language, audio quality; disparity reports and mitigation notes.
- **Evaluation**: Run STT (Chirp 3), translation, emotion detection via APIs on evaluation set → WER, BLEU, F1 vs targets (WER < 10%, BLEU > 0.40, F1 > 0.70).
- **Anomaly detection**: Missing/corrupt files, duration distribution, label imbalance; optional alerts.

## Environment Setup

- **Python**: 3.10+ recommended.
- **Airflow**: Install separately (see below). Use a dedicated venv for Airflow if you prefer.

### 1. Create virtual environment and install dependencies

```bash
cd data-pipeline
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Airflow (orchestration)

DAGs and Docker setup live in the **`airflow/`** directory at the repo root (not inside data-pipeline). The container mounts this directory’s `scripts`, `config`, and `data`.

From the repo root:

```bash
cd airflow
bash setup.sh          # one-time: .env + airflow-init
docker compose up      # start; then http://localhost:8080 (airflow / airflow)
```

See **`airflow/README.md`** for details. Allocate at least 4GB memory for Docker (ideally 8GB).

### 3. DVC (data versioning)

```bash
pip install dvc dvc-gs   # dvc-gs for Google Cloud Storage
cd data-pipeline
dvc init
# Optional: add remote (GCS or local)
# dvc remote add -d storage gs://your-bucket/dvc
```

## Running the Pipeline

### Without Airflow (scripts only)

Run stages in order:

```bash
cd data-pipeline
# 1. Download datasets (configure URLs in config/datasets.yaml)
python scripts/download_datasets.py [RAVDESS MELD ...]

# 2. Preprocess (inference-style: same as live API input) and build evaluation sets
python scripts/preprocess_audio.py
python scripts/stratified_split.py

# 3. Validate (API-input: schema + quality report, then Great Expectations + statistics)
python scripts/validate_schema.py
python scripts/run_great_expectations.py

# 3b. Gemini verification (optional: spot-check that preprocessed WAVs work with Gemini API)
#     Set GEMINI_API_KEY or GOOGLE_API_KEY; use --force to run without RUN_GEMINI_VERIFICATION.
python scripts/verify_gemini_audio.py [data/processed] [--max-files 2] [--force]

# 4. Bias report (uses Fairlearn for per-group slicing)
python scripts/detect_bias.py

# 5. Evaluation (run APIs on evaluation set; placeholder metrics if live APIs disabled)
python scripts/evaluate_models.py

# 6. Anomaly check (fails on anomalies; use anomaly_detection_dag for email alert)
python scripts/anomaly_check.py

# 7. Legal glossary (from repo data/legal_glossary)
python scripts/legal_glossary_prep.py
```

### Data folders and verifying acquisition

- **Data directories** are kept in git via `.gitkeep`:
  - `data-pipeline/data/.gitkeep`
  - `data-pipeline/data/raw/.gitkeep`
- **Raw downloads** (RAVDESS, MELD, EMO-DB, etc.) go into **`data-pipeline/data/raw/`**. When using Airflow (Docker), the container writes to `/opt/airflow/data/raw`, which is mounted from this `data/` folder, so files appear here after **data_acquisition_dag** runs.
- **To confirm the DAG is working:** After running **data_acquisition_dag** in the Airflow UI, check that `data/raw/` contains files (e.g. `RAVDESS/`, `RAVDESS.zip`, `MELD/`, etc.). In the UI, open the run → **download_datasets** task → **Log** and look for `RAW_DIR=...` and any download or error lines.
- **To test download without Airflow** (same script the DAG uses):
  ```bash
  cd data-pipeline
  python -m scripts.download_datasets
  # or: python scripts/download_datasets.py
  ```
  Then list raw: `ls -la data/raw/`.

### With DVC

```bash
cd data-pipeline
dvc repro
```

### With Airflow

1. **Start Airflow from this project** so the UI uses this repo’s DAGs:
   ```bash
   cd data-pipeline
   ./run_airflow.sh
   ```
   Or manually: `export AIRFLOW_HOME="$(pwd)/airflow_home"` then `airflow standalone`. If you start Airflow from elsewhere or without `AIRFLOW_HOME`, the UI will use a different `dags_folder` and **full_pipeline_dag will not appear**.
2. In the UI, open the **DAGs** tab (main nav). Search for `full` or `pipeline` if the list is long. Unpause the DAG with the toggle if needed.
3. To confirm Airflow can see the DAG: from `data-pipeline` run `./check_dags.sh`. **If the CLI shows `full_pipeline_dag` but the UI does not**, the browser is almost certainly connected to a *different* Airflow (e.g. one started without `AIRFLOW_HOME` or from another directory). **Fix:** stop any other Airflow (close other terminals running `airflow standalone` or `airflow webserver`), then from `data-pipeline` run `./run_airflow.sh` and open http://localhost:8080 → click **DAGs** in the top nav (not Home) → search for `full` or `pipeline`.
4. Ensure `dags_folder` in `airflow_home/airflow.cfg` points to this repo’s `data-pipeline/dags` (done by `./setup_airflow.sh`).
5. **Single flow (recommended):** Trigger the full pipeline DAG to run all stages in sequence:
   ```bash
   export AIRFLOW_HOME="$(pwd)/airflow_home"
   airflow dags trigger full_pipeline_dag
   ```
   If tasks fail in the UI with **empty logs** (Airflow 3 executor/scheduler issue), run the DAG in-process to see output and trigger the stage DAGs:
   ```bash
   export AIRFLOW_HOME="$(pwd)/airflow_home"
   airflow dags test full_pipeline_dag 2025-01-01
   ```
   Or trigger each stage DAG individually (modular):
   ```bash
   airflow dags trigger data_acquisition_dag
   airflow dags trigger preprocessing_dag
   airflow dags trigger validation_dag
   airflow dags trigger gemini_verification_dag   # optional; set RUN_GEMINI_VERIFICATION=true to run check
   airflow dags trigger bias_detection_dag
   airflow dags trigger evaluation_dag
   airflow dags trigger anomaly_detection_dag
   ```

6. Use the Airflow UI (Gantt chart) to parallelize independent tasks and optimize bottlenecks (e.g. parallel dataset downloads).
7. **Gemini verification (optional)**: The pipeline includes an optional stage that sends one or two preprocessed WAVs to the Gemini API to confirm the format is accepted end-to-end. By default it **does not run** (no-op success). To enable: set env `RUN_GEMINI_VERIFICATION=true` and `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the Airflow task environment, or run manually: `python scripts/verify_gemini_audio.py --force` from `data-pipeline`.
8. **Alerts**: When anomalies are detected, `anomaly_detection_dag` fails and Airflow sends an **email** (configure SMTP in `[email]` in `airflow.cfg`). For **Slack**, add an `on_failure_callback` that posts to a Slack webhook (see Airflow docs).

## Data Sources

| Dataset        | Description                    | URL / note                          |
|----------------|--------------------------------|-------------------------------------|
| RAVDESS        | Emotional speech/song          | Zenodo (see `config/datasets.yaml`) |
| IEMOCAP        | Multimodal emotions            | License required                     |
| CREMA-D        | Emotion recognition            | License required                     |
| MELD           | Multimodal EmotionLines        | GitHub                               |
| TESS / SAVEE / EMO-DB | Emotion datasets       | Configure in `config/datasets.yaml`  |
| Common Voice   | Multilingual speech            | Hugging Face / Mozilla               |

Add or update URLs and checksums in `config/datasets.yaml`.

## DVC Commands

- **Track data** (after downloads and processing):

  ```bash
  dvc add data/raw/RAVDESS data/raw/IEMOCAP
  dvc add data/processed/dev data/processed/test data/processed/holdout   # evaluation sets
  dvc add data/legal_glossary
  git add *.dvc .gitignore
  git commit -m "Track pipeline data with DVC"
  ```

- **Pull data** (e.g. on another machine):

  ```bash
  dvc pull
  ```

- **Reproduce pipeline**:

  ```bash
  dvc repro
  ```

## Testing

```bash
cd data-pipeline
pip install -r requirements.txt
pytest tests/ -v
```

- `tests/test_preprocessing.py`: Audio normalization, resampling, `process_one`, `collect_audio_files`.
- `tests/test_validation.py`: Schema validation, manifest label checks.
- `tests/test_splitting.py`: Stratified split, no speaker overlap, `infer_speaker_id` / `infer_emotion`.

## Folder Structure

```
data-pipeline/
├── dags/                          # DAGs moved to repo root airflow/dags/ (see dags/README.md)
├── scripts/
│   ├── download_datasets.py
│   ├── preprocess_audio.py
│   ├── stratified_split.py
│   ├── validate_schema.py
│   ├── run_great_expectations.py   # Schema + statistics (PDF §2.7)
│   ├── verify_gemini_audio.py     # Optional: Gemini API spot-check (Option A/B)
│   ├── detect_bias.py              # Uses Fairlearn (PDF §3.2)
│   ├── evaluate_models.py
│   ├── legal_glossary_prep.py
│   ├── anomaly_check.py
│   └── utils.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   └── test_splitting.py
├── data/
│   ├── raw/              # DVC tracked
│   ├── processed/        # evaluation sets (dev, test, holdout) + reports (DVC tracked)
│   └── legal_glossary/   # DVC tracked
├── config/
│   └── datasets.yaml
├── logs/
├── dvc.yaml
├── requirements.txt
└── README.md
```

## Bias Detection Results

The bias detection step (`scripts/detect_bias.py`, `bias_detection_dag`) produces `data/processed/bias_report.json` with:

- Counts per emotion and per speaker.
- Disparities (e.g. strong class imbalance).
- Recommendations: stratified evaluation, confidence thresholding, re-sampling.
- **Fairlearn** (PDF §3.2): per-group metrics via `MetricFrame`; see `fairlearn_by_group` in the report.

Summarize any findings (e.g. “Female voices in TESS show X% better emotion detection than male in SAVEE”) in this section after running on your data.

---

## Bias Detection and Mitigation Process (PDF §3.4)

This section documents the steps we take to detect and mitigate bias, the types of bias we look for, how we address them, and any trade-offs.

### 1. Steps to detect bias

1. **Data slicing**: We slice the processed dataset by **sensitive features** (emotion label, speaker identity as a proxy for demographics). Slicing is implemented using **Fairlearn**’s `MetricFrame` (PDF §3.2) so that metrics (e.g. sample count) are computed per subgroup.
2. **Disparity detection**: We compare counts per slice to an expected balanced proportion (e.g. equal share per emotion). Any slice that deviates beyond a threshold (e.g. &gt;50% above/below expected) is flagged as a disparity in `bias_report.json`.
3. **Report generation**: The pipeline writes `data/processed/bias_report.json` with `by_emotion`, `by_speaker_count`, `fairlearn_by_group`, `disparities`, and `recommendations`.

### 2. Types of bias we look for

- **Emotion-class imbalance**: Some emotions (e.g. “neutral”, “happy”) may have many more samples than others (e.g. “disgust”), which can skew evaluation and deployment.
- **Speaker/demographic skew**: When speaker_id is correlated with demographics (e.g. gender or accent in CREMA-D), uneven representation across speakers can lead to worse performance for under-represented groups.
- **Split imbalance**: Evaluation sets (dev/test/holdout) may have different label or speaker distributions; we check that splits are stratified by speaker (no overlap) and report per-split counts.

### 3. How we address bias (mitigation)

- **Re-sampling**: We recommend re-sampling under-represented emotion classes or speakers when calibrating or evaluating APIs (documented in `recommendations` in the report).
- **Stratified evaluation**: We use stratified evaluation sets (by speaker) and recommend reporting metrics **per slice** (e.g. per emotion, per speaker group) so that API performance gaps are visible.
- **Confidence thresholding**: For deployment, we recommend confidence thresholding or calibration for under-represented classes to reduce disparate impact.

### 4. Trade-offs

- **Re-sampling** can improve balance but may over-represent rare classes and slightly change the effective distribution; we use it only where imbalance exceeds our threshold.
- **Stratified evaluation** adds reporting overhead but surfaces API performance gaps that need to be addressed by data collection, re-sampling, or API/prompt design.
- **Confidence thresholding** may reduce overall accuracy slightly in exchange for more equitable performance across groups; the exact thresholds should be tuned on holdout data.

## Pipeline Optimization

- **Bottlenecks**: Use Airflow’s Gantt chart to find long-running tasks (e.g. large dataset downloads). Parallelize independent downloads in the acquisition DAG.
- **Preprocessing**: Scripts support batch processing; for very large corpora, consider multiprocessing or chunked runs (e.g. by dataset or speaker).
- **Validation**: Run validation and anomaly checks after preprocessing so failures are caught before evaluation.

## Replicating on Another Machine

1. Clone the repo and go to `data-pipeline`.
2. Create a venv, install dependencies: `pip install -r requirements.txt`.
3. Install Airflow if using DAGs; set `AIRFLOW_HOME` and point `dags_folder` to `data-pipeline/dags`.
4. Configure `config/datasets.yaml` (and optional DVC remote).
5. Pull data: `dvc pull` (if remotes are set).
6. Run pipeline: `dvc repro` or trigger Airflow DAGs in order.
7. Run tests: `pytest tests/ -v`.

## Logging

Scripts use Python `logging`; logs are written under `logs/` with timestamps. Each DAG task logs start, progress, and completion; validation and anomaly results are also logged.

## License

Same as the parent Iikshana repository.
