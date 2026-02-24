# Iikshana Data Pipeline

Data pipeline for the Iikshana ADA-aligned courtroom accessibility system.  
Implements data acquisition, preprocessing, validation, testing, logging, DVC versioning, Airflow orchestration, anomaly detection, and bias analysis.

---

## 1. Goals and Design Principles

- **Inference-first**: Pipeline mirrors what the live system sends to APIs (16 kHz mono WAV, normalization, trimming). We do **not** train models here.
- **API-input validation**: Everything produced is checked for format, duration, sample rate, and schema before being sent to APIs.
- **Reproducible and versioned**: Code in git, data under DVC, pipeline defined in `dvc.yaml` and Airflow DAGs.
- **Bias- and anomaly-aware**: Built-in data slicing, fairness checks, anomaly detection, and alert hooks.

Stages:

1. **Data acquisition**
2. **Preprocessing**
3. **Schema & statistics generation**
4. **Validation + anomaly detection**
5. **Bias detection and mitigation analysis**
6. **Evaluation (WER / BLEU / F1)**
7. **Tracking, logging, and optimization**

---

## 2. Folder Structure (within `data-pipeline/`)

```text
data-pipeline/
├── scripts/           # All pipeline stages (see below)
├── tests/             # pytest unit tests for key components
├── config/            # Dataset configuration (URLs, checksums, splits)
├── logs/              # Pipeline logs (per run / per task)
├── dvc.yaml           # DVC pipeline definition and stages
├── requirements.txt   # Data-pipeline dependencies (see repo root)
└── README.md
```

Data itself lives at repo root in `data/` (see `data/README.md`).

Key scripts:

- `download_datasets.py` – **Data acquisition**
- `preprocess_audio.py` – **Preprocessing** (resample, normalize, trim)
- `stratified_split.py` – Build **dev / test / holdout** splits (no speaker overlap)
- `validate_schema.py` – Basic **schema and quality checks**
- `run_great_expectations.py` – Extended **statistics and validation**
- `verify_gemini_audio.py` – Optional **API-format smoke test**
- `detect_bias.py` – **Bias detection via slicing + Fairlearn**
- `evaluate_models.py` – **Evaluation** (WER, BLEU, F1)
- `anomaly_check.py` – **Anomaly detection + failure hook**
- `legal_glossary_prep.py` – Prepare legal glossary artefacts

---

## 3. Environment Setup & Dependencies

**Python**: 3.10+ recommended.

```bash
cd data-pipeline
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

This installs all pipeline requirements (numpy, pandas, librosa, Great Expectations-style validation, Fairlearn, DVC, pytest, etc.).  
Airflow and Docker are configured from the repo-root `airflow/` directory (see that README for details).

---

## 4. Running the Pipeline (Scripts Only)

From `data-pipeline/`, run stages in order:

```bash
cd data-pipeline

# 1. Data acquisition
python scripts/download_datasets.py [RAVDESS MELD ...]

# 2. Preprocessing + splits
python scripts/preprocess_audio.py
python scripts/stratified_split.py

# 3. Schema, statistics, and validation
python scripts/validate_schema.py
python scripts/run_great_expectations.py

# 4. Optional Gemini/ API verification
python scripts/verify_gemini_audio.py [data/processed] [--max-files 2] [--force]

# 5. Bias detection
python scripts/detect_bias.py

# 6. Evaluation (WER / BLEU / F1)
python scripts/evaluate_models.py

# 7. Anomaly detection
python scripts/anomaly_check.py

# 8. Legal glossary prep
python scripts/legal_glossary_prep.py
```

All paths are relative to repo root `data/` (see below).

---

## 5. Data Layout and DVC (Reproducibility)

Data lives at **repo root**:

```text
data/
├── raw/          # Raw downloads (RAVDESS, MELD, EMO-DB, etc.)
├── processed/    # Preprocessed audio + splits + reports
└── legal_glossary/
```

DVC is initialized in `data-pipeline/`:

```bash
cd data-pipeline

# Track data artefacts
dvc add ../data/raw/RAVDESS ../data/raw/IEMOCAP
dvc add ../data/processed/dev ../data/processed/test ../data/processed/holdout
dvc add ../data/legal_glossary
git add *.dvc .gitignore
git commit -m "Track pipeline data with DVC"

# Reproduce entire pipeline
dvc repro

# Pull data on a new machine
dvc pull

---

## 6. Airflow Orchestration

Airflow DAGs and Docker setup live in repo-root `airflow/`. They orchestrate the same stages described above:

- `data_acquisition_dag`
- `preprocessing_dag`
- `validation_dag`
- `gemini_verification_dag` (optional)
- `bias_detection_dag`
- `evaluation_dag`
- `anomaly_detection_dag`
- `full_pipeline_dag` (end-to-end)

See `airflow/README.md` for:

- How to run `bash setup.sh && docker compose up`
- How data is mounted at `/workspace/data`
- How to use the **Gantt chart** to identify bottlenecks and optimize flow (parallel downloads, etc.)

---

## 7. Testing, Logging, and Error Handling

### Unit tests

```bash
cd data-pipeline
pip install -r requirements.txt
pytest tests/ -v
```

Coverage:

- `tests/test_preprocessing.py` – normalization, resampling, and helper functions.
- `tests/test_validation.py` – schema validation, label/manifest checks.
- `tests/test_splitting.py` – stratified splits, no speaker overlap.

### Logging and error handling

- All scripts use Python `logging` and write to `data-pipeline/logs/`.
- Each Airflow task logs start, progress, and completion.
- `anomaly_check.py` and the anomaly DAG **fail fast** on schema/quality violations so Airflow can trigger alerts (email/Slack as configured).

---

## 8. Bias Detection and Mitigation

`scripts/detect_bias.py` and `bias_detection_dag` implement the **data slicing and bias analysis** requested in the rubric:

- Slices by emotion and speaker (proxy for demographic groups).
- Computes per-slice metrics via **Fairlearn** `MetricFrame`.
- Writes `data/processed/bias_report.json` with:
  - Counts per emotion and per speaker
  - Detected disparities
  - Recommendations (re-sampling, stratified evaluation, confidence thresholds)


---

## 9. Replicating on Another Machine

1. Clone the repo and `cd data-pipeline`.
2. Create a virtual environment and `pip install -r requirements.txt`.
3. Configure `config/datasets.yaml` and any credentials (if using private datasets or GCS).
4. (Optional) Configure DVC remote and run `dvc pull`.
5. Run the script-only pipeline or trigger Airflow DAGs.
6. Run tests: `pytest tests/ -v`.

