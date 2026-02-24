# Data Directory

---

## Structure

- `raw/`  
  Raw datasets used for benchmarking and bias analysis (e.g., RAVDESS, MELD, EMO-DB, Common Voice).  
  These are **not committed to git**; they are managed via **DVC** from `data-pipeline/`.

- `processed/`  
  Outputs of the data pipeline, including:
  - Preprocessed audio (16 kHz mono, normalized, trimmed).
  - `dev` / `test` / `holdout` splits (no speaker overlap).
  - Evaluation artefacts (metrics, reports).
  - `bias_report.json` and anomaly reports.


---

## Data Usage and Policy

- All datasets are used **only for offline testing, evaluation, and bias analysis**.
- **No live courtroom audio or operational data will be stored** here.
- No data from real proceedings will be reused for training.

The pipeline reads from `data/raw/` and writes to `data/processed/`, while the backend consumes selected artefacts (e.g., legal glossary, evaluation insights) at runtime.

For details on how this directory is wired into the pipeline and Airflow DAGs, see:

- `data-pipeline/README.md`
- `airflow/README.md`
