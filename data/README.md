# Data Directory

---

## Structure

- `raw/`  
  Raw datasets used for benchmarking and bias analysis (e.g., RAVDESS, MELD, EMO-DB, Common Voice).  
  These are **not committed to git**; they are managed via **DVC** from `data-pipeline/`.

- `processed/`  
  Outputs of the data pipeline:
  - **`emotions/`** — emotion datasets (staged + `dev` / `test` / `holdout`), manifests, and what the model pipeline reads by default.
  - **`stt/`** — speech-only corpora (Common Voice, LibriSpeech, VoxPopuli), same split layout, kept separate from emotion benchmarks.
  - **`legal/`** — reserved for courtroom/legal audio (e.g. Oyez) when added.
  - Top-level JSON reports: `quality_report.json`, `bias_report.json`, `anomaly_report.json`, `evaluation_metrics.json`, etc.


---

## Data Usage and Policy

- All datasets are used **only for offline testing, evaluation, and bias analysis**.
- **No live courtroom audio or operational data will be stored** here.
- No data from real proceedings will be reused for training.

The pipeline reads from `data/raw/` and writes under `data/processed/emotions/` and `data/processed/stt/` (plus top-level reports under `data/processed/`). The backend consumes selected artefacts (e.g., legal glossary, evaluation insights) at runtime.

For details on how this directory is wired into the pipeline and Airflow DAGs, see:

- `data-pipeline/README.md`
- `airflow/README.md`
