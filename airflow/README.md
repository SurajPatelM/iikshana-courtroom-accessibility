# Airflow (Iikshana Pipeline Orchestration)

This directory contains **Apache Airflow** configuration and DAGs for the Iikshana data pipeline

- End-to-end **pipeline orchestration** (data acquisition → preprocessing → validation → evaluation → bias / anomaly checks).
- **Tracking & logging** of each task.
- **Flow optimization** using Airflow’s Gantt chart.

Pipeline code (scripts, tests, config) lives in `../data-pipeline/` and is mounted into the container. Data lives at repo root `../data/` and is mounted as `/workspace/data`.

---

## Quick Start (Docker Compose)

From this directory (`airflow/`):

```bash
bash setup.sh          # one-time: .env + airflow-init
docker compose up      # start webserver, scheduler, DB
```

Then open `http://localhost:8080` in your browser.

- Default login (can be changed in `.env`):  
  - **Username**: `airflow`  
  - **Password**: `airflow`

Stop: `Ctrl+C`, then:

```bash
docker compose down          # stop containers
docker compose down -v       # stop + reset DB (if needed)
```

Allocate **4–8 GB RAM** to Docker for smooth dataset downloads and processing.

---

## Layout

- `dags/` – All DAG definitions:
  - `full_pipeline_dag.py`
  - `data_acquisition_dag.py`
  - `preprocessing_dag.py`
  - `validation_dag.py`
  - `gemini_verification_dag.py` (optional)
  - `bias_detection_dag.py`
  - `evaluation_dag.py`
  - `anomaly_detection_dag.py`
- `docker-compose.yaml` – Postgres, webserver, scheduler; mounts:
  - Repo root as `/workspace`
  - `../data-pipeline/` scripts, config, and logs
- `setup.sh` – One-time initialization (`.env`, `AIRFLOW_UID`, `airflow-init`)
- `.env.example` – Template for environment variables (copy to `.env`)

These DAGs orchestrate the same stages described in `data-pipeline/README.md`.

### DAGs Overview

- **`full_pipeline_dag`**: Runs the entire pipeline end-to-end (acquisition → preprocessing → validation → evaluation → bias + anomaly checks, plus optional DVC pull/push).
- **`data_acquisition_dag`**: Downloads all required datasets into `data/raw/` according to `data-pipeline/config/datasets.yaml`.
- **`preprocessing_dag`**: Runs inference-style preprocessing and creates stratified `dev` / `test` / `holdout` splits.
- **`validation_dag`**: Generates schema and statistics, runs validation checks (including Great-Expectations-style reports).
- **`gemini_verification_dag`** (optional): Sends a small sample of preprocessed audio to the Gemini API to verify that the pipeline output is accepted end-to-end.
- **`bias_detection_dag`**: Performs data slicing and bias analysis, writing `bias_report.json` under `data/processed/`.
- **`evaluation_dag`**: Runs API-based evaluation (e.g., STT, translation, emotion) and computes metrics such as WER, BLEU, and F1.
- **`anomaly_detection_dag`**: Detects anomalies (missing files, distribution shifts, schema violations) and fails the DAG so Airflow can trigger alerts.

---

## Data Locations

Inside the container:

- Code: `/workspace/data-pipeline/`
- Data: `/workspace/data/`

On your machine (repo root):

- `data/raw/` – Raw downloads (RAVDESS, MELD, EMO-DB, etc.)
- `data/processed/` – Preprocessed audio, splits, reports

When **`data_acquisition_dag`** runs, it writes directly into `data/raw/`.  
If `data/raw/` stays empty, inspect the **`download_datasets`** task logs in the Airflow UI and compare the `RAW_DIR` path with your repo.

---

## DVC & GCS Integration (Optional)

The `full_pipeline_dag` can run **DVC pull / push** steps against a Google Cloud Storage remote so that data is versioned and shared across machines.

1. **Credentials**

   - Place your service account key at repo root: `.secrets/gcp-dvc-key.json` (gitignored).
   - In `airflow/.env`:

   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=/workspace/.secrets/gcp-dvc-key.json
   ```

2. **Bucket name**

   In `airflow/.env`:

   ```bash
   DVC_GCS_BUCKET=your-bucket-name   # without gs://
   ```

   DAGs will use `gs://${DVC_GCS_BUCKET}/dvc` as the remote.

3. **Install DVC in the image**

   Ensure `dvc` and `dvc-gs` are available via `docker-compose.yaml` or `_PIP_ADDITIONAL_REQUIREMENTS`, then rebuild:

   ```bash
   docker compose build --no-cache
   ```

If `DVC_GCS_BUCKET` is not set, DVC tasks (`dvc_pull`, `dvc_push`) are treated as no-ops and the rest of the pipeline still runs.

---

## Monitoring, Logging, and Optimization

- Use the **DAG graph** and **Gantt chart** to:
  - Inspect task durations.
  - Identify bottlenecks (e.g., slow dataset downloads).
  - Parallelize independent tasks where safe.
- Each task logs to Airflow’s logging system; see per-task logs in the UI.

### Alerts on failure (anomaly_detection_dag and task failures)

When a task fails (e.g. anomalies detected), alerts can be sent via **email** and **Slack**:

1. **Email**  
   In `airflow/.env` set SMTP and recipient list:
   - `AIRFLOW__SMTP__SMTP_HOST`, `AIRFLOW__SMTP__SMTP_PORT`, `AIRFLOW__SMTP__SMTP_USER`, `AIRFLOW__SMTP__SMTP_PASSWORD`, `AIRFLOW__SMTP__SMTP_MAIL_FROM` (see [Airflow email config](https://airflow.apache.org/docs/apache-airflow/stable/howto/email-config.html)).
   - `ALERT_EMAIL=you@example.com,other@example.com` (comma-separated; these receive failure emails).
   Restart: `docker compose down && docker compose up -d`.

2. **Slack**  
   Create an [Incoming Webhook](https://api.slack.com/messaging/webhooks) in your Slack workspace, then in `airflow/.env` set:
   - `SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL`
   The `anomaly_detection_dag` uses `on_failure_callback` in `dags/callbacks.py` to post a message on task failure. Restart containers after changing `.env`.


---

## References

- Airflow Docker docs: `https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html`
- Example lab / tutorial: `https://www.mlwithramin.com/blog/airflow-lab1`
