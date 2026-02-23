# Airflow (Iikshana pipeline orchestration)

DAGs and Docker setup for running the data pipeline. Pipeline code (scripts, config) lives in **`../data-pipeline/`** and is mounted into the container; data is stored at **repo root `../data/`** and mounted as `/workspace/data`.

## Quick start

From this directory (`airflow/`):

```bash
bash setup.sh          # one-time: .env + airflow-init
docker compose build   # build image with DVC + pipeline deps (first time or after Dockerfile change)
docker compose up      # start; then http://localhost:8080 (airflow / airflow)
```

Stop: `Ctrl+C`, then `docker compose down`. Reset DB: `docker compose down -v`.

## Full pipeline DAG (end-to-end with DVC)

**`full_pipeline_dag`** implements: **dvc pull → if no data → acquisition → preprocessing → … → dvc push**; if data is already present (e.g. after pull), acquisition is skipped and no push is run.

1. **dvc_pull** – Restore DVC-tracked data from GCS. First run: nothing to pull (task still succeeds).
2. **branch_on_data** – If `data/raw/` or `data/processed/dev/` has content → **skip_acquisition**. Else → **trigger_data_acquisition**.
3. **When no data:** trigger_data_acquisition → trigger_preprocessing → trigger_validation → trigger_anomaly_detection → trigger_bias_detection → trigger_gemini_verification → **dvc_commit_push** (commit + push to GCS).
4. **When data present:** skip_acquisition only; no acquisition or dvc push.

So: **first run (DVC empty)** → pull does nothing → branch sees no data → acquisition runs → download, process, validate, etc. → dvc push. **Later runs** → pull restores data → branch skips acquisition. Task logs for **branch_on_data** and **download_datasets** show the paths checked and where data is written (`REPO_ROOT/data/raw`).

### GCP credentials for DVC in Docker

For **dvc_pull** and **dvc_commit_push** to access GCS, the container needs GCP credentials. Mount a service account key and set `GOOGLE_APPLICATION_CREDENTIALS` (e.g. in `.env` or docker-compose `environment`). Example: add to your compose override or env: `GOOGLE_APPLICATION_CREDENTIALS=/workspace/.gcp/key.json` and mount the key file into the container. Without credentials, DVC tasks will fail with 401. See **docs/DVC_GCS_SETUP.md** for GCS setup.

## Verifying DAGs via CLI and logs

Each DAG task logs **where it reads from and where it writes to** using a `[DAG_LOG]` prefix so you can confirm paths and that the task ran.

**Trigger a single DAG** (with Airflow already up):

```bash
cd airflow
docker compose exec airflow-scheduler airflow dags trigger <dag_id>
```

**Test a single task** (dry run; uses current date as execution date):

```bash
docker compose exec airflow-scheduler airflow dags test <dag_id> <task_id> <execution_date>
# e.g. airflow dags test data_acquisition_dag download_datasets 2026-02-23
```

**In task logs** (Airflow UI → DAG → run → task → Log), look for:

- `[DAG_LOG] DAG_ID=... TASK_ID=...`
- `[DAG_LOG] DATA_ROOT=...` (and `RAW_DIR`, `PROCESSED_DIR` where used)
- `[DAG_LOG] READ_FROM:` / `[DAG_LOG] WRITE_TO:` listing paths

This lets you verify each DAG is running and using `data-pipeline/data/` as intended.

## Layout

- **`dags/`** – Airflow DAGs (full_pipeline_dag, data_acquisition_dag, etc.)
- **`docker-compose.yaml`** – Postgres + webserver + scheduler; builds **`Dockerfile`** (DVC + pipeline deps); mounts repo root as `/workspace`, and `../data-pipeline/scripts`, `config`, `logs`
- **`Dockerfile`** – Extends Airflow image with certifi, dvc, dvc-gs, and pipeline Python deps
- **`setup.sh`** – One-time setup (AIRFLOW_UID, credentials, `docker compose up airflow-init`)
- **`.env.example`** – Template for `.env` (credentials; `.env` is gitignored)

## Where does data go?

When **data_acquisition_dag** runs, it downloads datasets into **repo root `data/raw/`** (RAVDESS, MELD, EMO-DB, etc. — see `data-pipeline/config/datasets.yaml` for URLs). Inside the container the path is `/workspace/data/raw`; the repo root is mounted as `/workspace`, so files appear in **`data/raw/`** on your machine. The download task log starts with "Writing data to …/data/raw" so you can confirm the path. The repo keeps those folders in git via **`.gitkeep`** files: `data/.gitkeep`, `data/raw/.gitkeep`, `data/processed/.gitkeep`. If the folder stays empty, confirm **branch_on_data** chose **trigger_data_acquisition** (not skip), then check **download_datasets** task logs for errors and that at least one dataset has a valid URL in `datasets.yaml`; see **data-pipeline/README.md** (“Data folders and verifying acquisition”) to test the download script without Airflow.

## Login

Default after `setup.sh`: **Username** `airflow`, **Password** `airflow` (override in `.env`).

## References

- [Airflow Docker docs](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Airflow Lab tutorial](https://www.mlwithramin.com/blog/airflow-lab1)
- [DVC + GCS setup](../docs/DVC_GCS_SETUP.md)
