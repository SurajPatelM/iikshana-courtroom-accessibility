# Airflow (Iikshana pipeline orchestration)

DAGs and Docker setup for running the data pipeline. Pipeline code (scripts, config) lives in **`../data-pipeline/`** and is mounted into the container; data is stored at **repo root `../data/`** and mounted as `/workspace/data`.

## Quick start

From this directory (`airflow/`):

```bash
bash setup.sh          # one-time: .env + airflow-init
docker compose up      # start; then http://localhost:8080 (airflow / airflow)
```

Stop: `Ctrl+C`, then `docker compose down`. Reset DB: `docker compose down -v`.

## Layout

- **`dags/`** – Airflow DAGs (full_pipeline_dag, data_acquisition_dag, etc.)
- **`docker-compose.yaml`** – Postgres + webserver + scheduler; mounts repo root as `/workspace`, and `../data-pipeline/scripts`, `config`, `logs`
- **`setup.sh`** – One-time setup (AIRFLOW_UID, credentials, `docker compose up airflow-init`)
- **`.env.example`** – Template for `.env` (credentials; `.env` is gitignored)

## Where does data go?

When **data_acquisition_dag** runs, it downloads datasets into **repo root `data/raw/`** (RAVDESS, MELD, EMO-DB, etc. — see `data-pipeline/config/datasets.yaml` for URLs). Inside the container the path is `/workspace/data/raw`; the repo root is mounted as `/workspace`, so files appear in **`data/raw/`** on your machine. The repo keeps those folders in git via **`.gitkeep`** files: `data/.gitkeep`, `data/raw/.gitkeep`, `data/processed/.gitkeep`. If the folder stays empty, check the **download_datasets** task logs in the Airflow UI for errors and the logged `RAW_DIR` path; see **data-pipeline/README.md** (“Data folders and verifying acquisition”) to test the download script without Airflow.

## Login

Default after `setup.sh`: **Username** `airflow`, **Password** `airflow` (override in `.env`).

## DVC and GCS (optional)

The **full_pipeline_dag** can pull data from GCS at the start and push it at the end. This only runs when the following are set.

### Where to put the GCS bucket name and project

| What | Where |
|------|--------|
| **GCS bucket name** | In **`airflow/.env`**: set `DVC_GCS_BUCKET=your-bucket-name` (no `gs://`). The DAG uses it as `gs://${DVC_GCS_BUCKET}/dvc`. |
| **GCP project** | In your **service account JSON key file** (field `project_id`). You do not set the project name in Airflow; DVC/GCS use the project from the credentials. |

### Setup

1. **Credentials**  
   Place your GCP service account key at **repo root**: **`.secrets/gcp-dvc-key.json`** (the repo root is mounted as `/workspace` in Docker).  
   In **`airflow/.env`** set (or leave default):
   ```bash
   GOOGLE_APPLICATION_CREDENTIALS=/workspace/.secrets/gcp-dvc-key.json
   ```
   Do not commit `.secrets/` (it is in `.gitignore`).

2. **Bucket name**  
   In **`airflow/.env`** set:
   ```bash
   DVC_GCS_BUCKET=your-actual-bucket-name
   ```

3. **Rebuild**  
   After adding `dvc` and `dvc-gs` (they are in `docker-compose.yaml`), (re)build so the image has them:
   ```bash
   docker compose build --no-cache
   ```
   Or rely on the image’s `_PIP_ADDITIONAL_REQUIREMENTS` if your setup installs them at startup.

If `DVC_GCS_BUCKET` is not set, **dvc_pull** and **dvc_push** tasks no-op (exit 0) and the rest of the pipeline runs unchanged.

## References

- [Airflow Docker docs](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Airflow Lab tutorial](https://www.mlwithramin.com/blog/airflow-lab1)
