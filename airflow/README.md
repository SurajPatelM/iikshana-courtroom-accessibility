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

## References

- [Airflow Docker docs](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Airflow Lab tutorial](https://www.mlwithramin.com/blog/airflow-lab1)
