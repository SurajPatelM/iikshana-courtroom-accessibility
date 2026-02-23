"""
Full pipeline DAG: DVC pull → [branch] → acquisition (or skip) → preprocessing → … → DVC commit & push.

Flow:
  - dvc_pull: restore DVC-tracked data from GCS (first run: no data to pull, task still succeeds).
  - branch_on_data: if REPO_ROOT/data/raw or data/processed/dev has content → skip_acquisition; else → trigger_data_acquisition.
  - If no data: run data acquisition (download) → preprocessing → validation → anomaly → bias → gemini → dvc_commit_push.
  - If data present: skip acquisition; no dvc push (data already in DVC from pull).
Data written by acquisition lives at REPO_ROOT/data/ (in Docker /workspace/data/ = repo root data/ on host).
"""
import os
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

try:
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# Env for DVC tasks: ensure GCP credentials and REPO_ROOT are passed to the subprocess (BashOperator may not inherit container env).
REPO_ROOT = os.environ.get("REPO_ROOT", "/workspace")
# Data is stored under data-pipeline/data/ inside the repo
PIPELINE_ROOT = Path(os.environ.get("PIPELINE_ROOT", os.path.join(REPO_ROOT, "data-pipeline")))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PIPELINE_ROOT / "data")))
DVC_TASK_ENV = {
    **os.environ,
    "GOOGLE_APPLICATION_CREDENTIALS": "/workspace/data-pipeline/iikshana-mlops-dac50f075ba0.json",
    "REPO_ROOT": REPO_ROOT,
}


def _data_already_present(**kwargs) -> str:
    """Return 'skip_acquisition' if data/raw or data/processed has real content; else 'trigger_data_acquisition'."""
    import sys
    _dags_dir = Path(__file__).resolve().parent
    if str(_dags_dir) not in sys.path:
        sys.path.insert(0, str(_dags_dir))
    from dag_logging import log_dag_task_paths
    log_dag_task_paths(
        "full_pipeline_dag", "branch_on_data",
        read_from=[str(DATA_ROOT / "raw"), str(DATA_ROOT / "processed" / "dev")],
        write_to=[],
        extra={"action": "check_if_data_present"},
    )
    # Look in data-pipeline/data/ (via DATA_ROOT), not repo root data/
    raw_dir = DATA_ROOT / "raw"
    processed_dev = DATA_ROOT / "processed" / "dev"
    # Consider data present if raw has any file/dir besides .gitkeep, or processed dev exists with content
    raw_entries = list(raw_dir.iterdir()) if raw_dir.exists() else []
    raw_has_data = any(p.name != ".gitkeep" for p in raw_entries)
    processed_has_data = processed_dev.exists() and any(processed_dev.iterdir())
    if raw_has_data or processed_has_data:
        print(f"[branch_on_data] DATA_ROOT={DATA_ROOT} | data present (raw={raw_has_data}, processed/dev={processed_has_data}) -> skip_acquisition")
        return "skip_acquisition"
    print(f"[branch_on_data] DATA_ROOT={DATA_ROOT} | no data in {raw_dir} / {processed_dev} -> trigger_data_acquisition")
    return "trigger_data_acquisition"


default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
        dag_id="full_pipeline_dag",
        default_args=default_args,
        description="DVC pull → acquisition → preprocessing → validation → anomaly → bias → gemini → DVC commit & push",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["iikshana", "data", "pipeline", "full", "dvc"],
) as dag:
    # Restore DVC-tracked data from GCS; do not fail task when nothing to pull or checkout fails (first run).
    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=(
            'echo "[DAG_LOG] ========== DAG run ==========" && '
            'echo "[DAG_LOG] DAG_ID=full_pipeline_dag TASK_ID=dvc_pull" && '
            'echo "[DAG_LOG] READ_FROM: DVC remote (GCS). WRITE_TO: ${REPO_ROOT:-.} (repo root, data-pipeline/data)" && '
            'echo "[DAG_LOG] DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data-pipeline/data}" && '
            'echo "[DAG_LOG] ================================" && '
            'cd "${REPO_ROOT:-.}" && '
            '(python3 -c "import certifi; print(certifi.where())" 2>/dev/null | while read p; do export SSL_CERT_FILE="$p" REQUESTS_CA_BUNDLE="$p"; done); '
            "dvc pull --force || true"
        ),
        env=DVC_TASK_ENV,
        execution_timeout=timedelta(minutes=15),
        dag=dag,
    )

    # Decide whether to run acquisition or skip (data already present from pull).
    branch = BranchPythonOperator(
        task_id="branch_on_data",
        python_callable=_data_already_present,
        dag=dag,
    )

    skip_acquisition = BashOperator(
        task_id="skip_acquisition",
        bash_command=(
            'echo "[DAG_LOG] ========== DAG run ==========" && '
            'echo "[DAG_LOG] DAG_ID=full_pipeline_dag TASK_ID=skip_acquisition" && '
            'echo "[DAG_LOG] READ_FROM: already in DATA_ROOT (no new read). WRITE_TO: none (skip)." && '
            'echo "[DAG_LOG] DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data-pipeline/data}" && '
            'echo "[DAG_LOG] ================================" && '
            'echo "Data already present from DVC pull; skipping acquisition."'
        ),
        dag=dag,
    )

    trigger_acquisition = TriggerDagRunOperator(
        task_id="trigger_data_acquisition",
        trigger_dag_id="data_acquisition_dag",
        wait_for_completion=True,
        dag=dag,
    )

    trigger_preprocessing = TriggerDagRunOperator(
        task_id="trigger_preprocessing",
        trigger_dag_id="preprocessing_dag",
        wait_for_completion=True,
        dag=dag,
    )

    trigger_validation = TriggerDagRunOperator(
        task_id="trigger_validation",
        trigger_dag_id="validation_dag",
        wait_for_completion=True,
        dag=dag,
    )

    trigger_anomaly = TriggerDagRunOperator(
        task_id="trigger_anomaly_detection",
        trigger_dag_id="anomaly_detection_dag",
        wait_for_completion=True,
        dag=dag,
    )

    trigger_bias_detection = TriggerDagRunOperator(
        task_id="trigger_bias_detection",
        trigger_dag_id="bias_detection_dag",
        wait_for_completion=True,
        dag=dag,
    )

    trigger_gemini_verification = TriggerDagRunOperator(
        task_id="trigger_gemini_verification",
        trigger_dag_id="gemini_verification_dag",
        wait_for_completion=True,
        dag=dag,
    )

    # Update DVC lock from pipeline outputs and push to GCS
    dvc_commit_push = BashOperator(
        task_id="dvc_commit_push",
        bash_command=(
            'echo "[DAG_LOG] ========== DAG run ==========" && '
            'echo "[DAG_LOG] DAG_ID=full_pipeline_dag TASK_ID=dvc_commit_push" && '
            'echo "[DAG_LOG] READ_FROM: ${REPO_ROOT:-.}/data-pipeline (dvc.yaml, data/). WRITE_TO: DVC remote (GCS)." && '
            'echo "[DAG_LOG] DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data-pipeline/data}" && '
            'echo "[DAG_LOG] ================================" && '
            'cd "${REPO_ROOT:-.}" && '
            '(python3 -c "import certifi; print(certifi.where())" 2>/dev/null | while read p; do export SSL_CERT_FILE="$p" REQUESTS_CA_BUNDLE="$p"; done); '
            'cd data-pipeline && dvc commit && cd "${REPO_ROOT:-.}" && dvc push'
        ),
        env=DVC_TASK_ENV,
        execution_timeout=timedelta(minutes=30),
        dag=dag,
    )

    # dvc_pull → branch → [trigger_acquisition → … → dvc_commit_push] or [skip_acquisition]
    dvc_pull >> branch
    branch >> trigger_acquisition
    branch >> skip_acquisition
    (
        trigger_acquisition
        >> trigger_preprocessing
        >> trigger_validation
        >> trigger_anomaly
        >> trigger_bias_detection
        >> trigger_gemini_verification
        >> dvc_commit_push
    )