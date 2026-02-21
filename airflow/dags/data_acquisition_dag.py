"""
Data Acquisition DAG: download emotion & speech datasets, validate checksums, store in data/raw/ (DVC tracked).
Runs the same download command as CLI (python -m scripts.download_datasets) so behavior matches manual runs.
"""
from datetime import datetime, timedelta
import sys
from pathlib import Path

from airflow import DAG  # type: ignore[import-untyped]
from airflow.operators.bash import BashOperator  # type: ignore[import-untyped]
from airflow.operators.python import PythonOperator  # type: ignore[import-untyped]

# In Docker, pipeline root is /opt/airflow (scripts, config, data mounted there)
_DAG_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = _DAG_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _ensure_scripts_on_path():
    """Ensure pipeline root (so 'scripts' can be imported) is on sys.path at task runtime."""
    # Prefer known Docker path so scripts are found regardless of which process runs the task
    for root in [
        Path("/opt/airflow"),
        PIPELINE_ROOT,
        _DAG_DIR.parent.parent / "data-pipeline",
    ]:
        if root.exists() and (root / "scripts" / "utils.py").exists():
            s = str(root)
            if s not in sys.path:
                sys.path.insert(0, s)
            return root
    return PIPELINE_ROOT


# Same command as: cd data-pipeline && python -m scripts.download_datasets (in container: /opt/airflow)
DOWNLOAD_CMD = "cd /opt/airflow && PYTHONPATH=/opt/airflow python -m scripts.download_datasets"


def _validate_checksums():
    # Resolve pipeline root at runtime so scripts are found (task may run in different process)
    _ensure_scripts_on_path()
    from scripts.download_datasets import compute_sha256
    from scripts.utils import RAW_DIR
    out = {}
    for p in RAW_DIR.rglob("*"):
        if p.is_file() and p.suffix != ".dvc":
            try:
                out[str(p.relative_to(RAW_DIR))] = compute_sha256(p)
            except Exception:
                pass
    return out


_default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=0),
}


def get_tasks(dag: DAG):
    """Return tasks for this stage (for use in full_pipeline_dag). Last task is the final one in the chain."""
    download_task = BashOperator(
        task_id="download_datasets",
        bash_command=DOWNLOAD_CMD,
        execution_timeout=timedelta(hours=2),
        dag=dag,
    )
    validate_checksums = PythonOperator(
        task_id="validate_checksums",
        python_callable=_validate_checksums,
        dag=dag,
    )
    download_task >> validate_checksums
    return [download_task, validate_checksums]


with DAG(
    dag_id="data_acquisition_dag",
    default_args=_default_args,
    description="Download RAVDESS, IEMOCAP, CREMA-D, MELD, Common Voice, etc. into data/raw/",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "acquisition"],
) as dag:
    _tasks = get_tasks(dag)
