"""
Preprocessing DAG: inference-style audio (16kHz mono, normalize, trim — same as API input);
then build evaluation sets (dev/test/holdout) for running APIs and measuring quality.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_preprocess(**kwargs):
    import sys
    _dags_dir = Path(__file__).resolve().parent
    if str(_dags_dir) not in sys.path:
        sys.path.insert(0, str(_dags_dir))
    from dag_logging import log_dag_task_paths
    from scripts.utils import RAW_DIR, PROCESSED_DIR
    staged = PROCESSED_DIR / "staged"
    log_dag_task_paths(
        "preprocessing_dag", "preprocess_audio",
        read_from=[str(RAW_DIR)],
        write_to=[str(staged)],
    )
    from scripts.preprocess_audio import run_preprocessing
    ok, fail = run_preprocessing(raw_subdir=None)
    if fail > 0 and ok == 0:
        raise RuntimeError("All preprocessing failed")
    return {"ok": ok, "fail": fail}


def _run_split(**kwargs):
    import sys
    _dags_dir = Path(__file__).resolve().parent
    if str(_dags_dir) not in sys.path:
        sys.path.insert(0, str(_dags_dir))
    from dag_logging import log_dag_task_paths
    from scripts.stratified_split import run_split
    from scripts.utils import PROCESSED_DIR
    staged = PROCESSED_DIR / "staged"
    dev = PROCESSED_DIR / "dev"
    test = PROCESSED_DIR / "test"
    holdout = PROCESSED_DIR / "holdout"
    log_dag_task_paths(
        "preprocessing_dag", "stratified_split",
        read_from=[str(staged)],
        write_to=[str(dev), str(test), str(holdout)],
    )
    return run_split(staged_dir=staged, out_dir=PROCESSED_DIR)


_default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def get_tasks(dag: DAG):
    """Return tasks for this stage (for use in full_pipeline_dag)."""
    preprocess = PythonOperator(
        task_id="preprocess_audio",
        python_callable=_run_preprocess,
        dag=dag,
    )
    stratified_split = PythonOperator(
        task_id="stratified_split",
        python_callable=_run_split,
        dag=dag,
    )
    preprocess >> stratified_split
    return [preprocess, stratified_split]


with DAG(
    dag_id="preprocessing_dag",
    default_args=_default_args,
    description="Inference-style preprocessing + evaluation-set split (dev/test/holdout)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "preprocessing"],
) as dag:
    _tasks = get_tasks(dag)
