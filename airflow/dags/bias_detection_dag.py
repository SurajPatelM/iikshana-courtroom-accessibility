"""
Bias Detection DAG: slice by demographics, emotion, language, audio quality; report disparities and mitigation notes.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_bias_detection(**kwargs):
    from scripts.detect_bias import run_bias_analysis
    return run_bias_analysis(data_dir=PIPELINE_ROOT / "data" / "processed", report_path=PIPELINE_ROOT / "data" / "processed" / "bias_report.json")


_default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def get_tasks(dag: DAG):
    """Return tasks for this stage (for use in full_pipeline_dag)."""
    bias_task = PythonOperator(
        task_id="detect_bias_and_report",
        python_callable=_run_bias_detection,
        dag=dag,
    )
    return [bias_task]


with DAG(
    dag_id="bias_detection_dag",
    default_args=_default_args,
    description="Data slicing and bias/disparity analysis",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "bias"],
) as dag:
    _tasks = get_tasks(dag)
