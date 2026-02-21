"""
Validation DAG: API-input validation (sample rate, duration, format, emotion labels);
quality report and JSON schema. Same checks as for live inference.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_validation(**kwargs):
    from scripts.validate_schema import run_validation
    from scripts.utils import PROCESSED_DIR
    r = run_validation(data_dir=PIPELINE_ROOT / "data" / "processed", schema_out=PROCESSED_DIR / "audio_schema.json", report_out=PROCESSED_DIR / "quality_report.json")
    if r.get("failed", 0) > 0 and r.get("passed", 0) == 0:
        raise RuntimeError("All validation checks failed")
    return r


def _run_great_expectations(**kwargs):
    """Run Great Expectations for schema and statistics (PDF §2.7)."""
    from scripts.run_great_expectations import run_ge_validation_and_statistics
    from scripts.utils import PROCESSED_DIR
    out = run_ge_validation_and_statistics(
        data_dir=PIPELINE_ROOT / "data" / "processed",
        result_path=PROCESSED_DIR / "ge_validation_result.json",
        statistics_path=PROCESSED_DIR / "data_statistics.json",
    )
    if not out.get("success", True):
        raise RuntimeError("Great Expectations validation failed")
    return out


_default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def get_tasks(dag: DAG):
    """Return tasks for this stage (for use in full_pipeline_dag)."""
    validate = PythonOperator(
        task_id="validate_schema_and_quality",
        python_callable=_run_validation,
        dag=dag,
    )
    ge_validate = PythonOperator(
        task_id="run_great_expectations",
        python_callable=_run_great_expectations,
        dag=dag,
    )
    validate >> ge_validate
    return [validate, ge_validate]


with DAG(
    dag_id="validation_dag",
    default_args=_default_args,
    description="API-input validation and schema (Great Expectations style)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "validation"],
) as dag:
    _tasks = get_tasks(dag)
