"""
Gemini verification DAG: optional smoke test that pipeline output works with Gemini API.
Runs only when RUN_GEMINI_VERIFICATION=true (and GEMINI_API_KEY or GOOGLE_API_KEY is set);
otherwise the single task no-ops successfully so the main pipeline is not blocked.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_gemini_verification(**kwargs):
    from scripts.verify_gemini_audio import run_verification, EXIT_FAILURE
    data_dir = PIPELINE_ROOT / "data" / "processed"
    result = run_verification(data_dir=data_dir, max_files=2, prefer_split="staged", force_run=False)
    if result.get("skipped"):
        return result
    if result.get("exit_code") == EXIT_FAILURE or not result.get("success"):
        raise RuntimeError(
            "Gemini verification failed (exit_code=%s): %s"
            % (result.get("exit_code"), result.get("results", result))
        )
    return result  # full result (exit_code, results with api_response per file) for logs


_default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="gemini_verification_dag",
    default_args=_default_args,
    description="Optional: verify preprocessed audio with Gemini API (set RUN_GEMINI_VERIFICATION=true)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "gemini", "verification"],
) as dag:
    verify_gemini_audio = PythonOperator(
        task_id="verify_gemini_audio",
        python_callable=_run_gemini_verification,
        dag=dag,
    )
