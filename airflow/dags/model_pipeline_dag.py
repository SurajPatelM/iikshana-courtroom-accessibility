"""
Model pipeline DAG: run translation evaluation on already-available processed data.

This DAG is separate from the data pipeline. It assumes data has already been
pulled / produced (for example by the full_pipeline_dag's initial DVC pull and
subsequent stages) and therefore **does not perform any DVC operations itself**.

- schedule: None (manual trigger only)
- Not triggered by full_pipeline_dag
- Tags: model, model-pipeline
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# Run translation eval from repo root so backend and config are on PYTHONPATH.
RUN_TRANSLATION_EVAL = """
set -e
echo "=== Model pipeline: translation evaluation ==="
export PYTHONPATH=/workspace
cd /workspace
python model-pipeline/scripts/model_setup.py \
  --split "{{ params.get('split', 'dev') }}" \
  --config-id "{{ params.get('config_id', 'translation_flash_v1') }}"
echo "=== Model pipeline: done ==="
"""

with DAG(
    dag_id="model_pipeline_dag",
    default_args=default_args,
    description="Pull data from GCS then apply Gemini translation (run manually).",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "model", "model-pipeline"],
    params={
        "split": "dev",
        "config_id": "translation_flash_v1",
    },
) as dag:

    run_translation_eval = BashOperator(
        task_id="mode_setup",
        bash_command=RUN_TRANSLATION_EVAL,
        dag=dag,
    )

    run_translation_eval
