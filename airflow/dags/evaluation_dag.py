"""
Evaluation DAG: run APIs (STT, translation, emotion) on evaluation set; compute WER, BLEU, F1; compare to targets.
Inference-style: we do not train models; we measure API quality on the evaluation sets.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_evaluation(**kwargs):
    from scripts.evaluate_models import run_evaluation
    return run_evaluation(data_dir=PIPELINE_ROOT / "data" / "processed", metrics_path=PIPELINE_ROOT / "data" / "processed" / "evaluation_metrics.json", use_live_apis=False)


_default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def get_tasks(dag: DAG):
    """Return tasks for this stage (for use in full_pipeline_dag)."""
    eval_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=_run_evaluation,
        dag=dag,
    )
    return [eval_task]


with DAG(
    dag_id="evaluation_dag",
    default_args=_default_args,
    description="Run APIs on evaluation set: WER, BLEU, F1 vs targets",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "evaluation"],
) as dag:
    _tasks = get_tasks(dag)
