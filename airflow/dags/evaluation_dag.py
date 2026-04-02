"""
Evaluation DAG: run APIs (STT, translation, emotion) on evaluation set; compute WER, BLEU, F1; compare to targets.
Inference-style: we do not train models; we measure API quality on the evaluation sets.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path

# Resolve model-pipeline root — works both locally and inside Docker (/workspace mount)
_DAG_DIR = Path(__file__).resolve().parent
for _candidate in [
    Path("/workspace/model-pipeline"),
    _DAG_DIR.parent / "model-pipeline",
    _DAG_DIR.parent.parent / "model-pipeline",
]:
    if _candidate.exists() and (_candidate / "scripts" / "run_validation.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

# Also add repo root so backend services (gemini_translation etc.) are importable
for _root in [Path("/workspace"), _DAG_DIR.parent, _DAG_DIR.parent.parent]:
    if _root.exists() and (_root / "backend").exists():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break


def _run_evaluation(**kwargs):
    from scripts.run_validation import main as run_validation_main
    import sys as _sys
    # Run validation on dev split with default config
    _sys.argv = [
    "run_validation.py",
    "--split", "dev",
    "--configs", "translation_flash_court",
    "--inputs-basename", "translation_inputs",
    "--no-plots",
    "--max-rows", "20",
    "--delay", "0.5",
    ]
    run_validation_main()


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