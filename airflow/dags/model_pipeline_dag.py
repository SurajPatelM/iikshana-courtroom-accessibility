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

from airflow import DAG  # type: ignore[import-untyped]
from airflow.operators.bash import BashOperator  # type: ignore[import-untyped]

default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# Run translation eval from repo root so backend and config are on PYTHONPATH.
RUN_BUILD_TRANSLATION_INPUTS_FROM_AUDIO = """
set -e
echo "=== Model pipeline: build translation_inputs.csv (audio -> STT -> reference mapping) ==="
export PYTHONPATH=/workspace
cd /workspace
SPLIT="{{ params.get('split', 'dev') }}"
INPUTS_PATH="data/processed/${SPLIT}/translation_inputs.csv"
if [ -s "$INPUTS_PATH" ]; then
  echo "[SKIP] translation_inputs already exists: $INPUTS_PATH"
  exit 0
fi
python model-pipeline/scripts/build_translation_inputs_from_audio.py \
  --split "${SPLIT}" \
  --max-rows 20 \
  --delay 0.5
echo "=== build translation_inputs: done ==="
"""

RUN_CONFIG_SEARCH = """
set -e
echo "=== Model pipeline: config search (select best config_id) ==="
export PYTHONPATH=/workspace
cd /workspace
SPLIT="{{ params.get('split', 'dev') }}"
RESULTS_JSON="data/processed/${SPLIT}/config_search_results.json"
if [ -s "$RESULTS_JSON" ]; then
  echo "[SKIP] config_search_results already exists: $RESULTS_JSON"
  exit 0
fi
CONFIG_IDS="translation_flash_v1,translation_flash_glossary,translation_flash_court,translation_flash_short_prompt,translation_flash_temp03,translation_groq_llama70b_v1"
python model-pipeline/scripts/run_config_search.py \
  --split "${SPLIT}" \
  --configs "${CONFIG_IDS}" \
  --metric bleu \
  --delay 0.0 \
  --output "data/processed/${SPLIT}/config_search_results.json"
echo "=== config search: done ==="
"""

RUN_MODEL_SETUP = """
set -e
echo "=== Model pipeline: translation evaluation (best config selected) ==="
export PYTHONPATH=/workspace
cd /workspace
SPLIT="{{ params.get('split', 'dev') }}"
RESULTS_JSON="data/processed/${SPLIT}/config_search_results.json"
BEST_CONFIG_ID=$(python -c "import json; print(json.load(open('${RESULTS_JSON}', 'r', encoding='utf-8'))['best_config_id'])" 2>/dev/null || echo "{{ params.get('config_id', 'translation_groq_llama70b_v1') }}")
echo "Best config_id = ${BEST_CONFIG_ID}"
python model-pipeline/scripts/model_setup.py \
  --split "${SPLIT}" \
  --config-id "${BEST_CONFIG_ID}"
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
        "config_id": "translation_groq_llama70b_v1",
    },
) as dag:

    build_translation_inputs_from_audio = BashOperator(
        task_id="build_translation_inputs_from_audio",
        bash_command=RUN_BUILD_TRANSLATION_INPUTS_FROM_AUDIO,
        dag=dag,
    )

    run_config_search = BashOperator(
        task_id="run_config_search",
        bash_command=RUN_CONFIG_SEARCH,
        dag=dag,
    )

    run_model_setup = BashOperator(
        task_id="mode_setup",
        bash_command=RUN_MODEL_SETUP,
        dag=dag,
    )

    build_translation_inputs_from_audio >> run_config_search >> run_model_setup
