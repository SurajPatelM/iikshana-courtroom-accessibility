"""
Model pipeline DAG: run translation evaluation on already-available processed data.

This DAG is separate from the data pipeline. It assumes data has already been
pulled / produced (for example by the full_pipeline_dag's initial DVC pull and
subsequent stages) and therefore **does not perform any DVC operations itself**.

For **low-latency EXPO / production-style inference** without config search, use
``expo_translation_dag`` instead (triggered by default from ``demo/gradio_expo_app.py``).

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

# Trigger conf examples:
#   Expo (rebuild inputs only, reuse best config): {"split":"dev","refresh_inputs":true,"refresh_config_search":false}
#   Full refresh including config search: add "refresh_config_search":true
# Jinja: dag_run.conf may be None on first parse in some contexts; use (dag_run.conf or {}).
_RUN_TMPL_SPLIT = "{{ (dag_run.conf or {}).get('split', params.get('split', 'dev')) }}"
_RUN_TMPL_REFRESH_INPUTS = "{{ 'true' if (dag_run.conf or {}).get('refresh_inputs') else 'false' }}"
_RUN_TMPL_REFRESH_CFG = "{{ 'true' if (dag_run.conf or {}).get('refresh_config_search') else 'false' }}"
_RUN_TMPL_TLANG = "{{ (dag_run.conf or {}).get('target_language', 'es') }}"
# Last N manifest entries for build (higher = more rows in translation_inputs; not append-per-run).
_RUN_TMPL_MANIFEST_TAIL = "{{ (dag_run.conf or {}).get('manifest_tail', 200) }}"
# Space Groq translation calls in mode_setup to avoid 429 when translation_inputs has many rows.
_RUN_TMPL_TRANSLATE_DELAY = "{{ (dag_run.conf or {}).get('translate_delay', 0.6) }}"

# Run translation eval from repo root so backend and config are on PYTHONPATH.
RUN_BUILD_TRANSLATION_INPUTS_FROM_AUDIO = f"""
set -e
echo "=== Model pipeline: build translation_inputs.csv (audio -> STT -> reference mapping) ==="
export PYTHONPATH=/workspace
cd /workspace
SPLIT="{_RUN_TMPL_SPLIT}"
REFRESH_INPUTS="{_RUN_TMPL_REFRESH_INPUTS}"
TARGET_LANG="{_RUN_TMPL_TLANG}"
MANIFEST_TAIL="{_RUN_TMPL_MANIFEST_TAIL}"
MR_INPUTS="/workspace/data/model_runs/${{SPLIT}}/translation_inputs.csv"
PR_INPUTS="/workspace/data/processed/${{SPLIT}}/translation_inputs.csv"
if [ "$REFRESH_INPUTS" = "true" ]; then
  echo "[INFO] refresh_inputs: removing stale translation_inputs (UI / manual follow-up)"
  rm -f "$MR_INPUTS" "$PR_INPUTS" || true
fi
if [ -s "$MR_INPUTS" ] || [ -s "$PR_INPUTS" ]; then
  echo "[SKIP] translation_inputs already exists under model_runs or processed"
  exit 0
fi
python model-pipeline/scripts/build_translation_inputs_from_audio.py \
  --split "${{SPLIT}}" \
  --tail "${{MANIFEST_TAIL}}" \
  --delay 0.5 \
  --target-language "${{TARGET_LANG}}"
echo "=== build translation_inputs: done ==="
"""

RUN_CONFIG_SEARCH = f"""
set -e
echo "=== Model pipeline: config search (select best config_id) ==="
export PYTHONPATH=/workspace
cd /workspace
SPLIT="{_RUN_TMPL_SPLIT}"
REFRESH_CFG="{_RUN_TMPL_REFRESH_CFG}"
RESULTS_MR="/workspace/data/model_runs/${{SPLIT}}/config_search_results.json"
RESULTS_PR="/workspace/data/processed/${{SPLIT}}/config_search_results.json"
if [ "$REFRESH_CFG" = "true" ]; then
  echo "[INFO] refresh_config_search: removing stale config_search_results"
  rm -f "$RESULTS_MR" "$RESULTS_PR" || true
fi
if [ -s "$RESULTS_MR" ] || [ -s "$RESULTS_PR" ]; then
  echo "[SKIP] config_search_results already exists"
  exit 0
fi
CONFIG_IDS="translation_flash_v1,translation_flash_glossary,translation_flash_court,translation_flash_short_prompt,translation_flash_temp03,translation_groq_llama70b_v1"
python model-pipeline/scripts/run_config_search.py \
  --split "${{SPLIT}}" \
  --configs "${{CONFIG_IDS}}" \
  --metric bleu \
  --delay 0.0 \
  --output "data/processed/${SPLIT}/config_search_results.json"
echo "=== config search: done ==="
"""

RUN_MODEL_SETUP = f"""
set -e
echo "=== Model pipeline: translation evaluation (best config selected) ==="
export PYTHONPATH=/workspace
cd /workspace
SPLIT="{{ params.get('split', 'dev') }}"
RESULTS_JSON="data/processed/${SPLIT}/config_search_results.json"
BEST_CONFIG_ID=$(python -c "import json; print(json.load(open('${RESULTS_JSON}', 'r', encoding='utf-8'))['best_config_id'])" 2>/dev/null || echo "{{ params.get('config_id', 'translation_groq_llama70b_v1') }}")
echo "Best config_id = ${BEST_CONFIG_ID}"
python model-pipeline/scripts/model_setup.py \
  --split "${{SPLIT}}" \
  --config-id "$BEST_CONFIG_ID" \
  --translate-delay "$TRANSLATE_DELAY"
echo "=== Model pipeline: done ==="
"""

with DAG(
    dag_id="model_pipeline_dag",
    default_args=default_args,
    description="Model stages: build translation_inputs, config search, mode_setup. "
    "Conf: split, refresh_inputs (rebuild translation_inputs), refresh_config_search (re-run search; default omit/false), "
    "target_language, manifest_tail, translate_delay (seconds between Groq translation calls in mode_setup; default 0.6).",
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
