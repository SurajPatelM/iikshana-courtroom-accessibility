"""
Minimal translation path for live EXPO / production-style inference.

Runs only:
  1. build_translation_inputs_from_audio  (manifest tail + STT + sidecars)
  2. model_setup with a **fixed** ``config_id`` from trigger conf

**Does not** run ``run_config_search`` (offline / evaluation only). Use ``model_pipeline_dag`` when you
need BLEU-based config selection.

Conf (all optional except split semantics):
  - split (default dev)
  - refresh_inputs (default true via UI) — drop stale translation_inputs before build
  - manifest_tail (default 1) — last N manifest rows; use 1 when EXPO row is last
  - target_language (default es)
  - config_id (default translation_flash_v1) — must match a file under config/models/
  - translate_delay (default 0) — seconds between translation API calls in model_setup
  - stt_delay (default 0) — seconds between STT calls in build step
"""
from datetime import datetime, timedelta

from airflow import DAG  # type: ignore[import-untyped]
from airflow.operators.bash import BashOperator  # type: ignore[import-untyped]

default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 0,
}

_RUN_TMPL_SPLIT = "{{ (dag_run.conf or {}).get('split', params.get('split', 'dev')) }}"
_RUN_TMPL_REFRESH_INPUTS = "{{ 'true' if (dag_run.conf or {}).get('refresh_inputs', True) else 'false' }}"
_RUN_TMPL_TLANG = "{{ (dag_run.conf or {}).get('target_language', 'es') }}"
_RUN_TMPL_MANIFEST_TAIL = "{{ (dag_run.conf or {}).get('manifest_tail', 1) }}"
_RUN_TMPL_TRANSLATE_DELAY = "{{ (dag_run.conf or {}).get('translate_delay', 0) }}"
_RUN_TMPL_STT_DELAY = "{{ (dag_run.conf or {}).get('stt_delay', 0) }}"
_RUN_TMPL_CONFIG_ID = (
    "{{ (dag_run.conf or {}).get('config_id', params.get('config_id', 'translation_flash_v1')) }}"
)

RUN_BUILD = f"""
set -e
echo "=== Expo: build translation_inputs.csv (no config search) ==="
export PYTHONPATH=/workspace
cd /workspace
SPLIT="{_RUN_TMPL_SPLIT}"
REFRESH_INPUTS="{_RUN_TMPL_REFRESH_INPUTS}"
TARGET_LANG="{_RUN_TMPL_TLANG}"
MANIFEST_TAIL="{_RUN_TMPL_MANIFEST_TAIL}"
STT_DELAY="{_RUN_TMPL_STT_DELAY}"
MR_INPUTS="/workspace/data/model_runs/${{SPLIT}}/translation_inputs.csv"
PR_INPUTS="/workspace/data/processed/${{SPLIT}}/translation_inputs.csv"
if [ "$REFRESH_INPUTS" = "true" ]; then
  echo "[INFO] refresh_inputs: removing stale translation_inputs"
  rm -f "$MR_INPUTS" "$PR_INPUTS" || true
fi
if [ -s "$MR_INPUTS" ] || [ -s "$PR_INPUTS" ]; then
  echo "[SKIP] translation_inputs already exists under model_runs or processed"
  exit 0
fi
python model-pipeline/scripts/build_translation_inputs_from_audio.py \\
  --split "${{SPLIT}}" \\
  --tail "${{MANIFEST_TAIL}}" \\
  --delay "${{STT_DELAY}}" \\
  --target-language "${{TARGET_LANG}}"
echo "=== build translation_inputs: done ==="
"""

RUN_TRANSLATE = f"""
set -e
echo "=== Expo: model_setup (fixed config_id) ==="
export PYTHONPATH=/workspace
cd /workspace
SPLIT="{_RUN_TMPL_SPLIT}"
CONFIG_ID="{_RUN_TMPL_CONFIG_ID}"
TRANSLATE_DELAY="{_RUN_TMPL_TRANSLATE_DELAY}"
echo "config_id = $CONFIG_ID"
python model-pipeline/scripts/model_setup.py \\
  --split "${{SPLIT}}" \\
  --config-id "$CONFIG_ID" \\
  --translate-delay "$TRANSLATE_DELAY"
echo "=== expo translation: done ==="
"""

with DAG(
    dag_id="expo_translation_dag",
    default_args=default_args,
    description="Minimal path: build translation_inputs → translate with fixed config_id. "
    "No config search. Conf: split, refresh_inputs, manifest_tail, target_language, "
    "config_id, translate_delay, stt_delay.",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "model", "expo", "infer"],
    params={
        "split": "dev",
        "config_id": "translation_flash_v1",
    },
) as dag:
    build_inputs = BashOperator(
        task_id="build_translation_inputs_from_audio",
        bash_command=RUN_BUILD,
        dag=dag,
    )
    translate = BashOperator(
        task_id="mode_setup",
        bash_command=RUN_TRANSLATE,
        dag=dag,
    )
    build_inputs >> translate
