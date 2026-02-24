"""
Full pipeline DAG: triggers each stage DAG in order.
  dvc_pull (optional) → acquisition → preprocessing → validation → anomaly → bias_detection → gemini_verification → dvc_push (optional).
  DVC steps run only when DVC_GCS_BUCKET is set in the environment (see airflow/.env).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

try:
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
except ImportError:
    from airflow.operators.trigger_dagrun import TriggerDagRunOperator

default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# Bash snippet: set DVC remote from DVC_GCS_BUCKET and explicitly set credential path
# (--local so nothing is committed to git)
_DVC_SET_REMOTE = (
    'dvc remote add -d storage --local "gs://${DVC_GCS_BUCKET}/dvc" 2>/dev/null '
    '|| dvc remote modify storage url "gs://${DVC_GCS_BUCKET}/dvc" --local && '
    'dvc remote modify storage --local credentialpath "${GOOGLE_APPLICATION_CREDENTIALS}"'
)

# dvc pull exits with code 1 if any tracked file is missing from cache (e.g. pipeline-generated
# outputs like bias_report.json that have never been pushed). We capture the output, print it,
# and only fail if zero files were fetched — meaning a real auth/connectivity failure.
_DVC_PULL = """\
set +e
DVC_OUT=$(dvc pull 2>&1)
DVC_RC=$?
echo "$DVC_OUT"
set -e
if echo "$DVC_OUT" | grep -qE "[0-9]+ (file|files) (added|modified|fetched|pulled)|Everything is up to date"; then
    echo "[OK] dvc pull succeeded (some pipeline-generated files skipped — expected)"
    exit 0
fi
if [ $DVC_RC -ne 0 ]; then
    echo "[ERROR] dvc pull failed — check credentials/cache"
    exit $DVC_RC
fi
"""

# dvc push exits with code 1 if any tracked output has no cache entry yet (e.g. files that
# will be created later in the pipeline). Treat partial push as success; only fail on
# auth/connectivity errors (i.e. when nothing was pushed at all).
_DVC_PUSH = """\
set +e
DVC_OUT=$(dvc push 2>&1)
DVC_RC=$?
echo "$DVC_OUT"
set -e
if echo "$DVC_OUT" | grep -qE "[0-9]+ (file|files) pushed"; then
    echo "[OK] dvc push succeeded (some pipeline-generated files skipped — expected)"
    exit 0
fi
if [ $DVC_RC -ne 0 ]; then
    echo "[ERROR] dvc push failed with no files pushed — check credentials/cache"
    exit $DVC_RC
fi
"""

with DAG(
        dag_id="full_pipeline_dag",
        default_args=default_args,
        description="DVC pull (optional) → acquisition → preprocessing → validation → anomaly → bias → gemini → DVC push (optional)",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["iikshana", "data", "pipeline", "full"],
) as dag:

    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=f"""
set -e
echo "=== DVC pull task: start ==="
echo "DVC_GCS_BUCKET=${{DVC_GCS_BUCKET:-<not set>}}"
if [ -z "${{DVC_GCS_BUCKET}}" ]; then echo "[SKIP] DVC_GCS_BUCKET not set, skipping dvc pull"; exit 0; fi
cd /workspace/data-pipeline && {_DVC_SET_REMOTE}
echo "Remote configured: gs://${{DVC_GCS_BUCKET}}/dvc"
echo "Checking local data dirs: /workspace/data/raw and /workspace/data/processed"
echo "  raw exists:" $( [ -d /workspace/data/raw ] && echo yes || echo no ) "non-empty:" $( [ -d /workspace/data/raw ] && [ -n "$(ls -A /workspace/data/raw 2>/dev/null)" ] && echo yes || echo no )
echo "  processed exists:" $( [ -d /workspace/data/processed ] && echo yes || echo no ) "non-empty:" $( [ -d /workspace/data/processed ] && [ -n "$(ls -A /workspace/data/processed 2>/dev/null)" ] && echo yes || echo no )
if [ -d /workspace/data/raw ] && [ -n "$(ls -A /workspace/data/raw 2>/dev/null)" ] && [ -d /workspace/data/processed ] && [ -n "$(ls -A /workspace/data/processed 2>/dev/null)" ]; then
  echo "[SKIP] Data already present locally (raw + processed non-empty), skipping dvc pull"
  exit 0
fi
echo "Running: dvc status (before pull)"
dvc status || true
echo "Running: dvc pull"
{_DVC_PULL}
echo "Running: dvc status (after pull)"
dvc status || true
echo "=== DVC pull task: done ==="
""",
        dag=dag,
    )

    trigger_acquisition = TriggerDagRunOperator(
        task_id="trigger_data_acquisition",
        trigger_dag_id="data_acquisition_dag",
        wait_for_completion=True,
    )

    trigger_preprocessing = TriggerDagRunOperator(
        task_id="trigger_preprocessing",
        trigger_dag_id="preprocessing_dag",
        wait_for_completion=True,
    )

    trigger_validation = TriggerDagRunOperator(
        task_id="trigger_validation",
        trigger_dag_id="validation_dag",
        wait_for_completion=True,
    )

    trigger_anomaly = TriggerDagRunOperator(
        task_id="trigger_anomaly_detection",
        trigger_dag_id="anomaly_detection_dag",
        wait_for_completion=True,
    )

    trigger_bias_detection = TriggerDagRunOperator(
        task_id="trigger_bias_detection",
        trigger_dag_id="bias_detection_dag",
        wait_for_completion=True,
    )

    trigger_gemini_verification = TriggerDagRunOperator(
        task_id="trigger_gemini_verification",
        trigger_dag_id="gemini_verification_dag",
        wait_for_completion=True,
    )

    dvc_push = BashOperator(
        task_id="dvc_push",
        bash_command=f"""
set -e
echo "=== DVC push task: start ==="
echo "DVC_GCS_BUCKET=${{DVC_GCS_BUCKET:-<not set>}}"
if [ -z "${{DVC_GCS_BUCKET}}" ]; then echo "[SKIP] DVC_GCS_BUCKET not set, skipping dvc push"; exit 0; fi
cd /workspace/data-pipeline && {_DVC_SET_REMOTE}
echo "Remote configured: gs://${{DVC_GCS_BUCKET}}/dvc"
echo "Running: dvc status (before push)"
dvc status || true
echo "Running: dvc push"
{_DVC_PUSH}
echo "Running: dvc status (after push)"
dvc status || true
echo "=== DVC push task: done ==="
""",
        dag=dag,
    )

    (
            dvc_pull
            >> trigger_acquisition
            >> trigger_preprocessing
            >> trigger_validation
            >> trigger_anomaly
            >> trigger_bias_detection
            >> trigger_gemini_verification
            >> dvc_push
    )