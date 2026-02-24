"""
Anomaly Detection DAG (PDF §2.8, §6): run anomaly checks; on failure trigger alert via email and Slack.
Configure SMTP in .env (AIRFLOW__SMTP__*) and optional ALERT_EMAIL, SLACK_WEBHOOK_URL.
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

_DAG_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = _DAG_DIR.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from callbacks import slack_alert_on_failure


def _ensure_scripts_on_path():
    """Ensure pipeline root (so 'scripts' can be imported) is on sys.path at task runtime."""
    for root in [
        Path("/workspace/data-pipeline"),
        Path("/opt/airflow"),
        PIPELINE_ROOT,
        _DAG_DIR.parent.parent / "data-pipeline",
    ]:
        if root.exists() and (root / "scripts" / "utils.py").exists():
            s = str(root)
            if s not in sys.path:
                sys.path.insert(0, s)
            return root
    return PIPELINE_ROOT


def _run_anomaly_checks(**kwargs):
    """Run anomaly detection; raise on anomalies so Airflow triggers email_on_failure (PDF: alert on anomalies)."""
    _ensure_scripts_on_path()
    from scripts.anomaly_check import run_anomaly_checks
    from scripts.utils import RAW_DIR, PROCESSED_DIR
    report = run_anomaly_checks(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        report_path=PROCESSED_DIR / "anomaly_report.json",
    )
    if not report.get("passed", True):
        raise RuntimeError(
            "Anomalies detected. Pipeline triggers alert (configure Airflow email/Slack). Anomalies: %s"
            % report.get("anomalies", [])
        )
    return report


def _alert_emails() -> list[str]:
    """Parse ALERT_EMAIL env (comma-separated) for email_on_failure recipients."""
    raw = os.environ.get("ALERT_EMAIL", "").strip()
    if not raw:
        return []
    return [e.strip() for e in raw.split(",") if e.strip()]


_default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": True,  # PDF §2.8: trigger alert when anomalies detected
    "email_on_retry": False,
    "on_failure_callback": slack_alert_on_failure,
}
# Add email recipients if configured (Airflow sends to these when task fails)
_emails = _alert_emails()
if _emails:
    _default_args["email"] = _emails


def get_tasks(dag: DAG):
    """Return tasks for this stage (for use in full_pipeline_dag)."""
    anomaly_task = PythonOperator(
        task_id="run_anomaly_checks_and_alert",
        python_callable=_run_anomaly_checks,
        dag=dag,
    )
    return [anomaly_task]


with DAG(
    dag_id="anomaly_detection_dag",
    default_args=_default_args,
    description="Anomaly detection and alerts (missing files, duration, label imbalance, schema violations)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iikshana", "data", "anomaly", "alerts"],
) as dag:
    _tasks = get_tasks(dag)
