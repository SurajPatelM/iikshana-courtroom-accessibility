"""
Anomaly Detection DAG (PDF §2.8, §6): run anomaly checks; on failure trigger alert via Airflow email.
Configure email in Airflow (smtp) so that when anomalies are detected the task fails and email is sent.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _run_anomaly_checks(**kwargs):
    """Run anomaly detection; raise on anomalies so Airflow triggers email_on_failure (PDF: alert on anomalies)."""
    from scripts.anomaly_check import run_anomaly_checks
    report = run_anomaly_checks(
        raw_dir=PIPELINE_ROOT / "data" / "raw",
        processed_dir=PIPELINE_ROOT / "data" / "processed",
        report_path=PIPELINE_ROOT / "data" / "processed" / "anomaly_report.json",
    )
    if not report.get("passed", True):
        raise RuntimeError(
            "Anomalies detected. Pipeline triggers alert (configure Airflow email/Slack). Anomalies: %s"
            % report.get("anomalies", [])
        )
    return report


_default_args = {
    "owner": "iikshana",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": True,  # PDF §2.8: trigger alert (e.g. email) when anomalies detected
    "email_on_retry": False,
}


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
