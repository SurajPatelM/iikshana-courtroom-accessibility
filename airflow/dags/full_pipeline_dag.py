"""
Full pipeline DAG: triggers each stage DAG in order.
  acquisition → preprocessing → validation → anomaly_detection → gemini_verification (last).
  Gemini runs only if anomaly detection completes with no anomaly (wait_for_completion on anomaly).

Bias detection removed from this flow. Uses TriggerDagRunOperator.
"""
from datetime import datetime, timedelta

from airflow import DAG

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

with DAG(
        dag_id="full_pipeline_dag",
        default_args=default_args,
        description="Triggers acquisition → preprocessing → validation → anomaly → bias_detection → gemini_verification",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["iikshana", "data", "pipeline", "full"],
) as dag:
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
        wait_for_completion=True,  # Gemini only runs on bias-checked data
    )

    trigger_gemini_verification = TriggerDagRunOperator(
        task_id="trigger_gemini_verification",
        trigger_dag_id="gemini_verification_dag",
        wait_for_completion=True,  # Final gate
    )

    # acquisition → preprocessing → validation → anomaly → bias_detection → gemini (final)
    (
            trigger_acquisition
            >> trigger_preprocessing
            >> trigger_validation
            >> trigger_anomaly
            >> trigger_bias_detection
            >> trigger_gemini_verification
    )