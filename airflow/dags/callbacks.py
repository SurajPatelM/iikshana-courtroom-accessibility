"""Shared Airflow callbacks; lives next to DAGs (mounted at /opt/airflow/dags)."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)


def slack_alert_on_failure(context: dict[str, Any]) -> None:
    """Post to Slack when SLACK_WEBHOOK_URL is set; otherwise no-op."""
    url = (os.environ.get("SLACK_WEBHOOK_URL") or "").strip()
    if not url:
        logger.info("SLACK_WEBHOOK_URL not set; skipping Slack alert.")
        return

    dag = context.get("dag")
    ti = context.get("task_instance")
    dag_id = dag.dag_id if dag is not None else str(context.get("dag_id", "?"))
    task_id = ti.task_id if ti is not None else "?"
    run_id = getattr(ti, "run_id", None) if ti is not None else context.get("run_id", "?")
    log_url = getattr(ti, "log_url", None) if ti is not None else None
    lines = [
        ":rotating_light: *Airflow task failed*",
        f"*DAG:* `{dag_id}`",
        f"*Task:* `{task_id}`",
        f"*Run:* `{run_id}`",
    ]
    if log_url:
        lines.append(f"*Log:* {log_url}")
    try:
        r = requests.post(url, json={"text": "\n".join(lines)}, timeout=10)
        r.raise_for_status()
    except Exception as exc:
        logger.warning("Slack webhook post failed: %s", exc)
