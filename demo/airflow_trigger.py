"""
Trigger an Airflow DAG from the host (e.g. Gradio expo UI) while Docker Compose is running.

Uses ``docker compose -f airflow/docker-compose.yaml exec airflow-scheduler airflow ...``
from the **airflow/** directory (compose project context).

Before trigger, calls ``airflow dags unpause`` so runs actually execute (new DAGs start paused).

Environment overrides:
  AIRFLOW_COMPOSE_DIR   — directory containing docker-compose.yaml (default: <repo>/airflow)
  AIRFLOW_SCHEDULER_SERVICE — service name (default: airflow-scheduler)
  AIRFLOW_MODEL_DAG_ID  — DAG to trigger (default: model_pipeline_dag)
  AIRFLOW_SKIP_UNPAUSE  — set to ``1`` to skip unpause before trigger
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose_exec(
    *,
    compose_file: Path,
    compose_dir: Path,
    service: str,
    airflow_args: list[str],
    timeout_sec: int,
) -> subprocess.CompletedProcess:
    cmd = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "exec",
        "-T",
        service,
        "airflow",
        *airflow_args,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=str(compose_dir),
    )


def _combined_out(proc: subprocess.CompletedProcess) -> str:
    parts = [proc.stdout or "", proc.stderr or ""]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def trigger_model_pipeline_dag(
    *,
    split: str = "dev",
    refresh_inputs: bool = True,
    refresh_config_search: bool = False,
    manifest_tail: int = 200,
    target_language: str = "es",
    compose_dir: Path | None = None,
    service: str | None = None,
    dag_id: str | None = None,
    timeout_sec: int = 180,
    trigger_retries: int = 3,
    trigger_retry_delay_sec: float = 2.5,
) -> tuple[int, str, str]:
    """
    Unpause the DAG (unless skipped), then ``airflow dags trigger`` with JSON conf.

    ``refresh_config_search`` — when True, deletes ``config_search_results.json`` so
    ``run_config_search`` runs again. Default False: reuse existing best config after the first run.

    Returns
    -------
    (returncode, combined_log, "")
        Third element reserved; kept for backwards compatibility.
    """
    compose_dir = Path(
        compose_dir
        or os.environ.get("AIRFLOW_COMPOSE_DIR", "")
        or (REPO_ROOT / "airflow")
    ).resolve()
    compose_file = compose_dir / "docker-compose.yaml"
    if not compose_file.is_file():
        return 127, "", f"Compose file not found: {compose_file}"

    service = service or os.environ.get("AIRFLOW_SCHEDULER_SERVICE", "airflow-scheduler")
    dag_id = dag_id or os.environ.get("AIRFLOW_MODEL_DAG_ID", "model_pipeline_dag")

    tail_n = int(manifest_tail) if manifest_tail else 200
    if tail_n < 1:
        tail_n = 200
    conf: dict[str, Any] = {
        "split": split,
        "refresh_inputs": refresh_inputs,
        "refresh_config_search": refresh_config_search,
        "target_language": (target_language or "es").strip() or "es",
        "manifest_tail": tail_n,
    }
    conf_json = json.dumps(conf, separators=(",", ":"))

    logs: list[str] = []

    try:
        if os.environ.get("AIRFLOW_SKIP_UNPAUSE", "").strip() != "1":
            up = _compose_exec(
                compose_file=compose_file,
                compose_dir=compose_dir,
                service=service,
                airflow_args=["dags", "unpause", dag_id],
                timeout_sec=timeout_sec,
            )
            logs.append(f"=== airflow dags unpause {dag_id} (exit {up.returncode}) ===\n{_combined_out(up)}")
            # unpause may return non-zero if already unpaused on some versions; continue anyway

        last_proc: subprocess.CompletedProcess | None = None
        for attempt in range(max(1, trigger_retries)):
            last_proc = _compose_exec(
                compose_file=compose_file,
                compose_dir=compose_dir,
                service=service,
                airflow_args=["dags", "trigger", dag_id, "--conf", conf_json],
                timeout_sec=timeout_sec,
            )
            out = _combined_out(last_proc)
            logs.append(
                f"=== airflow dags trigger (attempt {attempt + 1}/{trigger_retries}, exit {last_proc.returncode}) ===\n{out}"
            )
            if last_proc.returncode == 0:
                return 0, "\n\n".join(logs), ""
            if attempt + 1 < trigger_retries:
                time.sleep(trigger_retry_delay_sec)

        return last_proc.returncode if last_proc else 1, "\n\n".join(logs), ""

    except FileNotFoundError:
        return 127, "", "docker CLI not found. Install Docker and use Docker Compose v2."
    except subprocess.TimeoutExpired:
        return 124, "\n\n".join(logs), f"Timeout after {timeout_sec}s waiting for docker compose exec."
