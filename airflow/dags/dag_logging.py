"""
Shared logging for DAGs: log DAG/task name and read/write paths so runs are verifiable via CLI or UI.
All output uses [DAG_LOG] prefix for easy grep. Print to stdout so Airflow captures it in task logs.
"""
import os
from pathlib import Path


def _get_data_paths():
    """Resolve DATA_ROOT, RAW_DIR, PROCESSED_DIR from env or scripts.utils if available."""
    repo_root = os.environ.get("REPO_ROOT", "/workspace")
    pipeline_root = os.environ.get("PIPELINE_ROOT", os.path.join(repo_root, "data-pipeline"))
    data_root = os.environ.get("DATA_ROOT", os.path.join(pipeline_root, "data"))
    data_root = Path(data_root)
    return {
        "DATA_ROOT": data_root,
        "RAW_DIR": data_root / "raw",
        "PROCESSED_DIR": data_root / "processed",
    }


def log_dag_task_paths(
    dag_id: str,
    task_id: str,
    read_from: list[str] | None = None,
    write_to: list[str] | None = None,
    extra: dict | None = None,
) -> None:
    """
    Log where this DAG task reads from and writes to. Call at the start of each task.
    read_from / write_to: list of path strings or Paths (will be str()'d).
    extra: optional dict of key=value to log (e.g. config).
    """
    paths = _get_data_paths()
    lines = [
        "[DAG_LOG] ========== DAG run ==========",
        f"[DAG_LOG] DAG_ID={dag_id} TASK_ID={task_id}",
        f"[DAG_LOG] DATA_ROOT={paths['DATA_ROOT']}",
        f"[DAG_LOG] RAW_DIR={paths['RAW_DIR']}",
        f"[DAG_LOG] PROCESSED_DIR={paths['PROCESSED_DIR']}",
    ]
    if read_from:
        lines.append("[DAG_LOG] READ_FROM:")
        for p in read_from:
            lines.append(f"  - {p}")
    if write_to:
        lines.append("[DAG_LOG] WRITE_TO:")
        for p in write_to:
            lines.append(f"  - {p}")
    if extra:
        lines.append("[DAG_LOG] EXTRA:")
        for k, v in extra.items():
            lines.append(f"  {k}={v}")
    lines.append("[DAG_LOG] ================================")
    print("\n".join(lines))


def log_dag_task_start(dag_id: str, task_id: str, message: str = "") -> None:
    """Log that a DAG task has started (for Bash-only tasks or simple triggers)."""
    paths = _get_data_paths()
    lines = [
        "[DAG_LOG] ========== DAG run ==========",
        f"[DAG_LOG] DAG_ID={dag_id} TASK_ID={task_id}",
        f"[DAG_LOG] DATA_ROOT={paths['DATA_ROOT']}",
    ]
    if message:
        lines.append(f"[DAG_LOG] {message}")
    lines.append("[DAG_LOG] ================================")
    print("\n".join(lines))
