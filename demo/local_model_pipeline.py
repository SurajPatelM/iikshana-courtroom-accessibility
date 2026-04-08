"""
Run the same model stages as ``model_pipeline_dag`` locally (no Docker).

Order: ``build_translation_inputs_from_audio`` → ``run_config_search`` → ``model_setup``.
Translations shown in the Gradio expo UI are read from ``translation_predictions_<best_config>.csv`` only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

# Default candidate list aligned with ``airflow/dags/model_pipeline_dag.py``
DEFAULT_CONFIG_SEARCH_IDS = (
    "translation_flash_v1,translation_flash_glossary,translation_flash_court,"
    "translation_flash_short_prompt,translation_flash_temp03"
)


@dataclass
class LocalPipelineResult:
    best_config_id: str
    predictions_csv: Path
    stdout_log: str


def _split_paths(repo_root: Path, split: str) -> tuple[Path, Path]:
    pipeline_split = repo_root / "data" / "processed" / split
    model_split = repo_root / "data" / "model_runs" / split
    return pipeline_split, model_split


def clear_translation_stage_artifacts(
    repo_root: Path,
    split: str,
    *,
    refresh_inputs: bool = True,
    refresh_config_search: bool = True,
) -> None:
    """Mirror DAG refresh flags: which artifacts to delete before a local run."""
    for base in (
        repo_root / "data" / "model_runs" / split,
        repo_root / "data" / "processed" / split,
    ):
        if refresh_inputs:
            p = base / "translation_inputs.csv"
            if p.is_file():
                p.unlink()
        if refresh_config_search:
            p = base / "config_search_results.json"
            if p.is_file():
                p.unlink()


def _run_script(repo_root: Path, rel_script: str, args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    cmd = [sys.executable, str(repo_root / rel_script), *args]
    return subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )


def run_full_model_pipeline(
    repo_root: Path,
    split: str,
    *,
    refresh: bool = True,
    refresh_config_search: bool | None = None,
    target_language: str = "es",
    build_delay: float = 0.5,
    config_search_delay: float = 0.0,
    configs: str = DEFAULT_CONFIG_SEARCH_IDS,
    manifest_tail: int = 64,
) -> LocalPipelineResult:
    """
    Execute the three model stages and return paths + best config.

    Raises
    ------
    RuntimeError
        If a stage exits non-zero or predictions / best config cannot be read.
    """
    repo_root = repo_root.resolve()
    _, model_split = _split_paths(repo_root, split)
    log_parts: list[str] = []

    if refresh:
        rcfg = refresh_config_search if refresh_config_search is not None else True
        clear_translation_stage_artifacts(
            repo_root, split, refresh_inputs=True, refresh_config_search=rcfg
        )

    build_args = [
        "--split",
        split,
        "--delay",
        str(build_delay),
        "--target-language",
        target_language,
        "--tail",
        str(max(1, manifest_tail)),
    ]
    r1 = _run_script(repo_root, "model-pipeline/scripts/build_translation_inputs_from_audio.py", build_args)
    log_parts.append("=== build_translation_inputs_from_audio ===\n" + (r1.stdout or "") + (r1.stderr or ""))
    if r1.returncode != 0:
        raise RuntimeError(
            "build_translation_inputs_from_audio failed:\n" + log_parts[-1]
        )

    r2 = _run_script(
        repo_root,
        "model-pipeline/scripts/run_config_search.py",
        [
            "--split",
            split,
            "--configs",
            configs,
            "--metric",
            "bleu",
            "--delay",
            str(config_search_delay),
        ],
    )
    log_parts.append("=== run_config_search ===\n" + (r2.stdout or "") + (r2.stderr or ""))
    if r2.returncode != 0:
        raise RuntimeError("run_config_search failed:\n" + log_parts[-1])

    results_json = model_split / "config_search_results.json"
    if not results_json.is_file():
        fallback = repo_root / "data" / "processed" / split / "config_search_results.json"
        results_json = fallback if fallback.is_file() else results_json

    if not results_json.is_file():
        raise RuntimeError(
            f"config search did not write config_search_results.json under model_runs or processed ({split})."
        )

    with results_json.open(encoding="utf-8") as f:
        payload = json.load(f)
    best_config_id = str(payload.get("best_config_id") or "").strip()
    if not best_config_id:
        raise RuntimeError("config_search_results.json missing best_config_id")

    r3 = _run_script(
        repo_root,
        "model-pipeline/scripts/model_setup.py",
        ["--split", split, "--config-id", best_config_id],
    )
    log_parts.append("=== model_setup ===\n" + (r3.stdout or "") + (r3.stderr or ""))
    if r3.returncode != 0:
        raise RuntimeError("model_setup failed:\n" + log_parts[-1])

    predictions = model_split / f"translation_predictions_{best_config_id}.csv"
    if not predictions.is_file():
        predictions = (
            repo_root / "data" / "processed" / split / f"translation_predictions_{best_config_id}.csv"
        )
    if not predictions.is_file():
        raise RuntimeError(
            f"model_setup did not write translation_predictions_{best_config_id}.csv under model_runs or processed."
        )

    return LocalPipelineResult(
        best_config_id=best_config_id,
        predictions_csv=predictions,
        stdout_log="\n".join(log_parts),
    )


def _read_prediction_row(
    pred: Path,
    key: str,
    config_label: str,
) -> tuple[str, Path, str] | None:
    try:
        df = pd.read_csv(pred)
        if "file" not in df.columns or "translated_text_model" not in df.columns:
            return None
        sub = df[df["file"].astype(str) == key]
        if sub.empty:
            return None
        return (
            str(sub.iloc[0]["translated_text_model"]).strip(),
            pred.resolve(),
            config_label,
        )
    except (OSError, ValueError):
        return None


def try_read_pipeline_translation(
    repo_root: Path,
    split: str,
    manifest_file: str,
    *,
    config_id: str | None = None,
) -> tuple[str, Path, str] | None:
    """
    If Airflow (or a local run) has finished, read the translation for ``manifest_file`` from disk.
    Checks ``data/model_runs/<split>`` then ``data/processed/<split>``.

    When ``config_id`` is set (``expo_translation_dag``), reads
    ``translation_predictions_<config_id>.csv`` first. Otherwise uses ``config_search_results.json``
    to pick the best config (``model_pipeline_dag``).
    """
    repo_root = repo_root.resolve()
    model_split = repo_root / "data" / "model_runs" / split
    processed_split = repo_root / "data" / "processed" / split
    key = Path(manifest_file).name

    cid = (config_id or "").strip()
    if cid:
        for pred_parent in (model_split, processed_split):
            pred = pred_parent / f"translation_predictions_{cid}.csv"
            if pred.is_file():
                got = _read_prediction_row(pred, key, cid)
                if got is not None:
                    return got

    for cfg_parent in (model_split, processed_split):
        cfg_path = cfg_parent / "config_search_results.json"
        if not cfg_path.is_file():
            continue
        try:
            with cfg_path.open(encoding="utf-8") as f:
                best = str(json.load(f).get("best_config_id") or "").strip()
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if not best:
            continue
        for pred_parent in (model_split, processed_split):
            pred = pred_parent / f"translation_predictions_{best}.csv"
            if not pred.is_file():
                continue
            got = _read_prediction_row(pred, key, best)
            if got is not None:
                return got
    return None


def read_translation_for_file(predictions_csv: Path, manifest_file: str) -> str:
    """Return ``translated_text_model`` for the manifest ``file`` value (basename)."""
    df = pd.read_csv(predictions_csv)
    if "file" not in df.columns or "translated_text_model" not in df.columns:
        raise ValueError(f"Predictions CSV missing file/translated_text_model: {predictions_csv}")
    key = Path(manifest_file).name
    sub = df[df["file"].astype(str) == key]
    if sub.empty:
        raise ValueError(f"No prediction row for file={key!r} in {predictions_csv}")
    return str(sub.iloc[0]["translated_text_model"]).strip()
