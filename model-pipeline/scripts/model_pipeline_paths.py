"""
Split data pipeline vs model pipeline directories.

- **Pipeline root** (default ``data/processed/emotions``): emotion-benchmark manifests and WAVs — read-only for model scripts. Use ``PIPELINE_DATA_DIR`` for STT-only data under ``data/processed/stt``.
- **Model output root** (default ``data/model_runs``): translation_inputs, predictions, validation, bias, etc.

Legacy: ``--data-dir`` sets a single root for both (tests / old workflows).

Environment (optional): ``PIPELINE_DATA_DIR``, ``MODEL_OUTPUT_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_PIPELINE_REL = "data/processed/emotions"
DEFAULT_MODEL_OUTPUT_REL = "data/model_runs"


def _path_under_repo(repo_root: Path, s: str) -> Path:
    p = Path(s.strip())
    return p if p.is_absolute() else repo_root / p


def resolve_pipeline_and_model_roots(
    repo_root: Path,
    data_dir_legacy: str = "",
    pipeline_data_dir: str = "",
    model_output_root: str = "",
) -> Tuple[Path, Path]:
    if (data_dir_legacy or "").strip():
        r = _path_under_repo(repo_root, data_dir_legacy.strip())
        return r, r

    p = (pipeline_data_dir or "").strip() or os.environ.get("PIPELINE_DATA_DIR", "").strip()
    pipeline_root = _path_under_repo(repo_root, p) if p else repo_root / DEFAULT_PIPELINE_REL

    m = (model_output_root or "").strip() or os.environ.get("MODEL_OUTPUT_ROOT", "").strip()
    model_root = _path_under_repo(repo_root, m) if m else repo_root / DEFAULT_MODEL_OUTPUT_REL

    return pipeline_root, model_root


def split_dirs(pipeline_root: Path, model_root: Path, split: str) -> Tuple[Path, Path]:
    return pipeline_root / split, model_root / split


def find_file_basename(
    split_dirs_ordered: Tuple[Path, ...],
    basename: str,
    extensions: Tuple[str, ...] = (".parquet", ".csv"),
) -> Optional[Path]:
    for split_dir in split_dirs_ordered:
        if not split_dir.is_dir():
            continue
        for ext in extensions:
            p = split_dir / f"{basename}{ext}"
            if p.is_file():
                return p
    return None


def find_translation_inputs(
    model_split_dir: Path,
    pipeline_split_dir: Path,
    basename: str = "translation_inputs",
) -> Optional[Path]:
    return find_file_basename((model_split_dir, pipeline_split_dir), basename)


def find_predictions_file(
    model_split_dir: Path,
    pipeline_split_dir: Path,
    config_id: str,
) -> Optional[Path]:
    base = f"translation_predictions_{config_id}"
    for split_dir in (model_split_dir, pipeline_split_dir):
        if not split_dir.is_dir():
            continue
        for ext in (".csv", ".parquet"):
            p = split_dir / f"{base}{ext}"
            if p.is_file():
                return p
    return None
