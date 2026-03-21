"""
Task 2.1 — Evaluation module: load pipeline output and split into features vs labels.

Implements:
- Load translation eval tables written by model-pipeline scripts (default under
  ``data/model_runs/<split>/``), with fallback to ``data/processed/<split>/`` for older layouts.
- Split into:
  - features → prompt input (whatever text/structured info you feed into the LLM),
  - labels   → expected output (golden answer / class / numeric score you compare against).

Supported sources:
- Per-split table: ``translation_inputs.csv`` or ``.parquet`` under
  ``data/model_runs/<split>/`` first, then ``data/processed/<split>/`` (split=dev|test|holdout).
  Legacy: pass ``data_dir=`` to use one root for both. Env: ``PIPELINE_DATA_DIR``, ``MODEL_OUTPUT_ROOT``.
- Single file: any path (e.g. ``data/processed/val.parquet``) via ``load_eval_dataset_from_file``.
- In-memory: DataFrame from BigQuery or elsewhere (use ``split_features_labels``).

Required columns in the table:
- source_text, source_language, target_language  → used as features / prompt input.
- reference_translation                           → label (golden answer).

Optional task-specific tables (same load pattern, different schema):
- Emotion: emotion_inputs.csv with (file or source_text) + emotion → use load_emotion_eval_data.
- ASR:      asr_inputs.csv with file + reference_transcript → use load_asr_eval_data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from model_pipeline_paths import find_translation_inputs, resolve_pipeline_and_model_roots, split_dirs


VALID_SPLITS = ("dev", "test", "holdout")
TRANSLATION_INPUTS_BASENAME = "translation_inputs"
REQUIRED_COLUMNS = ["source_text", "source_language", "target_language", "reference_translation"]
FEATURE_COLUMNS = ["source_text", "source_language", "target_language"]
LABEL_COLUMN = "reference_translation"


def _resolve_processed_dir(repo_root: Path, override: str | None) -> Path:
    """Resolve the root of processed data (default: repo_root / data / processed)."""
    if override:
        p = Path(override)
        return p if p.is_absolute() else repo_root / p
    return repo_root / "data" / "processed"


def _validate_and_split(df: pd.DataFrame, source_hint: str = "table") -> Tuple[pd.DataFrame, pd.Series]:
    """Check required columns and return (features_df, labels)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source_hint} is missing required column(s): {missing}. "
            f"Expected: {', '.join(REQUIRED_COLUMNS)}."
        )
    extra_cols = [c for c in df.columns if c not in FEATURE_COLUMNS + [LABEL_COLUMN]]
    features_df = df[FEATURE_COLUMNS + extra_cols].copy()
    labels = df[LABEL_COLUMN].copy()
    return features_df, labels


def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame from any source (CSV, parquet, BigQuery) into features and labels.

    Use this when you load data yourself (e.g. from BigQuery) so it matches the
    same schema as the rest of the evaluation pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: source_text, source_language, target_language, reference_translation.

    Returns
    -------
    features_df : pd.DataFrame
        Columns used as LLM prompt input (source_text, source_language, target_language + any extra).
    labels : pd.Series
        Ground-truth reference translations (or class / numeric score).
    """
    return _validate_and_split(df, source_hint="DataFrame")


def load_eval_dataset_from_file(path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed data from a single file (e.g. data/processed/val.parquet) and return (features, labels).

    Parameters
    ----------
    path : str or Path
        Path to a CSV or Parquet file with required columns (see REQUIRED_COLUMNS).

    Returns
    -------
    features_df, labels
        Same as load_eval_dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return _validate_and_split(df, source_hint=str(path))


def load_eval_dataset(
    split: str,
    *,
    data_dir: str | Path | None = None,
    model_output_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed data from the pipeline output for a split; return (features, labels).

    This is the main entry point when using per-split tables produced by the
    data pipeline (e.g. data/processed/dev/translation_inputs.csv).

    Parameters
    ----------
    split : str
        One of dev, test, holdout.
    data_dir : str or Path, optional
        Legacy: single root for both pipeline and model tables (old layout).
    model_output_root : str or Path, optional
        Override model artifacts root (default: data/model_runs). Ignored if ``data_dir`` is set.
    repo_root : str or Path, optional
        Repository root (default: inferred from this file).

    Returns
    -------
    features_df : pd.DataFrame
        Input to the LLM: source_text, source_language, target_language, plus any extra columns.
    labels : pd.Series
        Golden answer / expected output (reference_translation).
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    dd = str(data_dir).strip() if data_dir else ""
    mor = str(model_output_root).strip() if model_output_root else ""
    pipeline_root, model_root = resolve_pipeline_and_model_roots(
        repo_root,
        data_dir_legacy=dd,
        pipeline_data_dir="",
        model_output_root=mor,
    )
    pipeline_split, model_split = split_dirs(pipeline_root, model_root, split)

    if not pipeline_split.is_dir() and not model_split.is_dir():
        raise FileNotFoundError(
            f"No split directories at {pipeline_split} or {model_split}. "
            "Run the data pipeline and/or build_translation_inputs_from_audio.py."
        )

    inputs_path = find_translation_inputs(model_split, pipeline_split, TRANSLATION_INPUTS_BASENAME)
    if inputs_path is None:
        raise FileNotFoundError(
            f"No {TRANSLATION_INPUTS_BASENAME}.csv/.parquet under {model_split} or {pipeline_split}."
        )
    if inputs_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(inputs_path)
    else:
        df = pd.read_csv(inputs_path)

    return _validate_and_split(df, source_hint=str(inputs_path))


def load_emotion_eval_data(
    split: str,
    *,
    data_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load emotion eval data: features (e.g. file or source_text), labels (emotion class).
    Expects data/processed/<split>/emotion_inputs.csv with columns: emotion, and either file or source_text.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    processed_root = _resolve_processed_dir(repo_root, str(data_dir) if data_dir else None)
    split_dir = processed_root / split
    path = split_dir / "emotion_inputs.csv"
    if not path.exists():
        raise FileNotFoundError(f"emotion_inputs.csv not found in {split_dir}")
    df = pd.read_csv(path)
    if "emotion" not in df.columns:
        raise ValueError("emotion_inputs must have column: emotion")
    feature_col = "file" if "file" in df.columns else "source_text"
    if feature_col not in df.columns:
        raise ValueError("emotion_inputs must have column: file or source_text")
    extra = [c for c in df.columns if c not in (feature_col, "emotion")]
    return df[[feature_col] + extra].copy(), df["emotion"].copy()


def load_asr_eval_data(
    split: str,
    *,
    data_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load ASR eval data: features (file paths), labels (reference_transcript).
    Expects data/processed/<split>/asr_inputs.csv with columns: file, reference_transcript.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    processed_root = _resolve_processed_dir(repo_root, str(data_dir) if data_dir else None)
    split_dir = processed_root / split
    for ext in (".csv", ".parquet"):
        p = split_dir / f"asr_inputs{ext}"
        if p.exists():
            df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            if "file" not in df.columns or "reference_transcript" not in df.columns:
                raise ValueError("asr_inputs must have columns: file, reference_transcript")
            return df[["file"]].copy(), df["reference_transcript"].copy()
    raise FileNotFoundError(f"asr_inputs.csv or .parquet not found in {split_dir}. Run build_asr_inputs_from_audio.py.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 2.1: Load pipeline output and split into features (prompt input) vs labels (golden output)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=VALID_SPLITS,
        help="Split name when using per-split tables (dev, test, holdout).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Legacy: single root for pipeline+model tables.",
    )
    parser.add_argument(
        "--model-output-root",
        type=str,
        default="",
        help="Model artifacts root when not using --data-dir (default: data/model_runs).",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Load from a single file instead of split (e.g. data/processed/val.parquet). Overrides --split.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI: load data and print features/labels shape (Task 2.1)."""
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    if args.file:
        path = Path(args.file) if Path(args.file).is_absolute() else repo_root / args.file
        features_df, labels = load_eval_dataset_from_file(path)
        print(f"Loaded from file: {path}")
    else:
        features_df, labels = load_eval_dataset(
            split=args.split,
            data_dir=args.data_dir or None,
            model_output_root=args.model_output_root or None,
            repo_root=repo_root,
        )
        print(f"Loaded split={args.split!r}")

    print(f"Features shape: {features_df.shape}  (prompt input)")
    print(f"Labels shape:   {labels.shape}  (golden output)")
    print("Feature columns:", list(features_df.columns))


if __name__ == "__main__":
    main()

