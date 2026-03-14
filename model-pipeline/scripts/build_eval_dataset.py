"""
Task 2.1 — Evaluation module: load pipeline output and split into features vs labels.

Implements:
- Load processed data from your existing pipeline output
  (e.g. data/processed/<split>/translation_inputs.csv, data/processed/val.parquet).
- Split into:
  - features → prompt input (whatever text/structured info you feed into the LLM),
  - labels   → expected output (golden answer / class / numeric score you compare against).

Supported sources:
- Per-split table: data/processed/<split>/translation_inputs.csv or .parquet (split=dev|test|holdout).
- Single file:      data/processed/val.parquet (use load_eval_dataset_from_file).
- In-memory:        DataFrame from BigQuery or elsewhere (use split_features_labels).

Required columns in the table:
- source_text, source_language, target_language  → used as features / prompt input.
- reference_translation                           → label (golden answer).

Optional task-specific tables (same load pattern, different schema):
- Emotion: emotion_inputs.csv with (file or source_text) + emotion → use load_emotion_eval_data.
- ASR:      asr_inputs.csv with file + reference_transcript → use load_asr_eval_data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


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


def _find_translation_inputs(split_dir: Path) -> Path:
    """Return path to translation_inputs.parquet or .csv for a split."""
    for ext in (".parquet", ".csv"):
        p = split_dir / f"{TRANSLATION_INPUTS_BASENAME}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No {TRANSLATION_INPUTS_BASENAME}.parquet or .csv in {split_dir}. "
        f"Required columns: {', '.join(REQUIRED_COLUMNS)}."
    )


def load_eval_dataset(
    split: str,
    *,
    data_dir: str | Path | None = None,
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
        Override for data/processed (default: <repo_root>/data/processed).
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
    processed_root = _resolve_processed_dir(repo_root, str(data_dir) if data_dir else None)
    split_dir = processed_root / split

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Processed split directory not found: {split_dir}. "
            "Run the data pipeline (or DVC pull) so that data/processed/<split>/ exists."
        )

    inputs_path = _find_translation_inputs(split_dir)
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
        help="Override processed data directory (default: data/processed).",
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
            repo_root=repo_root,
        )
        print(f"Loaded split={args.split!r}")

    print(f"Features shape: {features_df.shape}  (prompt input)")
    print(f"Labels shape:   {labels.shape}  (golden output)")
    print("Feature columns:", list(features_df.columns))


if __name__ == "__main__":
    main()

