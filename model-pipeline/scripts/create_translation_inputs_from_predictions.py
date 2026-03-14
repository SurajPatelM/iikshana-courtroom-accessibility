"""
Helper script to bootstrap labelled translation data for Task 2.1.

Idea:
- You may already have model outputs from the model pipeline in:

    data/processed/<split>/translation_predictions_<config_id>.csv

  (created by ``model_setup.py`` / ``run_translation_eval.py``).

- This script converts that file into a ``translation_inputs.csv`` table with:
    - source_text
    - source_language
    - target_language
    - reference_translation
  plus any extra metadata columns (emotion, speaker_id, etc.).

This gives you an initial "pseudo‑labelled" table that you can then
manually review / correct. It also matches the shape expected by
``build_eval_dataset.load_eval_dataset``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VALID_SPLITS = ("dev", "test", "holdout")


def _resolve_processed_dir(repo_root: Path, override: str | None) -> Path:
    if override:
        p = Path(override)
        return p if p.is_absolute() else repo_root / p
    return repo_root / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create translation_inputs.csv from translation_predictions_<config_id>.csv."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=VALID_SPLITS,
        help="Split name (dev, test, holdout).",
    )
    parser.add_argument(
        "--config-id",
        type=str,
        default="translation_flash_v1",
        help="Config id used when generating translation_predictions_<config_id>.csv.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Override processed data directory (default: data/processed).",
    )
    parser.add_argument(
        "--source-language",
        type=str,
        default="en",
        help="Source language code to record in the table (default: en).",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default="es",
        help="Target language code to record in the table (default: es).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    processed_root = _resolve_processed_dir(repo_root, args.data_dir or None)
    split_dir = processed_root / args.split

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Processed split directory not found: {split_dir}. "
            "Run the data/model pipeline first so translation_predictions exist."
        )

    pred_path = split_dir / f"translation_predictions_{args.config_id}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {pred_path}. "
            "Run model-pipeline/scripts/model_setup.py (or the Airflow task) "
            "to generate translation_predictions_<config_id>.csv first."
        )

    df = pd.read_csv(pred_path)

    if "source_phrase" not in df.columns or "translated_text_model" not in df.columns:
        raise ValueError(
            f"{pred_path} does not have expected columns 'source_phrase' and "
            "'translated_text_model'. Check the model_setup.py output format."
        )

    out = df.copy()
    out["source_text"] = out["source_phrase"]
    out["source_language"] = args.source_language
    out["target_language"] = args.target_language
    # Use model output as initial "reference_translation" that you can later edit.
    out["reference_translation"] = out["translated_text_model"]

    # Reorder columns to put features / label first.
    front_cols = [
        "source_text",
        "source_language",
        "target_language",
        "reference_translation",
    ]
    remaining_cols = [c for c in out.columns if c not in front_cols]
    out = out[front_cols + remaining_cols]

    out_path = split_dir / "translation_inputs.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote labelled table to {out_path}")
    print(
        "You can now manually review / correct the reference_translation column, "
        "and use build_eval_dataset.load_eval_dataset for Task 2.1+."
    )


if __name__ == "__main__":
    main()

