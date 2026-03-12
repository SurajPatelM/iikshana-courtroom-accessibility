"""
Offline evaluation script for Gemini-based translation models.

This script is intended to be run from the ``data-pipeline`` directory.
It loads a processed validation or test split from ``../data/processed``,
calls the configured Gemini translation model, and writes predictions.

Example:
    python scripts/gemini_translation_eval.py \\
        --split translation_val \\
        --config-id translation_flash_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from backend.src.services.gemini_translation import translate_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gemini translation model.")
    parser.add_argument(
        "--split",
        type=str,
        default="translation_val",
        help="Name of the processed split file (without extension).",
    )
    parser.add_argument(
        "--config-id",
        type=str,
        default="translation_flash_v1",
        help="Translation model configuration identifier.",
    )
    return parser.parse_args()


def _load_split(split_name: str) -> pd.DataFrame:
    """Load a processed split from the data directory."""
    for extension in (".parquet", ".csv"):
        candidate = DATA_PROCESSED_DIR / f"{split_name}{extension}"
        if candidate.exists():
            if extension == ".parquet":
                return pd.read_parquet(candidate)
            return pd.read_csv(candidate)

    raise FileNotFoundError(
        f"Could not find split '{split_name}' as Parquet or CSV "
        f"under {DATA_PROCESSED_DIR}."
    )


def _validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def main() -> None:
    args = _parse_args()

    df = _load_split(args.split)
    required_columns = ["source_text", "source_language", "target_language"]
    _validate_required_columns(df, required_columns)

    translations: List[str] = []
    for _, row in df.iterrows():
        translated = translate_text(
            source_text=str(row["source_text"]),
            source_language=str(row["source_language"]),
            target_language=str(row["target_language"]),
            config_id=args.config_id,
        )
        translations.append(translated)

    df["translated_text_model"] = translations

    output_path = DATA_PROCESSED_DIR / f"{args.split}_{args.config_id}_predictions.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()

