"""
Create translation_inputs.csv from translation_predictions_<config_id>.csv.

Reads predictions from ``data/model_runs/<split>/`` (or legacy ``data/processed``),
writes ``translation_inputs.csv`` under ``data/model_runs/<split>/`` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

VALID_SPLITS = ("dev", "test", "holdout")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create translation_inputs.csv from translation_predictions_<config_id>.csv."
    )
    parser.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    parser.add_argument("--config-id", type=str, default="translation_flash_v1")
    parser.add_argument("--data-dir", type=str, default="", help="Legacy: single root for read+write.")
    parser.add_argument("--pipeline-data-dir", type=str, default="")
    parser.add_argument("--model-output-root", type=str, default="")
    parser.add_argument("--source-language", type=str, default="en")
    parser.add_argument("--target-language", type=str, default="es")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = Path(__file__).resolve().parent
    for p in (repo_root, scripts_dir):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from model_pipeline_paths import (
        find_predictions_file,
        resolve_pipeline_and_model_roots,
        split_dirs,
    )

    pipeline_root, model_root = resolve_pipeline_and_model_roots(
        repo_root,
        data_dir_legacy=args.data_dir,
        pipeline_data_dir=args.pipeline_data_dir,
        model_output_root=args.model_output_root,
    )
    pipeline_split, model_split = split_dirs(pipeline_root, model_root, args.split)

    pred_path = find_predictions_file(model_split, pipeline_split, args.config_id)
    if pred_path is None or not pred_path.is_file():
        raise FileNotFoundError(
            f"Predictions file not found for config {args.config_id!r} under {model_split} or {pipeline_split}. "
            "Run model_setup.py first."
        )

    df = pd.read_csv(pred_path)

    if "source_phrase" not in df.columns or "translated_text_model" not in df.columns:
        raise ValueError(
            f"{pred_path} does not have expected columns 'source_phrase' and 'translated_text_model'."
        )

    out = df.copy()
    out["source_text"] = out["source_phrase"]
    out["source_language"] = args.source_language
    out["target_language"] = args.target_language
    out["reference_translation"] = out["translated_text_model"]

    front_cols = [
        "source_text",
        "source_language",
        "target_language",
        "reference_translation",
    ]
    remaining_cols = [c for c in out.columns if c not in front_cols]
    out = out[front_cols + remaining_cols]

    model_split.mkdir(parents=True, exist_ok=True)
    out_path = model_split / "translation_inputs.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote labelled table to {out_path}")


if __name__ == "__main__":
    main()
