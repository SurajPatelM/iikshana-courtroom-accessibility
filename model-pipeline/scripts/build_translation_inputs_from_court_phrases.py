"""
Build translation_inputs from court-related phrases (optional complement to audio-based inputs).

Reads data/court_phrases.csv (source_text, source_language, target_language, reference_translation)
and writes data/processed/<split>/court_translation_inputs.csv so you can:
- Run config search on court content (legal glossary, formal language).
- Merge with audio-based translation_inputs for a combined eval set.

Run from repo root. On Windows PowerShell, set PYTHONPATH first if needed:
    $env:PYTHONPATH = "."; python model-pipeline/scripts/build_translation_inputs_from_court_phrases.py --split dev
On Linux/macOS:
    PYTHONPATH=. python model-pipeline/scripts/build_translation_inputs_from_court_phrases.py --split dev
    # Optional: merge into main translation_inputs (e.g. concatenate CSVs) for combined eval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model_pipeline_paths import resolve_pipeline_and_model_roots

VALID_SPLITS = ("dev", "test", "holdout")
COURT_PHRASES_BASENAME = "court_phrases.csv"
OUTPUT_BASENAME = "court_translation_inputs.csv"
REQUIRED_COLS = ["source_text", "source_language", "target_language", "reference_translation"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build court_translation_inputs.csv from data/court_phrases.csv."
    )
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Legacy: single root for outputs.")
    p.add_argument("--model-output-root", type=str, default="", help="Model artifacts root (default: data/model_runs).")
    p.add_argument(
        "--phrases-path",
        type=str,
        default="",
        help="Path to court phrases CSV (default: data/court_phrases.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _, model_root = resolve_pipeline_and_model_roots(
        REPO_ROOT,
        data_dir_legacy=args.data_dir,
        pipeline_data_dir="",
        model_output_root=args.model_output_root,
    )
    model_split = model_root / args.split

    phrases_path = Path(args.phrases_path) if args.phrases_path else REPO_ROOT / "data" / COURT_PHRASES_BASENAME
    if not phrases_path.exists():
        print(f"[ERROR] Court phrases file not found: {phrases_path}")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(phrases_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[ERROR] Court phrases CSV missing columns: {missing}. Required: {REQUIRED_COLS}")
        sys.exit(1)
    df = df[REQUIRED_COLS].dropna(how="all").reset_index(drop=True)
    if df.empty:
        print("[ERROR] No rows in court phrases CSV.")
        sys.exit(1)

    model_split.mkdir(parents=True, exist_ok=True)
    out_path = model_split / OUTPUT_BASENAME
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} court-phrase rows to {out_path}")
    print("Use this file for config search on court content, or merge with translation_inputs.csv for combined eval.")


if __name__ == "__main__":
    main()
