"""
Merge translation_inputs.csv (e.g. RAVDESS/audio-based) and court_translation_inputs.csv
into combined_translation_inputs.csv so you can run config search on both in one go.

Run from repo root. Creates data/processed/<split>/combined_translation_inputs.csv when
both (or either) source file exists. Then run config search with:
  --inputs-basename combined_translation_inputs

PowerShell:
  $env:PYTHONPATH = "."; python model-pipeline/scripts/build_combined_translation_inputs.py --split dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VALID_SPLITS = ("dev", "test", "holdout")
REQUIRED_COLS = ["source_text", "source_language", "target_language", "reference_translation"]
SOURCES = ["translation_inputs", "court_translation_inputs"]
OUTPUT_BASENAME = "combined_translation_inputs"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge translation_inputs + court_translation_inputs into combined_translation_inputs.csv")
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Override data/processed")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    processed_root = REPO_ROOT / "data" / "processed"
    if args.data_dir:
        processed_root = Path(args.data_dir) if Path(args.data_dir).is_absolute() else REPO_ROOT / args.data_dir
    split_dir = processed_root / args.split
    if not split_dir.is_dir():
        print(f"[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)

    import pandas as pd
    frames = []
    for basename in SOURCES:
        for ext in (".csv", ".parquet"):
            path = split_dir / f"{basename}{ext}"
            if path.exists():
                df = pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
                missing = [c for c in REQUIRED_COLS if c not in df.columns]
                if missing:
                    print(f"[WARN] Skipping {path}: missing columns {missing}")
                    break
                df = df[REQUIRED_COLS].dropna(subset=["source_text", "reference_translation"], how="all")
                if not df.empty:
                    frames.append(df)
                break

    if not frames:
        print(f"[ERROR] No translation inputs found in {split_dir}. Run build_translation_inputs_from_audio.py and/or build_translation_inputs_from_court_phrases.py first.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["source_text", "source_language", "target_language"], keep="first")
    out_path = split_dir / f"{OUTPUT_BASENAME}.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows to {out_path}")
    print("Run config search with: --inputs-basename combined_translation_inputs")

if __name__ == "__main__":
    main()
