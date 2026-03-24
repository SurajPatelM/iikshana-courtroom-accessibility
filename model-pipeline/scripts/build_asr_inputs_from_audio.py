"""
Build asr_inputs.csv from pipeline audio (manifest + known script for reference transcript).

Writes data/processed/<split>/asr_inputs.csv with columns: file, reference_transcript.
Used by run_asr_eval.py to compute WER (STT output vs reference). RAVDESS entries
get reference from the known two phrases; other datasets need a mapping or separate file.

Run from repo root with PYTHONPATH set.

Example:
    PYTHONPATH=. python model-pipeline/scripts/build_asr_inputs_from_audio.py --split dev --max-rows 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VALID_SPLITS = ("dev", "test", "holdout")
MANIFEST_FILENAME = "manifest.json"
ASR_INPUTS_BASENAME = "asr_inputs"

# RAVDESS script (reference = what was said)
RAVDESS_STATEMENT_TO_REF: Dict[str, str] = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build asr_inputs.csv from manifest + reference mapping.")
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Override data/processed")
    p.add_argument("--max-rows", type=int, default=200, help="Max manifest entries (0 = all).")
    return p.parse_args()


def _ravdess_statement_from_file(file_name: str) -> str | None:
    if not file_name.startswith("RAVDESS_") or not file_name.endswith(".wav"):
        return None
    base = file_name.replace("RAVDESS_", "").replace(".wav", "")
    parts = base.split("-")
    return parts[4] if len(parts) >= 5 else None


def main() -> None:
    args = _parse_args()
    processed_root = REPO_ROOT / "data" / "processed"
    if args.data_dir:
        processed_root = Path(args.data_dir) if Path(args.data_dir).is_absolute() else REPO_ROOT / args.data_dir
    split_dir = processed_root / args.split
    if not split_dir.is_dir():
        print(f"[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)
    manifest_path = split_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(f"[ERROR] No {MANIFEST_FILENAME} in {split_dir}")
        sys.exit(1)
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    manifest = data if isinstance(data, list) else []
    if not manifest:
        print("[ERROR] Empty manifest.")
        sys.exit(1)
    subset = manifest[: args.max_rows] if args.max_rows > 0 else manifest
    rows: List[Dict[str, Any]] = []
    for entry in subset:
        file_name = entry.get("file", "")
        if not file_name:
            continue
        ref = ""
        if (entry.get("dataset") or "").strip().upper() == "RAVDESS":
            stmt = _ravdess_statement_from_file(file_name)
            if stmt and stmt in RAVDESS_STATEMENT_TO_REF:
                ref = RAVDESS_STATEMENT_TO_REF[stmt]
        if not ref:
            continue
        rows.append({"file": file_name, "reference_transcript": ref})
    if not rows:
        print("[ERROR] No rows with reference_transcript (e.g. only non-RAVDESS?). Add mapping or use RAVDESS.")
        sys.exit(1)
    import pandas as pd
    df = pd.DataFrame(rows)
    out_path = split_dir / f"{ASR_INPUTS_BASENAME}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}. Run run_asr_eval.py to compute WER.")


if __name__ == "__main__":
    main()
