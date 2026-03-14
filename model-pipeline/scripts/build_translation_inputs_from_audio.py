"""
Build translation_inputs.csv from pipeline audio (manifest + WAVs).

This ties the evaluation set to the actual data pipeline output:
- Load manifest.json from data/processed/<split>/ (file, dataset, speaker_id, emotion).
- For each WAV, run STT to get source_text (what was said in the audio).
- Assign reference_translation from known script (e.g. RAVDESS has two phrases).
- Write translation_inputs.csv so 2.1 and run_translation_eval use audio-derived inputs and labels.

Requires: GROQ_API_KEY (for STT). Run from repo root with PYTHONPATH set.

Example:
    PYTHONPATH=. python model-pipeline/scripts/build_translation_inputs_from_audio.py --split dev --max-rows 30
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.src.services.groq_stt_service import (
    transcribe_audio,
    AUDIO_EXTENSIONS,
    DEFAULT_STT_MODEL,
)

VALID_SPLITS = ("dev", "test", "holdout")
MANIFEST_FILENAME = "manifest.json"
TRANSLATION_INPUTS_BASENAME = "translation_inputs"

# RAVDESS: statement 01 = "Kids are talking by the door.", 02 = "Dogs are sitting by the door."
# English source (script) and gold Spanish for evaluation.
RAVDESS_STATEMENT_TO_SOURCE: Dict[str, str] = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}
RAVDESS_STATEMENT_TO_REF: Dict[str, str] = {
    "01": "Los niños están hablando cerca de la puerta.",
    "02": "Los perros están sentados junto a la puerta.",
}
DEFAULT_SOURCE_LANGUAGE = "en"
DEFAULT_TARGET_LANGUAGE = "es"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build translation_inputs.csv from pipeline audio (manifest + STT + reference mapping)."
    )
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Override data/processed")
    p.add_argument(
        "--max-rows",
        type=int,
        default=50,
        help="Max manifest entries to process (default 50).",
    )
    p.add_argument("--stt-model", type=str, default=DEFAULT_STT_MODEL)
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between STT API calls to avoid rate limits.",
    )
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many manifest entries before taking max-rows (e.g. 100 to use entries 100–199).",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle manifest before selecting entries (use with --seed for reproducibility).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --shuffle (e.g. 42). Omit for different files each run.",
    )
    return p.parse_args()


def _resolve_processed_dir(repo_root: Path, override: str) -> Path:
    if override:
        path = Path(override)
        return path if path.is_absolute() else repo_root / path
    return repo_root / "data" / "processed"


def _load_manifest(split_dir: Path) -> List[Dict[str, Any]]:
    path = split_dir / MANIFEST_FILENAME
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _ravdess_statement_from_file(file_name: str) -> str | None:
    """Extract RAVDESS statement id (01 or 02) from filename. Returns None if not RAVDESS or parse fails."""
    if not file_name.startswith("RAVDESS_") or not file_name.endswith(".wav"):
        return None
    base = file_name.replace("RAVDESS_", "").replace(".wav", "")
    parts = base.split("-")
    if len(parts) >= 5:
        return parts[4]  # statement 01 or 02
    return None


def _reference_translation_for_entry(entry: Dict[str, Any], file_name: str) -> str:
    """Get gold reference translation for this manifest entry (e.g. from RAVDESS script)."""
    dataset = (entry.get("dataset") or "").strip().upper()
    if dataset == "RAVDESS":
        stmt = _ravdess_statement_from_file(file_name)
        if stmt and stmt in RAVDESS_STATEMENT_TO_REF:
            return RAVDESS_STATEMENT_TO_REF[stmt]
    # Fallback: no known script; caller can skip or use empty (we still write source_text from STT)
    return ""


def main() -> None:
    args = _parse_args()
    processed_root = _resolve_processed_dir(REPO_ROOT, args.data_dir)
    split_dir = processed_root / args.split

    if not split_dir.is_dir():
        print(f"[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)

    manifest = _load_manifest(split_dir)
    if not manifest:
        print(f"[ERROR] No {MANIFEST_FILENAME} in {split_dir}")
        sys.exit(1)

    if args.shuffle:
        if args.seed is not None:
            random.seed(args.seed)
        manifest = list(manifest)
        random.shuffle(manifest)

    start = min(args.offset, len(manifest))
    subset = manifest[start : start + args.max_rows]
    if start > 0 or args.shuffle:
        print(f"Using {len(subset)} entries (offset={start}, shuffle={args.shuffle})")
    rows: List[Dict[str, Any]] = []

    for i, entry in enumerate(subset):
        file_name = entry.get("file", "")
        if not file_name:
            continue
        wav_path = split_dir / file_name
        if not wav_path.exists() or wav_path.suffix.lower() not in AUDIO_EXTENSIONS:
            print(f"[SKIP] Missing or not audio: {wav_path}")
            continue

        if i > 0:
            time.sleep(args.delay)

        try:
            source_text = transcribe_audio(wav_path, model=args.stt_model).strip() or ""
        except Exception as e:
            print(f"[WARN] STT failed for {file_name}: {e}")
            source_text = ""

        # Fallback: for RAVDESS we know the script, so fill source_text if STT failed or was skipped
        if not source_text and (entry.get("dataset") or "").strip().upper() == "RAVDESS":
            stmt = _ravdess_statement_from_file(file_name)
            if stmt and stmt in RAVDESS_STATEMENT_TO_SOURCE:
                source_text = RAVDESS_STATEMENT_TO_SOURCE[stmt]

        ref = _reference_translation_for_entry(entry, file_name)
        if not ref and entry.get("dataset", "").strip().upper() == "RAVDESS":
            print(f"[WARN] No reference for RAVDESS file {file_name}; ref will be empty")

        rows.append({
            "source_text": source_text,
            "source_language": DEFAULT_SOURCE_LANGUAGE,
            "target_language": DEFAULT_TARGET_LANGUAGE,
            "reference_translation": ref,
            "file": file_name,
            "dataset": entry.get("dataset", ""),
            "speaker_id": entry.get("speaker_id", ""),
            "emotion": entry.get("emotion", ""),
        })

    if not rows:
        print("[ERROR] No rows produced. Check manifest and WAV paths.")
        sys.exit(1)

    import pandas as pd
    df = pd.DataFrame(rows)
    # Keep only rows that have a reference translation (so eval is well-defined)
    df = df[df["reference_translation"].astype(str).str.len() > 0].reset_index(drop=True)
    if df.empty:
        print("[ERROR] No rows with reference_translation (e.g. only non-RAVDESS?). Add ref mapping or use RAVDESS.")
        sys.exit(1)

    out_path = split_dir / f"{TRANSLATION_INPUTS_BASENAME}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path} (from pipeline audio + STT + reference mapping).")
    print("Run build_eval_dataset.py --split dev and run_translation_eval.py to use this table.")


if __name__ == "__main__":
    main()
