"""
ASR evaluation: load asr_inputs (file, reference_transcript), run STT, compute WER.

Proposal target (Section 7.2.2): WER < 10%. Same 2.1 pattern: load eval table from
pipeline output, features = audio file paths, labels = reference_transcript.

Requires: data/processed/<split>/asr_inputs.csv (build with build_asr_inputs_from_audio.py).
Optional: ELEVENLABS_API_KEY for Scribe v2 STT. jiwer for WER.

Example:
    PYTHONPATH=. python model-pipeline/scripts/run_asr_eval.py --split dev
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from backend.src.services.elevenlabs_stt_service import AUDIO_EXTENSIONS, transcribe_audio

VALID_SPLITS = ("dev", "test", "holdout")
ASR_INPUTS_BASENAME = "asr_inputs"
PROPOSAL_WER_TARGET_PCT = 10.0  # WER < 10%


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ASR on asr_inputs, compute WER (proposal target < 10%).")
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Override data/processed")
    p.add_argument("--max-rows", type=int, default=0, help="Cap rows (0 = all).")
    p.add_argument("--delay", type=float, default=2.0, help="Seconds between STT API calls.")
    return p.parse_args()


def _load_asr_inputs(split_dir: Path, max_rows: int) -> tuple[pd.Series, pd.Series]:
    path = None
    for ext in (".csv", ".parquet"):
        p = split_dir / f"{ASR_INPUTS_BASENAME}{ext}"
        if p.exists():
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"No {ASR_INPUTS_BASENAME}.csv or .parquet in {split_dir}. Run build_asr_inputs_from_audio.py first.")
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "file" not in df.columns or "reference_transcript" not in df.columns:
        raise ValueError("asr_inputs must have columns: file, reference_transcript")
    if max_rows > 0:
        df = df.head(max_rows)
    return df["file"], df["reference_transcript"]


def _compute_wer(references: list[str], hypotheses: list[str]) -> float:
    try:
        import jiwer
        return jiwer.wer(references, hypotheses)
    except ImportError:
        # Fallback: simple word error rate (insertions+deletions+substitutions) / ref words
        total_ref_words = 0
        total_errors = 0
        for ref, hyp in zip(references, hypotheses):
            rw = (ref or "").split()
            hw = (hyp or "").split()
            total_ref_words += len(rw)
            if not rw:
                total_errors += len(hw)
                continue
            # Levenshtein-like at word level (simplified: symmetric diff size / ref len)
            rs, hs = set(rw), set(hw)
            total_errors += max(0, len(hs - rs) + len(rs - hs))
        return total_errors / total_ref_words if total_ref_words else 0.0


def main() -> None:
    args = _parse_args()
    processed_root = REPO_ROOT / "data" / "processed"
    if args.data_dir:
        processed_root = Path(args.data_dir) if Path(args.data_dir).is_absolute() else REPO_ROOT / args.data_dir
    split_dir = processed_root / args.split
    if not split_dir.is_dir():
        print(f"[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)
    files_ser, refs_ser = _load_asr_inputs(split_dir, args.max_rows)
    files = files_ser.astype(str).tolist()
    references = refs_ser.astype(str).tolist()
    n = len(files)
    print(f"Running STT on {n} files...")
    hypotheses = []
    for i, fname in enumerate(files):
        if i > 0 and args.delay > 0:
            time.sleep(args.delay)
        wav_path = split_dir / fname
        if not wav_path.exists() or wav_path.suffix.lower() not in AUDIO_EXTENSIONS:
            hypotheses.append("")
            continue
        try:
            hyp = transcribe_audio(wav_path).strip() or ""
            hypotheses.append(hyp)
        except Exception as e:
            hypotheses.append(f"(error: {e})")
    wer = _compute_wer(references, hypotheses)
    wer_pct = wer * 100
    meets_target = wer_pct < PROPOSAL_WER_TARGET_PCT
    print(f"WER = {wer_pct:.2f}%  (proposal target < {PROPOSAL_WER_TARGET_PCT}%: {'yes' if meets_target else 'no'})")
    out_path = split_dir / "asr_predictions.csv"
    pd.DataFrame({"file": files, "reference_transcript": references, "hypothesis": hypotheses}).to_csv(out_path, index=False)
    print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
