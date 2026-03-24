"""
Emotion evaluation: load emotion_inputs (features + emotion label), compute F1 from predictions.

Proposal target (Section 7.2.2): F1 > 0.70. Same 2.1 pattern: load eval table,
features = audio path or transcript, labels = emotion class.

Two modes:
1. From predictions file: emotion_predictions.csv with columns (file or id, predicted_emotion, emotion).
2. From emotion_inputs.csv: file, emotion. You run your emotion model elsewhere and produce
   emotion_predictions.csv, then this script computes F1.

Schema for emotion_inputs.csv: file, emotion (required). Optional: source_text, dataset.
Build from manifest (manifest already has file, emotion) or from a dedicated builder.

Example:
    # After producing emotion_predictions.csv (e.g. from your emotion model):
    PYTHONPATH=. python model-pipeline/scripts/run_emotion_eval.py --split dev --predictions data/processed/dev/emotion_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

VALID_SPLITS = ("dev", "test", "holdout")
EMOTION_PREDICTIONS_BASENAME = "emotion_predictions"
PROPOSAL_F1_TARGET = 0.70


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute emotion F1 from predictions (proposal target > 0.70).")
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Override data/processed")
    p.add_argument(
        "--predictions",
        type=str,
        default="",
        help="Path to emotion_predictions CSV (columns: predicted_emotion, emotion). Default: data/processed/<split>/emotion_predictions.csv",
    )
    return p.parse_args()


def _compute_f1(labels: list[str], preds: list[str]) -> float:
    from sklearn.metrics import f1_score
    # macro F1 over classes
    return float(f1_score(labels, preds, average="macro", zero_division=0.0))


def main() -> None:
    args = _parse_args()
    processed_root = REPO_ROOT / "data" / "processed"
    if args.data_dir:
        processed_root = Path(args.data_dir) if Path(args.data_dir).is_absolute() else REPO_ROOT / args.data_dir
    split_dir = processed_root / args.split
    pred_path = Path(args.predictions) if args.predictions else split_dir / f"{EMOTION_PREDICTIONS_BASENAME}.csv"
    if not pred_path.is_absolute():
        pred_path = REPO_ROOT / pred_path
    if not pred_path.exists():
        print(f"[INFO] Predictions file not found: {pred_path}")
        print("Produce emotion_predictions.csv with columns: predicted_emotion, emotion (and optionally file/id).")
        print("Then run: python model-pipeline/scripts/run_emotion_eval.py --split dev --predictions <path>")
        sys.exit(0)
    df = pd.read_csv(pred_path)
    if "emotion" not in df.columns or "predicted_emotion" not in df.columns:
        print("[ERROR] Predictions CSV must have columns: emotion, predicted_emotion")
        sys.exit(1)
    labels = df["emotion"].astype(str).str.strip().str.lower().tolist()
    preds = df["predicted_emotion"].astype(str).str.strip().str.lower().tolist()
    f1 = _compute_f1(labels, preds)
    meets = f1 >= PROPOSAL_F1_TARGET
    print(f"Emotion macro F1 = {f1:.4f}  (proposal target >= {PROPOSAL_F1_TARGET}: {'yes' if meets else 'no'})")


if __name__ == "__main__":
    main()
