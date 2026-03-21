"""
Per-slice translation performance with Fairlearn (model development fairness check).

This script takes:
- The translation eval table (e.g. data/processed/<split>/translation_inputs.csv)
- A single config (or list of configs) to evaluate
- One or more sensitive attributes (columns) such as emotion, speaker_id, dataset

For each config it:
- Runs the translation model to produce predictions (same as run_validation.py)
- Computes exact-match accuracy overall
- Uses Fairlearn's MetricFrame to compute per-group exact-match accuracy
- Writes a JSON report with overall and per-group metrics

PowerShell:
  $env:PYTHONPATH = "."
  python model-pipeline/scripts/run_translation_bias_analysis.py --split dev --config-id translation_flash_v1 --group-cols emotion,speaker_id
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from fairlearn.metrics import MetricFrame

    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    MetricFrame = None  # type: ignore

from backend.src.services.gemini_translation import translate_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model_pipeline_paths import find_translation_inputs, resolve_pipeline_and_model_roots, split_dirs

VALID_SPLITS = ("dev", "test", "holdout")
TRANSLATION_INPUTS_BASENAME = "translation_inputs"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-group translation metrics using Fairlearn MetricFrame (bias analysis)."
    )
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Legacy: single root.")
    p.add_argument("--pipeline-data-dir", type=str, default="")
    p.add_argument("--model-output-root", type=str, default="")
    p.add_argument(
        "--config-id",
        type=str,
        default="translation_flash_v1",
        help="Single config id to evaluate.",
    )
    p.add_argument(
        "--inputs-basename",
        type=str,
        default=TRANSLATION_INPUTS_BASENAME,
        help="Eval table basename (default: translation_inputs).",
    )
    p.add_argument(
        "--group-cols",
        type=str,
        default="emotion,speaker_id",
        help="Comma-separated sensitive attribute columns (must exist in eval table).",
    )
    p.add_argument("--max-rows", type=int, default=0, help="Cap rows (0 = all).")
    p.add_argument("--delay", type=float, default=2.0, help="Seconds between API calls.")
    return p.parse_args()


def _load_eval(inputs_path: Path, max_rows: int) -> pd.DataFrame:
    df = pd.read_parquet(inputs_path) if inputs_path.suffix.lower() == ".parquet" else pd.read_csv(inputs_path)
    for col in ["source_text", "source_language", "target_language", "reference_translation"]:
        if col not in df.columns:
            raise ValueError(f"Eval table must have column {col}")
    if max_rows > 0:
        df = df.head(max_rows)
    return df


def _exact_match_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    matches = sum(
        1
        for r, h in zip(y_true, y_pred)
        if (r or "").strip().lower() == (h or "").strip().lower()
    )
    return round(matches / len(y_true), 4)


def _run_config(config_id: str, features: pd.DataFrame, delay: float) -> List[str]:
    preds: List[str] = []
    for i, (_, row) in enumerate(features.iterrows()):
        if i > 0 and delay > 0:
            time.sleep(delay)
        try:
            t = translate_text(
                source_text=str(row["source_text"]),
                source_language=str(row["source_language"]),
                target_language=str(row["target_language"]),
                config_id=config_id,
            )
            preds.append((t or "").strip())
        except Exception as e:  # pragma: no cover - defensive
            preds.append(f"(error: {e})")
    return preds


def main() -> None:
    args = _parse_args()
    split = args.split
    pipeline_root, model_root = resolve_pipeline_and_model_roots(
        REPO_ROOT,
        data_dir_legacy=args.data_dir,
        pipeline_data_dir=args.pipeline_data_dir,
        model_output_root=args.model_output_root,
    )
    pipeline_split, model_split = split_dirs(pipeline_root, model_root, split)
    inputs_path = find_translation_inputs(model_split, pipeline_split, args.inputs_basename)
    if inputs_path is None:
        print(f"[ERROR] No {args.inputs_basename} under {model_split} or {pipeline_split}.")
        sys.exit(1)

    model_split.mkdir(parents=True, exist_ok=True)
    out_dir = model_split

    df = _load_eval(inputs_path, args.max_rows)
    features = df[["source_text", "source_language", "target_language"]]
    y_true = df["reference_translation"].astype(str).tolist()
    n = len(y_true)

    print(f"Running bias analysis for {args.config_id} on {n} example(s) from {inputs_path}...")
    y_pred = _run_config(args.config_id, features, args.delay)
    if len(y_pred) < len(y_true):
        y_pred = y_pred + [""] * (len(y_true) - len(y_pred))
    y_pred = y_pred[: len(y_true)]

    overall_acc = _exact_match_accuracy(y_true, y_pred)

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing group columns in eval table: {missing}. They will be ignored.")
        group_cols = [c for c in group_cols if c in df.columns]

    per_group: Dict[str, Any] = {}
    if FAIRLEARN_AVAILABLE and MetricFrame is not None and group_cols:
        print(f"Using Fairlearn MetricFrame over group columns: {group_cols}")
        try:
            sensitive = df[group_cols]
            mf = MetricFrame(
                metrics=_exact_match_accuracy,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
            per_group = {
                "overall": mf.overall,
                "by_group": mf.by_group.reset_index().to_dict(orient="records"),
            }
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Fairlearn MetricFrame failed: {exc}")
            per_group = {"error": str(exc)}
    else:
        if not FAIRLEARN_AVAILABLE:
            print("[WARN] fairlearn not installed; install fairlearn to enable MetricFrame.")
        if not group_cols:
            print("[WARN] No valid group columns provided; skipping per-group metrics.")

    out_path = out_dir / f"translation_bias_metrics_{args.config_id}.json"
    payload: Dict[str, Any] = {
        "task": "translation",
        "split": split,
        "config_id": args.config_id,
        "n_samples": n,
        "overall_exact_match_accuracy": overall_acc,
        "group_columns": group_cols,
        "fairlearn_metricframe": per_group,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Bias metrics JSON: {out_path}")


if __name__ == "__main__":
    main()

