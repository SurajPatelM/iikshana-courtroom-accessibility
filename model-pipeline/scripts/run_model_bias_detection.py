#!/usr/bin/env python3
"""
Modeling bias detection (API translation): slice metrics + disparities + mitigations.

This script is **additive** alongside ``run_translation_bias_analysis.py`` and does not
modify other pipeline code.

**Inputs**
  - Eval table: ``data/model_runs/<split>/translation_inputs.csv`` (or under ``data/processed`` if legacy), or
    ``translation_predictions_<config_id>.csv`` from ``model_setup.py`` when using ``--from-predictions``.
  - Required columns: ``source_text``, ``source_language``, ``target_language``,
    ``reference_translation``.
  - Slice columns (e.g. ``dataset``, ``emotion``) must exist on that table for Fairlearn.

**Outputs** (under ``data/model_runs/<split>/`` by default)
  - ``model_bias_report_<config_id>.json`` — overall metrics, Fairlearn by-group table,
    disparity list, mitigation recommendations.
  - ``model_bias_by_dataset_<config_id>.png`` — optional bar chart (if ``dataset`` is in
    ``--group-cols`` and aggregation is possible).
  - **MLflow** (unless ``--no-mlflow``): experiment ``iikshana-translation``, run name
    ``bias_<config_id>_<split>_<group_suffix>``; metrics prefixed ``bias_*``; JSON/PNG artifacts.

**Examples** (repo root, ``PYTHONPATH=.``)::

  python model-pipeline/scripts/run_model_bias_detection.py \\
    --split dev --config-id translation_flash_v1 --group-cols dataset,emotion

  python model-pipeline/scripts/run_model_bias_detection.py \\
    --split dev --config-id translation_flash_v1 --from-predictions \\
    --group-cols dataset,emotion
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (REPO_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd

from model_pipeline_paths import (
    find_predictions_file,
    find_translation_inputs,
    resolve_pipeline_and_model_roots,
    split_dirs,
)

from model_bias_detection_core import (
    assert_required_columns,
    build_metric_frame,
    disparities_exact_match,
    exact_match_list,
    load_eval_table,
    mitigation_recommendations,
    save_dataset_bar_plot,
    write_report_json,
)

VALID_SPLITS = ("dev", "test", "holdout")
DEFAULT_INPUTS_BASENAME = "translation_inputs"

def _group_cols_suffix(group_cols: List[str]) -> str:
    """
    Build a filesystem-safe suffix from the final slice columns.

    Example:
      ["dataset", "emotion"] -> "dataset_emotion"
    """
    if not group_cols:
        return "nogroups"

    raw = "_".join([c.strip() for c in group_cols if c.strip()])
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    # Keep filenames readable, but cap length deterministically.
    if len(safe) > 80:
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:60]}_{h}"
    return safe


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Modeling bias detection: Fairlearn slices on translation eval / predictions."
    )
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Legacy: single root for pipeline+model outputs.")
    p.add_argument("--pipeline-data-dir", type=str, default="", help="Pipeline root (default: data/processed).")
    p.add_argument("--model-output-root", type=str, default="", help="Model artifacts root (default: data/model_runs).")
    p.add_argument("--config-id", type=str, default="translation_flash_v1")
    p.add_argument(
        "--inputs-basename",
        type=str,
        default=DEFAULT_INPUTS_BASENAME,
        help="Basename for eval table when not using --from-predictions.",
    )
    p.add_argument(
        "--group-cols",
        type=str,
        default="dataset,emotion",
        help="Comma-separated slice columns (must exist in the table).",
    )
    p.add_argument("--max-rows", type=int, default=0, help="Limit rows (0 = all).")
    p.add_argument("--delay", type=float, default=2.0, help="Delay between API calls (API mode).")
    p.add_argument(
        "--from-predictions",
        action="store_true",
        help="Read translation_predictions_<config>.csv|.parquet instead of calling the API.",
    )
    p.add_argument(
        "--predictions-path",
        type=str,
        default="",
        help="Explicit path to predictions CSV/Parquet (overrides default name).",
    )
    p.add_argument("--ref-col", type=str, default="reference_translation")
    p.add_argument("--pred-col", type=str, default="translated_text_model")
    p.add_argument(
        "--disparity-threshold",
        type=float,
        default=0.15,
        help="Flag slices where |overall exact_match - group exact_match| > this.",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip PNG bar chart.")
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip MLflow logging (default: log when mlflow is installed).",
    )
    return p.parse_args()


def _build_bias_mlflow_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    """Flatten bias report fields to MLflow scalars (prefix bias_*)."""
    metrics: Dict[str, float] = {
        "bias_overall_exact_match": float(payload.get("overall_exact_match_accuracy", 0.0)),
        "bias_n_samples": float(payload.get("n_samples", 0)),
        "bias_disparity_threshold": float(payload.get("disparity_threshold_exact_match", 0.0)),
    }
    disparities = payload.get("disparities") or []
    metrics["bias_disparity_count"] = float(len(disparities))
    gaps = [
        float(d["absolute_gap"])
        for d in disparities
        if isinstance(d, dict) and "absolute_gap" in d
    ]
    if gaps:
        metrics["bias_max_disparity_gap"] = max(gaps)

    per_group = payload.get("fairlearn_metricframe") or {}
    overall = per_group.get("overall") if isinstance(per_group, dict) else None
    if isinstance(overall, dict):
        if "exact_match" in overall:
            metrics["bias_fairlearn_exact_match"] = float(overall["exact_match"])
        if "mean_sentence_bleu" in overall:
            metrics["bias_fairlearn_mean_sentence_bleu"] = float(overall["mean_sentence_bleu"])

    records = per_group.get("by_group") if isinstance(per_group, dict) else None
    if isinstance(records, list):
        ems = [
            float(r["exact_match"])
            for r in records
            if isinstance(r, dict) and "exact_match" in r
        ]
        if ems:
            metrics["bias_min_slice_exact_match"] = min(ems)
            metrics["bias_max_slice_exact_match"] = max(ems)

    return metrics


def _log_bias_to_mlflow(
    *,
    args: argparse.Namespace,
    group_suffix: str,
    report_path: Path,
    plot_path: Optional[Path],
    payload: Dict[str, Any],
) -> None:
    if args.no_mlflow:
        return
    try:
        import mlflow
    except ImportError:
        print("[info] mlflow not installed; skipping MLflow logging.")
        return

    metrics = _build_bias_mlflow_metrics(payload)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    run_name = f"bias_{args.config_id}_{args.split}_{group_suffix}"
    if len(run_name) > 200:
        run_name = run_name[:190] + "_trunc"

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("iikshana-translation")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "config_id": args.config_id,
                    "split": args.split,
                    "bias_mode": str(payload.get("mode", "")),
                    "bias_group_suffix": group_suffix,
                    "bias_group_cols": ",".join(payload.get("group_columns") or []),
                    "inputs_basename": args.inputs_basename,
                    "run_type": "model_bias_detection",
                }
            )
            mlflow.log_metrics(metrics)
            if report_path.is_file():
                mlflow.log_artifact(str(report_path), artifact_path="bias_report")
            if plot_path is not None and plot_path.is_file():
                mlflow.log_artifact(str(plot_path), artifact_path="bias_plots")
        print(f"[info] MLflow: logged bias run '{run_name}' to {tracking_uri}")
    except Exception as exc:
        print(f"[WARN] MLflow logging failed (continuing): {exc}")


def _run_api_predictions(config_id: str, features: pd.DataFrame, delay: float) -> List[str]:
    preds: List[str] = []
    for i, (_, row) in enumerate(features.iterrows()):
        if i > 0 and delay > 0:
            time.sleep(delay)
        try:
            # Lazy import so `--from-predictions` does not require Gemini deps.
            from backend.src.services.gemini_translation import translate_text

            t = translate_text(
                source_text=str(row["source_text"]),
                source_language=str(row["source_language"]),
                target_language=str(row["target_language"]),
                config_id=config_id,
            )
            preds.append((t or "").strip())
        except Exception as e:
            preds.append(f"(error: {e})")
    return preds


def main() -> None:
    args = _parse_args()
    pipeline_root, model_root = resolve_pipeline_and_model_roots(
        REPO_ROOT,
        data_dir_legacy=args.data_dir,
        pipeline_data_dir=args.pipeline_data_dir,
        model_output_root=args.model_output_root,
    )
    pipeline_split, model_split = split_dirs(pipeline_root, model_root, args.split)
    if not pipeline_split.is_dir() and not model_split.is_dir():
        print(f"[ERROR] Neither pipeline nor model split dir exists:\n  {pipeline_split}\n  {model_split}")
        sys.exit(1)

    model_split.mkdir(parents=True, exist_ok=True)
    out_dir = model_split

    pred_col = args.pred_col
    ref_col = args.ref_col

    if args.from_predictions:
        if args.predictions_path.strip():
            pp = Path(args.predictions_path)
            pred_path = pp if pp.is_absolute() else REPO_ROOT / pp
        else:
            pred_path = find_predictions_file(model_split, pipeline_split, args.config_id)
            if pred_path is None:
                pred_path = model_split / f"translation_predictions_{args.config_id}.csv"
        if pred_path is None or not pred_path.is_file():
            print(f"[ERROR] Predictions file not found: {pred_path}")
            sys.exit(1)
        df = load_eval_table(pred_path, args.max_rows)
        assert_required_columns(df, extra=[pred_col])
        y_pred = df[pred_col].astype(str).tolist()
        print(f"[info] Loaded {len(y_pred)} row(s) from {pred_path}")
    else:
        path = find_translation_inputs(model_split, pipeline_split, args.inputs_basename)
        if path is None:
            print(
                f"[ERROR] No {args.inputs_basename}.csv/.parquet under {model_split} or {pipeline_split}."
            )
            sys.exit(1)
        df = load_eval_table(path, args.max_rows)
        assert_required_columns(df)
        feats = df[["source_text", "source_language", "target_language"]]
        print(f"[info] Calling translation API ({args.config_id}) for {len(df)} row(s)...")
        y_pred = _run_api_predictions(args.config_id, feats, args.delay)

    y_true = df[ref_col].astype(str).tolist()
    if len(y_pred) < len(y_true):
        y_pred.extend([""] * (len(y_true) - len(y_pred)))
    y_pred = y_pred[: len(y_true)]

    overall_em = exact_match_list(y_true, y_pred)

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing slice columns (ignored): {missing}")
        group_cols = [c for c in group_cols if c in df.columns]

    mf_obj: Any = None
    per_group: Dict[str, Any] = {}
    if group_cols:
        sensitive = df[group_cols].fillna("unknown").astype(str)
        mf_obj, per_group = build_metric_frame(y_true, y_pred, sensitive)
        if per_group.get("error") and "overall" not in per_group:
            print(f"[WARN] Fairlearn: {per_group.get('error')}")
    else:
        print("[WARN] No valid group columns; skipping Fairlearn MetricFrame.")

    disparities = disparities_exact_match(mf_obj, args.disparity_threshold)
    mitigations = mitigation_recommendations(disparities, group_cols)

    plot_path: Optional[Path] = None
    if not args.no_plots and "dataset" in group_cols:
        records = per_group.get("by_group") if isinstance(per_group, dict) else []
        if isinstance(records, list) and records:
            group_suffix = _group_cols_suffix(group_cols)
            plot_path = out_dir / f"model_bias_by_dataset_{args.config_id}__{group_suffix}.png"
            if not save_dataset_bar_plot(records, plot_path):
                plot_path = None
            elif plot_path is not None:
                print(f"[info] Plot: {plot_path}")

    group_suffix = _group_cols_suffix(group_cols)
    out_path = out_dir / f"model_bias_report_{args.config_id}__{group_suffix}.json"
    payload: Dict[str, Any] = {
        "schema": "iikshana.model_bias_detection.v1",
        "task": "translation",
        "split": args.split,
        "config_id": args.config_id,
        "mode": "from_predictions" if args.from_predictions else "api",
        "n_samples": len(y_true),
        "overall_exact_match_accuracy": overall_em,
        "group_columns": group_cols,
        "disparity_threshold_exact_match": args.disparity_threshold,
        "disparities": disparities,
        "mitigation_recommendations": mitigations,
        "fairlearn_metricframe": per_group,
        "plot_path": str(plot_path) if plot_path is not None else None,
    }
    write_report_json(out_path, payload)
    print(f"[info] Wrote {out_path}")

    _log_bias_to_mlflow(
        args=args,
        group_suffix=group_suffix,
        report_path=out_path,
        plot_path=plot_path,
        payload=payload,
    )


if __name__ == "__main__":
    main()
