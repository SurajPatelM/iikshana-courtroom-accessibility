"""
Core logic for modeling (API) bias detection on translation eval tables.

Used by ``run_model_bias_detection.py``. Safe to import from tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from fairlearn.metrics import MetricFrame

    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    MetricFrame = None  # type: ignore

REQUIRED_COLS = ("source_text", "source_language", "target_language", "reference_translation")


def load_eval_table(path: Path, max_rows: int = 0) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if max_rows > 0:
        df = df.head(max_rows)
    return df


def assert_required_columns(df: pd.DataFrame, extra: Optional[List[str]] = None) -> None:
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Table must include column {col!r}")
    if extra:
        for col in extra:
            if col not in df.columns:
                raise ValueError(f"Table must include column {col!r}")


def exact_match_list(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    n = sum(
        1
        for r, h in zip(y_true, y_pred)
        if (r or "").strip().lower() == (h or "").strip().lower()
    )
    return round(n / len(y_true), 4)


def _exact_match_metric(y_true: Any, y_pred: Any) -> float:
    yt = [str(x) for x in np.asarray(y_true).flatten()]
    yp = [str(x) for x in np.asarray(y_pred).flatten()]
    return exact_match_list(yt, yp)


def _mean_sentence_bleu_metric(y_true: Any, y_pred: Any) -> float:
    refs = [str(x) for x in np.asarray(y_true).flatten()]
    hyps = [str(x) for x in np.asarray(y_pred).flatten()]
    if not refs:
        return 0.0
    try:
        import sacrebleu

        scores = [
            sacrebleu.sentence_bleu(h, [r]).score
            for r, h in zip(refs, hyps)
            if "(error:" not in h
        ]
        return round(float(np.mean(scores)), 4) if scores else 0.0
    except ImportError:
        return 0.0


def build_metric_frame(
    y_true: List[str],
    y_pred: List[str],
    sensitive: pd.DataFrame,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    if not FAIRLEARN_AVAILABLE or MetricFrame is None:
        return None, {"error": "fairlearn not installed; pip install fairlearn"}
    metrics: Dict[str, Callable[..., float]] = {
        "exact_match": _exact_match_metric,
        "mean_sentence_bleu": _mean_sentence_bleu_metric,
    }
    try:
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
        )
        bg = mf.by_group
        records = bg.reset_index().to_dict(orient="records") if hasattr(bg, "reset_index") else []
        overall = {k: float(v) if hasattr(v, "item") else float(v) for k, v in mf.overall.items()}
        return mf, {"overall": overall, "by_group": records}
    except Exception as exc:
        return None, {"error": str(exc)}


def disparities_exact_match(
    mf: Any,
    threshold: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if mf is None or mf.overall is None:
        return out
    overall_em = float(mf.overall.get("exact_match", 0.0))
    bg = mf.by_group
    if bg is None or not hasattr(bg, "iterrows"):
        return out
    names = list(bg.index.names)
    if names == [None]:
        names = []
    for idx, row in bg.iterrows():
        g_em = float(row.get("exact_match", 0.0))
        gap = abs(overall_em - g_em)
        if gap <= threshold:
            continue
        if isinstance(idx, tuple) and names and len(names) == len(idx):
            gv = {names[i]: idx[i] for i in range(len(idx))}
        else:
            gv = {"group": idx}
        out.append(
            {
                "metric": "exact_match",
                "overall": overall_em,
                "group_value": gv,
                "group_metric": g_em,
                "absolute_gap": round(gap, 4),
                "note": "Slice differs from overall exact-match rate beyond threshold.",
            }
        )
    return out


def mitigation_recommendations(
    disparities: List[Dict[str, Any]],
    group_cols: List[str],
) -> List[str]:
    rec = [
        "Report translation quality per slice (e.g. per corpus `dataset`, per `emotion`); do not rely on a single global score.",
        "For legal/court wording, evaluate with glossary/court configs and court-phrase tables under `data/` / `model-pipeline/scripts/build_translation_inputs_from_court_phrases.py`.",
        "Where a slice underperforms, add human review, confidence gating, or more in-domain eval examples before treating the system as uniform.",
    ]
    if "dataset" in group_cols and disparities:
        rec.append(
            "Cross-corpus gaps suggest domain shift vs `data-pipeline/config/datasets.yaml` sources; "
            "mitigate with domain-specific prompts or balanced eval coverage."
        )
    return rec


def aggregate_exact_match_by_dataset(by_group_records: List[Dict[str, Any]]) -> pd.Series:
    """Mean exact_match per dataset when rows include both columns."""
    if not by_group_records:
        return pd.Series(dtype=float)
    df = pd.DataFrame(by_group_records)
    if "dataset" not in df.columns or "exact_match" not in df.columns:
        return pd.Series(dtype=float)
    return df.groupby(df["dataset"].astype(str), dropna=False)["exact_match"].mean()


def save_dataset_bar_plot(
    by_group_records: List[Dict[str, Any]],
    out_path: Path,
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = aggregate_exact_match_by_dataset(by_group_records)
    if series.empty or len(series) < 1:
        return False
    fig, ax = plt.subplots(figsize=(max(6, len(series) * 0.7), 4))
    series.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_ylabel("Mean exact-match (within slice groups)")
    ax.set_xlabel("dataset (corpus)")
    ax.set_title("Model bias snapshot: exact-match by corpus")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def write_report_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
