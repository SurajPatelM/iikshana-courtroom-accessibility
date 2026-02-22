"""
Bias detection via data slicing using Fairlearn (PDF §3.2: SliceFinder, TFMA, or Fairlearn).
Slice by demographics (gender, accent), emotion, language, audio quality; document disparities and mitigation.
"""
import json
from pathlib import Path

import pandas as pd

from scripts.utils import get_logger, load_config, PROCESSED_DIR

logger = get_logger("detect_bias")

try:
    from fairlearn.metrics import MetricFrame
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    MetricFrame = None  # type: ignore


def load_manifests(data_dir: Path) -> list[dict]:
    """Load all manifest.json entries from dev/test/holdout."""
    entries = []
    for split in ("dev", "test", "holdout"):
        manifest_path = data_dir / split / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        for item in (data if isinstance(data, list) else data.get("items", [])):
            item["split"] = split
            entries.append(item)
    return entries


def slice_by_emotion(entries: list[dict]) -> dict[str, list[dict]]:
    """Slice by emotion label."""
    by_emotion: dict[str, list[dict]] = {}
    for e in entries:
        label = e.get("emotion") or e.get("label") or "unknown"
        by_emotion.setdefault(label, []).append(e)
    return by_emotion


def slice_by_speaker(entries: list[dict]) -> dict[str, list[dict]]:
    """Slice by speaker_id (proxy for demographics when no explicit gender/accent)."""
    by_speaker: dict[str, list[dict]] = {}
    for e in entries:
        sid = e.get("speaker_id", "unknown")
        by_speaker.setdefault(sid, []).append(e)
    return by_speaker

def slice_by_duration(entries: list[dict]) -> dict[str, list[dict]]:
    """Slice audio into short (< 2s), medium (2-5s), and long (> 5s) buckets."""
    by_duration: dict[str, list[dict]] = {"short": [], "medium": [], "long": []}
    for e in entries:
        # Assuming duration is added to the manifest during stratified_split
        dur = e.get("duration_sec", 0)
        if dur < 2.0:
            by_duration["short"].append(e)
        elif dur <= 5.0:
            by_duration["medium"].append(e)
        else:
            by_duration["long"].append(e)
    return by_duration


def compute_counts_per_slice(slices: dict[str, list[dict]]) -> dict[str, int]:
    """Return count per slice for imbalance check."""
    return {k: len(v) for k, v in slices.items()}


def run_bias_analysis(
    data_dir: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    """Run slicing, compute counts, and write bias report."""
    if data_dir is None:
        data_dir = PROCESSED_DIR
    data_dir = Path(data_dir)
    if report_path is None:
        report_path = data_dir / "bias_report.json"

    entries = load_manifests(data_dir)
    if not entries:
        logger.warning("No manifest entries found under %s", data_dir)
        return {"slices": {}, "disparities": [], "recommendations": [], "fairlearn_by_group": {}}

    by_emotion = slice_by_emotion(entries)
    by_speaker = slice_by_speaker(entries)
    by_duration = slice_by_duration(entries)
    emotion_counts = compute_counts_per_slice(by_emotion)
    speaker_counts = compute_counts_per_slice(by_speaker)
    duration_counts = compute_counts_per_slice(by_duration)

    cfg = load_config().get("bias_detection", {})
    disparity_threshold = cfg.get("disparity_threshold", 0.5)

    total = len(entries)
    expected_per_emotion = total / len(by_emotion) if by_emotion else 0

    expected_per_duration = total / len(by_duration) if by_duration else 0

    disparities = []
    for label, count in emotion_counts.items():
        if expected_per_emotion and abs(count - expected_per_emotion) / expected_per_emotion > disparity_threshold:
            disparities.append({
                "slice": "emotion",
                "value": label,
                "count": count,
                "expected_approx": round(expected_per_emotion, 1),
                "note": f"Imbalanced emotion class exceeds {disparity_threshold*100}% threshold.",
            })

    for bucket, count in duration_counts.items():
            # Using a 0.5 (50%) threshold, but you can adjust this
        if expected_per_duration and abs(count - expected_per_duration) / expected_per_duration > 0.5:
            disparities.append({
                "slice": "duration",
                "value": bucket,
                "count": count,
                "expected_approx": round(expected_per_duration, 1),
                "note": f"Imbalanced duration bucket ({bucket}); model may be biased against this length.",
            })

    # PDF §3.2: Use Fairlearn for data slicing (MetricFrame per-group metrics)
    fairlearn_by_group = {}
    if FAIRLEARN_AVAILABLE and MetricFrame is not None and entries:
        try:
            n = len(entries)
            emotions = [e.get("emotion") or e.get("label") or "unknown" for e in entries]
            speakers = [e.get("speaker_id", "unknown") for e in entries]
            sensitive_df = pd.DataFrame({"emotion": emotions, "speaker_id": speakers})
            # Count metric: per-group sample count (data slicing for bias analysis)
            count_metric = lambda y_true, y_pred: len(y_true)  # noqa: E731
            mf = MetricFrame(
                metrics={"count": count_metric},
                y_true=list(range(n)),
                y_pred=list(range(n)),
                sensitive_features=sensitive_df,
            )
            overall_count = int(mf.overall["count"]) if mf.overall is not None else n
            by_group_obj = mf.by_group
            by_group_list = []
            if by_group_obj is not None:
                if hasattr(by_group_obj, "reset_index"):
                    by_group_df = by_group_obj.reset_index()
                    by_group_list = by_group_df.to_dict(orient="records") if hasattr(by_group_df, "to_dict") else []
                elif hasattr(by_group_obj, "to_dict"):
                    by_group_list = [{"group": k, "count": int(v) if hasattr(v, "item") else v} for k, v in by_group_obj.to_dict().items()]
                for row in by_group_list:
                    for k, v in list(row.items()):
                        if hasattr(v, "item"):
                            row[k] = v.item()
                        elif isinstance(v, float) and v == int(v):
                            row[k] = int(v)
            fairlearn_by_group = {
                "overall_count": overall_count,
                "by_group": by_group_list,
            }
            logger.info("Fairlearn MetricFrame computed per-group counts for %d entries", n)
        except Exception as e:
            logger.warning("Fairlearn MetricFrame failed (using custom slices): %s", e)
            fairlearn_by_group = {"error": str(e)}

    report = {
        "total_entries": total,
        "by_emotion": emotion_counts,
        "by_duration": duration_counts, # NEW: Expose the duration stats
        "by_speaker_count": len(by_speaker),
        "speaker_sample_counts": dict(list(speaker_counts.items())[:20]),
        "disparities": disparities,
        "fairlearn_by_group": fairlearn_by_group, #
        "recommendations": [ #
            "Use stratified evaluation and report metrics per emotion slice.",
            "If deploying, consider confidence thresholding for under-represented classes.",
            "Ensure duration distribution matches expected real-world deployment." # NEW
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True) #
    with open(report_path, "w", encoding="utf-8") as f: #
        json.dump(report, f, indent=2) #

    logger.info("Bias report written to %s; %d disparities noted", report_path, len(disparities)) #
    return report


def main() -> None:
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    # Run the analysis
    report = run_bias_analysis(data_dir=data_dir)

    # Fail the Airflow task if there are severe disparities
    if len(report.get("disparities", [])) > 0:
        logger.error("Pipeline halted: Bias disparities detected.")
        sys.exit(1) # Airflow will mark the task as failed

if __name__ == "__main__":
    main()
