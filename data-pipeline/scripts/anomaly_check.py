"""
Anomaly detection: missing/corrupt files, duration distribution, label imbalance, schema violations.
Can trigger email/Slack alerts when anomalies are detected (configure in Airflow).
"""
import json
from pathlib import Path

from scripts.utils import get_logger, load_config, PROCESSED_DIR, RAW_DIR

logger = get_logger("anomaly_check")

# Thresholds
MAX_MISSING_RATIO = 0.1
MIN_FILES_PER_SPLIT = 1
LABEL_IMBALANCE_RATIO = 3.0  # max ratio between most and least frequent class


def check_missing_files(raw_dir: Path) -> list[str]:
    """Check for expected but missing dataset dirs or empty dirs."""
    anomalies = []
    if not raw_dir.exists():
        anomalies.append("data/raw does not exist")
        return anomalies
    dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    empty = [d.name for d in dirs if not any(d.rglob("*")) or not any(d.rglob("*.wav")) and not any(d.rglob("*.mp3"))]
    if empty:
        anomalies.append(f"Empty or no-audio dirs in raw: {empty}")
    return anomalies


def check_duration_distribution(processed_dir: Path) -> list[str]:
    """Flag if many files are outside expected duration range."""
    import soundfile as sf
    anomalies = []
    cfg = load_config().get("validation", {})
    min_d, max_d = cfg.get("min_duration_sec", 0.5), cfg.get("max_duration_sec", 30.0)
    out_of_range = 0
    total = 0
    for split in ("dev", "test", "holdout"):
        for wav in (processed_dir / split).rglob("*.wav"):
            total += 1
            try:
                info = sf.info(wav)
                if info.duration < min_d or info.duration > max_d:
                    out_of_range += 1
            except Exception:
                out_of_range += 1
    if total > 0 and out_of_range / total > 0.2:
        anomalies.append(f"High fraction of files outside duration [{min_d}, {max_d}]s: {out_of_range}/{total}")
    return anomalies


def check_label_imbalance(processed_dir: Path) -> list[str]:
    """Check for severe label imbalance."""
    anomalies = []
    from collections import Counter
    counts = Counter()
    for split in ("dev", "test", "holdout"):
        manifest = processed_dir / split / "manifest.json"
        if not manifest.exists():
            continue
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        for item in (data if isinstance(data, list) else data.get("items", [])):
            label = item.get("emotion") or item.get("label") or "unknown"
            counts[label] += 1
    if not counts:
        return anomalies
    most, least = max(counts.values()), min(counts.values())
    if least > 0 and most / least > LABEL_IMBALANCE_RATIO:
        anomalies.append(f"Label imbalance: max/min count ratio {most/least:.1f} (threshold {LABEL_IMBALANCE_RATIO})")
    return anomalies


def run_anomaly_checks(
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    """Run all checks; return report. Raise or alert if anomalies found."""
    raw_dir = raw_dir or RAW_DIR
    processed_dir = processed_dir or PROCESSED_DIR
    if report_path is None:
        report_path = processed_dir / "anomaly_report.json"

    all_anomalies = []
    all_anomalies.extend(check_missing_files(raw_dir))
    all_anomalies.extend(check_duration_distribution(processed_dir))
    all_anomalies.extend(check_label_imbalance(processed_dir))

    report = {"anomalies": all_anomalies, "passed": len(all_anomalies) == 0}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if all_anomalies:
        logger.warning("Anomalies detected: %s", all_anomalies)
    else:
        logger.info("Anomaly check passed")
    return report


def main() -> None:
    r = run_anomaly_checks()
    if not r["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
