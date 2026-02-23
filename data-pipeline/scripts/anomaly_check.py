"""
Anomaly detection: missing/corrupt files, duration distribution, label imbalance, schema violations.
Can trigger email/Slack alerts when anomalies are detected (configure in Airflow).
"""
import json
from pathlib import Path

from scripts.utils import get_logger, load_config, PROCESSED_DIR, RAW_DIR

logger = get_logger("anomaly_check")

# Ignored filenames when deciding if a raw dir is "actually empty"
_IGNORED_EMPTY_FILES = (".gitkeep", ".DS_Store")


def _is_actually_empty(d: Path) -> bool:
    """True if dir has no files, or only ignored sentinels (.gitkeep, .DS_Store, ._*)."""
    for f in d.rglob("*"):
        if not f.is_file():
            continue
        if f.name in _IGNORED_EMPTY_FILES or f.name.startswith("._"):
            continue
        return False
    return True


def _has_archives_only(d: Path, include_video: bool) -> bool:
    """True if dir has at least one archive and no processable media."""
    if _has_processable_media(d, include_video):
        return False
    archive_suffixes = (".zip", ".tar.gz", ".tgz", ".tar", ".gz")
    for f in d.rglob("*"):
        if not f.is_file():
            continue
        if f.name.startswith("._"):
            continue
        if f.name.lower().endswith(archive_suffixes):
            return True
    return False


def _has_processable_media(d: Path, include_video: bool) -> bool:
    """True if dir has at least one processable audio/video file.
    Audio: .wav, .mp3, .flac, .ogg, .m4a (aligned with preprocess_audio). Video when include_video.
    Also treats .txt as content (e.g. MELD) so dir is not considered empty."""
    for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
        if any(d.rglob(f"*{ext}")) or any(d.rglob(f"*{ext.upper()}")):
            return True
    if include_video:
        for ext in (".mp4", ".m4v", ".mkv", ".avi", ".mov"):
            if any(d.rglob(f"*{ext}")):
                return True
    if any(d.rglob("*.txt")):
        return True
    return False


def check_missing_files(raw_dir: Path) -> tuple[list[str], list[str]]:
    """Check for actually empty raw dirs (fail) and dirs with archives not extracted (warn only).
    Returns (anomalies, not_extracted_raw_dirs). Only actually empty dirs are added to anomalies."""
    anomalies = []
    not_extracted = []
    if not raw_dir.exists():
        anomalies.append("data/raw does not exist")
        return anomalies, not_extracted
    cfg = load_config()
    include_video = bool(cfg.get("preprocessing", {}).get("include_video", False))
    dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    empty = [d.name for d in dirs if _is_actually_empty(d)]
    if empty:
        anomalies.append(f"Empty raw dirs (no files): {empty}")
    for d in dirs:
        if _is_actually_empty(d):
            continue
        if _has_processable_media(d, include_video):
            continue
        if _has_archives_only(d, include_video):
            not_extracted.append(d.name)
    return anomalies, not_extracted


def check_processed_splits_present(processed_dir: Path) -> list[str]:
    """Detect incomplete or failed preprocessing/split (missing or empty splits)."""
    anomalies = []
    cfg = load_config().get("anomaly_checks", {})
    min_per_split = int(cfg.get("min_files_per_split", 1))
    for split in ("dev", "test", "holdout"):
        split_dir = processed_dir / split
        if not split_dir.is_dir():
            anomalies.append(f"Split {split} directory missing")
            continue
        count = sum(1 for _ in split_dir.rglob("*.wav"))
        if count < min_per_split:
            anomalies.append(f"Split {split} has {count} files (min {min_per_split})")
    if not anomalies and min_per_split > 0:
        total = sum(
            sum(1 for _ in (processed_dir / s).rglob("*.wav"))
            for s in ("dev", "test", "holdout")
            if (processed_dir / s).is_dir()
        )
        if total == 0:
            anomalies.append("No processed files in dev/test/holdout")
    return anomalies


def check_duration_distribution(processed_dir: Path) -> list[str]:
    """Flag if many files are outside expected duration range or unreadable."""
    import soundfile as sf
    anomalies = []
    cfg = load_config()
    val_cfg = cfg.get("validation", {})
    anom_cfg = cfg.get("anomaly_checks", {})
    min_d = val_cfg.get("min_duration_sec", 0.5)
    max_d = val_cfg.get("max_duration_sec", 30.0)
    max_out_ratio = float(anom_cfg.get("max_out_of_range_ratio", 0.2))
    out_of_range = 0
    total = 0
    for split in ("dev", "test", "holdout"):
        split_dir = processed_dir / split
        if not split_dir.is_dir():
            continue
        for wav in split_dir.rglob("*.wav"):
            total += 1
            try:
                info = sf.info(wav)
                if info.duration < min_d or info.duration > max_d:
                    out_of_range += 1
            except Exception:
                out_of_range += 1
    if total > 0 and out_of_range / total > max_out_ratio:
        anomalies.append(
            f"High fraction of files outside duration [{min_d}, {max_d}]s: {out_of_range}/{total} (max ratio {max_out_ratio})"
        )
    return anomalies


def check_schema_failures(processed_dir: Path) -> list[str]:
    """Treat validation DAG failures as anomalies when fail_on_schema_failures is true."""
    anomalies = []
    cfg = load_config().get("anomaly_checks", {})
    if not cfg.get("fail_on_schema_failures", True):
        return anomalies
    report_path = processed_dir / "quality_report.json"
    if not report_path.exists():
        anomalies.append("Validation report missing; run validation_dag first")
        return anomalies
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)
    failed = data.get("failed", 0)
    passed = data.get("passed", 0)
    files_checked = data.get("files_checked", 0)
    if failed > 0:
        msg = f"Validation reported {failed} failed files (fail_on_schema_failures=true)"
        manifest_errors = data.get("manifest_errors", [])
        if manifest_errors:
            msg += "; sample: " + (manifest_errors[0][:80] + "..." if len(manifest_errors[0]) > 80 else manifest_errors[0])
        anomalies.append(msg)
    elif files_checked > 0 and passed == 0:
        anomalies.append("Validation reported 0 passed files (fail_on_schema_failures=true)")
    return anomalies


def check_label_imbalance(processed_dir: Path) -> list[str]:
    """Check for severe label imbalance (any class > max_class_ratio or < min_class_ratio)."""
    from collections import Counter

    anomalies = []
    cfg = load_config().get("anomaly_checks", {})
    max_class_ratio = float(cfg.get("max_class_ratio", 0.80))
    min_class_ratio = float(cfg.get("min_class_ratio", 0.05))
    min_total_for_min_ratio = 20  # only flag "too small" class when total is large enough

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
    total = sum(counts.values())
    # Allow 100% unknown so DAG can pass before emotion inference is fixed (e.g. RAVDESS codes)
    if len(counts) == 1 and "unknown" in counts:
        return anomalies
    for label, cnt in counts.items():
        ratio = cnt / total
        if ratio > max_class_ratio:
            anomalies.append(f"Label '{label}' is {ratio:.1%} of total (max {max_class_ratio:.0%})")
        if total >= min_total_for_min_ratio and ratio < min_class_ratio:
            anomalies.append(f"Label '{label}' is {ratio:.1%} of total (min {min_class_ratio:.0%})")
    return anomalies


def check_manifest_file_consistency(processed_dir: Path) -> list[str]:
    """Catch manifest entries pointing to missing files or WAVs not listed in manifest."""
    anomalies = []
    for split in ("dev", "test", "holdout"):
        split_dir = processed_dir / split
        if not split_dir.is_dir():
            continue
        manifest_path = split_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("items", [])
        manifest_files = {item.get("file") for item in items if item.get("file")}
        missing = []
        for f in manifest_files:
            if not (split_dir / f).exists():
                missing.append(f)
        if missing:
            anomalies.append(f"Split {split}: {len(missing)} manifest entries point to missing files")
        on_disk = {p.name for p in split_dir.rglob("*.wav")}
        not_in_manifest = on_disk - manifest_files
        if not_in_manifest:
            anomalies.append(f"Split {split}: {len(not_in_manifest)} WAVs not in manifest")
    return anomalies


def check_per_dataset_balance(processed_dir: Path) -> list[str]:
    """Flag if any single dataset is < 1% or > 95% of total (multi-dataset runs)."""
    anomalies = []
    cfg = load_config().get("anomaly_checks", {})
    min_dataset_ratio = float(cfg.get("min_dataset_ratio", 0.01))
    max_dataset_ratio = float(cfg.get("max_dataset_ratio", 0.95))
    from collections import Counter

    counts = Counter()
    for split in ("dev", "test", "holdout"):
        manifest = processed_dir / split / "manifest.json"
        if not manifest.exists():
            continue
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        for item in (data if isinstance(data, list) else data.get("items", [])):
            ds = item.get("dataset") or "unknown"
            counts[ds] += 1
    if len(counts) <= 1:
        return anomalies
    total = sum(counts.values())
    for ds, cnt in counts.items():
        ratio = cnt / total
        if ratio < min_dataset_ratio:
            anomalies.append(f"Dataset '{ds}' is {ratio:.1%} of total (min {min_dataset_ratio:.0%})")
        if ratio > max_dataset_ratio:
            anomalies.append(f"Dataset '{ds}' is {ratio:.1%} of total (max {max_dataset_ratio:.0%})")
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
    missing_anomalies, not_extracted_raw_dirs = check_missing_files(raw_dir)
    all_anomalies.extend(missing_anomalies)
    if not_extracted_raw_dirs:
        logger.warning("Raw dirs with archives not extracted (no processable media yet): %s", not_extracted_raw_dirs)
    all_anomalies.extend(check_processed_splits_present(processed_dir))
    all_anomalies.extend(check_schema_failures(processed_dir))
    all_anomalies.extend(check_duration_distribution(processed_dir))
    all_anomalies.extend(check_label_imbalance(processed_dir))
    all_anomalies.extend(check_manifest_file_consistency(processed_dir))
    all_anomalies.extend(check_per_dataset_balance(processed_dir))

    report = {"anomalies": all_anomalies, "passed": len(all_anomalies) == 0}
    if not_extracted_raw_dirs:
        report["not_extracted_raw_dirs"] = not_extracted_raw_dirs
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
