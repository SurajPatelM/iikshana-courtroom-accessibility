"""
API-input validation: expected audio format (16kHz, mono), duration bounds, emotion labels.
Ensures audio is valid for consumption by our APIs (Gemini, STT, emotion, etc.). Same checks for
pipeline evaluation data and for live inference. Outputs JSON schema and quality report.
"""
import json
from pathlib import Path

import soundfile as sf

from scripts.utils import get_logger, load_config, PROCESSED_DIR

logger = get_logger("validate_schema")

# Default schema
AUDIO_SCHEMA = {
    "type": "object",
    "properties": {
        "sample_rate": {"type": "integer", "minimum": 8000, "maximum": 48000},
        "channels": {"type": "integer", "minimum": 1, "maximum": 2},
        "duration_sec": {"type": "number", "minimum": 0.1, "maximum": 120},
        "format": {"type": "string", "enum": ["WAV", "FLAC"]},
        "subtype": {"type": "string"},
    },
    "required": ["sample_rate", "channels", "duration_sec"],
}


def validate_one_audio(
    path: Path,
    expected_sr: int = 16000,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> tuple[bool, dict]:
    """Validate one audio file; return (passed, report_dict)."""
    report = {"path": str(path), "errors": []}
    try:
        info = sf.info(path)
        sr = info.samplerate
        ch = info.channels
        dur = info.duration
        report["sample_rate"] = sr
        report["channels"] = ch
        report["duration_sec"] = round(dur, 3)
        report["format"] = info.format
        if sr != expected_sr:
            report["errors"].append(f"sample_rate {sr} != expected {expected_sr}")
        if ch != 1:
            report["errors"].append(f"channels {ch} != 1 (mono)")
        if dur < min_duration:
            report["errors"].append(f"duration {dur:.2f}s < min {min_duration}s")
        if dur > max_duration:
            report["errors"].append(f"duration {dur:.2f}s > max {max_duration}s")
        passed = len(report["errors"]) == 0
        report["passed"] = passed
        return passed, report
    except Exception as e:
        report["errors"].append(str(e))
        report["passed"] = False
        return False, report


def validate_manifest_labels(manifest_path: Path, allowed: list[str] | None) -> tuple[bool, list[str]]:
    """Validate emotion labels in manifest.json against allowed set."""
    errors = []
    if not manifest_path.exists():
        return False, ["manifest not found"]
    if allowed is None:
        return True, []
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    for item in data if isinstance(data, list) else data.get("items", []):
        label = item.get("emotion") or item.get("label")
        if label and label not in allowed:
            errors.append(f"invalid label '{label}' (allowed: {allowed})")
    return len(errors) == 0, errors


def run_validation(
    data_dir: Path | None = None,
    schema_out: Path | None = None,
    report_out: Path | None = None,
) -> dict:
    """Validate processed splits and write schema + quality report."""
    cfg = load_config()
    val_cfg = cfg.get("validation", {})
    expected_sr = val_cfg.get("expected_sr", 16000)
    min_dur = val_cfg.get("min_duration_sec", 0.5)
    max_dur = val_cfg.get("max_duration_sec", 30.0)

    if data_dir is None:
        data_dir = PROCESSED_DIR
    data_dir = Path(data_dir)
    if schema_out is None:
        schema_out = data_dir / "audio_schema.json"
    if report_out is None:
        report_out = data_dir / "quality_report.json"

    results = {"files_checked": 0, "passed": 0, "failed": 0, "file_reports": [], "manifest_errors": []}
    for split in ("dev", "test", "holdout"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        manifest_path = split_dir / "manifest.json"
        allowed = val_cfg.get("allowed_emotion_labels", {}).get("RAVDESS")  # or from config per-dataset
        m_ok, m_errors = validate_manifest_labels(manifest_path, allowed)
        if m_errors:
            results["manifest_errors"].extend(m_errors)
        for wav in split_dir.glob("*.wav"):
            passed, report = validate_one_audio(wav, expected_sr, min_dur, max_dur)
            results["files_checked"] += 1
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
            results["file_reports"].append(report)

    with open(schema_out, "w", encoding="utf-8") as f:
        json.dump({**AUDIO_SCHEMA, "expected_sr": expected_sr, "min_duration_sec": min_dur, "max_duration_sec": max_dur}, f, indent=2)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Validation: %d checked, %d passed, %d failed", results["files_checked"], results["passed"], results["failed"])
    return results


def main() -> None:
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_validation(data_dir=data_dir)


if __name__ == "__main__":
    main()
