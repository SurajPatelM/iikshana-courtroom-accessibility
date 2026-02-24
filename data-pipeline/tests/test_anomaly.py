"""Unit tests for anomaly_check: missing files, splits, schema, labels, manifest consistency."""
import json
from pathlib import Path

import pytest

import sys
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from scripts.anomaly_check import (
    check_missing_files,
    check_processed_splits_present,
    check_schema_failures,
    check_label_imbalance,
    check_manifest_file_consistency,
    check_per_dataset_balance,
    run_anomaly_checks,
)


def _minimal_config(
    include_video=False,
    min_files_per_split=1,
    fail_on_schema=True,
    max_class_ratio=0.80,
    min_class_ratio=0.05,
    max_out_of_range_ratio=0.2,
    min_dataset_ratio=0.01,
    max_dataset_ratio=0.95,
):
    return {
        "preprocessing": {"include_video": include_video},
        "validation": {"min_duration_sec": 0.5, "max_duration_sec": 30.0},
        "anomaly_checks": {
            "min_files_per_split": min_files_per_split,
            "fail_on_schema_failures": fail_on_schema,
            "max_class_ratio": max_class_ratio,
            "min_class_ratio": min_class_ratio,
            "max_out_of_range_ratio": max_out_of_range_ratio,
            "min_dataset_ratio": min_dataset_ratio,
            "max_dataset_ratio": max_dataset_ratio,
        },
    }


def test_check_missing_files_raw_dir_does_not_exist(monkeypatch, temp_data_dir):
    """When raw_dir does not exist, anomaly is reported."""
    monkeypatch.setattr("scripts.anomaly_check.load_config", lambda: _minimal_config())
    anomalies, not_extracted = check_missing_files(temp_data_dir / "nonexistent")
    assert "data/raw does not exist" in anomalies or any("exist" in a.lower() for a in anomalies)
    assert not_extracted == []


def test_check_missing_files_empty_dirs_are_anomaly(monkeypatch, temp_data_dir):
    """Empty raw subdirs (no files except .gitkeep) are reported as anomaly."""
    raw = temp_data_dir / "raw"
    raw.mkdir()
    (raw / "empty_dataset").mkdir()
    (raw / "with_gitkeep").mkdir()
    (raw / "with_gitkeep" / ".gitkeep").write_text("")
    monkeypatch.setattr("scripts.anomaly_check.load_config", lambda: _minimal_config())
    anomalies, not_extracted = check_missing_files(raw)
    assert any("Empty raw dirs" in a or "empty" in a.lower() for a in anomalies)
    assert not_extracted == []


def test_check_missing_files_with_wav_not_anomaly(monkeypatch, temp_data_dir):
    """Raw dir with at least one .wav is not considered empty."""
    raw = temp_data_dir / "raw"
    raw.mkdir()
    ds = raw / "RAVDESS"
    ds.mkdir()
    (ds / "sample.wav").write_bytes(b"x")
    monkeypatch.setattr("scripts.anomaly_check.load_config", lambda: _minimal_config())
    anomalies, _ = check_missing_files(raw)
    assert not any("Empty raw dirs" in a for a in anomalies)


def test_check_processed_splits_present_missing_split(monkeypatch, temp_data_dir):
    """Missing dev/test/holdout directory is reported."""
    processed = temp_data_dir / "processed"
    processed.mkdir()
    (processed / "dev").mkdir()
    # test and holdout missing
    monkeypatch.setattr("scripts.anomaly_check.load_config", lambda: _minimal_config())
    anomalies = check_processed_splits_present(processed)
    assert any("missing" in a.lower() for a in anomalies)


def test_check_processed_splits_present_min_files(monkeypatch, temp_data_dir):
    """Split with fewer than min_files_per_split WAVs is reported when min > 0."""
    processed = temp_data_dir / "processed"
    for split in ("dev", "test", "holdout"):
        (processed / split).mkdir(parents=True)
    # dev has 0 wavs; min_files_per_split=2
    monkeypatch.setattr(
        "scripts.anomaly_check.load_config",
        lambda: _minimal_config(min_files_per_split=2),
    )
    anomalies = check_processed_splits_present(processed)
    assert any("min" in a.lower() or "0 files" in a for a in anomalies)


def test_check_schema_failures_report_missing(monkeypatch, temp_data_dir):
    """Missing quality_report.json yields anomaly when fail_on_schema_failures is True."""
    processed = temp_data_dir / "processed"
    processed.mkdir()
    monkeypatch.setattr(
        "scripts.anomaly_check.load_config",
        lambda: _minimal_config(fail_on_schema=True),
    )
    anomalies = check_schema_failures(processed)
    assert any("Validation report missing" in a or "validation_dag" in a.lower() for a in anomalies)


def test_check_schema_failures_failed_count(monkeypatch, temp_data_dir):
    """quality_report with failed > 0 yields anomaly."""
    processed = temp_data_dir / "processed"
    processed.mkdir()
    (processed / "quality_report.json").write_text(
        json.dumps({"failed": 3, "passed": 10, "files_checked": 13})
    )
    monkeypatch.setattr(
        "scripts.anomaly_check.load_config",
        lambda: _minimal_config(fail_on_schema=True),
    )
    anomalies = check_schema_failures(processed)
    assert any("failed" in a.lower() for a in anomalies)


def test_check_schema_failures_disabled(monkeypatch, temp_data_dir):
    """When fail_on_schema_failures is False, schema failures are not reported."""
    processed = temp_data_dir / "processed"
    processed.mkdir()
    (processed / "quality_report.json").write_text(
        json.dumps({"failed": 1, "passed": 0, "files_checked": 1})
    )
    monkeypatch.setattr(
        "scripts.anomaly_check.load_config",
        lambda: _minimal_config(fail_on_schema=False),
    )
    anomalies = check_schema_failures(processed)
    assert not anomalies


def test_check_label_imbalance_dominant_class(monkeypatch, temp_data_dir):
    """One label > max_class_ratio is reported."""
    processed = temp_data_dir / "processed"
    for split in ("dev", "test"):
        (processed / split).mkdir(parents=True)
        manifest = processed / split / "manifest.json"
        # 90% happy, 10% sad
        items = [{"emotion": "happy", "file": f"a{i}.wav"} for i in range(9)]
        items.append({"emotion": "sad", "file": "b.wav"})
        manifest.write_text(json.dumps(items))
    monkeypatch.setattr(
        "scripts.anomaly_check.load_config",
        lambda: _minimal_config(max_class_ratio=0.80),
    )
    anomalies = check_label_imbalance(processed)
    assert any("happy" in a or "80%" in a or "90%" in a for a in anomalies)


def test_check_label_imbalance_all_unknown_allowed(monkeypatch, temp_data_dir):
    """Single label 'unknown' for all items does not trigger imbalance (allowed)."""
    processed = temp_data_dir / "processed"
    (processed / "dev").mkdir(parents=True)
    (processed / "dev" / "manifest.json").write_text(
        json.dumps([{"emotion": "unknown", "file": "x.wav"}] * 5)
    )
    monkeypatch.setattr("scripts.anomaly_check.load_config", lambda: _minimal_config())
    anomalies = check_label_imbalance(processed)
    assert not anomalies


def test_check_manifest_file_consistency_missing_file(monkeypatch, temp_data_dir):
    """Manifest entry pointing to non-existent file is reported."""
    processed = temp_data_dir / "processed"
    split_dir = processed / "dev"
    split_dir.mkdir(parents=True)
    (split_dir / "manifest.json").write_text(
        json.dumps([{"file": "missing.wav", "emotion": "happy"}])
    )
    # do not create missing.wav
    anomalies = check_manifest_file_consistency(processed)
    assert any("missing" in a.lower() for a in anomalies)


def test_check_manifest_file_consistency_extra_wav(monkeypatch, temp_data_dir):
    """WAV on disk not in manifest is reported."""
    processed = temp_data_dir / "processed"
    split_dir = processed / "dev"
    split_dir.mkdir(parents=True)
    (split_dir / "manifest.json").write_text(
        json.dumps([{"file": "a.wav", "emotion": "happy"}])
    )
    (split_dir / "a.wav").write_bytes(b"x")
    (split_dir / "orphan.wav").write_bytes(b"y")
    anomalies = check_manifest_file_consistency(processed)
    assert any("not in manifest" in a or "WAVs" in a for a in anomalies)


def test_check_per_dataset_balance_two_datasets(monkeypatch, temp_data_dir):
    """One dataset > max_dataset_ratio is reported."""
    processed = temp_data_dir / "processed"
    for split in ("dev", "test"):
        (processed / split).mkdir(parents=True)
        items = [{"dataset": "A", "file": f"a{i}.wav"} for i in range(98)]
        items.extend([{"dataset": "B", "file": f"b{i}.wav"} for i in range(2)])
        (processed / split / "manifest.json").write_text(json.dumps(items))
    monkeypatch.setattr(
        "scripts.anomaly_check.load_config",
        lambda: _minimal_config(max_dataset_ratio=0.95),
    )
    anomalies = check_per_dataset_balance(processed)
    assert any("A" in a or "95%" in a or "98%" in a for a in anomalies)


def test_run_anomaly_checks_writes_report(monkeypatch, temp_data_dir):
    """run_anomaly_checks writes report JSON and returns passed=False when anomalies exist."""
    raw = temp_data_dir / "raw"
    raw.mkdir()
    (raw / "empty").mkdir()
    processed = temp_data_dir / "processed"
    processed.mkdir()
    report_path = temp_data_dir / "anomaly_report.json"
    monkeypatch.setattr("scripts.anomaly_check.load_config", lambda: _minimal_config())
    monkeypatch.setattr("scripts.anomaly_check.RAW_DIR", raw)
    monkeypatch.setattr("scripts.anomaly_check.PROCESSED_DIR", processed)
    report = run_anomaly_checks(
        raw_dir=raw, processed_dir=processed, report_path=report_path
    )
    assert report["passed"] is False
    assert "anomalies" in report
    assert len(report["anomalies"]) > 0
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["passed"] is False
    assert "anomalies" in data
