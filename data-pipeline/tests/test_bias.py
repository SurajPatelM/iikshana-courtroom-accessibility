"""Unit tests for detect_bias: load_manifests, slicing, compute_counts, run_bias_analysis."""
import json
from pathlib import Path

import pytest

import sys
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from scripts.detect_bias import (
    load_manifests,
    slice_by_emotion,
    slice_by_speaker,
    slice_by_duration,
    compute_counts_per_slice,
    run_bias_analysis,
)


def test_load_manifests_empty_dir(temp_data_dir):
    """Empty data_dir returns empty list."""
    data_dir = temp_data_dir / "processed"
    data_dir.mkdir()
    entries = load_manifests(data_dir)
    assert entries == []


def test_load_manifests_list_format(temp_data_dir):
    """Manifest as list of items is loaded with split attached."""
    data_dir = temp_data_dir / "processed"
    (data_dir / "dev").mkdir(parents=True)
    (data_dir / "dev" / "manifest.json").write_text(
        json.dumps([{"file": "a.wav", "emotion": "happy"}, {"file": "b.wav", "emotion": "sad"}])
    )
    entries = load_manifests(data_dir)
    assert len(entries) == 2
    assert all(e["split"] == "dev" for e in entries)
    emotions = {e.get("emotion") for e in entries}
    assert emotions == {"happy", "sad"}


def test_load_manifests_items_key(temp_data_dir):
    """Manifest as dict with 'items' key is loaded."""
    data_dir = temp_data_dir / "processed"
    (data_dir / "test").mkdir(parents=True)
    (data_dir / "test" / "manifest.json").write_text(
        json.dumps({"items": [{"file": "x.wav", "label": "neutral"}]})
    )
    entries = load_manifests(data_dir)
    assert len(entries) == 1
    assert entries[0]["split"] == "test"
    assert entries[0].get("label") == "neutral"


def test_load_manifests_multiple_splits(temp_data_dir):
    """All three splits are merged."""
    data_dir = temp_data_dir / "processed"
    for split in ("dev", "test", "holdout"):
        (data_dir / split).mkdir(parents=True)
        (data_dir / split / "manifest.json").write_text(
            json.dumps([{"file": f"{split}.wav", "emotion": "calm"}])
        )
    entries = load_manifests(data_dir)
    assert len(entries) == 3
    assert {e["split"] for e in entries} == {"dev", "test", "holdout"}


def test_slice_by_emotion():
    """Entries are grouped by emotion (or label) key."""
    entries = [
        {"emotion": "happy"},
        {"emotion": "sad"},
        {"label": "happy"},
        {},
    ]
    by_emotion = slice_by_emotion(entries)
    assert set(by_emotion.keys()) == {"happy", "sad", "unknown"}
    assert len(by_emotion["happy"]) == 2
    assert len(by_emotion["sad"]) == 1
    assert len(by_emotion["unknown"]) == 1


def test_slice_by_speaker():
    """Entries are grouped by speaker_id."""
    entries = [
        {"speaker_id": "01"},
        {"speaker_id": "01"},
        {"speaker_id": "02"},
    ]
    by_speaker = slice_by_speaker(entries)
    assert len(by_speaker["01"]) == 2
    assert len(by_speaker["02"]) == 1
    assert len(by_speaker.get("unknown", [])) == 0


def test_slice_by_duration():
    """Entries are bucketed into short/medium/long by duration_sec (missing -> 0 -> short)."""
    entries = [
        {"duration_sec": 1.0},
        {"duration_sec": 3.0},
        {"duration_sec": 6.0},
        {},
    ]
    by_duration = slice_by_duration(entries)
    # 1.0 and missing (0) both < 2 -> short
    assert len(by_duration["short"]) == 2
    assert len(by_duration["medium"]) == 1  # 2 <= 3 <= 5
    assert len(by_duration["long"]) == 1     # 6 > 5
    assert sum(len(v) for v in by_duration.values()) == 4


def test_compute_counts_per_slice():
    """Counts are lengths of each slice list."""
    slices = {"a": [1, 2], "b": [3], "c": []}
    counts = compute_counts_per_slice(slices)
    assert counts == {"a": 2, "b": 1, "c": 0}


def test_run_bias_analysis_empty_manifests(monkeypatch, temp_data_dir):
    """When no manifest entries, returns minimal report and does not raise."""
    data_dir = temp_data_dir / "processed"
    data_dir.mkdir()
    (data_dir / "dev").mkdir()
    report_path = temp_data_dir / "bias_report.json"
    monkeypatch.setattr("scripts.detect_bias.PROCESSED_DIR", data_dir)
    report = run_bias_analysis(data_dir=data_dir, report_path=report_path)
    assert report.get("total_entries", 0) == 0 or "slices" in report
    assert "disparities" in report
    assert "recommendations" in report


def test_run_bias_analysis_with_entries(monkeypatch, temp_data_dir):
    """With manifest entries, report contains by_emotion, by_duration, and is written to file."""
    data_dir = temp_data_dir / "processed"
    (data_dir / "dev").mkdir(parents=True)
    (data_dir / "dev" / "manifest.json").write_text(
        json.dumps([
            {"file": "a.wav", "emotion": "happy", "speaker_id": "01", "duration_sec": 1.0},
            {"file": "b.wav", "emotion": "happy", "speaker_id": "01", "duration_sec": 3.0},
            {"file": "c.wav", "emotion": "sad", "speaker_id": "02", "duration_sec": 5.0},
        ])
    )
    report_path = temp_data_dir / "bias_report.json"
    monkeypatch.setattr("scripts.detect_bias.load_config", lambda: {"bias_detection": {"disparity_threshold": 0.5}})
    report = run_bias_analysis(data_dir=data_dir, report_path=report_path)
    assert report["total_entries"] == 3
    assert "by_emotion" in report
    assert "happy" in report["by_emotion"]
    assert report["by_emotion"]["happy"] == 2
    assert report["by_emotion"]["sad"] == 1
    assert "by_duration" in report
    assert report_path.exists()
    loaded = json.loads(report_path.read_text())
    assert loaded["total_entries"] == 3
    assert "recommendations" in loaded
