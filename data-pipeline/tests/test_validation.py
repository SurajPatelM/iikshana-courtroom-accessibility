"""Tests for schema validation logic."""
import json
import pytest
from pathlib import Path

import sys
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from scripts.validate_schema import (
    validate_one_audio,
    validate_manifest_labels,
    AUDIO_SCHEMA,
)


def test_validate_one_audio_pass(sample_wav_16k):
    passed, report = validate_one_audio(sample_wav_16k, expected_sr=16000, min_duration=0.1, max_duration=120)
    assert passed
    assert report["sample_rate"] == 16000
    assert report["channels"] == 1
    assert report["duration_sec"] == pytest.approx(1.0, abs=0.01)


def test_validate_one_audio_wrong_sr(sample_wav_16k):
    passed, report = validate_one_audio(sample_wav_16k, expected_sr=8000, min_duration=0.1, max_duration=120)
    assert not passed
    assert any("sample_rate" in e for e in report["errors"])


def test_validate_one_audio_nonexistent():
    passed, report = validate_one_audio(Path("/nonexistent/file.wav"))
    assert not passed
    assert report["passed"] is False


def test_validate_manifest_labels_no_allowed(temp_data_dir):
    manifest = temp_data_dir / "manifest.json"
    manifest.write_text(json.dumps([{"emotion": "happy"}, {"emotion": "sad"}]))
    ok, errors = validate_manifest_labels(manifest, allowed=None)
    assert ok
    assert len(errors) == 0


def test_validate_manifest_labels_invalid(temp_data_dir):
    manifest = temp_data_dir / "manifest.json"
    manifest.write_text(json.dumps([{"emotion": "happy"}, {"emotion": "invalid_label"}]))
    ok, errors = validate_manifest_labels(manifest, allowed=["happy", "sad", "neutral"])
    assert not ok
    assert len(errors) >= 1


def test_audio_schema_structure():
    assert "properties" in AUDIO_SCHEMA
    assert "sample_rate" in AUDIO_SCHEMA["properties"]
    assert "required" in AUDIO_SCHEMA
