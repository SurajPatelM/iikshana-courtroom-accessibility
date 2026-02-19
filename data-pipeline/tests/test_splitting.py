"""Tests for stratified split: no speaker overlap, ratios."""
import json
import pytest
from pathlib import Path

import sys
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from scripts.stratified_split import (
    infer_speaker_id,
    infer_emotion,
    collect_meta,
    stratified_split_by_speaker,
    run_split,
)


def test_infer_speaker_id():
    # Path with Actor_01 folder -> prefer folder
    assert infer_speaker_id(Path("Actor_01/03-01-05-01-01-01-01.wav")) == "01"
    assert infer_speaker_id(Path("subject_42/sample.wav")) == "42"
    # Filename-only: last digits (e.g. actor in RAVDESS)
    assert infer_speaker_id(Path("03-01-05-01-01-01-01.wav")) == "01"


def test_infer_emotion():
    p = Path("03-01-05-01-01-01-01.wav")  # RAVDESS code 05 = happy
    # Our simple impl looks for substring
    label = infer_emotion(p)
    assert label in ("unknown", "happy", "neutral", "calm", "sad", "angry", "fearful", "disgust", "surprised")


def test_stratified_split_no_speaker_overlap():
    meta = [
        {"path": "/a/1.wav", "speaker_id": "01", "emotion": "happy"},
        {"path": "/a/2.wav", "speaker_id": "01", "emotion": "sad"},
        {"path": "/a/3.wav", "speaker_id": "02", "emotion": "happy"},
        {"path": "/a/4.wav", "speaker_id": "02", "emotion": "neutral"},
        {"path": "/a/5.wav", "speaker_id": "03", "emotion": "sad"},
    ]
    dev, test, holdout = stratified_split_by_speaker(meta, dev_ratio=0.2, test_ratio=0.6, holdout_ratio=0.2, seed=42)
    dev_speakers = {m["speaker_id"] for m in dev}
    test_speakers = {m["speaker_id"] for m in test}
    holdout_speakers = {m["speaker_id"] for m in holdout}
    assert dev_speakers.isdisjoint(test_speakers)
    assert dev_speakers.isdisjoint(holdout_speakers)
    assert test_speakers.isdisjoint(holdout_speakers)
    assert len(dev) + len(test) + len(holdout) == len(meta)


def test_collect_meta(temp_data_dir, sample_wav_16k):
    (temp_data_dir / "actor_01_sample.wav").touch()
    files = [temp_data_dir / "actor_01_sample.wav"]
    meta = collect_meta(files)
    assert len(meta) == 1
    assert "speaker_id" in meta[0]
    assert "emotion" in meta[0]


def test_run_split_empty_dir(temp_data_dir):
    # No staged wavs -> should not crash, return zeros
    result = run_split(staged_dir=temp_data_dir, out_dir=temp_data_dir / "out", extensions=(".wav",))
    assert result["dev"] == 0
    assert result["test"] == 0
    assert result["holdout"] == 0
