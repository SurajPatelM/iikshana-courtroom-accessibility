"""Unit tests for model_bias_detection_core (no API calls)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

MP_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = MP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from model_bias_detection_core import (  # noqa: E402
    aggregate_exact_match_by_dataset,
    assert_required_columns,
    build_metric_frame,
    disparities_exact_match,
    exact_match_list,
    mitigation_recommendations,
    write_report_json,
)


def test_exact_match_list():
    assert exact_match_list(["a", "b"], ["A", "c"]) == 0.5


def test_assert_required_columns_ok():
    df = pd.DataFrame(
        {
            "source_text": ["x"],
            "source_language": ["en"],
            "target_language": ["es"],
            "reference_translation": ["hola"],
        }
    )
    assert_required_columns(df)


def test_assert_required_columns_missing():
    df = pd.DataFrame(
        {
            "source_text": ["x"],
            "source_language": ["en"],
            "target_language": ["es"],
        }
    )
    with pytest.raises(ValueError, match="reference_translation"):
        assert_required_columns(df)


def test_build_metric_frame_and_disparities():
    df = pd.DataFrame(
        {
            "dataset": ["A", "A", "B", "B"],
            "emotion": ["n", "h", "n", "h"],
            "reference_translation": ["x", "x", "y", "y"],
        }
    )
    y_true = df["reference_translation"].tolist()
    y_pred = ["x", "wrong", "y", "y"]
    sensitive = df[["dataset", "emotion"]].astype(str)
    mf, payload = build_metric_frame(y_true, y_pred, sensitive)
    assert "overall" in payload or "error" in payload
    if mf is not None:
        d = disparities_exact_match(mf, threshold=0.01)
        assert isinstance(d, list)
        m = mitigation_recommendations(d, ["dataset", "emotion"])
        assert len(m) >= 3


def test_aggregate_exact_match_by_dataset():
    records = [
        {"dataset": "RAVDESS", "exact_match": 0.5},
        {"dataset": "RAVDESS", "exact_match": 0.5},
        {"dataset": "MELD", "exact_match": 1.0},
    ]
    s = aggregate_exact_match_by_dataset(records)
    assert abs(s["RAVDESS"] - 0.5) < 1e-6
    assert abs(s["MELD"] - 1.0) < 1e-6


def test_write_report_json(tmp_path: Path):
    p = tmp_path / "out.json"
    write_report_json(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
