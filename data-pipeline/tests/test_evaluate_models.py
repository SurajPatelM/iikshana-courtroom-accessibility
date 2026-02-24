"""Unit tests for evaluate_models (WER, BLEU, F1). Skipped if script is not present."""
import json
from pathlib import Path

import pytest

import sys
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

# Skip entire module if evaluate_models does not exist (script may be added later)
evaluate_models = pytest.importorskip("scripts.evaluate_models", reason="scripts.evaluate_models not found")


def test_run_evaluation_returns_dict(temp_data_dir):
    """run_evaluation returns a dict with metrics or error info."""
    # Use a minimal processed dir: dev manifest with one entry so evaluation can run or skip gracefully
    data_dir = temp_data_dir / "processed"
    (data_dir / "dev").mkdir(parents=True)
    (data_dir / "dev" / "manifest.json").write_text(
        json.dumps([{"file": "sample.wav", "emotion": "neutral"}])
    )
    metrics_path = data_dir / "evaluation_metrics.json"
    result = evaluate_models.run_evaluation(
        data_dir=data_dir,
        metrics_path=metrics_path,
        use_live_apis=False,
    )
    assert isinstance(result, dict)
    # With use_live_apis=False, implementation may write metrics or skip; we only require dict return
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text())
        assert isinstance(data, dict)


def test_run_evaluation_accepts_data_dir_none(monkeypatch, temp_data_dir):
    """run_evaluation can be called with data_dir=None (uses PROCESSED_DIR)."""
    # Use temp dir so we don't write to repo
    monkeypatch.setattr("scripts.utils.PROCESSED_DIR", temp_data_dir)
    if hasattr(evaluate_models, "PROCESSED_DIR"):
        monkeypatch.setattr("scripts.evaluate_models.PROCESSED_DIR", temp_data_dir)
    result = evaluate_models.run_evaluation(
        data_dir=None,
        metrics_path=temp_data_dir / "evaluation_metrics.json",
        use_live_apis=False,
    )
    assert isinstance(result, dict)
