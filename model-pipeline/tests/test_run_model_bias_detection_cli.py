"""End-to-end CLI test: --from-predictions, no Gemini API."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "model-pipeline" / "scripts" / "run_model_bias_detection.py"


@pytest.mark.skipif(not RUNNER.is_file(), reason="runner missing")
def test_cli_from_predictions_writes_report(tmp_path: Path):
    processed = tmp_path / "processed" / "dev"
    processed.mkdir(parents=True)
    csv_path = processed / "translation_predictions_translation_flash_v1.csv"
    csv_path.write_text(
        "source_text,source_language,target_language,reference_translation,dataset,emotion,translated_text_model\n"
        "hi,en,es,hola,RAVDESS,calm,hola\n"
        "bye,en,es,adios,MELD,happy,adios\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    r = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--split",
            "dev",
            "--data-dir",
            str(tmp_path / "processed"),
            "--config-id",
            "translation_flash_v1",
            "--from-predictions",
            "--group-cols",
            "dataset,emotion",
            "--no-plots",
        ],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    out = processed / "model_bias_report_translation_flash_v1__dataset_emotion.json"
    assert out.is_file(), r.stdout + r.stderr
    data = json.loads(out.read_text())
    assert data["schema"] == "iikshana.model_bias_detection.v1"
    assert data["n_samples"] == 2
    assert data["overall_exact_match_accuracy"] == 1.0
    assert "dataset" in data["group_columns"]
