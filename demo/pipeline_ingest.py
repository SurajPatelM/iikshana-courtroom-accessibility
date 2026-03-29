"""
Ingest expo / Streamlit recordings into the same layout as the batch data pipeline.

Flow (matches offline pipeline intent):
  1. Persist bytes under ``data/raw/expo_ui/`` (audit / DVC-friendly raw hook).
  2. Run ``process_one`` from ``data-pipeline/scripts/preprocess_audio.py`` (16 kHz mono,
     normalization, trim — same as ``run_preprocessing``).
  3. Write ``EXPO_<utc_timestamp>.wav`` under ``data/processed/<split>/``.
  4. Append one entry to ``data/processed/<split>/manifest.json`` so downstream DAGs
     (``build_translation_inputs_from_audio``, validation, anomaly manifest checks ) see the file.

Call from repo root with ``PYTHONPATH`` including the repo (Streamlit app sets this).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# data-pipeline on path for ``scripts.*``
REPO_ROOT = Path(__file__).resolve().parents[1]
_DP = REPO_ROOT / "data-pipeline"
if str(_DP) not in sys.path:
    sys.path.insert(0, str(_DP))

from scripts.preprocess_audio import process_one  # noqa: E402
from scripts.utils import PROCESSED_DIR, RAW_DIR, load_config  # noqa: E402

VALID_SPLITS = ("dev", "test", "holdout")


def ingest_expo_recording(
    source_path: Path,
    *,
    split: str = "dev",
    source_suffix: str = ".wav",
) -> tuple[Path, dict[str, Any]]:
    """
    Process ``source_path`` (temporary file from UI) into ``data/processed/<split>/``
    and update manifest.

    Returns
    -------
    out_wav : Path
        Final WAV path in processed split dir.
    manifest_row : dict
        The appended manifest entry.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

    cfg = load_config()
    preproc = cfg.get("preprocessing", {})
    target_sr = int(preproc.get("target_sr", 16000))
    mono = bool(preproc.get("mono", True))
    normalize = bool(preproc.get("normalize_loudness", True))
    trim = bool(preproc.get("trim_silence", True))
    log_memory = bool(preproc.get("log_memory", False))
    court = preproc.get("courtroom_robust", {}) or {}
    high_pass_hz = float(court.get("high_pass_hz", 0) or 0)
    peak_limit_db = court.get("peak_limit_db")
    if peak_limit_db is not None:
        peak_limit_db = float(peak_limit_db)
    noise_reduction = bool(court.get("noise_reduction", False))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = RAW_DIR / "expo_ui"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_copy = raw_dir / f"recording_{ts}{source_suffix}"
    raw_copy.write_bytes(source_path.read_bytes())

    split_dir = PROCESSED_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    dest_name = f"EXPO_{ts}.wav"
    out_wav = split_dir / dest_name

    ok = process_one(
        raw_copy,
        out_wav,
        target_sr=target_sr,
        mono=mono,
        normalize=normalize,
        trim=trim,
        high_pass_hz=high_pass_hz,
        peak_limit_db=peak_limit_db,
        noise_reduction=noise_reduction,
        log_memory=log_memory,
    )
    if not ok or not out_wav.is_file():
        raise RuntimeError(
            f"Preprocessing failed for expo clip (see data-pipeline logs). Raw copy: {raw_copy}"
        )

    manifest_path = split_dir / "manifest.json"
    entries: list[Any] = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entries = []
    if not isinstance(entries, list):
        entries = []

    row = {
        "file": dest_name,
        "dataset": "EXPO",
        "speaker_id": "live_ui",
        "emotion": "neutral",
    }
    entries.append(row)
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    return out_wav, row
