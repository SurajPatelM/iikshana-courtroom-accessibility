"""
Shared utilities: logging, paths, config loading.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

# Base paths: pipeline lives in data-pipeline/, data lives at repo root data/
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(PIPELINE_ROOT.parent)))
DATA_ROOT = REPO_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
LEGAL_GLOSSARY_DIR = DATA_ROOT / "legal_glossary"
LOGS_DIR = PIPELINE_ROOT / "logs"
CONFIG_DIR = PIPELINE_ROOT / "config"


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger that writes to logs/ with timestamps."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if log_file is not None or True:
        fh = logging.FileHandler(
            LOGS_DIR / (log_file if log_file else f"{name}.log"),
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_config(filename: str = "datasets.yaml") -> dict:
    """Load YAML config from config/."""
    path = CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _processed_layout_subdir(cfg_key: str, default: str, env_var: str) -> Path:
    """Resolve data/processed/<name>/ from datasets.yaml processed_layout and optional env override."""
    cfg = load_config().get("processed_layout") or {}
    name = (os.environ.get(env_var) or "").strip() or cfg.get(cfg_key, default)
    name = str(name).strip().strip("/\\")
    return PROCESSED_DIR / name


# Split processed outputs: emotion benchmarks vs STT-only vs future legal (see processed_layout in datasets.yaml).
# Global reports (quality_report.json, anomaly_report.json, …) stay under PROCESSED_DIR.
PROCESSED_EMOTION_DIR = _processed_layout_subdir("emotion_subdir", "emotions", "PIPELINE_EMOTION_SUBDIR")
PROCESSED_STT_DIR = _processed_layout_subdir("stt_subdir", "stt", "PIPELINE_STT_SUBDIR")
PROCESSED_LEGAL_DIR = _processed_layout_subdir("legal_subdir", "legal", "PIPELINE_LEGAL_SUBDIR")
