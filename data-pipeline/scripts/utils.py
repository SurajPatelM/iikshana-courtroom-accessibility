"""
Shared utilities: logging, paths, config loading.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

# Base paths relative to data-pipeline/
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PIPELINE_ROOT / "data"
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
