"""
Dataset-specific emotion label extraction and unified canonical mapping.

Each emotion dataset encodes labels differently (filename codes, CSVs, folder names).
This module provides per-dataset extractors and a single canonicalization layer
that maps every raw label into the Unified-7 label set used for evaluation.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from scripts.utils import get_logger, load_config, RAW_DIR

logger = get_logger("label_extractors")

# ---------------------------------------------------------------------------
# Unified-7 canonical label set
# ---------------------------------------------------------------------------
CANONICAL_LABELS = frozenset(
    ("neutral", "happy", "sad", "angry", "fear", "disgust", "surprised")
)

_LABEL_MAP: dict[str, str] = {
    # identity
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "surprised": "surprised",
    # RAVDESS variants
    "calm": "neutral",
    "fearful": "fear",
    # MELD variants
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "surprise": "surprised",
    # IEMOCAP variants (future-proofing)
    "happiness": "happy",
    "frustrated": "angry",
}

# ---------------------------------------------------------------------------
# Dataset task classification
# ---------------------------------------------------------------------------
_EMOTION_DATASETS = {"RAVDESS", "MELD", "IEMOCAP", "CREMA-D", "TESS", "SAVEE", "EMO-DB", "AESDD", "MSP-Podcast"}
_STT_ONLY_DATASETS = {"common_voice", "librispeech", "voxpopuli"}


def dataset_task(dataset_name: str) -> str:
    """Return 'emotion' or 'stt_only' based on dataset identity."""
    if dataset_name in _EMOTION_DATASETS:
        return "emotion"
    if dataset_name in _STT_ONLY_DATASETS:
        return "stt_only"
    cfg = load_config()
    if dataset_name in cfg.get("emotion_datasets", {}):
        return "emotion"
    if dataset_name in cfg.get("multilingual_speech", {}):
        return "stt_only"
    return "unknown"


def canonicalize_emotion(raw_label: str) -> Optional[str]:
    """Map a raw emotion string to the Unified-7 canonical set, or None."""
    mapped = _LABEL_MAP.get(raw_label.lower().strip())
    if mapped and mapped in CANONICAL_LABELS:
        return mapped
    return None


# ---------------------------------------------------------------------------
# RAVDESS extractor
# ---------------------------------------------------------------------------
_RAVDESS_CODE_TO_RAW = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "surprised",
    8: "disgust",
}


def extract_ravdess(path: Path) -> Optional[str]:
    """Extract canonical emotion from a RAVDESS filename like 03-01-05-01-01-01-01.wav."""
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    try:
        code = int(parts[2])
    except ValueError:
        return None
    raw = _RAVDESS_CODE_TO_RAW.get(code)
    if raw is None:
        return None
    return canonicalize_emotion(raw)


# ---------------------------------------------------------------------------
# MELD extractor — builds a lookup index from the three CSV files
# ---------------------------------------------------------------------------
_MELD_SUBFOLDER_TO_CSV = {
    "train_splits": "train_sent_emo.csv",
    "dev_splits_complete": "dev_sent_emo.csv",
    "output_repeated_splits_test": "test_sent_emo.csv",
}

_DIA_UTT_RE = re.compile(r"dia(\d+)_utt(\d+)")
# Alternate naming in MELD test folder, e.g. final_videos_testdia93_utt6.wav
_FINAL_VIDEOS_TEST_RE = re.compile(r"final_videos_testdia(\d+)_utt(\d+)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _build_meld_index() -> dict[tuple[str, int, int], str]:
    """Build {(csv_basename, dialogue_id, utterance_id) -> raw_emotion} from MELD CSVs.

    Dialogue IDs are local per MELD split, so the CSV basename is part of the key.
    """
    index: dict[tuple[str, int, int], str] = {}
    meld_root = RAW_DIR / "MELD"
    if not meld_root.exists():
        logger.warning("MELD raw directory not found at %s", meld_root)
        return index

    csv_files = list(meld_root.rglob("*_sent_emo.csv"))
    if not csv_files:
        logger.warning("No MELD CSV files found under %s", meld_root)
        return index

    for csv_path in csv_files:
        csv_name = csv_path.name
        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dia_id = int(row["Dialogue_ID"])
                        utt_id = int(row["Utterance_ID"])
                        emotion = row["Emotion"].strip().lower()
                        index[(csv_name, dia_id, utt_id)] = emotion
                    except (KeyError, ValueError):
                        continue
        except Exception as exc:
            logger.warning("Failed to read MELD CSV %s: %s", csv_path, exc)

    logger.info("MELD index built: %d entries from %d CSVs", len(index), len(csv_files))
    return index


def _meld_subfolder_from_path(path: Path) -> Optional[str]:
    """Determine which MELD subfolder (train_splits, dev_splits_complete, etc.) a file is in."""
    parts_lower = [p.lower() for p in path.parts]
    for subfolder in _MELD_SUBFOLDER_TO_CSV:
        if subfolder.lower() in parts_lower:
            return subfolder
    return None


def _parse_meld_dia_utt(stem: str) -> tuple[int, int] | None:
    """Parse Dialogue_ID and Utterance_ID from a clip filename stem.

    Handles:
    - dia12_utt3.wav -> (12, 3)
    - dia12_utt3_1.wav (dedup suffix from split) -> (12, 3); trailing _<digits> after utt is ignored
    - final_videos_testdia93_utt6.wav -> (93, 6)
    """
    m = _FINAL_VIDEOS_TEST_RE.search(stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = _DIA_UTT_RE.search(stem)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def extract_meld(path: Path) -> Optional[str]:
    """Extract canonical emotion for a MELD clip using its staged path and the CSV index."""
    parsed = _parse_meld_dia_utt(path.stem)
    if parsed is None:
        return None
    dia_id, utt_id = parsed

    subfolder = _meld_subfolder_from_path(path)
    # final_videos_* clips live under test split; map to test CSV
    if subfolder is None and _FINAL_VIDEOS_TEST_RE.search(path.stem):
        subfolder = "output_repeated_splits_test"
    if subfolder is None:
        return None

    csv_name = _MELD_SUBFOLDER_TO_CSV.get(subfolder)
    if csv_name is None:
        return None

    index = _build_meld_index()
    raw_emotion = index.get((csv_name, dia_id, utt_id))
    if raw_emotion is None:
        logger.debug("MELD label not found: csv=%s dia=%d utt=%d path=%s", csv_name, dia_id, utt_id, path)
        return None

    return canonicalize_emotion(raw_emotion)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_EXTRACTORS: dict[str, callable] = {
    "RAVDESS": extract_ravdess,
    "MELD": extract_meld,
}


def detect_dataset_name(path: Path, staged_root: Path) -> str:
    """Infer the dataset name from the top-level directory under staged_root."""
    try:
        rel = path.relative_to(staged_root)
        return rel.parts[0] if rel.parts else "unknown"
    except ValueError:
        return "unknown"


def extract_emotion(dataset: str, path: Path) -> Optional[str]:
    """Extract canonical emotion for a file, given its dataset and staged path.

    Returns None when:
      - dataset is stt_only, or
      - no extractor exists for this dataset, or
      - the extractor could not determine a label.
    """
    task = dataset_task(dataset)
    if task != "emotion":
        return None

    extractor = _EXTRACTORS.get(dataset)
    if extractor is None:
        logger.debug("No label extractor for dataset '%s' (file: %s)", dataset, path)
        return None

    return extractor(path)
