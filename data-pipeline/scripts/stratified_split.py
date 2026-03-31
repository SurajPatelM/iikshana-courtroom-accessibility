"""
Stratified split: Dev 20%, Test 70%, Holdout 10% by speaker identity (no overlap),
and optionally by language, emotion, demographics, audio quality.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Allow running as `python3 stratified_split.py` from data-pipeline/scripts/ or from repo root
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from scripts.utils import get_logger, load_config, PROCESSED_EMOTION_DIR, PROCESSED_STT_DIR
from scripts.label_extractors import (
    detect_dataset_name,
    extract_emotion,
    dataset_task,
)

logger = get_logger("stratified_split")


def infer_speaker_id(path: Path) -> str:
    """Infer speaker ID from filename/dir (dataset-dependent). E.g. RAVDESS: Actor_01 -> 01."""
    # Prefer parent folder like Actor_01, speaker_02
    for p in path.parents:
        if p.name in ("staged", "processed", "extracted", "wav", "data", "raw"):
            continue
        m = re.search(r"(?:actor_?|speaker_?|subject_?)_?(\d{2,3})$", p.name.lower())
        if m:
            return m.group(1)
    name = path.stem
    # RAVDESS filename: 03-01-05-01-01-01-01 -> last two digits are actor
    m = re.search(r"(\d{2,3})$", name)
    if m:
        return m.group(1)
    m = re.search(r"(?:actor_?|speaker_?|subject_?)_?(\d{2,3})", name.lower())
    if m:
        return m.group(1)
    for p in path.parents:
        if p.name not in ("staged", "processed", "extracted", "wav"):
            return p.name
    return name[:32]


# RAVDESS filename: XX-XX-NN-XX-XX-XX-XX -> third segment (index 2) is emotion code 01-08
RAVDESS_EMOTION_BY_CODE = (
    "neutral",   # 01
    "calm",      # 02
    "happy",     # 03
    "sad",       # 04
    "angry",     # 05
    "fearful",   # 06
    "surprised", # 07
    "disgust",   # 08
)


def infer_emotion(path: Path) -> str:
    """Infer emotion label from path/filename if possible."""
    stem = path.stem
    parts_lower = [p.lower() for p in path.parts]

    # RAVDESS: 03-01-05-01-01-01-01 -> third segment is emotion code (01-08)
    if "ravdess" in parts_lower or re.match(r"^\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", stem):
        segments = stem.split("-")
        if len(segments) >= 3:
            try:
                code = int(segments[2])
                if 1 <= code <= 8:
                    return RAVDESS_EMOTION_BY_CODE[code - 1]
            except ValueError:
                pass

    stem_lower = stem.lower()
    for label in ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"):
        if label in stem_lower or label in parts_lower:
            return label
    return "unknown"


def collect_meta(files: list[Path], staged_root: Path | None = None) -> list[dict]:
    """Collect path, speaker_id, emotion, dataset, and task for each file.

    Uses dataset-specific label extractors when available, falling back to
    the legacy infer_emotion() heuristic.
    """
    results = []
    for f in files:
        dataset = detect_dataset_name(f, staged_root) if staged_root else "unknown"
        task = dataset_task(dataset)

        emotion = extract_emotion(dataset, f)
        # MELD labels come only from CSV join; orphan clips (audio without a row for
        # this split) are skipped so manifests stay free of bogus "unknown".
        if dataset == "MELD" and emotion is None:
            logger.debug("Skipping MELD clip with no CSV label: %s", f)
            continue
        if emotion is None and task == "emotion":
            emotion = infer_emotion(f)

        results.append({
            "path": str(f),
            "speaker_id": infer_speaker_id(f),
            "emotion": emotion or "unknown",
            "dataset": dataset,
            "task": task,
        })
    return results


def stratified_split_by_speaker(
    meta: list[dict],
    dev_ratio: float = 0.20,
    test_ratio: float = 0.70,
    holdout_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split so that each speaker appears in only one split (no overlap)."""
    import random
    rng = random.Random(seed)
    by_speaker: dict[str, list[dict]] = {}
    for m in meta:
        sid = m["speaker_id"]
        by_speaker.setdefault(sid, []).append(m)
    speakers = list(by_speaker.keys())
    rng.shuffle(speakers)
    n = len(speakers)
    n_dev = max(1, int(n * dev_ratio))
    n_holdout = max(1, int(n * holdout_ratio))
    n_test = n - n_dev - n_holdout
    if n_test < 1:
        n_test = 1
        n_dev = max(0, n - 2)
        n_holdout = n - n_dev - n_test
    dev_speakers = set(speakers[:n_dev])
    holdout_speakers = set(speakers[n_dev : n_dev + n_holdout])
    test_speakers = set(speakers[n_dev + n_holdout :])
    dev = [m for m in meta if m["speaker_id"] in dev_speakers]
    holdout = [m for m in meta if m["speaker_id"] in holdout_speakers]
    test = [m for m in meta if m["speaker_id"] in test_speakers]
    return dev, test, holdout


def run_split(
    staged_dir: Path | None = None,
    out_dir: Path | None = None,
    extensions: tuple[str, ...] = (".wav",),
) -> dict[str, int]:
    """Build splits from staged preprocessed audio and write manifest + copy/symlink."""
    cfg = load_config()
    splits_cfg = cfg.get("splits", {})
    dev_ratio = float(splits_cfg.get("dev", 0.20))
    test_ratio = float(splits_cfg.get("test", 0.70))
    holdout_ratio = float(splits_cfg.get("holdout", 0.10))

    if staged_dir is None:
        staged_dir = PROCESSED_EMOTION_DIR / "staged"
    if out_dir is None:
        out_dir = PROCESSED_EMOTION_DIR
    staged_dir = Path(staged_dir)
    out_dir = Path(out_dir)

    files = []
    for ext in extensions:
        files.extend(staged_dir.rglob(f"*{ext}"))
    files = sorted(set(files))
    if not files:
        logger.warning("No audio files in %s", staged_dir)
        return {"dev": 0, "test": 0, "holdout": 0}

    meta = collect_meta(files, staged_root=staged_dir)
    dev_meta, test_meta, holdout_meta = stratified_split_by_speaker(
        meta, dev_ratio=dev_ratio, test_ratio=test_ratio, holdout_ratio=holdout_ratio
    )

    for split_name, split_meta in (("dev", dev_meta), ("test", test_meta), ("holdout", holdout_meta)):
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        used_names = set()
        for m in split_meta:
            src = Path(m["path"])
            if not src.exists():
                continue
            dataset = re.sub(r"[\s/\\]+", "_", m.get("dataset", "unknown"))
            dest_name = f"{dataset}_{src.stem}.wav"
            idx = 0
            while dest_name in used_names:
                idx += 1
                dest_name = f"{dataset}_{src.stem}_{idx}.wav"
            used_names.add(dest_name)
            dest = split_dir / dest_name
            if dest != src:
                import shutil
                shutil.copy2(src, dest)
            manifest.append({
                "file": dest.name,
                "dataset": dataset,
                "speaker_id": m["speaker_id"],
                "emotion": m["emotion"],
                "task": m.get("task", "emotion"),
            })
        manifest_path = split_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Split %s: %d files -> %s", split_name, len(manifest), split_dir)

    return {"dev": len(dev_meta), "test": len(test_meta), "holdout": len(holdout_meta)}


def main() -> None:
    import sys
    if len(sys.argv) >= 3:
        run_split(staged_dir=Path(sys.argv[1]), out_dir=Path(sys.argv[2]))
    elif len(sys.argv) == 2:
        run_split(staged_dir=Path(sys.argv[1]))
    else:
        emo = run_split()
        stt = run_split(
            staged_dir=PROCESSED_STT_DIR / "staged",
            out_dir=PROCESSED_STT_DIR,
        )
        logger.info("stratified_split emotion: %s; stt: %s", emo, stt)


if __name__ == "__main__":
    main()
