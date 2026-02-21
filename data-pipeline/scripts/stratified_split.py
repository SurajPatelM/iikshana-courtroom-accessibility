"""
Stratified split: Dev 20%, Test 70%, Holdout 10% by speaker identity (no overlap),
and optionally by language, emotion, demographics, audio quality.
"""
import json
import re
from pathlib import Path

from scripts.utils import get_logger, load_config, PROCESSED_DIR

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


def infer_emotion(path: Path) -> str:
    """Infer emotion label from path/filename if possible."""
    stem = path.stem.lower()
    # RAVDESS: 03-01-05-01-01-01-01 -> 05 is emotion code
    for label in ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"):
        if label in stem or label in path.parts:
            return label
    return "unknown"


def collect_meta(files: list[Path]) -> list[dict]:
    """Collect path, speaker_id, emotion for each file."""
    return [
        {"path": str(f), "speaker_id": infer_speaker_id(f), "emotion": infer_emotion(f)}
        for f in files
    ]


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
        staged_dir = PROCESSED_DIR / "staged"
    if out_dir is None:
        out_dir = PROCESSED_DIR
    staged_dir = Path(staged_dir)
    out_dir = Path(out_dir)

    files = []
    for ext in extensions:
        files.extend(staged_dir.rglob(f"*{ext}"))
    files = sorted(set(files))
    if not files:
        logger.warning("No audio files in %s", staged_dir)
        return {"dev": 0, "test": 0, "holdout": 0}

    meta = collect_meta(files)
    dev_meta, test_meta, holdout_meta = stratified_split_by_speaker(
        meta, dev_ratio=dev_ratio, test_ratio=test_ratio, holdout_ratio=holdout_ratio
    )

    for split_name, split_meta in (("dev", dev_meta), ("test", test_meta), ("holdout", holdout_meta)):
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for m in split_meta:
            src = Path(m["path"])
            if not src.exists():
                continue
            dest = split_dir / src.name
            if dest != src:
                import shutil
                shutil.copy2(src, dest)
            manifest.append({"file": dest.name, "speaker_id": m["speaker_id"], "emotion": m["emotion"]})
        manifest_path = split_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Split %s: %d files -> %s", split_name, len(manifest), split_dir)

    return {"dev": len(dev_meta), "test": len(test_meta), "holdout": len(holdout_meta)}


def main() -> None:
    import sys
    staged = sys.argv[1] if len(sys.argv) > 1 else None
    run_split(staged_dir=Path(staged) if staged else None)


if __name__ == "__main__":
    main()
