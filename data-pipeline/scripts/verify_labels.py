"""
Verify emotion labels across processed manifests.

Loads manifest.json from each split (dev/test/holdout), filters to task=emotion
entries, and reports:
  - per-class counts and percentages
  - unknown count (should be 0 after proper extraction)
  - per-dataset breakdown

Exit code 1 if any emotion-task entry still has label 'unknown'.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

from scripts.utils import PROCESSED_EMOTION_DIR, get_logger
from scripts.label_extractors import CANONICAL_LABELS

logger = get_logger("verify_labels")


def load_manifests(processed_dir: Path) -> list[dict]:
    items: list[dict] = []
    for split in ("dev", "test", "holdout"):
        manifest_path = processed_dir / split / "manifest.json"
        if not manifest_path.exists():
            logger.warning("Missing manifest: %s", manifest_path)
            continue
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else data.get("items", [])
        for entry in entries:
            entry["_split"] = split
        items.extend(entries)
    return items


def verify(processed_dir: Path | None = None) -> bool:
    processed_dir = processed_dir or PROCESSED_EMOTION_DIR
    items = load_manifests(processed_dir)
    if not items:
        logger.error("No manifest entries found")
        return False

    emotion_items = [i for i in items if i.get("task", "emotion") == "emotion"]
    stt_items = [i for i in items if i.get("task") == "stt_only"]
    other_items = [i for i in items if i.get("task") not in ("emotion", "stt_only", None)]

    print(f"\n{'='*60}")
    print(f"  Label Verification Report")
    print(f"{'='*60}")
    print(f"  Total manifest entries : {len(items)}")
    print(f"  Emotion-task entries   : {len(emotion_items)}")
    print(f"  STT-only entries       : {len(stt_items)}")
    if other_items:
        print(f"  Other task entries     : {len(other_items)}")
    print()

    if not emotion_items:
        logger.warning("No emotion-task entries to verify")
        return True

    emotion_counts = Counter(i.get("emotion", "unknown") for i in emotion_items)
    total_emotion = sum(emotion_counts.values())

    print(f"  Emotion class distribution (task=emotion only):")
    print(f"  {'Label':<15} {'Count':>8} {'Pct':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8}")
    for label in sorted(emotion_counts, key=emotion_counts.get, reverse=True):
        cnt = emotion_counts[label]
        pct = 100.0 * cnt / total_emotion
        marker = " **" if label == "unknown" else ""
        print(f"  {label:<15} {cnt:>8} {pct:>7.1f}%{marker}")
    print()

    dataset_counts = Counter(i.get("dataset", "?") for i in emotion_items)
    print(f"  Per-dataset breakdown (emotion only):")
    print(f"  {'Dataset':<20} {'Count':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for ds in sorted(dataset_counts, key=dataset_counts.get, reverse=True):
        print(f"  {ds:<20} {dataset_counts[ds]:>8}")
    print()

    unknown_count = emotion_counts.get("unknown", 0)
    non_canonical = {
        label for label in emotion_counts if label not in CANONICAL_LABELS and label != "unknown"
    }

    ok = True
    if unknown_count > 0:
        logger.error(
            "%d emotion-task entries still labeled 'unknown' (%.1f%%)",
            unknown_count,
            100.0 * unknown_count / total_emotion,
        )
        ok = False
    if non_canonical:
        logger.warning(
            "Non-canonical labels found in emotion entries: %s — "
            "these should be mapped via canonicalize_emotion()",
            non_canonical,
        )
        ok = False

    if ok:
        print("  RESULT: All emotion labels are canonical and non-unknown.")
    else:
        print("  RESULT: Issues detected (see warnings above).")
    print(f"{'='*60}\n")
    return ok


def main() -> None:
    processed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not verify(processed_dir):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
