"""
Rollback guard for CI/CD.

Compares current validation metrics against a checked-in baseline file.
Fails the pipeline when key metrics regress.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CURRENT_METRICS_CANDIDATES = [
    Path("data/model_runs/dev/validation_metrics.json"),
    Path("data/model_runs/test/validation_metrics.json"),
    Path("data/model_runs/holdout/validation_metrics.json"),
]
BASELINE_PATH = Path("data/baseline_metrics.json")
TOLERANCE = 0.0


def _find_current_metrics_file() -> Path | None:
    for p in CURRENT_METRICS_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _extract_metrics(payload: dict[str, Any]) -> dict[str, float]:
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        data = payload["metrics"]
    elif "configs" in payload and payload["configs"]:
        data = payload["configs"][0]
    else:
        data = payload
    return {
        "bleu": float(data.get("bleu", 0.0)),
        "chrf": float(data.get("chrf", 0.0)),
        "exact_match_accuracy": float(data.get("exact_match_accuracy", 0.0)),
    }


def main() -> None:
    if len(sys.argv) > 1:
        current_path = Path(sys.argv[1])
    else:
        current_path = _find_current_metrics_file()

    if current_path is None or not current_path.exists():
        print("❌ Rollback check failed: current validation metrics file not found.")
        print(f"   Searched: {[str(p) for p in CURRENT_METRICS_CANDIDATES]}")
        sys.exit(1)

    if not BASELINE_PATH.exists():
        print("⚠️  Baseline metrics file not found. Passing rollback check for initial setup.")
        print(f"   Expected baseline path: {BASELINE_PATH}")
        sys.exit(0)

    current = _extract_metrics(_load_json(current_path))
    baseline = _extract_metrics(_load_json(BASELINE_PATH))

    print("========== Rollback Check ==========")
    print(f"Current metrics:  {current}")
    print(f"Baseline metrics: {baseline}")
    print("====================================")

    failures: list[str] = []
    for k in ("bleu", "chrf", "exact_match_accuracy"):
        if current[k] + TOLERANCE < baseline[k]:
            failures.append(f"{k} regressed: current={current[k]:.4f}, baseline={baseline[k]:.4f}")

    if failures:
        print("\n❌ ROLLBACK CHECK FAILED")
        for f in failures:
            print(f" - {f}")
        print("\nSuggested action: keep previous model version and investigate regressions.")
        sys.exit(1)

    print("\n✅ Rollback check passed. No metric regression detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
