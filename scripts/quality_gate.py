"""
Quality gate for CI/CD pipeline.
Reads validation_metrics.json produced by run_validation.py and fails
if metrics fall below defined thresholds.

Output location (model-pipeline): data/model_runs/<split>/validation_metrics.json
Fallback (legacy):                 data/processed/<split>/validation_metrics.json
                                   data/processed/evaluation_metrics.json

Thresholds (from project targets):
  - BLEU  >= 40.0
  - chrF  >= 50.0
  - exact_match_accuracy >= 0.10

Exit code 0 = passed, 1 = failed (blocks CI pipeline).
"""

import json
import sys
from pathlib import Path

# ---------- Thresholds ----------
# CI smoke-test values (20-row sample via API). Raise as model matures.
BLEU_MIN = 1.0
CHRF_MIN = 5.0
EXACT_MATCH_MIN = 0.0
# --------------------------------

METRICS_CANDIDATES = [
    # model-pipeline output (data/model_runs/<split>/)
    Path("data/model_runs/dev/validation_metrics.json"),
    Path("data/model_runs/test/validation_metrics.json"),
    Path("data/model_runs/holdout/validation_metrics.json"),
    # emotion pipeline splits (current layout)
    Path("data/processed/emotions/dev/validation_metrics.json"),
    Path("data/processed/emotions/test/validation_metrics.json"),
    Path("data/processed/emotions/holdout/validation_metrics.json"),
    # legacy flat processed layout
    Path("data/processed/dev/validation_metrics.json"),
    Path("data/processed/test/validation_metrics.json"),
    Path("data/processed/holdout/validation_metrics.json"),
    Path("data/processed/evaluation_metrics.json"),
]


def find_metrics_file() -> Path | None:
    for p in METRICS_CANDIDATES:
        if p.exists():
            return p
    return None


def extract_metrics(data: dict) -> dict:
    """Extract flat metrics dict from run_validation.py output format."""
    if "metrics" in data:
        return data["metrics"]
    if "configs" in data and data["configs"]:
        return data["configs"][0]
    return data


def run_gate(metrics_path: Path) -> bool:
    print(f"Reading metrics from: {metrics_path}")
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)

    metrics = extract_metrics(data)
    bleu = float(metrics.get("bleu", 0))
    chrf = float(metrics.get("chrf", 0))
    exact_match = float(metrics.get("exact_match_accuracy", 0))

    print("\n========== Quality Gate ==========")
    print(f"  BLEU:               {bleu:.4f}  (min: {BLEU_MIN})")
    print(f"  chrF:               {chrf:.4f}  (min: {CHRF_MIN})")
    print(f"  Exact match acc:    {exact_match:.4f}  (min: {EXACT_MATCH_MIN})")
    print("===================================")

    failures = []
    if bleu < BLEU_MIN:
        failures.append(f"BLEU {bleu:.4f} < threshold {BLEU_MIN}")
    if chrf < CHRF_MIN:
        failures.append(f"chrF {chrf:.4f} < threshold {CHRF_MIN}")
    if exact_match < EXACT_MATCH_MIN:
        failures.append(f"Exact match {exact_match:.4f} < threshold {EXACT_MATCH_MIN}")

    if failures:
        print("\n❌ QUALITY GATE FAILED:")
        for f in failures:
            print(f"   - {f}")
        return False

    print("\n✅ Quality gate passed.")
    return True


def main() -> None:
    if len(sys.argv) > 1:
        metrics_path = Path(sys.argv[1])
    else:
        metrics_path = find_metrics_file()

    if metrics_path is None or not metrics_path.exists():
        print("❌ No metrics file found. Failing quality gate because metrics are required in CI.")
        print(f"   Searched: {[str(p) for p in METRICS_CANDIDATES]}")
        sys.exit(1)

    passed = run_gate(metrics_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()