"""
Bias gate for CI/CD pipeline.
Reads model_bias_report_*.json produced by run_model_bias_detection.py and fails
if disparity counts exceed the defined threshold.

Output location (model-pipeline):
  data/model_runs/<split>/model_bias_report_<config_id>__<group_suffix>.json

Fallback (data-pipeline detect_bias.py):
  data/processed/bias_report.json

Report schema (model-pipeline v1):
  {
    "schema": "iikshana.model_bias_detection.v1",
    "disparities": [
      {
        "metric": "exact_match",
        "overall": 0.45,
        "group_value": {"dataset": "RAVDESS"},
        "group_metric": 0.20,
        "absolute_gap": 0.25,
        "note": "..."
      }
    ],
    "mitigation_recommendations": [...],
    "overall_exact_match_accuracy": 0.45
  }

Rules:
  - 0 disparities            → PASS
  - <= MAX_DISPARITIES        → WARN (pass with warning)
  - >  MAX_DISPARITIES        → FAIL (blocks CI)

Exit code 0 = passed/warned, 1 = failed.
"""

import json
import sys
from pathlib import Path

# ---------- Thresholds ----------
MAX_DISPARITIES = 2
# --------------------------------

def find_bias_report() -> Path | None:
    """Search model_runs first, then processed fallback."""
    # model-pipeline output: glob for any config_id and group_suffix
    for split in ("dev", "test", "holdout"):
        model_split = Path(f"data/model_runs/{split}")
        if model_split.is_dir():
            reports = sorted(model_split.glob("model_bias_report_*.json"))
            if reports:
                return reports[0]  # use first found

    # data-pipeline fallback
    fallback = Path("data/processed/bias_report.json")
    if fallback.exists():
        return fallback

    return None


def run_gate(report_path: Path) -> bool:
    print(f"Reading bias report from: {report_path}")
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    schema = data.get("schema", "legacy")
    disparities = data.get("disparities", [])
    n = len(disparities)

    print("\n========== Bias Gate ==========")
    print(f"  Schema:             {schema}")
    print(f"  Config:             {data.get('config_id', 'N/A')}")
    print(f"  Split:              {data.get('split', 'N/A')}")
    print(f"  Samples:            {data.get('n_samples', 'N/A')}")
    print(f"  Overall exact match:{data.get('overall_exact_match_accuracy', 'N/A')}")
    print(f"  Disparities found:  {n}  (max allowed: {MAX_DISPARITIES})")

    if n > 0:
        print("\n  Flagged disparities:")
        for d in disparities:
            metric = d.get("metric", "unknown")
            group = d.get("group_value", {})
            overall = d.get("overall", "?")
            group_metric = d.get("group_metric", "?")
            gap = d.get("absolute_gap", "?")
            note = d.get("note", "")
            print(f"   - [{metric}] group={group}")
            print(f"     overall={overall}, group={group_metric}, gap={gap}")
            print(f"     {note}")

    print("================================")

    if n > MAX_DISPARITIES:
        print(f"\n❌ BIAS GATE FAILED: {n} disparities exceed threshold of {MAX_DISPARITIES}.")
        print("   Review bias report and apply mitigation strategies:")
        mitigations = data.get("mitigation_recommendations", [])
        if mitigations:
            print("\n   Recommendations from report:")
            for r in mitigations:
                print(f"   • {r}")
        return False

    if n > 0:
        print(f"\n⚠️  WARNING: {n} disparity/disparities detected (within allowed threshold).")
        print("   Monitor these and address before next release.")
    else:
        print("\n✅ Bias gate passed. No disparities detected.")

    return True


def main() -> None:
    if len(sys.argv) > 1:
        report_path = Path(sys.argv[1])
    else:
        report_path = find_bias_report()

    if report_path is None or not report_path.exists():
        print("⚠️  No bias report found. Skipping bias gate (report not yet generated).")
        sys.exit(0)

    passed = run_gate(report_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()