"""
Config search gate for CI/CD pipeline.
Reads config_search_results.json produced by run_config_search.py and fails
if the best config does not meet the BLEU target.

Output location: data/model_runs/<split>/config_search_results.json
Fallback:        data/processed/<split>/config_search_results.json

Expected JSON structure (run_config_search.py output):
{
  "results": [...],
  "best_config_id": "translation_flash_v1",
  "metric": "bleu",
  "best_meets_proposal_bleu_target": true,
  "proposal": {
    "translation_bleu_target": 40.0,
    "translation_glossary_enforcement_target": 0.95
  }
}

Exit code 0 = passed, 1 = failed (blocks CI pipeline).
"""

import json
import sys
from pathlib import Path

SEARCH_RESULTS_CANDIDATES = [
    # model-pipeline output
    Path("data/model_runs/dev/config_search_results.json"),
    Path("data/model_runs/test/config_search_results.json"),
    # legacy / data-pipeline output
    Path("data/processed/dev/config_search_results.json"),
    Path("data/processed/test/config_search_results.json"),
]


def find_results_file() -> Path | None:
    for p in SEARCH_RESULTS_CANDIDATES:
        if p.exists():
            return p
    return None


def run_gate(results_path: Path) -> bool:
    print(f"Reading config search results from: {results_path}")
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    best_config = data.get("best_config_id", "unknown")
    metric = data.get("metric", "bleu")
    meets_target = data.get("best_meets_proposal_bleu_target", None)
    proposal = data.get("proposal", {})
    bleu_target = proposal.get("translation_bleu_target", 40.0)
    glossary_target = proposal.get("translation_glossary_enforcement_target", 0.95)

    # Find best score from results list
    results = data.get("results", [])
    best_score = next(
        (r.get("score", 0) for r in results if r.get("config_id") == best_config), 0
    )
    best_glossary = data.get("best_glossary_enforcement", None)

    print("\n========== Config Search Gate ==========")
    print(f"  Best config:        {best_config}")
    print(f"  Metric:             {metric}")
    print(f"  Best score:         {best_score:.4f}  (target: >= {bleu_target})")
    if best_glossary is not None:
        print(f"  Glossary enforce:   {best_glossary:.2%}  (target: > {glossary_target:.0%})")
    print(f"  Configs evaluated:  {len(results)}")
    print("=========================================")

    failures = []
    if meets_target is False:
        failures.append(
            f"Best config '{best_config}' scored {best_score:.4f} — below BLEU target of {bleu_target}"
        )
    if best_glossary is not None and best_glossary < glossary_target:
        failures.append(
            f"Glossary enforcement {best_glossary:.2%} below target {glossary_target:.0%}"
        )

    if failures:
        print("\n❌ CONFIG SEARCH GATE FAILED:")
        for f in failures:
            print(f"   - {f}")
        print("\n   Suggestions:")
        print("   - Try additional prompt variants in run_config_search.py")
        print("   - Add court-specific prompt templates under config/models/")
        print("   - Increase training data coverage for low-scoring languages")
        return False

    print("\n✅ Config search gate passed.")
    return True


def main() -> None:
    if len(sys.argv) > 1:
        results_path = Path(sys.argv[1])
    else:
        results_path = find_results_file()

    if results_path is None or not results_path.exists():
        print("❌ No config search results found. Failing gate because config search must run in CI.")
        print(f"   Searched: {[str(p) for p in SEARCH_RESULTS_CANDIDATES]}")
        sys.exit(1)

    passed = run_gate(results_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()