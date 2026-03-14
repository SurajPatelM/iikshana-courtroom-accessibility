"""
Task 2.2 — Config & prompt search: compare configs, compute metrics, select best.

Loops over candidate configs (different models, prompts, parameters), runs each
on the validation set, computes a translation metric (BLEU), and selects the best
config by that metric. Treat each combination as a "model" for the assignment.

Requires: eval table at data/processed/<split>/translation_inputs.csv (with
source_text, reference_translation). Optional: sacrebleu for BLEU (pip install sacrebleu).

Example (PowerShell):
    $env:PYTHONPATH = "."; python model-pipeline/scripts/run_config_search.py --split dev --inputs-basename court_translation_inputs
Example (Linux/macOS):
    PYTHONPATH=. python model-pipeline/scripts/run_config_search.py --split dev --configs translation_flash_v1,translation_flash_glossary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from backend.src.services.gemini_translation import translate_text

VALID_SPLITS = ("dev", "test", "holdout")
TRANSLATION_INPUTS_BASENAME = "translation_inputs"

# Proposal (Section 7.2.2): Translation BLEU target > 0.40 (sacrebleu 0–100 scale: > 40).
PROPOSAL_BLEU_TARGET = 40.0
# Proposal: Legal glossary enforcement rate target > 95%.
PROPOSAL_GLOSSARY_TARGET = 0.95

# Default candidate configs: different models and/or prompts and/or params.
DEFAULT_CONFIG_IDS = [
    "translation_flash_v1",           # Groq + baseline prompt, temp=0
    "translation_flash_glossary",     # Groq + glossary prompt
    "translation_flash_court",        # Groq + court-phrase equivalences (better on court_translation_inputs)
    "translation_flash_short_prompt", # Groq + minimal prompt
    "translation_flash_temp03",       # Groq + baseline, temp=0.3
    "translation_hf_v1",              # HuggingFace opus-mt (if available)
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 2.2: Run config/prompt search, compute metrics, select best config."
    )
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Override data/processed")
    p.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma-separated config_ids to try (default: translation_flash_v1,translation_flash_glossary,translation_flash_short_prompt,translation_flash_temp03). Use 'all' for all YAMLs in config/models.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap rows per config for a quick run (0 = use all).",
    )
    p.add_argument("--delay", type=float, default=2.0, help="Seconds between API calls per config.")
    p.add_argument(
        "--metric",
        type=str,
        default="bleu",
        choices=("bleu", "chrf"),
        help="Metric to maximize (bleu or chrf).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Write results JSON here (default: data/processed/<split>/config_search_results.json).",
    )
    p.add_argument(
        "--no-glossary",
        action="store_true",
        help="Skip glossary enforcement computation even if data/legal_glossary/legal_terms.json exists.",
    )
    p.add_argument(
        "--inputs-basename",
        type=str,
        default=TRANSLATION_INPUTS_BASENAME,
        help="Basename of eval table (default: translation_inputs). Use court_translation_inputs for court-phrase-only eval.",
    )
    p.add_argument(
        "--bleu-target",
        type=float,
        default=PROPOSAL_BLEU_TARGET,
        help="BLEU target (0-100 scale). Best config is reported as meeting target if score >= this. Default 40 (proposal). Use e.g. 30 for 'Good', 20 for 'Fair'.",
    )
    return p.parse_args()


def _resolve_processed_dir(repo_root: Path, override: str) -> Path:
    if override:
        path = Path(override)
        return path if path.is_absolute() else repo_root / path
    return repo_root / "data" / "processed"


def _find_translation_inputs(split_dir: Path, basename: str = TRANSLATION_INPUTS_BASENAME) -> Path:
    for ext in (".parquet", ".csv"):
        p = split_dir / f"{basename}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No {basename}.parquet or .csv in {split_dir}")


def _load_eval_data(
    split_dir: Path, max_rows: int, inputs_basename: str = TRANSLATION_INPUTS_BASENAME
) -> tuple[pd.DataFrame, pd.Series]:
    path = _find_translation_inputs(split_dir, basename=inputs_basename)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for col in ["source_text", "source_language", "target_language", "reference_translation"]:
        if col not in df.columns:
            raise ValueError(f"Eval table must have column {col}")
    if max_rows > 0:
        df = df.head(max_rows)
    features = df[["source_text", "source_language", "target_language"]]
    labels = df["reference_translation"]
    return features, labels


def _compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    try:
        import sacrebleu
        # One reference per segment: ref_streams = [references]
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score
    except ImportError:
        # Fallback: simple token F1 overlap (no extra dep)
        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tok = set(str(ref).lower().split())
            hyp_tok = set(str(hyp).lower().split())
            if not ref_tok:
                scores.append(1.0 if not hyp_tok else 0.0)
                continue
            overlap = len(ref_tok & hyp_tok)
            prec = overlap / len(hyp_tok) if hyp_tok else 0.0
            rec = overlap / len(ref_tok)
            scores.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        return sum(scores) / len(scores) * 100.0 if scores else 0.0


def _compute_chrf(references: List[str], hypotheses: List[str]) -> float:
    try:
        import sacrebleu
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        return chrf.score
    except ImportError:
        return _compute_bleu(references, hypotheses)  # fallback


def _load_glossary_terms(repo_root: Path) -> List[str]:
    """Load legal term strings from data/legal_glossary/legal_terms.json. Returns [] if missing."""
    path = repo_root / "data" / "legal_glossary" / "legal_terms.json"
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        terms = data.get("terms") or []
        return [t.get("term", "").strip().lower() for t in terms if isinstance(t, dict) and t.get("term")]
    except (json.JSONDecodeError, OSError):
        return []


def _compute_glossary_enforcement(
    references: List[str], hypotheses: List[str], terms: List[str]
) -> Optional[float]:
    """
    Fraction of legal terms that appear in ref and are preserved in pred (case-insensitive).
    Proposal target: > 95%. Returns None if no terms in any reference.
    """
    if not terms:
        return None
    term_set = set(terms)
    preserved, total = 0, 0
    for ref, pred in zip(references, hypotheses):
        ref_lower = (ref or "").lower()
        pred_lower = (pred or "").lower()
        in_ref = [t for t in term_set if t in ref_lower]
        if not in_ref:
            continue
        total += len(in_ref)
        preserved += sum(1 for t in in_ref if t in pred_lower)
    if total == 0:
        return None
    return round(preserved / total, 4)


def _run_config(
    config_id: str,
    features: pd.DataFrame,
    delay: float,
) -> List[str]:
    predictions: List[str] = []
    for i, (_, row) in enumerate(features.iterrows()):
        if i > 0 and delay > 0:
            time.sleep(delay)
        try:
            t = translate_text(
                source_text=str(row["source_text"]),
                source_language=str(row["source_language"]),
                target_language=str(row["target_language"]),
                config_id=config_id,
            )
            predictions.append((t or "").strip())
        except Exception as e:
            predictions.append(f"(error: {e})")
    return predictions


def main() -> None:
    args = _parse_args()
    processed_root = _resolve_processed_dir(REPO_ROOT, args.data_dir)
    split_dir = processed_root / args.split

    if not split_dir.is_dir():
        print(f"[ERROR] Split directory not found: {split_dir}")
        sys.exit(1)

    inputs_basename = (args.inputs_basename or TRANSLATION_INPUTS_BASENAME).strip()
    if not inputs_basename.endswith(".csv") and not inputs_basename.endswith(".parquet"):
        pass  # use as basename
    else:
        inputs_basename = Path(inputs_basename).stem

    if args.configs.strip().lower() == "all":
        config_dir = REPO_ROOT / "config" / "models"
        config_ids = [p.stem for p in config_dir.glob("*.yaml")]
        config_ids.sort()
    elif args.configs.strip():
        config_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        config_ids = DEFAULT_CONFIG_IDS

    print(f"Loading eval data from {split_dir} (basename={inputs_basename})...")
    features, labels = _load_eval_data(split_dir, args.max_rows, inputs_basename)
    n = len(features)
    print(f"Evaluating {len(config_ids)} config(s) on {n} row(s), metric={args.metric}")

    references = labels.astype(str).tolist()
    glossary_terms = [] if args.no_glossary else _load_glossary_terms(REPO_ROOT)
    if glossary_terms:
        print(f"  Glossary: {len(glossary_terms)} terms (proposal target: > {PROPOSAL_GLOSSARY_TARGET * 100:.0f}%)")
    results: List[Dict[str, Any]] = []

    for config_id in config_ids:
        print(f"  Running config: {config_id}...")
        preds = _run_config(config_id, features, args.delay)
        if len(preds) != len(references):
            preds = preds + [""] * (len(references) - len(preds))
        preds = preds[: len(references)]

        if args.metric == "chrf":
            score = _compute_chrf(references, preds)
        else:
            score = _compute_bleu(references, preds)

        row: Dict[str, Any] = {
            "config_id": config_id,
            "metric": args.metric,
            "score": round(score, 4),
            "n_samples": n,
        }
        if glossary_terms:
            gloss = _compute_glossary_enforcement(references, preds, glossary_terms)
            if gloss is not None:
                row["glossary_enforcement"] = gloss
                print(f"    glossary_enforcement = {gloss:.2%}")
        results.append(row)
        print(f"    {args.metric} = {score:.4f}")

    best = max(results, key=lambda x: x["score"])
    print(f"\nBest config: {best['config_id']} ({args.metric}={best['score']:.4f})")

    bleu_target = args.bleu_target
    payload: Dict[str, Any] = {
        "results": results,
        "best_config_id": best["config_id"],
        "metric": args.metric,
        "proposal": {
            "translation_bleu_target": bleu_target,
            "translation_glossary_enforcement_target": PROPOSAL_GLOSSARY_TARGET,
        },
    }
    if args.metric == "bleu":
        payload["best_meets_proposal_bleu_target"] = best["score"] >= bleu_target
        print(f"  BLEU target >= {bleu_target}: {'yes' if payload['best_meets_proposal_bleu_target'] else 'no'}")
    if glossary_terms and "glossary_enforcement" in best:
        payload["best_glossary_enforcement"] = best["glossary_enforcement"]
        payload["best_meets_proposal_glossary_target"] = best["glossary_enforcement"] >= PROPOSAL_GLOSSARY_TARGET
        print(f"  Proposal glossary target > {PROPOSAL_GLOSSARY_TARGET:.0%}: {'yes' if payload['best_meets_proposal_glossary_target'] else 'no'}")

    out_path = Path(args.output) if args.output else split_dir / (
        "config_search_results.json" if inputs_basename == TRANSLATION_INPUTS_BASENAME else f"config_search_results_{inputs_basename}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
