"""
Sensitivity analysis for the translation stack (no weight training).

1) Hyperparameter (OAT) sweeps: temperature, top_p, max_output_tokens vs corpus BLEU/chrF.
2) Stochastic decoding: repeated runs at T>0 → mean ± std of corpus metric.
3) Input stratification: source length quartiles, glossary term in reference, emotion (from manifest).
4) Optional local ablation: leave-one-word-out → Δ sentence-BLEU vs full source.
5) Optional STT vs gold script: RAVDESS rows — compare translating STT text vs script English.

Writes JSON: data/model_runs/<split>/sensitivity_analysis.json (override with --output).

Example:
  PYTHONPATH=. python model-pipeline/scripts/run_sensitivity_analysis.py --split dev --max-rows 25
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model_pipeline_paths import (  # noqa: E402
    find_translation_inputs,
    resolve_pipeline_and_model_roots,
    split_dirs,
)

from backend.src.services.gemini_translation import translate_text  # noqa: E402

VALID_SPLITS = ("dev", "test", "holdout")
TRANSLATION_INPUTS_BASENAME = "translation_inputs"

RAVDESS_STATEMENT_TO_SOURCE: Dict[str, str] = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}


def _parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _ravdess_statement_from_file(file_name: str) -> Optional[str]:
    if not file_name.startswith("RAVDESS_") or not file_name.endswith(".wav"):
        return None
    base = file_name.replace("RAVDESS_", "").replace(".wav", "")
    parts = base.split("-")
    if len(parts) >= 5:
        return parts[4]
    return None


def _script_source_for_row(file_name: str) -> Optional[str]:
    stmt = _ravdess_statement_from_file(file_name or "")
    if stmt and stmt in RAVDESS_STATEMENT_TO_SOURCE:
        return RAVDESS_STATEMENT_TO_SOURCE[stmt]
    return None


def _load_manifest_emotion_map(pipeline_split: Path) -> Dict[str, str]:
    path = pipeline_split / "manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: Dict[str, str] = {}
    if not isinstance(data, list):
        return out
    for entry in data:
        if not isinstance(entry, dict):
            continue
        fn = (entry.get("file") or "").strip()
        if fn:
            out[fn] = str(entry.get("emotion", "") or "").strip()
    return out


def _load_glossary_terms(repo_root: Path) -> List[str]:
    path = repo_root / "data" / "legal_glossary" / "legal_terms.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    terms = data.get("terms") or []
    return [
        t.get("term", "").strip().lower()
        for t in terms
        if isinstance(t, dict) and t.get("term")
    ]


def _ref_has_glossary_term(ref: str, term_set: set[str]) -> bool:
    r = (ref or "").lower()
    return any(t in r for t in term_set if t)


def _compute_corpus_bleu(references: List[str], hypotheses: List[str]) -> float:
    try:
        import sacrebleu

        return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)
    except ImportError:
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


def _compute_corpus_chrf(references: List[str], hypotheses: List[str]) -> float:
    try:
        import sacrebleu

        return float(sacrebleu.corpus_chrf(hypotheses, [references]).score)
    except ImportError:
        return _compute_corpus_bleu(references, hypotheses)


def _sentence_bleu_scores(references: List[str], hypotheses: List[str]) -> List[float]:
    try:
        import sacrebleu

        return [
            float(sacrebleu.sentence_bleu(hyp, [ref]).score)
            for ref, hyp in zip(references, hypotheses)
        ]
    except ImportError:
        n = len(references)
        return [_compute_corpus_bleu([references[i]], [hypotheses[i]]) for i in range(n)]


def _sentence_chrf_scores(references: List[str], hypotheses: List[str]) -> List[float]:
    try:
        import sacrebleu

        return [
            float(sacrebleu.sentence_chrf(hyp, [ref]).score)
            for ref, hyp in zip(references, hypotheses)
        ]
    except Exception:  # noqa: BLE001
        return _sentence_bleu_scores(references, hypotheses)


def _load_eval_df(path: Path, max_rows: int) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for col in ["source_text", "source_language", "target_language", "reference_translation"]:
        if col not in df.columns:
            raise ValueError(f"Eval table must have column {col}")
    if max_rows > 0:
        df = df.head(max_rows).copy()
    return df.reset_index(drop=True)


def _predict_batch(
    df: pd.DataFrame,
    config_id: str,
    delay: float,
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    source_col: str = "source_text",
) -> List[str]:
    preds: List[str] = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i > 0 and delay > 0:
            time.sleep(delay)
        kwargs: Dict[str, Any] = {"config_id": config_id}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        try:
            t = translate_text(
                str(row[source_col]),
                str(row["source_language"]),
                str(row["target_language"]),
                **kwargs,
            )
            preds.append((t or "").strip())
        except Exception as e:  # noqa: BLE001
            preds.append(f"(error: {e})")
    return preds


def _emotion_groups(df: pd.DataFrame, pipeline_split: Path) -> pd.Series:
    if "emotion" in df.columns:
        emo = df["emotion"].fillna("").astype(str).str.strip()
        if emo.ne("").any():
            return emo.replace("", "unknown")
    emap = _load_manifest_emotion_map(pipeline_split)
    if "file" not in df.columns or not emap:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return df["file"].map(lambda f: emap.get(str(f), "unknown")).fillna("unknown")


def run_hyperparameter_oat(
    df: pd.DataFrame,
    config_id: str,
    delay: float,
    metric: str,
    temperatures: Sequence[float],
    top_ps: Sequence[float],
    max_tokens_list: Sequence[int],
) -> Dict[str, Any]:
    references = df["reference_translation"].astype(str).tolist()
    base_block: Dict[str, Any] = {}

    def score(hyps: List[str]) -> float:
        if metric == "chrf":
            return _compute_corpus_chrf(references, hyps)
        return _compute_corpus_bleu(references, hyps)

    temp_rows = []
    for t in temperatures:
        hyps = _predict_batch(df, config_id, delay, temperature=t)
        temp_rows.append({"temperature": t, metric: round(score(hyps), 4), "n": len(df)})

    top_rows = []
    for tp in top_ps:
        hyps = _predict_batch(df, config_id, delay, top_p=tp)
        top_rows.append({"top_p": tp, metric: round(score(hyps), 4), "n": len(df)})

    tok_rows = []
    for mt in max_tokens_list:
        hyps = _predict_batch(df, config_id, delay, max_output_tokens=mt)
        tok_rows.append({"max_output_tokens": mt, metric: round(score(hyps), 4), "n": len(df)})

    base_block["temperature"] = temp_rows
    base_block["top_p"] = top_rows
    base_block["max_output_tokens"] = tok_rows
    return base_block


def run_stochastic_block(
    df: pd.DataFrame,
    config_id: str,
    delay: float,
    metric: str,
    temps: Sequence[float],
    repeats: int,
) -> List[Dict[str, Any]]:
    references = df["reference_translation"].astype(str).tolist()

    def score(hyps: List[str]) -> float:
        if metric == "chrf":
            return _compute_corpus_chrf(references, hyps)
        return _compute_corpus_bleu(references, hyps)

    out: List[Dict[str, Any]] = []
    for t in temps:
        if t <= 0:
            continue
        run_scores: List[float] = []
        for _ in range(repeats):
            hyps = _predict_batch(df, config_id, delay, temperature=t)
            run_scores.append(score(hyps))
        mean = sum(run_scores) / len(run_scores) if run_scores else 0.0
        var = (
            sum((x - mean) ** 2 for x in run_scores) / (len(run_scores) - 1)
            if len(run_scores) > 1
            else 0.0
        )
        std = math.sqrt(var) if var > 0 else 0.0
        out.append(
            {
                "temperature": t,
                "repeats": repeats,
                f"{metric}_per_run": [round(x, 4) for x in run_scores],
                "mean": round(mean, 4),
                "std": round(std, 4),
            }
        )
    return out


def run_stratification(
    df: pd.DataFrame,
    preds: List[str],
    glossary_terms: List[str],
    emotion_series: pd.Series,
    metric: str,
) -> Dict[str, Any]:
    references = df["reference_translation"].astype(str).tolist()
    if metric == "chrf":
        seg_scores = _sentence_chrf_scores(references, preds)
    else:
        seg_scores = _sentence_bleu_scores(references, preds)

    ntok = df["source_text"].astype(str).str.split().str.len().clip(lower=1)
    try:
        bins = pd.qcut(ntok, q=4, duplicates="drop")
        labels = [str(b) for b in bins.astype(str)]
    except ValueError:
        labels = ["all"] * len(df)

    by_len: Dict[str, Any] = {}
    ulabels = sorted(set(labels), key=lambda x: (len(x), x))
    for lab in ulabels:
        idx = [i for i, l in enumerate(labels) if l == lab]
        if not idx:
            continue
        ss = [seg_scores[i] for i in idx]
        by_len[lab] = {
            "n": len(idx),
            f"mean_segment_{metric}": round(sum(ss) / len(ss), 4),
        }

    term_set = set(glossary_terms)
    by_glossary: Dict[str, Any]
    if not term_set:
        by_glossary = {"note": "No glossary terms loaded (missing or empty legal_terms.json)."}
    else:
        has = [
            _ref_has_glossary_term(references[i], term_set) for i in range(len(references))
        ]
        by_glossary = {}
        for flag, key in ((True, "with_glossary_term_in_ref"), (False, "without_glossary_term")):
            idx = [i for i, h in enumerate(has) if h is flag]
            if not idx:
                by_glossary[key] = {"n": 0, f"mean_segment_{metric}": None}
                continue
            ss = [seg_scores[i] for i in idx]
            by_glossary[key] = {
                "n": len(idx),
                f"mean_segment_{metric}": round(sum(ss) / len(ss), 4),
            }

    by_emotion: Dict[str, Any] = {}
    emos = emotion_series.fillna("unknown").astype(str)
    for emo in sorted(emos.unique()):
        idx = [i for i in range(len(df)) if emos.iloc[i] == emo]
        if not idx:
            continue
        ss = [seg_scores[i] for i in idx]
        by_emotion[emo or "unknown"] = {
            "n": len(idx),
            f"mean_segment_{metric}": round(sum(ss) / len(ss), 4),
        }

    return {
        "by_source_length_quartile": by_len,
        "by_glossary_in_reference": by_glossary,
        "by_emotion": by_emotion,
    }


def run_ablation(
    df: pd.DataFrame,
    config_id: str,
    delay: float,
    metric: str,
    n_rows: int,
    max_words: int,
) -> List[Dict[str, Any]]:
    sub = df.head(n_rows)
    results: List[Dict[str, Any]] = []
    rows_list = list(sub.iterrows())
    for _, row in rows_list:
        ref = str(row["reference_translation"])
        src = str(row["source_text"]).strip()
        words = re.split(r"\s+", src) if src else []
        if len(words) < 2:
            continue
        if delay > 0:
            time.sleep(delay)
        pred_full = translate_text(
            src, str(row["source_language"]), str(row["target_language"]), config_id=config_id
        )
        if metric == "chrf":
            try:
                import sacrebleu

                full_score = float(sacrebleu.sentence_chrf(pred_full, [ref]).score)
            except Exception:  # noqa: BLE001
                try:
                    import sacrebleu as _sb

                    full_score = float(_sb.sentence_bleu(pred_full, [ref]).score)
                except Exception:  # noqa: BLE001
                    full_score = _compute_corpus_bleu([ref], [pred_full])
        else:
            try:
                import sacrebleu

                full_score = float(sacrebleu.sentence_bleu(pred_full, [ref]).score)
            except ImportError:
                full_score = _compute_corpus_bleu([ref], [pred_full])

        deltas: List[Dict[str, Any]] = []
        limit = min(len(words), max_words)
        for i in range(limit):
            ablated = " ".join(words[:i] + words[i + 1 :])
            if delay > 0:
                time.sleep(delay)
            pred_ab = translate_text(
                ablated,
                str(row["source_language"]),
                str(row["target_language"]),
                config_id=config_id,
            )
            if metric == "chrf":
                try:
                    import sacrebleu

                    ab_score = float(sacrebleu.sentence_chrf(pred_ab, [ref]).score)
                except Exception:  # noqa: BLE001
                    try:
                        import sacrebleu as _sb

                        ab_score = float(_sb.sentence_bleu(pred_ab, [ref]).score)
                    except Exception:  # noqa: BLE001
                        ab_score = _compute_corpus_bleu([ref], [pred_ab])
            else:
                try:
                    import sacrebleu

                    ab_score = float(sacrebleu.sentence_bleu(pred_ab, [ref]).score)
                except ImportError:
                    ab_score = _compute_corpus_bleu([ref], [pred_ab])
            deltas.append(
                {
                    "removed_index": i,
                    "removed_word": words[i],
                    "delta_segment_metric": round(full_score - ab_score, 4),
                }
            )
        results.append(
            {
                "source_preview": src[:120]
                + ("…" if len(src) > 120 else ""),
                "full_segment_metric": round(full_score, 4),
                "leave_one_word_out": deltas,
            }
        )
    return results


def run_stt_vs_script(
    df: pd.DataFrame,
    config_id: str,
    delay: float,
    metric: str,
) -> Optional[Dict[str, Any]]:
    if "file" not in df.columns:
        return None
    script_texts: List[Optional[str]] = []
    for _, row in df.iterrows():
        script_texts.append(_script_source_for_row(str(row.get("file", ""))))
    idx = [i for i, s in enumerate(script_texts) if s is not None]
    if len(idx) < 2:
        return {"enabled": False, "reason": "Fewer than 2 RAVDESS script rows in slice."}

    sub = df.iloc[idx].reset_index(drop=True).copy()
    sub["_gold_script_en"] = [script_texts[i] for i in idx]
    refs = sub["reference_translation"].astype(str).tolist()

    preds_stt = _predict_batch(sub, config_id, delay, source_col="source_text")
    preds_script = _predict_batch(sub, config_id, delay, source_col="_gold_script_en")

    if metric == "chrf":
        s_stt = _compute_corpus_chrf(refs, preds_stt)
        s_script = _compute_corpus_chrf(refs, preds_script)
    else:
        s_stt = _compute_corpus_bleu(refs, preds_stt)
        s_script = _compute_corpus_bleu(refs, preds_script)

    return {
        "enabled": True,
        "n_rows": len(sub),
        "description": "RAVDESS: translate STT source vs gold English script (same references).",
        f"corpus_{metric}_stt_source": round(s_stt, 4),
        f"corpus_{metric}_gold_script": round(s_script, 4),
        "delta_gold_minus_stt": round(s_script - s_stt, 4),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensitivity analysis for translation (API LLM, no training).")
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Legacy single root for pipeline+model.")
    p.add_argument("--pipeline-data-dir", type=str, default="")
    p.add_argument("--model-output-root", type=str, default="")
    p.add_argument("--config-id", type=str, default="translation_flash_v1")
    p.add_argument("--max-rows", type=int, default=25, help="Cap eval rows (0 = all).")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls.")
    p.add_argument("--metric", type=str, default="bleu", choices=("bleu", "chrf"))
    p.add_argument(
        "--temperatures",
        type=str,
        default="0,0.2,0.5,0.8",
        help="Comma-separated temperatures for OAT sweep.",
    )
    p.add_argument(
        "--top-ps",
        type=str,
        default="0.5,0.9,1.0",
        help="Comma-separated top_p values for OAT sweep.",
    )
    p.add_argument(
        "--max-output-tokens",
        type=str,
        default="64,128,256",
        help="Comma-separated max_output_tokens for OAT sweep.",
    )
    p.add_argument(
        "--stochastic-temperatures",
        type=str,
        default="0.5,0.8",
        help="Comma-separated T>0 for repeated runs (mean ± std). Empty to skip.",
    )
    p.add_argument("--stochastic-repeats", type=int, default=3, help="Repeats per stochastic temperature.")
    p.add_argument("--skip-hyperparam", action="store_true", help="Skip temperature/top_p/max_tokens sweeps.")
    p.add_argument("--skip-stochastic", action="store_true", help="Skip repeated decoding runs.")
    p.add_argument("--skip-stratification", action="store_true", help="Skip length/glossary/emotion breakdown.")
    p.add_argument("--skip-stt-vs-script", action="store_true")
    p.add_argument("--ablation-rows", type=int, default=0, help="Leave-one-word-out rows (0 = off).")
    p.add_argument("--ablation-max-words", type=int, default=8, help="Max words to ablate per sentence.")
    p.add_argument("--output", type=str, default="", help="JSON output path.")
    p.add_argument(
        "--inputs-basename",
        type=str,
        default=TRANSLATION_INPUTS_BASENAME,
        help="Eval table basename (default translation_inputs).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline_root, model_root = resolve_pipeline_and_model_roots(
        REPO_ROOT,
        data_dir_legacy=args.data_dir,
        pipeline_data_dir=args.pipeline_data_dir,
        model_output_root=args.model_output_root,
    )
    pipeline_split, model_split = split_dirs(pipeline_root, model_root, args.split)

    inputs_basename = args.inputs_basename.strip()
    if inputs_basename.endswith(".csv") or inputs_basename.endswith(".parquet"):
        inputs_basename = Path(inputs_basename).stem

    inputs_path = find_translation_inputs(model_split, pipeline_split, inputs_basename)
    if inputs_path is None:
        print(f"[ERROR] No {inputs_basename}.csv/.parquet under {model_split} or {pipeline_split}.")
        sys.exit(1)

    df = _load_eval_df(inputs_path, args.max_rows)
    n = len(df)
    if n == 0:
        print("[ERROR] No rows in eval table.")
        sys.exit(1)

    print(f"Loaded {n} row(s) from {inputs_path} (config_id={args.config_id}, metric={args.metric})")

    glossary_terms = _load_glossary_terms(REPO_ROOT)
    emotion_series = _emotion_groups(df, pipeline_split)

    payload: Dict[str, Any] = {
        "meta": {
            "split": args.split,
            "config_id": args.config_id,
            "metric": args.metric,
            "n_rows": n,
            "inputs_path": str(inputs_path),
            "description": "Sensitivity w.r.t. decoding hyperparameters and input strata (no weight training).",
        },
        "hyperparameter_oat": None,
        "stochastic_decoding": None,
        "input_stratification": None,
        "stt_vs_gold_script": None,
        "leave_one_word_out_ablation": None,
    }

    if not args.skip_hyperparam:
        temps = _parse_csv_floats(args.temperatures)
        tops = _parse_csv_floats(args.top_ps)
        mtoks = _parse_csv_ints(args.max_output_tokens)
        print("Running hyperparameter OAT sweeps...")
        payload["hyperparameter_oat"] = run_hyperparameter_oat(
            df, args.config_id, args.delay, args.metric, temps, tops, mtoks
        )

    if not args.skip_stochastic and (args.stochastic_temperatures or "").strip():
        stoch_temps = [t for t in _parse_csv_floats(args.stochastic_temperatures) if t > 0]
        if stoch_temps and args.stochastic_repeats > 0:
            print("Running stochastic decoding repeats...")
            payload["stochastic_decoding"] = run_stochastic_block(
                df,
                args.config_id,
                args.delay,
                args.metric,
                stoch_temps,
                args.stochastic_repeats,
            )

    if not args.skip_stratification:
        print("Baseline predictions for stratification...")
        preds_base = _predict_batch(df, args.config_id, args.delay)
        payload["input_stratification"] = run_stratification(
            df, preds_base, glossary_terms, emotion_series, args.metric
        )

    if not args.skip_stt_vs_script:
        payload["stt_vs_gold_script"] = run_stt_vs_script(
            df, args.config_id, args.delay, args.metric
        )

    if args.ablation_rows > 0:
        print(f"Running leave-one-word-out on first {args.ablation_rows} row(s)...")
        payload["leave_one_word_out_ablation"] = run_ablation(
            df,
            args.config_id,
            args.delay,
            args.metric,
            args.ablation_rows,
            args.ablation_max_words,
        )

    out_path = Path(args.output) if args.output else model_split / "sensitivity_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
