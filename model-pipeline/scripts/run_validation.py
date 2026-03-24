"""
Task 2.3 — Model validation.

- Reads the validation set (e.g. data/processed/<split>/translation_inputs.csv).
- For each example: builds the prompt, calls the model API, parses the response.
- Computes metrics (for translation: BLEU, chrF, exact-match accuracy; task-dependent).
- Saves: metrics JSON, metrics CSV, bar plot PNG, and (for classification tasks) confusion matrix PNG.

Translation task: metrics are BLEU, chrF, exact_match_accuracy, optional glossary_enforcement.
Classification tasks (e.g. emotion): accuracy, precision, recall, F1, AUC, confusion matrix.

PowerShell:
  $env:PYTHONPATH = "."; python model-pipeline/scripts/run_validation.py --split dev --config-id translation_flash_v1
"""

from __future__ import annotations

import os
import mlflow
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd

from model_pipeline_paths import (
    find_translation_inputs,
    resolve_pipeline_and_model_roots,
    split_dirs,
)

from backend.src.services.gemini_translation import translate_text

VALID_SPLITS = ("dev", "test", "holdout")
TRANSLATION_INPUTS_BASENAME = "translation_inputs"
VALIDATION_METRICS_JSON = "validation_metrics.json"
VALIDATION_METRICS_CSV = "validation_metrics.csv"
VALIDATION_BAR_PLOT = "validation_metrics_bar.png"
VALIDATION_SEGMENT_HIST = "validation_segment_bleu_hist.png"
VALIDATION_CM_PLOT = "validation_confusion_matrix.png"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 2.3: Run model validation on a split; compute metrics; save JSON, CSV, and plots."
    )
    p.add_argument("--split", type=str, default="dev", choices=VALID_SPLITS)
    p.add_argument("--data-dir", type=str, default="", help="Legacy: single root for pipeline+model outputs.")
    p.add_argument("--pipeline-data-dir", type=str, default="", help="Pipeline root (default: data/processed).")
    p.add_argument("--model-output-root", type=str, default="", help="Model artifacts root (default: data/model_runs).")
    p.add_argument(
        "--task",
        type=str,
        default="translation",
        choices=("translation",),
        help="Task type (translation only for now; emotion adds precision/recall/F1/confusion matrix).",
    )
    p.add_argument(
        "--config-id",
        type=str,
        default="translation_flash_v1",
        help="Single config to validate (e.g. translation_flash_v1).",
    )
    p.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma-separated configs to compare (overrides --config-id). Produces one bar per config.",
    )
    p.add_argument("--max-rows", type=int, default=0, help="Cap validation rows (0 = all).")
    p.add_argument("--delay", type=float, default=2.0, help="Seconds between API calls.")
    p.add_argument(
        "--inputs-basename",
        type=str,
        default=TRANSLATION_INPUTS_BASENAME,
        help="Eval table basename (default: translation_inputs).",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing PNGs (bar plot, histogram, confusion matrix).",
    )
    return p.parse_args()


def _load_translation_val(inputs_path: Path, max_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(inputs_path) if inputs_path.suffix.lower() == ".parquet" else pd.read_csv(inputs_path)
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
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score
    except ImportError:
        scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_tok = set(str(ref).lower().split())
            hyp_tok = set(str(hyp).lower().split())
            if not ref_tok:
                scores.append(100.0 if not hyp_tok else 0.0)
                continue
            overlap = len(ref_tok & hyp_tok)
            prec = overlap / len(hyp_tok) if hyp_tok else 0.0
            rec = overlap / len(ref_tok)
            scores.append(2 * prec * rec / (prec + rec) * 100 if (prec + rec) > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0


def _compute_chrf(references: List[str], hypotheses: List[str]) -> float:
    try:
        import sacrebleu
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])
        return chrf.score
    except ImportError:
        return _compute_bleu(references, hypotheses)


def _segment_bleus(references: List[str], hypotheses: List[str]) -> List[float]:
    try:
        import sacrebleu
        return [sacrebleu.sentence_bleu(hyp, [ref]).score for ref, hyp in zip(references, hypotheses)]
    except ImportError:
        return []
    except Exception:
        return []


def _exact_match_accuracy(references: List[str], hypotheses: List[str]) -> float:
    if not references:
        return 0.0
    matches = sum(
        1 for r, h in zip(references, hypotheses)
        if (r or "").strip().lower() == (h or "").strip().lower()
    )
    return round(matches / len(references), 4)


def _load_glossary_terms(repo_root: Path) -> List[str]:
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


def _glossary_enforcement(references: List[str], hypotheses: List[str], terms: List[str]) -> Optional[float]:
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

def _log_to_mlflow(
    *,
    config_id: str,
    split: str,
    inputs_basename: str,
    data_dir: str,
    n_samples: int,
    metrics: dict,
    artifacts: dict,
) -> None:
    """
    Log params, metrics, and artifacts for a single config into MLflow.
    Called once per config when --configs is used.
    """
    # Optional: respect MLFLOW_TRACKING_URI env var, else default to ./mlruns
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iikshana-translation")

    with mlflow.start_run(run_name=f"{config_id}_{split}_{inputs_basename}"):
        # Params
        mlflow.log_params(
            {
                "config_id": config_id,
                "split": split,
                "inputs_basename": inputs_basename,
                "data_dir": data_dir or "data/processed",
                "n_samples": n_samples,
            }
        )
        # Metrics
        mlflow.log_metrics(metrics)

        # Artifacts (if files exist)
        for name, path in artifacts.items():
            if path is None:
                continue
            p = Path(path)
            if p.is_file():
                mlflow.log_artifact(str(p), artifact_path=name)

def _run_config(config_id: str, features: pd.DataFrame, delay: float) -> List[str]:
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


def _save_bar_plot(metrics_per_config: List[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config_ids = [m["config_id"] for m in metrics_per_config]
    bleus = [m.get("bleu", 0) for m in metrics_per_config]
    chrfs = [m.get("chrf", 0) for m in metrics_per_config]

    x = range(len(config_ids))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(config_ids) * 1.2), 5))
    ax.bar([i - w / 2 for i in x], bleus, w, label="BLEU (0–100)", color="steelblue")
    ax.bar([i + w / 2 for i in x], chrfs, w, label="chrF", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(config_ids, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Validation metrics by config")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_segment_bleu_histogram(segment_bleus: List[float], out_path: Path) -> None:
    if not segment_bleus:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(segment_bleus, bins=min(30, max(10, len(segment_bleus) // 5)), color="steelblue", edgecolor="white")
    ax.set_xlabel("Segment BLEU (0–100)")
    ax.set_ylabel("Count")
    ax.set_title("Per-segment BLEU distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_confusion_matrix_plot(
    y_true: List[str], y_pred: List[str], labels: List[str], out_path: Path
) -> None:
    """Save confusion matrix PNG (for classification tasks)."""
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if cm.size == 0:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        plt.colorbar(im, ax=ax, label="Count")
        ax.set_title("Confusion matrix")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def _save_translation_cm_plot(references: List[str], predictions: List[str], out_path: Path) -> None:
    """2x2 confusion matrix for translation: Exact match (Y/N) vs Reference length (Short/Long)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        exact = [(r or "").strip().lower() == (p or "").strip().lower() for r, p in zip(references, predictions)]
        short_ref = [len((r or "").strip()) < 50 for r in references]
        # Rows: Reference length (Short, Long), Cols: Exact match (No, Yes)
        labels_row = ["Short ref", "Long ref"]
        labels_col = ["Not exact", "Exact"]
        cm = np.zeros((2, 2))
        for ex, short in zip(exact, short_ref):
            r = 0 if short else 1
            c = 1 if ex else 0
            cm[r, c] += 1
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_col)
        ax.set_yticklabels(labels_row)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Reference length")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
        plt.colorbar(im, ax=ax, label="Count")
        ax.set_title("Translation: exact match vs reference length")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def main() -> None:
    args = _parse_args()
    pipeline_root, model_root = resolve_pipeline_and_model_roots(
        REPO_ROOT,
        data_dir_legacy=args.data_dir,
        pipeline_data_dir=args.pipeline_data_dir,
        model_output_root=args.model_output_root,
    )
    pipeline_split, model_split = split_dirs(pipeline_root, model_root, args.split)

    if args.configs.strip():
        config_ids = [c.strip() for c in args.configs.split(",") if c.strip()]
    else:
        config_ids = [args.config_id]

    if args.task != "translation":
        print("[ERROR] Only task=translation is implemented. Use --task translation.")
        sys.exit(1)

    inputs_path = find_translation_inputs(model_split, pipeline_split, args.inputs_basename)
    if inputs_path is None:
        print(
            f"[ERROR] No {args.inputs_basename}.csv/.parquet under {model_split} or {pipeline_split}. "
            "Run build_translation_inputs_from_audio.py first."
        )
        sys.exit(1)

    print(f"Loading validation set from {inputs_path}...")
    features, labels = _load_translation_val(inputs_path, args.max_rows)
    references = labels.astype(str).tolist()
    n = len(references)
    print(f"Validating {len(config_ids)} config(s) on {n} example(s).")

    out_dir = model_split
    out_dir.mkdir(parents=True, exist_ok=True)

    glossary_terms = _load_glossary_terms(REPO_ROOT)
    metrics_per_config: List[Dict[str, Any]] = []
    last_predictions: Optional[List[str]] = None

    for config_id in config_ids:
        print(f"  Running config: {config_id}...")
        preds = _run_config(config_id, features, args.delay)
        if len(preds) < len(references):
            preds = preds + [""] * (len(references) - len(preds))
        preds = preds[: len(references)]

        bleu = _compute_bleu(references, preds)
        chrf = _compute_chrf(references, preds)
        exact_match = _exact_match_accuracy(references, preds)

        row: Dict[str, Any] = {
            "config_id": config_id,
            "task": args.task,
            "split": args.split,
            "n_samples": n,
            "bleu": round(bleu, 4),
            "chrf": round(chrf, 4),
            "exact_match_accuracy": exact_match,
        }
        if glossary_terms:
            gloss = _glossary_enforcement(references, preds, glossary_terms)
            if gloss is not None:
                row["glossary_enforcement"] = gloss
        metrics_per_config.append(row)
        last_predictions = preds
        print(f"    BLEU={bleu:.4f}  chrF={chrf:.4f}  exact_match={exact_match:.2%}")

        # When comparing multiple configs (--configs), log each config as a separate MLflow run.
        if args.configs.strip():
            artifacts = {
                "validation_metrics_json": out_dir / VALIDATION_METRICS_JSON,
                "validation_metrics_csv": out_dir / VALIDATION_METRICS_CSV,
                "bar_plot": out_dir / VALIDATION_BAR_PLOT,
                "segment_hist": out_dir / VALIDATION_SEGMENT_HIST,
                "confusion_plot": out_dir / VALIDATION_CM_PLOT,
            }
            metrics_payload: Dict[str, float] = {
                "bleu": float(row["bleu"]),
                "chrf": float(row["chrf"]),
                "exact_match_accuracy": float(row["exact_match_accuracy"]),
            }
            if "glossary_enforcement" in row:
                metrics_payload["glossary_enforcement"] = float(row["glossary_enforcement"])
            _log_to_mlflow(
                config_id=config_id,
                split=args.split,
                inputs_basename=args.inputs_basename,
                data_dir=args.data_dir,
                n_samples=n,
                metrics=metrics_payload,
                artifacts=artifacts,
            )

    payload: Dict[str, Any] = {
        "task": args.task,
        "split": args.split,
        "configs": metrics_per_config,
        "n_samples": n,
    }
    if len(metrics_per_config) == 1:
        payload["metrics"] = metrics_per_config[0]

    json_path = out_dir / VALIDATION_METRICS_JSON
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Metrics JSON: {json_path}")

    csv_path = out_dir / VALIDATION_METRICS_CSV
    pd.DataFrame(metrics_per_config).to_csv(csv_path, index=False)
    print(f"Metrics CSV:  {csv_path}")

    if not args.no_plots:
        bar_path = out_dir / VALIDATION_BAR_PLOT
        _save_bar_plot(metrics_per_config, bar_path)
        print(f"Bar plot:    {bar_path}")

        if len(config_ids) == 1 and last_predictions is not None:
            segment_bleus = _segment_bleus(references, last_predictions)
            if segment_bleus:
                hist_path = out_dir / VALIDATION_SEGMENT_HIST
                _save_segment_bleu_histogram(segment_bleus, hist_path)
                print(f"Histogram:   {hist_path}")
            cm_path = out_dir / VALIDATION_CM_PLOT
            _save_translation_cm_plot(references, last_predictions, cm_path)
            print(f"Confusion:   {cm_path}")

    print("Done.")


if __name__ == "__main__":
    main()
