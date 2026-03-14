"""
Model pipeline entry point: run translation on processed data (text or audio).

Uses the same splits as the data pipeline (dev, test, holdout). Reads
manifest.json (one entry per WAV: file, dataset, speaker_id, emotion). For
each entry: if the file is audio (WAV, etc.), transcribes it with Groq Whisper
(STT) then translates the transcribed text; otherwise generates a courtroom
phrase from emotion and translates it. Optionally supports translation_inputs
if you want to supply explicit text instead.

Run from repo root with PYTHONPATH set, or via Airflow run_translation_eval task.

Example:
    PYTHONPATH=/workspace python model-pipeline/scripts/run_translation_eval.py \\
        --split dev --config-id translation_flash_v1 --max-rows 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Repo root must be on path so backend and config are importable
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from backend.src.services.gemini_translation import (
    _load_model_config,
    _get_text_client,
    translate_text,
)
from backend.src.services.groq_stt_service import (
    transcribe_audio,
    AUDIO_EXTENSIONS,
    DEFAULT_STT_MODEL,
)

# Same split names as the data pipeline (stratified_split: dev, test, holdout)
VALID_SPLITS = ("dev", "test", "holdout")
MANIFEST_FILENAME = "manifest.json"
TRANSLATION_INPUTS_BASENAME = "translation_inputs"
DEFAULT_TARGET_LANGUAGE = "es"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model pipeline: run Gemini on data-pipeline split (manifest or translation_inputs)."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=VALID_SPLITS,
        help="Split name produced by the data pipeline (dev, test, or holdout).",
    )
    parser.add_argument(
        "--config-id",
        type=str,
        default="translation_flash_v1",
        help="Translation model configuration id (config/models/<id>.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Override processed data directory (default: data/processed).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10,
        help="Max manifest entries to process when using manifest.json (default 10).",
    )
    parser.add_argument(
        "--target-language",
        type=str,
        default=DEFAULT_TARGET_LANGUAGE,
        help="Target language for translation (default es).",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=DEFAULT_STT_MODEL,
        help="Groq Whisper model for speech-to-text (default whisper-large-v3-turbo).",
    )
    parser.add_argument(
        "--no-stt",
        action="store_true",
        help="Disable STT: always generate phrase from emotion instead of transcribing WAV.",
    )
    return parser.parse_args()


def _resolve_processed_dir(repo_root: Path, override: str) -> Path:
    """Resolve data/processed directory."""
    if override:
        p = Path(override)
        return p if p.is_absolute() else repo_root / p
    return repo_root / "data" / "processed"


def _load_manifest(split_dir: Path) -> List[Dict[str, Any]]:
    """Load manifest.json from the split directory."""
    path = split_dir / MANIFEST_FILENAME
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _find_translation_inputs(split_dir: Path) -> Path | None:
    """Return path to translation_inputs.parquet or .csv if it exists."""
    for ext in (".parquet", ".csv"):
        path = split_dir / f"{TRANSLATION_INPUTS_BASENAME}{ext}"
        if path.exists():
            return path
    return None


def _run_on_manifest(
    manifest: List[Dict[str, Any]],
    split_dir: Path,
    config_id: str,
    max_rows: int,
    target_language: str,
    use_stt: bool,
    stt_model: str,
) -> pd.DataFrame:
    """
    For each manifest entry: if use_stt and the file is audio (WAV etc.), transcribe
    with Groq Whisper then translate the text; otherwise generate a courtroom phrase
    from emotion and translate it.
    """
    config = _load_model_config(config_id)
    client = _get_text_client(config)

    prompt_template = (
        "Generate one short courtroom-appropriate sentence in English that a speaker with emotion \"{emotion}\" might say. "
        "Then translate that sentence to {target_language}. "
        "Output exactly two lines: first line 'EN: <sentence>', second line '{target_lang_label}: <translation>'. "
        "No other text."
    )
    target_lang_label = target_language.upper() if len(target_language) <= 3 else target_language

    rows: List[Dict[str, Any]] = []
    subset = manifest[:max_rows]
    for entry in subset:
        file_name = entry.get("file", "")
        dataset = entry.get("dataset", "")
        speaker_id = entry.get("speaker_id", "")
        emotion = entry.get("emotion", "neutral")
        wav_path = split_dir / file_name if file_name else None

        use_audio_stt = (
            use_stt
            and wav_path is not None
            and wav_path.exists()
            and wav_path.suffix.lower() in AUDIO_EXTENSIONS
        )

        if use_audio_stt:
            try:
                source_phrase = transcribe_audio(wav_path, model=stt_model).strip() or "(no speech)"
                translated_phrase = translate_text(
                    source_text=source_phrase,
                    source_language="en",
                    target_language=target_language,
                    config_id=config_id,
                )
                rows.append({
                    "file": file_name,
                    "dataset": dataset,
                    "speaker_id": speaker_id,
                    "emotion": emotion,
                    "source_phrase": source_phrase,
                    "translated_text_model": translated_phrase.strip(),
                })
            except Exception as e:
                rows.append({
                    "file": file_name,
                    "dataset": dataset,
                    "speaker_id": speaker_id,
                    "emotion": emotion,
                    "source_phrase": "",
                    "translated_text_model": f"(error: {e})",
                })
            continue

        prompt = prompt_template.format(
            emotion=emotion,
            target_language=target_language,
            target_lang_label=target_lang_label,
        )
        try:
            response = client.generate_text(
                prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_output_tokens,
            )
            source_phrase = ""
            translated_phrase = ""
            for line in response.strip().splitlines():
                line = line.strip()
                if re.match(r"^EN:\s*", line, re.I):
                    source_phrase = re.sub(r"^EN:\s*", "", line, flags=re.I).strip()
                elif re.match(rf"^{re.escape(target_lang_label)}:\s*", line, re.I):
                    translated_phrase = re.sub(
                        rf"^{re.escape(target_lang_label)}:\s*", "", line, flags=re.I
                    ).strip()
            if not source_phrase and not translated_phrase:
                source_phrase = response[:200]
                translated_phrase = "(parse failed)"
            rows.append({
                "file": file_name,
                "dataset": dataset,
                "speaker_id": speaker_id,
                "emotion": emotion,
                "source_phrase": source_phrase or response,
                "translated_text_model": translated_phrase or "",
            })
        except Exception as e:
            rows.append({
                "file": file_name,
                "dataset": dataset,
                "speaker_id": speaker_id,
                "emotion": emotion,
                "source_phrase": "",
                "translated_text_model": f"(error: {e})",
            })
    return pd.DataFrame(rows)


def _run_translation_inputs(path: Path, config_id: str) -> pd.DataFrame:
    """Load translation_inputs and call translate_text for each row."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for col in ["source_text", "source_language", "target_language"]:
        if col not in df.columns:
            raise ValueError(f"translation_inputs must have column: {col}")
    translations: List[str] = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i > 0:
            time.sleep(2.0)  # Rate limit: avoid 429 from Groq/API when many rows
        t = translate_text(
            source_text=str(row["source_text"]),
            source_language=str(row["source_language"]),
            target_language=str(row["target_language"]),
            config_id=config_id,
        )
        translations.append(t)
    df = df.copy()
    df["translated_text_model"] = translations
    return df


def main() -> None:
    args = _parse_args()
    processed_dir = _resolve_processed_dir(REPO_ROOT, args.data_dir)
    split_dir = processed_dir / args.split

    if not split_dir.is_dir():
        print(
            f"[SKIP] Split directory not found: {split_dir}. "
            "Ensure the data pipeline has produced data/processed/dev, test, and/or holdout."
        )
        sys.exit(0)

    # Prefer translation_inputs if present (explicit text table)
    inputs_path = _find_translation_inputs(split_dir)
    if inputs_path is not None:
        print(f"Using {inputs_path.name} for translation inputs...")
        df = _run_translation_inputs(inputs_path, args.config_id)
        n_rows = len(df)
        print(f"Calling Gemini API for {n_rows} row(s) (config={args.config_id})...")
    else:
        # Use manifest.json from pulled data (always present after DVC pull)
        manifest = _load_manifest(split_dir)
        if not manifest:
            print(
                f"[SKIP] No {MANIFEST_FILENAME} and no {TRANSLATION_INPUTS_BASENAME} in {split_dir}. "
                "Pull data from GCS first (dvc_pull task) or add translation_inputs.csv."
            )
            sys.exit(0)
        n_rows = min(len(manifest), args.max_rows)
        use_stt = not args.no_stt
        print(
            f"Using {MANIFEST_FILENAME} from pulled data: processing {n_rows} of {len(manifest)} entries "
            f"(config={args.config_id}, STT={'on' if use_stt else 'off'})..."
        )
        df = _run_on_manifest(
            manifest,
            split_dir=split_dir,
            config_id=args.config_id,
            max_rows=args.max_rows,
            target_language=args.target_language,
            use_stt=use_stt,
            stt_model=args.stt_model,
        )

    out_name = f"translation_predictions_{args.config_id}.csv"
    out_path = split_dir / out_name
    df.to_csv(out_path, index=False)
    print(f"Done: {len(df)} API call(s), CSV output saved to {out_path}")


if __name__ == "__main__":
    main()
