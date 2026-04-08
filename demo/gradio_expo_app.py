"""
IIKSHANA — courtroom accessibility expo UI (Gradio).

**Fast translate (default):** after **Scribe** and **local** gender/emotion analysis (**not** ElevenLabs — those use
**inaSpeechSegmenter** + **emotion2vec+** on the host), the UI can **translate on host** (no Airflow CSV wait).
Optional: **skip local ML** only if you explicitly want a quicker run without gender/emotion. **Batch** path uses
Airflow CSV translation instead of on-host translate.

Run from repo root::

    PYTHONPATH=. python demo/gradio_expo_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # override=True: values from these files replace vars already in the process env. That fixes the
    # common case where the shell has `export ELEVENLABS_API_KEY=` (empty) and python-dotenv would
    # otherwise refuse to load the real key from `.env` (override=False).
    for p in (REPO_ROOT / ".env", REPO_ROOT / ".secrets" / ".env"):
        if p.is_file():
            load_dotenv(p, override=True)


_load_repo_dotenv()

from demo.airflow_trigger import trigger_model_pipeline_dag
from demo.local_model_pipeline import try_read_pipeline_translation
from demo.pipeline_ingest import VALID_SPLITS, ingest_expo_recording

# Fast path: last manifest row only (EXPO is appended last), tight CSV polling, no artificial API spacing by default.
DEFAULT_MANIFEST_TAIL = 1
POLL_SEC = max(1, int(os.environ.get("EXPO_POLL_SEC", "1")))
MAX_WAIT_SEC = 45 * 60
TRANSLATE_DELAY_SEC = float(os.environ.get("EXPO_TRANSLATE_DELAY", "0"))
STT_DELAY_SEC = float(os.environ.get("EXPO_STT_DELAY", "0"))
DEFAULT_TRANSLATION_CONFIG_ID = (
    os.environ.get("EXPO_TRANSLATION_CONFIG_ID", "translation_flash_v1").strip() or "translation_flash_v1"
)

TRANSLATION_CONFIG_CHOICES = [
    "translation_flash_v1",
    "translation_flash_glossary",
    "translation_flash_court",
    "translation_flash_short_prompt",
    "translation_flash_temp03",
]

LANG_LABELS = {"es": "Spanish", "fr": "French", "de": "German"}


def _active_dag_id() -> str:
    return (os.environ.get("AIRFLOW_MODEL_DAG_ID", "") or "expo_translation_dag").strip()


def _default_fast_translate() -> bool:
    """On-host translation + no Airflow CSV wait (does not skip gender/emotion unless that box is checked)."""
    v = os.environ.get("IIKSHANA_REALTIME_MODE", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _default_skip_local_ml() -> bool:
    v = os.environ.get("IIKSHANA_SKIP_LOCAL_ML", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


_AIRFLOW_EMPTY_STT_HINT = (
    "\n\n---\n**Why `[EMPTY_TRANSCRIPT]`?** Batch STT runs **in Docker**. Ensure **`ELEVENLABS_API_KEY`** is in "
    "`airflow/.env` or `../.secrets/.env` (loaded by `docker-compose.yaml` via `env_file`) — **not** only under "
    "`environment: ${VAR:-}`**, which can inject an empty value and override the file. Then `docker compose down && "
    "docker compose up` from `airflow/`. The UI also writes **`EXPO_*.wav.scribe.txt`** with your **local** Scribe "
    "text; if that sidecar is missing and container STT is empty, you still see this placeholder."
)


def _airflow_chain_note_from_src(
    src: Path,
    *,
    split: str,
    target_language: str,
    rerun_config_search: bool,
    manifest_tail: float,
    wait_for_csv: bool,
    translation_config_id: str,
    local_scribe_transcript: str | None = None,
) -> tuple[str, str | None]:
    """Ingest same clip as EXPO + trigger DAG; optional poll CSV.

    Returns ``(status_markdown, batch_translation_text_or_none)``. The second value is the prediction
    string when CSV polling succeeds, so the UI can show batch output as the main translation.
    """
    suffix = src.suffix or ".wav"
    _, manifest_row = ingest_expo_recording(
        src,
        split=split,
        source_suffix=suffix,
        local_scribe_transcript=local_scribe_transcript,
        clear_previous_expo=True,
    )
    tail_n = int(manifest_tail) if manifest_tail else DEFAULT_MANIFEST_TAIL
    if tail_n < 1:
        tail_n = DEFAULT_MANIFEST_TAIL
    cfg_id = (translation_config_id or DEFAULT_TRANSLATION_CONFIG_ID).strip() or DEFAULT_TRANSLATION_CONFIG_ID
    code, log, _ = trigger_model_pipeline_dag(
        split=split,
        refresh_inputs=True,
        refresh_config_search=rerun_config_search,
        manifest_tail=tail_n,
        target_language=target_language,
        translate_delay=TRANSLATE_DELAY_SEC,
        config_id=cfg_id,
        stt_delay=STT_DELAY_SEC,
    )
    dag_id = _active_dag_id()
    lines = [
        "",
        "---",
        f"### Batch translation (`{dag_id}`)",
        f"- Ingested **`{manifest_row['file']}`** into split `{split}` (same clip as above).",
        f"- DAG trigger exit code: **{code}**.",
    ]
    if code != 0:
        lines.append(f"- Log:\n```\n{log or '(none)'}\n```")
        return "\n".join(lines), None
    if not wait_for_csv:
        lines.append(
            f"- *CSV poll skipped.* When the DAG finishes, check `data/model_runs/{split}/translation_predictions_*.csv`."
        )
        return "\n".join(lines), None
    deadline = time.monotonic() + MAX_WAIT_SEC
    got = None
    # expo_translation_dag writes translation_predictions_<config_id>.csv; model_pipeline_dag uses
    # config_search_results.json — poll that path when config_id is omitted.
    poll_config_id = cfg_id if dag_id == "expo_translation_dag" else None
    while time.monotonic() < deadline:
        got = try_read_pipeline_translation(
            REPO_ROOT,
            split,
            manifest_row["file"],
            config_id=poll_config_id,
        )
        if got is not None:
            break
        time.sleep(POLL_SEC)
    if got is None:
        lines.append("- *Timed out* waiting for a prediction row for this file (DAG may still be running).")
        return "\n".join(lines), None
    text, pred_path, best_cfg = got
    lines.append(f"- **Batch CSV translation** (config `{best_cfg}`): `{pred_path}`")
    lines.append(f"- Text:\n\n{text}")
    if "[EMPTY_TRANSCRIPT]" in text:
        lines.append(_AIRFLOW_EMPTY_STT_HINT)
    return "\n".join(lines), text


def _gradio_audio_to_temp_wav(audio: Any) -> tuple[Path | None, list[Path]]:
    """
    Normalize Gradio ``Audio`` values to a readable file path.

    Gradio 5 records to a server-side temp file and (after preprocessing) usually passes either a
    **filepath string** or ``(sample_rate, int16 ndarray)``. If the user clicks the action button
    **before** finishing the recording (mic still open / not stopped), the payload can still be
    ``None`` — the UI nudges them to press **Stop** first.
    """
    cleanup: list[Path] = []
    if audio is None:
        return None, cleanup

    # Dict-shaped FileData from API edges
    if isinstance(audio, dict):
        p = audio.get("path")
        if isinstance(p, str) and Path(p).is_file():
            return Path(p), cleanup
        return None, cleanup

    # Preprocessed filepath (Gradio 5 default for type="filepath")
    if isinstance(audio, (str, Path)):
        p = Path(audio)
        if p.is_file():
            return p, cleanup
        return None, cleanup

    # FileData model (defensive — normally preprocessed away)
    path_attr = getattr(audio, "path", None)
    if isinstance(path_attr, str) and Path(path_attr).is_file():
        return Path(path_attr), cleanup

    # type="numpy": (sample_rate, ndarray), int16 from browser mic per Gradio docs
    if not isinstance(audio, tuple) or len(audio) != 2:
        return None, cleanup
    sr_raw, y = audio
    if y is None:
        return None, cleanup
    y = np.asarray(y)
    if y.size == 0:
        return None, cleanup
    if np.issubdtype(y.dtype, np.integer):
        y = np.clip(y.astype(np.float32) / 32768.0, -1.0, 1.0)
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    sr = int(sr_raw) if sr_raw else 48_000
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    p = Path(name)
    sf.write(str(p), y, sr, subtype="PCM_16")
    cleanup.append(p)
    return p, cleanup


def _preload_models() -> str:
    try:
        from demo.audio_analysis_pipeline import get_emotion_model, get_gender_segmenter

        get_gender_segmenter()
        get_emotion_model()
    except Exception as e:  # noqa: BLE001
        return f"Model load failed: {e}"
    return "Models are ready in this server process."


def _run_unified_courtroom_demo(
    audio: Any,
    target_language: str,
    airflow_split: str,
    manifest_tail: float,
    airflow_rerun_config: bool,
    wait_for_airflow_csv: bool,
    translation_config_id: str,
    fast_translate: bool,
    airflow_background: bool,
    skip_local_ml: bool,
) -> tuple[str, str, str | None, str | None, str]:
    """
    ``fast_translate``: on-host ``translate_text`` + no CSV wait (optional background Airflow).
    Otherwise: translation from Airflow CSV. ``skip_local_ml`` skips inaSpeechSegmenter + emotion2vec+ only.
    """
    src, to_clean = _gradio_audio_to_temp_wav(audio)
    if src is None or not src.is_file():
        return (
            "**No audio on the server.** Record, press **Stop**, then run again. "
            "Use **http://127.0.0.1:7860** and allow the microphone.",
            "",
            None,
            None,
            "",
        )

    fd_w, wav_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd_w)
    tmp_wav = Path(wav_name)
    try:
        from demo.audio_analysis_pipeline import (
            normalize_to_wav_16k_mono,
            run_ui_audio_analysis,
            scribe_language_code_for_translation,
            synthesize_speech_mp3,
        )

        cfd_snap, cname_snap = tempfile.mkstemp(suffix=src.suffix or ".wav")
        os.close(cfd_snap)
        chain_ingest_path = Path(cname_snap)
        chain_ingest_path.write_bytes(src.read_bytes())
        to_clean.append(chain_ingest_path)

        normalize_to_wav_16k_mono(src, tmp_wav)
        log_lines: list[str] = []

        def status_cb(msg: str) -> None:
            log_lines.append(msg)

        result = run_ui_audio_analysis(
            tmp_wav, status=status_cb, skip_local_ml=bool(skip_local_ml)
        )
        progress = " → ".join(log_lines[-10:]) if log_lines else "(no log)"

        if result.scribe_error:
            return (f"**Scribe / analysis:** {result.scribe_error}", "", None, None, "")

        summaries = "(none)"
        if result.speaker_summaries:
            parts = [
                (
                    f"- **{s['speaker_label']}** — **{s['estimated_gender']}**, "
                    f"emotion **{s['dominant_emotion']}**, **{s['total_seconds']:.1f}** s"
                )
                for s in result.speaker_summaries
            ]
            summaries = "\n".join(parts)

        transcript_block = "\n\n".join(result.transcript_rich_lines) or "(empty)"

        label = LANG_LABELS.get(target_language, target_language)

        detail_md = (
            f"### Language\n\n**{result.language_display}** (scribe code: `{result.language_code or '?'}`)\n\n"
            f"### Speakers\n\n{summaries}\n\n### Transcript\n\n{transcript_block}"
        )

        status_md = f"**Pipeline:** {progress}"
        if skip_local_ml:
            status_md += "\n\n**Note:** local gender/emotion models were skipped (faster, less detail)."
        if fast_translate:
            status_md += "\n\n**Translation path:** on-host (no Airflow CSV wait)."
        if result.tts_error:
            status_md += f"\n\n**Summary TTS:** {result.tts_error}"

        sum_path: str | None = None
        if result.tts_audio_mp3:
            sfd, spath = tempfile.mkstemp(suffix=".mp3")
            os.close(sfd)
            Path(spath).write_bytes(result.tts_audio_mp3)
            sum_path = spath

        tr_path: str | None = None
        cfg_id = (translation_config_id or DEFAULT_TRANSLATION_CONFIG_ID).strip() or DEFAULT_TRANSLATION_CONFIG_ID

        if fast_translate:
            raw_t = (result.transcript_plain or "").strip()
            trans_md: str
            if not raw_t:
                trans_md = f"### Translation ({label})\n\n*(empty transcript — nothing to translate)*"
            else:
                from backend.src.services.gemini_translation import translate_text  # noqa: PLC0415

                status_md += "\n\n**Translating on host** (no Airflow wait)…"
                try:
                    sl = scribe_language_code_for_translation(result.language_code)
                    lt = translate_text(
                        source_text=raw_t,
                        source_language=sl,
                        target_language=target_language,
                        config_id=cfg_id,
                    )
                except Exception as e:  # noqa: BLE001
                    trans_md = f"### Translation ({label})\n\n**On-host translate failed:** `{e}`"
                else:
                    trans_md = f"### Translation ({label}, on-host / fast)\n\n{lt}"
                    b_mp3, b_err = synthesize_speech_mp3(lt)
                    if b_mp3:
                        tfd2, tpath2 = tempfile.mkstemp(suffix=".mp3")
                        os.close(tfd2)
                        Path(tpath2).write_bytes(b_mp3)
                        tr_path = tpath2
                    if b_err:
                        trans_md += f"\n\n*Translation TTS:* {b_err}"

            if airflow_background:
                try:
                    note, _ = _airflow_chain_note_from_src(
                        chain_ingest_path,
                        split=airflow_split,
                        target_language=target_language,
                        rerun_config_search=airflow_rerun_config,
                        manifest_tail=manifest_tail,
                        wait_for_csv=False,
                        translation_config_id=translation_config_id,
                        local_scribe_transcript=result.transcript_plain,
                    )
                    status_md += note
                except Exception as e:  # noqa: BLE001
                    status_md += f"\n\n**Background Airflow:** `{e}`"
            return (status_md, trans_md, sum_path, tr_path, detail_md)

        trans_md = f"### Translation ({label})\n\n*(waiting for Airflow `{_active_dag_id()}` / CSV…)*"

        def _no_dag_translation(msg: str) -> None:
            nonlocal trans_md, status_md
            status_md += f"\n\n**No DAG translation in UI** — {msg}"
            trans_md = (
                f"### Translation ({label})\n\n"
                f"*Translation in this mode comes from the Airflow DAG (**`{_active_dag_id()}`**) CSV.*\n\n"
                f"{msg}"
            )

        try:
            note, batch_text = _airflow_chain_note_from_src(
                chain_ingest_path,
                split=airflow_split,
                target_language=target_language,
                rerun_config_search=airflow_rerun_config,
                manifest_tail=manifest_tail,
                wait_for_csv=wait_for_airflow_csv,
                translation_config_id=translation_config_id,
                local_scribe_transcript=result.transcript_plain,
            )
            status_md += note

            use_batch = batch_text is not None and "[EMPTY_TRANSCRIPT]" not in batch_text
            if use_batch:
                trans_md = f"### Translation ({label}, from `{_active_dag_id()}`)\n\n{batch_text}"
                b_mp3, b_err = synthesize_speech_mp3(batch_text)
                if b_mp3:
                    tfd2, tpath2 = tempfile.mkstemp(suffix=".mp3")
                    os.close(tfd2)
                    Path(tpath2).write_bytes(b_mp3)
                    tr_path = tpath2
                if b_err:
                    trans_md += f"\n\n*Translation TTS:* {b_err}"
            elif batch_text is not None and "[EMPTY_TRANSCRIPT]" in batch_text:
                _no_dag_translation(
                    "Batch **`source_text`** was empty (container Scribe failed and no **`.scribe.txt`** sidecar was "
                    "used). Fix **`env_file`** keys in `airflow/docker-compose.yaml` (see **`airflow/.env.example`**), "
                    "recreate containers, and re-run — or confirm `data/processed/<split>/EXPO_*.wav.scribe.txt` exists "
                    "next to your WAV after ingest."
                )
            elif not wait_for_airflow_csv:
                _no_dag_translation(
                    "You turned off **Wait for DAG translation**. The DAG was still triggered, but this UI did not "
                    f"poll the CSV. Open `data/model_runs/{airflow_split}/translation_predictions_*.csv` after the run "
                    "finishes, or turn **Wait** back on."
                )
            else:
                _no_dag_translation(
                    "Trigger failed, poll timed out, or no prediction row matched this file yet. Check **Status** above "
                    "and the Airflow UI; logs live under `data/model_runs/`."
                )
        except Exception as e:  # noqa: BLE001
            status_md += f"\n\n**Batch translation step failed:** {e}"
            _no_dag_translation(f"Ingest / trigger / poll raised: `{e}`")

        return (status_md, trans_md, sum_path, tr_path, detail_md)
    except Exception as e:  # noqa: BLE001
        return (f"**Failed:** {e}", "", None, None, "")
    finally:
        tmp_wav.unlink(missing_ok=True)
        for p in to_clean:
            p.unlink(missing_ok=True)


def build_demo() -> gr.Blocks:
    def _audio_ready_hint() -> str:
        return "✓ Clip is on the server — run the action button below."

    with gr.Blocks(title="IIKSHANA — Courtroom Accessibility", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# IIKSHANA\n"
            "**ElevenLabs** provides **Scribe** (STT + diarization) and **TTS**. **Gender** (**inaSpeechSegmenter**) and "
            "**emotion** (**emotion2vec+**) run **on your machine** (see `requirements-demo-ui.txt`) — they are **not** "
            "ElevenLabs APIs.\n\n"
            "**Fast translate (default on):** after local analysis, **on-host translation** + **translation TTS** — **no** "
            "wait for Airflow CSV. Uncheck it for **batch** mode: translation from **`expo_translation_dag`** CSV.\n\n"
            "**Skip local gender/emotion** — optional; faster runs but **unknown** gender/emotion. Default **off** so "
            "behavior matches the full demo. Env: **`IIKSHANA_SKIP_LOCAL_ML=1`** to default the checkbox on.\n\n"
            "**Background Airflow** (with fast translate): ingest + trigger **without** CSV wait. Batch mode: **Wait for DAG "
            "translation** + **`EXPO_POLL_SEC`** (default **1** s). **`AIRFLOW_MODEL_DAG_ID=model_pipeline_dag`** for BLEU "
            "config search.\n\n"
            "Keys: **`ELEVENLABS_API_KEY`**, **`GROQ_API_KEY`**, Airflow **`airflow/.env`**. **`IIKSHANA_REALTIME_MODE=0`** "
            "defaults UI to **batch** translate path.\n\n"
            "Install **`requirements-demo-ui.txt`** + **`inaSpeechSegmenter`** (see that file) for gender/emotion; **Preload** "
            "warms models."
        )

        with gr.Accordion("Optional: preload gender / emotion models (first run can be slow)", open=False):
            preload_btn = gr.Button("Preload ML models")
            preload_msg = gr.Markdown()
            preload_btn.click(_preload_models, outputs=preload_msg)

        with gr.Row():
            fast_translate_cb = gr.Checkbox(
                label="Fast translate (on-host; no Airflow CSV wait)",
                value=_default_fast_translate(),
            )
            skip_ml_cb = gr.Checkbox(
                label="Skip local gender/emotion models (faster; optional)",
                value=_default_skip_local_ml(),
            )
            airflow_bg_cb = gr.Checkbox(
                label="Also ingest + trigger Airflow in background (fast-translate only; no wait)",
                value=False,
            )

        with gr.Row():
            target_language = gr.Dropdown(
                choices=["es", "fr", "de"],
                value="es",
                label="Translate to (target_language)",
            )
            translation_config = gr.Dropdown(
                choices=TRANSLATION_CONFIG_CHOICES,
                value=DEFAULT_TRANSLATION_CONFIG_ID,
                label="Translation config_id (on-host + Airflow; under config/models/)",
            )

        with gr.Row():
            chain_split = gr.Dropdown(
                list(VALID_SPLITS), value="dev", label="DAG split (ingest + translation DAG)"
            )
            chain_manifest_tail = gr.Number(
                label="Manifest tail (last N manifest rows; use 1 for single EXPO clip)",
                value=DEFAULT_MANIFEST_TAIL,
                precision=0,
                minimum=1,
                maximum=500,
            )
            chain_rerun_cfg = gr.Checkbox(
                label="model_pipeline_dag only: re-run config search",
                value=False,
            )
            chain_wait_csv = gr.Checkbox(
                label="Batch mode (fast translate off): wait for DAG translation",
                value=True,
            )

        main_audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Record or upload courtroom audio",
        )
        main_audio_hint = gr.Markdown("")
        main_audio.stop_recording(fn=_audio_ready_hint, outputs=main_audio_hint)
        main_audio.upload(fn=_audio_ready_hint, outputs=main_audio_hint)

        run_btn = gr.Button(
            "Run: Scribe → (optional ML) → summary TTS → translate → translation TTS",
            variant="primary",
        )

        gr.Markdown("### Status")
        unified_status = gr.Markdown()
        gr.Markdown("### Translation")
        unified_translation = gr.Markdown()
        with gr.Row():
            unified_tts_summary = gr.Audio(label="TTS: English summary", type="filepath")
            unified_tts_translation = gr.Audio(label="TTS: translated text (ElevenLabs)", type="filepath")
        gr.Markdown("### Language, speakers & transcript")
        unified_detail = gr.Markdown()

        run_btn.click(
            _run_unified_courtroom_demo,
            inputs=[
                main_audio,
                target_language,
                chain_split,
                chain_manifest_tail,
                chain_rerun_cfg,
                chain_wait_csv,
                translation_config,
                fast_translate_cb,
                airflow_bg_cb,
                skip_ml_cb,
            ],
            outputs=[
                unified_status,
                unified_translation,
                unified_tts_summary,
                unified_tts_translation,
                unified_detail,
            ],
        )

    return demo


def main() -> None:
    demo = build_demo()
    try:
        demo.queue(default_concurrency_limit=1)
    except TypeError:
        demo.queue()
    # Default 127.0.0.1 so browser microphone (getUserMedia) works; use GRADIO_SERVER_NAME=0.0.0.0 for LAN.
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
