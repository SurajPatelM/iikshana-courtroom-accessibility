"""
IIKSHANA — courtroom accessibility expo UI (Gradio).

**Main flow:** ElevenLabs **Scribe v2** (STT) → local speaker/gender/emotion → **ElevenLabs TTS** (spoken
summary) → **ingest + ``model_pipeline_dag``** (batch translation) → **ElevenLabs TTS** (translated text).
Translation text and translation TTS come **only** from the DAG output (predictions CSV), not from a separate
local translator.

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

POLL_SEC = 12
MAX_WAIT_SEC = 45 * 60

LANG_LABELS = {"es": "Spanish", "fr": "French", "de": "German"}

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
    tail_n = int(manifest_tail) if manifest_tail else 200
    code, log, _ = trigger_model_pipeline_dag(
        split=split,
        refresh_inputs=True,
        refresh_config_search=rerun_config_search,
        manifest_tail=tail_n,
        target_language=target_language,
    )
    lines = [
        "",
        "---",
        "### Batch translation (`model_pipeline_dag`)",
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
    while time.monotonic() < deadline:
        got = try_read_pipeline_translation(REPO_ROOT, split, manifest_row["file"])
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
) -> tuple[str, str, str | None, str | None, str]:
    """
    ElevenLabs Scribe + local ML + summary TTS, then ingest + ``model_pipeline_dag``; translation + translation
    TTS use **only** batch CSV output (no local Gemini/Groq).

    Return order: status, translation_md, summary_tts_path, translation_tts_path, detail_md.
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

        result = run_ui_audio_analysis(tmp_wav, status=status_cb)
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
        trans_md = f"### Translation ({label})\n\n*(waiting for `model_pipeline_dag` / CSV…)*"

        detail_md = (
            f"### Language\n\n**{result.language_display}** (scribe code: `{result.language_code or '?'}`)\n\n"
            f"### Speakers\n\n{summaries}\n\n### Transcript\n\n{transcript_block}"
        )

        status_md = f"**Pipeline:** {progress}"
        if result.tts_error:
            status_md += f"\n\n**Summary TTS:** {result.tts_error}"

        sum_path: str | None = None
        if result.tts_audio_mp3:
            sfd, spath = tempfile.mkstemp(suffix=".mp3")
            os.close(sfd)
            Path(spath).write_bytes(result.tts_audio_mp3)
            sum_path = spath

        tr_path: str | None = None

        def _no_dag_translation(msg: str) -> None:
            nonlocal trans_md, status_md
            status_md += f"\n\n**No DAG translation in UI** — {msg}"
            trans_md = (
                f"### Translation ({label})\n\n"
                "*This app only shows translation from **`model_pipeline_dag`** (predictions CSV).*\n\n"
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
                local_scribe_transcript=result.transcript_plain,
            )
            status_md += note

            use_batch = batch_text is not None and "[EMPTY_TRANSCRIPT]" not in batch_text
            if use_batch:
                trans_md = f"### Translation ({label}, from `model_pipeline_dag`)\n\n{batch_text}"
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
            "**Single pipeline:** ElevenLabs **Scribe v2** → **inaSpeechSegmenter** + **emotion2vec+** (host; needs "
            "`torchaudio`) → **ElevenLabs TTS** (spoken summary) → **ingest + `model_pipeline_dag`** (Docker: batch "
            "translation written to **predictions CSV**) → **ElevenLabs TTS** (that translated text). "
            "**Translation in the UI is only from the DAG** (no separate local translator).\n\n"
            "**Wait for DAG translation:** after triggering the DAG, the server **re-reads the CSV every ~12s** until it "
            "finds the row for **your** ingested file (or hits the long timeout). That is how the app gets the string for "
            "the Translation box and for translation TTS—Airflow writes the file asynchronously.\n\n"
            "Airflow: keys in **`airflow/.env`** or **`.secrets/.env`** (loaded by compose **`env_file`**); "
            "**`docker compose down && docker compose up`** after changes. Prior **EXPO** clips/CSV rows are cleared on each ingest; "
            "**.scribe.txt** supplies batch `source_text` if container STT is empty.\n\n"
            "Local: **`ELEVENLABS_API_KEY`** and **`GROQ_API_KEY`**. Optional **`ELEVENLABS_TTS_VOICE_ID`** or **`ELEVENLABS_TTS_MODEL_ID`**; TTS skips "
            "Voice Library voices and tries Flash/Turbo models first (free-tier friendly). **ffmpeg** on PATH. **Microphone:** "
            "**http://127.0.0.1:7860** — record, "
            "**Stop**, then run.\n\n"
            "Heavy ML extras: `pip install -r requirements-demo-ui.txt` + `inaSpeechSegmenter` line in that file; **Preload** once."
        )

        with gr.Accordion("Optional: preload gender / emotion models (first run can be slow)", open=False):
            preload_btn = gr.Button("Preload ML models")
            preload_msg = gr.Markdown()
            preload_btn.click(_preload_models, outputs=preload_msg)

        with gr.Row():
            target_language = gr.Dropdown(
                choices=["es", "fr", "de"],
                value="es",
                label="Translate to (passed to model_pipeline_dag)",
            )

        with gr.Row():
            chain_split = gr.Dropdown(
                list(VALID_SPLITS), value="dev", label="DAG split (ingest + model_pipeline_dag)"
            )
            chain_manifest_tail = gr.Number(
                label="Manifest tail (DAG)",
                value=200,
                precision=0,
                minimum=1,
                maximum=500,
            )
            chain_rerun_cfg = gr.Checkbox(label="DAG: re-run config search", value=False)
            chain_wait_csv = gr.Checkbox(
                label="Wait for DAG translation (poll predictions CSV until this file’s row appears)",
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
            "Run: Scribe → analysis → summary TTS → model_pipeline_dag translation → translation TTS",
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
