"""
Live translation module for IIKSHANA demo.

Energy-based VAD gates audio into utterances. Each complete utterance is
sent to ElevenLabs Scribe v2 (STT) then translated via the configured model.
Optionally synthesises TTS audio for the translated text (headphones mode).

Called from gradio_expo_app.py via the streaming audio .stream() event.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# VAD tuning constants
# ---------------------------------------------------------------------------
_ENERGY_THRESHOLD = 0.008       # RMS below this = silence
_SILENCE_FRAMES_TO_END = 6      # consecutive silent chunks → utterance complete (~3 s at 0.5 s/chunk)
_MIN_UTTERANCE_SECONDS = 0.4    # discard short noise bursts below this duration
_SAMPLE_RATE = 16_000

_LANG_LABELS = {"es": "Spanish", "fr": "French", "de": "German"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_initial_state() -> dict:
    """Return a fresh LiveTranslation state dict for gr.State."""
    return {
        "audio_buffer": [],       # list[np.ndarray] — 16 kHz mono float32 chunks
        "silence_frames": 0,      # consecutive silent chunk count
        "speech_detected": False, # True once speech seen in current utterance
        "full_output": "",        # accumulated markdown translation history
        "utterance_count": 0,     # total utterances processed this session
    }


def process_audio_chunk(
    chunk_data: Any,
    state: dict,
    target_language: str,
    config_id: str,
    tts_enabled: bool = False,
) -> tuple[dict, str, str, str | None]:
    """
    Process one streaming audio chunk from Gradio.

    Parameters
    ----------
    chunk_data     : (sample_rate, np.ndarray) tuple from gr.Audio streaming=True
    state          : current live-translation state (from gr.State)
    target_language: e.g. "es", "fr", "de"
    config_id      : translation config ID (e.g. "translation_flash_v1")
    tts_enabled    : if True, synthesise TTS audio for the translated text

    Returns
    -------
    (updated_state, live_output_markdown, status_message, tts_audio_path_or_none)
    tts_audio_path is a temp .mp3 filepath when tts_enabled and translation succeeded,
    otherwise None (caller should pass gr.update() to the Audio component).
    """
    if chunk_data is None:
        return state, state.get("full_output", ""), "", None

    if not (isinstance(chunk_data, (tuple, list)) and len(chunk_data) == 2):
        return state, state.get("full_output", ""), "Unexpected audio format — skipping chunk.", None

    src_sr, raw_samples = chunk_data
    if raw_samples is None:
        return state, state.get("full_output", ""), "", None

    samples = _to_16k_mono_float32(np.asarray(raw_samples), int(src_sr))
    if samples.size == 0:
        return state, state.get("full_output", ""), "", None

    rms = float(np.sqrt(np.mean(samples ** 2)))
    is_speech = rms > _ENERGY_THRESHOLD

    # Shallow copy so gr.State detects the mutation
    state = dict(state)
    state["audio_buffer"] = list(state.get("audio_buffer", []))
    status = ""

    if is_speech:
        state["speech_detected"] = True
        state["silence_frames"] = 0
        state["audio_buffer"].append(samples)
        return state, state.get("full_output", ""), "Listening…", None

    if state.get("speech_detected", False):
        # Trailing silence — accumulate, waiting for end threshold
        state["silence_frames"] = state.get("silence_frames", 0) + 1
        state["audio_buffer"].append(samples)

        if state["silence_frames"] >= _SILENCE_FRAMES_TO_END:
            utterance = np.concatenate(state["audio_buffer"])
            duration = len(utterance) / _SAMPLE_RATE

            # Reset VAD state before API calls
            state["audio_buffer"] = []
            state["speech_detected"] = False
            state["silence_frames"] = 0

            if duration < _MIN_UTTERANCE_SECONDS:
                return state, state.get("full_output", ""), "Short noise burst — ignored.", None

            state["utterance_count"] = state.get("utterance_count", 0) + 1
            n = state["utterance_count"]
            status = f"Processing utterance #{n}…"
            display_text, status, audio_path = _stt_and_translate(
                utterance, target_language, config_id, tts_enabled=tts_enabled
            )
            if display_text:
                state["full_output"] = state.get("full_output", "") + display_text
            return state, state.get("full_output", ""), status, audio_path

        return state, state.get("full_output", ""), "End of utterance detected…", None

    return state, state.get("full_output", ""), "Waiting for speech…", None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_16k_mono_float32(samples: np.ndarray, src_sr: int) -> np.ndarray:
    arr = samples.astype(np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    # int16 range → float32 [-1, 1]
    if np.max(np.abs(arr)) > 1.5:
        arr = arr / 32768.0
    if src_sr != _SAMPLE_RATE:
        try:
            import librosa  # noqa: PLC0415
            arr = librosa.resample(arr, orig_sr=src_sr, target_sr=_SAMPLE_RATE)
        except Exception:  # noqa: BLE001
            pass  # keep original sr; quality degrades but won't crash
    return arr


def _stt_and_translate(
    audio: np.ndarray,
    target_language: str,
    config_id: str,
    *,
    tts_enabled: bool = False,
) -> tuple[str, str, str | None]:
    """
    Send utterance to ElevenLabs STT then translate. Optionally synthesise TTS.

    Returns
    -------
    (display_markdown, status_message, tts_audio_path_or_none)
    display_markdown is empty string on failure.
    tts_audio_path is a .mp3 temp file path when tts_enabled and synthesis succeeded.
    """
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        sf.write(str(tmp_path), audio, _SAMPLE_RATE, subtype="PCM_16")

        # --- STT ---
        from backend.src.services.elevenlabs_stt_service import (  # noqa: PLC0415
            elevenlabs_api_key_from_env,
            transcribe_file_scribe_v2,
        )

        api_key = elevenlabs_api_key_from_env()
        if not api_key:
            return "", "ELEVENLABS_API_KEY not set — STT skipped.", None

        chunk = transcribe_file_scribe_v2(
            tmp_path,
            api_key=api_key,
            model_id="scribe_v2",
            diarize=False,
            tag_audio_events=False,
            timestamps_granularity="word",
        )
        transcript = (getattr(chunk, "text", None) or "").strip()
        if not transcript:
            return "", "STT returned empty (silence or noise).", None

        # --- Detect source language ---
        lang_code = getattr(chunk, "language_code", None) or ""
        from demo.audio_analysis_pipeline import scribe_language_code_for_translation  # noqa: PLC0415
        source_lang = scribe_language_code_for_translation(lang_code)

        # --- Translate ---
        from backend.src.services.gemini_translation import translate_text  # noqa: PLC0415
        translated = translate_text(
            source_text=transcript,
            source_language=source_lang,
            target_language=target_language,
            config_id=config_id,
        )

        lang_label = _LANG_LABELS.get(target_language, target_language.upper())
        display = (
            f"**[{source_lang.upper()} → {lang_label}]** {translated}\n\n"
            f"*Original: {transcript}*\n\n---\n\n"
        )
        status = f"Translated utterance ({source_lang} → {lang_label})."

        # --- TTS (headphones mode only) ---
        audio_path: str | None = None
        if tts_enabled and translated and "[EMPTY_TRANSCRIPT]" not in translated:
            try:
                from demo.audio_analysis_pipeline import synthesize_speech_mp3  # noqa: PLC0415
                mp3_bytes, tts_err = synthesize_speech_mp3(translated)
                if mp3_bytes:
                    tfd, tpath = tempfile.mkstemp(suffix=".mp3")
                    os.close(tfd)
                    Path(tpath).write_bytes(mp3_bytes)
                    audio_path = tpath
                elif tts_err:
                    status += f" (TTS: {tts_err})"
            except Exception as e:  # noqa: BLE001
                status += f" (TTS error: {e})"

        return display, status, audio_path

    except Exception as e:  # noqa: BLE001
        return "", f"Error: {e}", None
    finally:
        tmp_path.unlink(missing_ok=True)
