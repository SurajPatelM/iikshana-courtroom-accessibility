"""
Live translation module for IIKSHANA demo.

Energy-based VAD gates audio into utterances. Each complete utterance is
sent to ElevenLabs Scribe v2 (STT) then translated via the configured model.
Optionally synthesises TTS audio for the translated text (headphones mode).

Called from gradio_expo_app.py via the streaming audio .stream() event.
"""
from __future__ import annotations

import html
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger("iikshana.live_translation")

# ---------------------------------------------------------------------------
# VAD tuning constants
# ---------------------------------------------------------------------------
# Primary energy gate — used to *start* an utterance. Tuned for 16 kHz mono
# float32 mic input after browser AGC/noise-suppression.
_ENERGY_THRESHOLD = 0.0006
# Once we are inside an utterance, accept slightly softer frames as speech so
# normal intra-word dips don't immediately flip us into silence. Anything below
# this is treated as a silence frame for end-of-utterance detection.
_IN_UTTERANCE_ENERGY_THRESHOLD = 0.0005
_SILENCE_FRAMES_TO_END = 6
_SILENCE_SECONDS_TO_END = 0.5
_MIN_UTTERANCE_SECONDS = 0.4
_SAMPLE_RATE = 16_000

_LANG_LABELS = {"es": "Spanish", "fr": "French", "de": "German", "en": "English"}

# Speaker rotation for demo (in production, use actual diarization)
_SPEAKER_ROLES = ["Judge", "Witness", "Attorney", "Defendant"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_initial_state() -> dict:
    """Return a fresh LiveTranslation state dict for gr.State."""
    return {
        "audio_buffer": [],
        "silence_frames": 0,
        "silence_seconds": 0.0,
        "speech_detected": False,
        "utterances": [],  # List of utterance dicts for card display
        "utterance_count": 0,
        "full_output": "",  # Legacy text output
    }


def process_audio_chunk(
    chunk_data: Any,
    state: dict,
    target_language: str,
    config_id: str,
    tts_enabled: bool = False,
    source_language_override: str | None = None,
) -> tuple[dict, str, str, str | None]:
    """
    Process one streaming audio chunk from Gradio.

    Returns
    -------
    (updated_state, display_html, status_message, tts_audio_path_or_none)
    """
    if chunk_data is None:
        return state, _live_translations_display_html(state.get("utterances", [])), "", None

    if not (isinstance(chunk_data, (tuple, list)) and len(chunk_data) == 2):
        return state, _live_translations_display_html(state.get("utterances", [])), "", None

    src_sr, raw_samples = chunk_data
    if raw_samples is None:
        logger.info("process_audio_chunk: raw_samples is None")
        return state, _live_translations_display_html(state.get("utterances", [])), "", None

    samples = _to_16k_mono_float32(np.asarray(raw_samples), int(src_sr))
    if samples.size == 0:
        logger.info("process_audio_chunk: samples.size == 0 after conversion")
        return state, _live_translations_display_html(state.get("utterances", [])), "", None

    chunk_duration = len(samples) / _SAMPLE_RATE
    rms = float(np.sqrt(np.mean(samples ** 2)))
    # Use a stricter threshold to *start* speech, and a looser one once we are
    # already mid-utterance. This stops a single loud frame from being dropped
    # just because the next frame dipped slightly below the start threshold.
    already_in_utterance = bool(state.get("speech_detected", False))
    active_threshold = (
        _IN_UTTERANCE_ENERGY_THRESHOLD if already_in_utterance else _ENERGY_THRESHOLD
    )
    is_speech = rms > active_threshold
    logger.info(
        "process_audio_chunk: src_sr=%s raw_shape=%s samples=%s duration=%.3f rms=%.8f is_speech=%s threshold=%.4f",
        src_sr,
        np.asarray(raw_samples).shape,
        samples.shape,
        chunk_duration,
        rms,
        is_speech,
        active_threshold,
    )

    # Shallow copy state
    state = dict(state)
    state["audio_buffer"] = list(state.get("audio_buffer", []))
    state["utterances"] = list(state.get("utterances", []))
    state["silence_seconds"] = float(state.get("silence_seconds", 0.0))

    if is_speech:
        state["speech_detected"] = True
        state["silence_frames"] = 0
        state["audio_buffer"].append(samples)
        logger.info(
            "process_audio_chunk: speech detected; buffer_frames=%d buffer_samples=%d",
            len(state["audio_buffer"]),
            sum(s.shape[0] for s in state["audio_buffer"]),
        )

        # Show listening indicator
        listening_html = _live_translations_display_html(state["utterances"])
        return state, listening_html, "Listening...", None

    if state.get("speech_detected", False):
        state["silence_frames"] = state.get("silence_frames", 0) + 1
        state["silence_seconds"] = state.get("silence_seconds", 0.0) + chunk_duration
        state["audio_buffer"].append(samples)
        logger.info(
            "process_audio_chunk: continuing silence; silence_frames=%d silence_seconds=%.3f buffer_frames=%d",
            state["silence_frames"],
            state["silence_seconds"],
            len(state["audio_buffer"]),
        )

        if (
            state["silence_frames"] >= _SILENCE_FRAMES_TO_END
            or state["silence_seconds"] >= _SILENCE_SECONDS_TO_END
        ):
            utterance = np.concatenate(state["audio_buffer"])
            duration = len(utterance) / _SAMPLE_RATE

            # Reset VAD
            state["audio_buffer"] = []
            state["speech_detected"] = False
            state["silence_frames"] = 0
            state["silence_seconds"] = 0.0

            logger.info(
                "process_audio_chunk: ending utterance; duration=%.3f seconds buffer_frames=%d",
                duration,
                len(state["audio_buffer"]),
            )

            if duration < _MIN_UTTERANCE_SECONDS:
                return state, _live_translations_display_html(state["utterances"]), "", None

            state["utterance_count"] = state.get("utterance_count", 0) + 1
            
            # Process utterance
            utterance_data, status, audio_path = _stt_and_translate(
                utterance,
                target_language,
                config_id,
                utterance_num=state["utterance_count"],
                tts_enabled=tts_enabled,
                source_language_override=source_language_override,
            )
            logger.info(
                "process_audio_chunk: stt returned utterance_data=%s status=%s audio_path=%s",
                bool(utterance_data),
                status,
                audio_path,
            )
            
            if utterance_data:
                state["utterances"].append(utterance_data)
                # Keep only last 10 utterances for display
                if len(state["utterances"]) > 10:
                    state["utterances"] = state["utterances"][-10:]
            else:
                logger.info("process_audio_chunk: no utterance_data included after STT")
            
            display_html = _live_translations_display_html(state["utterances"])
            return state, display_html, status, audio_path

        # Still collecting silence
        listening_html = _live_translations_display_html(state["utterances"])
        return state, listening_html, "", None

    return state, _live_translations_display_html(state.get("utterances", [])), "", None


# ---------------------------------------------------------------------------
# Rendering functions
# ---------------------------------------------------------------------------

def _live_translations_display_html(utterances: list) -> str:
    """
    Live tab feed: translated lines only (escaped). Listening state is shown on the mic caption, not here, so it cannot get out of sync when you stop.
    """
    parts: list[str] = []
    for u in utterances:
        t = (u.get("translation") or "").strip()
        if t:
            safe = html.escape(t)
            parts.append(
                '<p style="margin:0 0 12px 0;font-size:clamp(14px,3.8vw,15px);'
                'color:#e5e7eb;line-height:1.55;">'
                f"{safe}</p>"
            )
    if not parts:
        return ""
    return (
        '<div class="live-translations-feed" style="padding:6px 2px 8px 2px;">'
        f'{"".join(parts)}</div>'
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_16k_mono_float32(samples: np.ndarray, src_sr: int) -> np.ndarray:
    arr = samples.astype(np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=1)
    if np.max(np.abs(arr)) > 1.5:
        arr = arr / 32768.0
    if src_sr != _SAMPLE_RATE:
        try:
            import librosa
            arr = librosa.resample(arr, orig_sr=src_sr, target_sr=_SAMPLE_RATE)
        except Exception:
            pass
    return arr


def _stt_and_translate(
    audio: np.ndarray,
    target_language: str,
    config_id: str,
    utterance_num: int = 1,
    tts_enabled: bool = False,
    source_language_override: str | None = None,
) -> tuple[dict | None, str, str | None]:
    """
    Send utterance to STT then translate.

    Returns
    -------
    (utterance_dict, status_message, tts_audio_path_or_none)
    utterance_dict contains: speaker, original, translation, timestamp
    """
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp_path = Path(tmp_name)
    
    try:
        sf.write(str(tmp_path), audio, _SAMPLE_RATE, subtype="PCM_16")

        # --- STT ---
        try:
            from backend.src.services.elevenlabs_stt_service import (  # type: ignore
                elevenlabs_api_key_from_env,
                transcribe_file_scribe_v2,
            )
        except ModuleNotFoundError:
            # Cloud Run backend container exposes services under src.services.
            from src.services.elevenlabs_stt_service import (  # type: ignore
                elevenlabs_api_key_from_env,
                transcribe_file_scribe_v2,
            )

        api_key = elevenlabs_api_key_from_env()
        if not api_key:
            logger.error("_stt_and_translate: ElevenLabs API key not set")
            return None, "API key not set", None

        logger.info(
            "_stt_and_translate: calling ElevenLabs STT; tmp_path=%s model_id=%s diarize=%s",
            tmp_path,
            "scribe_v2",
            False,
        )

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
            logger.info("_stt_and_translate: STT returned empty transcript; chunk=%s", chunk)
            return None, "", None

        # Get source language (STT-detect, unless user chose a fixed language)
        lang_code = getattr(chunk, "language_code", None) or ""
        logger.info(
            "_stt_and_translate: STT transcript=%s language_code=%s",
            transcript,
            lang_code,
        )
        from demo.audio_analysis_pipeline import scribe_language_code_for_translation

        detected = scribe_language_code_for_translation(lang_code)
        ov = (source_language_override or "").strip().lower()
        if ov and ov != "auto":
            source_lang = ov
        else:
            source_lang = detected

        # --- Translate ---
        try:
            from backend.src.services.gemini_translation import translate_text  # type: ignore
        except ModuleNotFoundError:
            from src.services.gemini_translation import translate_text  # type: ignore
        
        translated = translate_text(
            source_text=transcript,
            source_language=source_lang,
            target_language=target_language,
            config_id=config_id,
        )

        # Assign speaker role (rotate for demo, or use "Speaker" generically)
        speaker = _SPEAKER_ROLES[(utterance_num - 1) % len(_SPEAKER_ROLES)]
        
        # Create timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Build utterance data
        utterance_data = {
            "speaker": speaker,
            "original": transcript,
            "translation": translated,
            "timestamp": timestamp,
            "source_lang": source_lang,
            "target_lang": target_language,
        }

        lang_label = _LANG_LABELS.get(target_language, target_language)
        status = f"Translated to {lang_label}"

        # --- TTS ---
        audio_path: str | None = None
        tts_diag: str | None = None
        if tts_enabled and translated:
            try:
                from demo.audio_analysis_pipeline import synthesize_speech_mp3

                mp3_bytes, tts_err = synthesize_speech_mp3(translated)
                if mp3_bytes:
                    tfd, tpath = tempfile.mkstemp(suffix=".mp3")
                    os.close(tfd)
                    Path(tpath).write_bytes(mp3_bytes)
                    audio_path = tpath
                elif tts_err:
                    logger.warning("TTS not generated: %s", tts_err)
                    tts_diag = tts_err
            except Exception as exc:
                logger.exception("TTS failed with exception")
                tts_diag = str(exc) or "TTS failed"

        if tts_enabled and translated and audio_path is None:
            hint = ""
            if tts_diag:
                d = tts_diag.strip()
                if len(d) > 100:
                    d = d[:97] + "..."
                hint = f" — {d}"
            status = f"{status} (spoken output unavailable{hint})"

        return utterance_data, status, audio_path

    except Exception as e:
        return None, f"Error: {e}", None
    finally:
        tmp_path.unlink(missing_ok=True)