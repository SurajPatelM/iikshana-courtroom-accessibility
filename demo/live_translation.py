"""
Live translation module for IIKSHANA demo.

Energy-based VAD gates audio into utterances. Each complete utterance is
sent to ElevenLabs Scribe v2 (STT) then translated via the configured model.
Optionally synthesises TTS audio for the translated text (headphones mode).

Called from gradio_expo_app.py via the streaming audio .stream() event.
"""
from __future__ import annotations

import html
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# VAD tuning constants
# ---------------------------------------------------------------------------
_ENERGY_THRESHOLD = 0.008
_SILENCE_FRAMES_TO_END = 6
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
        return state, _live_translations_display_html(state.get("utterances", [])), "", None

    samples = _to_16k_mono_float32(np.asarray(raw_samples), int(src_sr))
    if samples.size == 0:
        return state, _live_translations_display_html(state.get("utterances", [])), "", None

    rms = float(np.sqrt(np.mean(samples ** 2)))
    is_speech = rms > _ENERGY_THRESHOLD

    # Shallow copy state
    state = dict(state)
    state["audio_buffer"] = list(state.get("audio_buffer", []))
    state["utterances"] = list(state.get("utterances", []))

    if is_speech:
        state["speech_detected"] = True
        state["silence_frames"] = 0
        state["audio_buffer"].append(samples)
        
        # Show listening indicator
        listening_html = _live_translations_display_html(state["utterances"], listening=True)
        return state, listening_html, "Listening...", None

    if state.get("speech_detected", False):
        state["silence_frames"] = state.get("silence_frames", 0) + 1
        state["audio_buffer"].append(samples)

        if state["silence_frames"] >= _SILENCE_FRAMES_TO_END:
            utterance = np.concatenate(state["audio_buffer"])
            duration = len(utterance) / _SAMPLE_RATE

            # Reset VAD
            state["audio_buffer"] = []
            state["speech_detected"] = False
            state["silence_frames"] = 0

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
            
            if utterance_data:
                state["utterances"].append(utterance_data)
                # Keep only last 10 utterances for display
                if len(state["utterances"]) > 10:
                    state["utterances"] = state["utterances"][-10:]
            
            display_html = _live_translations_display_html(state["utterances"])
            return state, display_html, status, audio_path

        # Still collecting silence
        listening_html = _live_translations_display_html(state["utterances"], listening=True)
        return state, listening_html, "", None

    return state, _live_translations_display_html(state.get("utterances", [])), "", None


# ---------------------------------------------------------------------------
# Rendering functions
# ---------------------------------------------------------------------------

def _live_translations_display_html(utterances: list, listening: bool = False) -> str:
    """
    Live tab feed: show translated lines only (user/model text escaped).
    Optional one-line “Listening…” while the VAD is open (no raw markup dump).
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
    if listening:
        parts.append('<p style="margin:0;font-size:13px;color:#10b981;">Listening…</p>')
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
        from backend.src.services.elevenlabs_stt_service import (
            elevenlabs_api_key_from_env,
            transcribe_file_scribe_v2,
        )

        api_key = elevenlabs_api_key_from_env()
        if not api_key:
            return None, "API key not set", None

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
            return None, "", None

        # Get source language (STT-detect, unless user chose a fixed language)
        lang_code = getattr(chunk, "language_code", None) or ""
        from demo.audio_analysis_pipeline import scribe_language_code_for_translation

        detected = scribe_language_code_for_translation(lang_code)
        ov = (source_language_override or "").strip().lower()
        if ov and ov != "auto":
            source_lang = ov
        else:
            source_lang = detected

        # --- Translate ---
        from backend.src.services.gemini_translation import translate_text
        
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
        if tts_enabled and translated:
            try:
                from demo.audio_analysis_pipeline import synthesize_speech_mp3
                mp3_bytes, tts_err = synthesize_speech_mp3(translated)
                if mp3_bytes:
                    tfd, tpath = tempfile.mkstemp(suffix=".mp3")
                    os.close(tfd)
                    Path(tpath).write_bytes(mp3_bytes)
                    audio_path = tpath
            except Exception:
                pass  # Silent fail for TTS

        return utterance_data, status, audio_path

    except Exception as e:
        return None, f"Error: {e}", None
    finally:
        tmp_path.unlink(missing_ok=True)