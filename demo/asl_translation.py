"""
ASL translation module for IIKSHANA demo.

Converts English text to ASL gloss notation via Groq LLM, then provides
fingerspelling data for display. Called from the Sign Language tab in
gradio_expo_app.py.
"""
from __future__ import annotations

import html
import logging
import os
import re
import tempfile
from typing import Any

import numpy as np
import requests
import soundfile as sf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VAD tuning (ASL tab only — faster end-of-utterance than demo/live_translation.py)
# ---------------------------------------------------------------------------
_ENERGY_THRESHOLD = 0.008
_SILENCE_FRAMES_TO_END = 4
_MIN_UTTERANCE_SECONDS = 0.3
_SAMPLE_RATE = 16_000
_SILENCE_SECONDS_TO_END = 2.0

_MAX_HISTORY_UTTERANCES = 10


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


def make_initial_asl_state() -> dict[str, Any]:
    return {
        "audio_buffer": [],
        "silence_frames": 0,
        "silence_seconds": 0.0,
        "speech_detected": False,
        "asl_history": [],
        "utterance_count": 0,
    }


def transcribe_with_groq_whisper(
    audio_samples: np.ndarray,
    sample_rate: int = 16_000,
    *,
    language: str | None = None,
) -> str:
    """Transcribe audio using Groq's Whisper API. Returns transcript text."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return ""

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        sf.write(tmp_path, audio_samples, sample_rate, subtype="PCM_16")

        data: dict[str, str] = {
            "model": "whisper-large-v3",
            "response_format": "text",
        }
        if language and language.lower() != "auto":
            data["language"] = language.lower()

        with open(tmp_path, "rb") as f:
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data=data,
                timeout=30,
            )
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        logger.warning("Groq Whisper STT failed: %s", e)
        return ""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def english_to_asl_gloss(english_text: str) -> str:
    """Convert transcript text to ASL gloss via Groq chat model."""
    text = (english_text or "").strip()
    if not text:
        return ""

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return re.sub(r"\s+", " ", text).upper()

    prompt = (
        "Convert the following English sentence to ASL gloss notation.\n"
        "Rules:\n"
        "- Remove articles (a, an, the)\n"
        "- Remove \"be\" verbs where possible\n"
        "- Use CAPS for all gloss words\n"
        "- Reorder to topic-comment structure where appropriate\n"
        "- Keep proper nouns as-is in CAPS\n"
        "- For legal terms, keep the English word in CAPS\n"
        "- Return ONLY the gloss words separated by spaces, nothing else\n\n"
        f"English: {text}\n"
        "ASL Gloss:"
    )

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an ASL gloss translator. Convert English to ASL gloss notation. "
                            "Rules: Remove articles (a, an, the). Remove 'be' verbs where possible. "
                            "Use ALL CAPS. Reorder to topic-comment structure. "
                            "Keep legal terms in CAPS as-is. "
                            "Return ONLY gloss words separated by spaces. No explanations."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.2,
            },
            timeout=15,
        )
        response.raise_for_status()
        gloss = response.json()["choices"][0]["message"]["content"].strip()
        gloss = gloss.strip("\"'")
        gloss = re.sub(r"^asl\s+gloss\s*:\s*", "", gloss, flags=re.IGNORECASE).strip()
        gloss = gloss.splitlines()[0].strip()
        gloss = re.sub(r"\s+", " ", gloss).upper()
        return gloss or re.sub(r"\s+", " ", text).upper()
    except Exception as e:
        logger.warning("ASL gloss translation failed: %s", e)
        return re.sub(r"\s+", " ", text).upper()


def gloss_to_fingerspell_data(gloss: str) -> list[dict[str, Any]]:
    """Split gloss into words; each word maps to a list of letter characters."""
    raw = (gloss or "").strip()
    if not raw:
        return []
    words = [w for w in raw.replace("\n", " ").split() if w]
    out: list[dict[str, Any]] = []
    for w in words:
        letters = [c for c in w.upper() if c.isalnum()]
        out.append({"word": w.upper(), "letters": letters})
    return out


def render_asl_html(asl_history: list[dict[str, Any]]) -> str:
    """Generate HTML for ASL fingerspelling display with staggered CSS pulse animation."""
    if not asl_history:
        return """<div style='text-align:center; padding:40px; color:#888;'>
            🤟 Start speaking to see ASL signs...
        </div>"""

    style = """
    <style>
    @keyframes asl-pulse {
        0% { transform: scale(1); background: rgba(0,200,150,0.15); border-color: rgba(0,200,150,0.3); }
        50% { transform: scale(1.2); background: rgba(0,200,150,0.4); border-color: rgba(0,200,150,0.8); }
        100% { transform: scale(1); background: rgba(0,200,150,0.15); border-color: rgba(0,200,150,0.3); }
    }
    .asl-letter {
        display: inline-flex; flex-direction: column; align-items: center;
        margin: 3px; padding: 10px 14px; background: rgba(0,200,150,0.15);
        border: 2px solid rgba(0,200,150,0.3); border-radius: 12px; min-width: 44px;
        animation: asl-pulse 0.6s ease-in-out 1;
        animation-fill-mode: both;
    }
    .asl-letter span.hand { font-size: 28px; }
    .asl-letter span.char { font-size: 20px; font-weight: bold; color: #fff; margin-top: 4px; }
    .asl-word-group { margin: 10px 0; }
    .asl-word-label { font-size: 12px; color: #888; margin-bottom: 4px; font-weight: 600; letter-spacing: 1px; }
    .asl-utterance {
        margin: 15px 0; padding: 12px;
        background: rgba(255,255,255,0.03); border-radius: 10px;
        border-left: 3px solid rgba(0,200,150,0.5);
    }
    .asl-english { font-size: 13px; color: #aaa; margin-bottom: 6px; }
    .asl-gloss {
        font-size: 18px; font-weight: bold; color: #00c896;
        letter-spacing: 2px; margin-bottom: 12px; padding: 8px;
        background: rgba(0,200,150,0.08); border-radius: 8px;
    }
    </style>
    """

    html_parts: list[str] = [style]
    delay_counter = 0

    for utterance in asl_history:
        html_parts.append("<div class='asl-utterance'>")

        english_escaped = html.escape((utterance.get("english") or "").strip())
        html_parts.append(f"<div class='asl-english'>🗣️ {english_escaped}</div>")

        gloss_escaped = html.escape((utterance.get("gloss") or "").strip())
        html_parts.append(f"<div class='asl-gloss'>🤟 {gloss_escaped}</div>")

        for word_data in utterance.get("words") or []:
            word = html.escape(str(word_data.get("word") or ""))
            letters = word_data.get("letters") or []

            html_parts.append("<div class='asl-word-group'>")
            html_parts.append(f"<div class='asl-word-label'>{word}</div>")
            html_parts.append("<div style='display:flex; flex-wrap:wrap; gap:2px;'>")

            for letter in letters:
                delay = delay_counter * 0.15
                letter_escaped = html.escape(str(letter))
                html_parts.append(
                    f"<div class='asl-letter' style='animation-delay:{delay:.2f}s;'>"
                    "<span class='hand'>🤟</span>"
                    f"<span class='char'>{letter_escaped}</span>"
                    "</div>"
                )
                delay_counter += 1

            html_parts.append("</div></div>")
            delay_counter += 2

        html_parts.append("</div>")

    return f"<div style='max-height:500px; overflow-y:auto; padding:10px;'>{''.join(html_parts)}</div>"


def _accum_lines(history: list[dict[str, Any]], key: str) -> str:
    return "\n".join((h.get(key) or "").strip() for h in history if (h.get(key) or "").strip())


def process_asl_chunk(
    chunk_data: Any,
    state: dict[str, Any],
    source_language_override: str | None = None,
) -> tuple[dict[str, Any], str, str, str]:
    """
    Streaming handler for the Sign Language tab microphone (VAD + Groq Whisper + gloss).

    VAD mirrors ``demo/live_translation.process_audio_chunk`` (same RMS gate, silence
    accumulation while ``speech_detected``, end-of-utterance on ``_SILENCE_FRAMES_TO_END``).

    Returns
    -------
    (updated_state, english_transcript_block, asl_gloss_block, asl_html)
    """
    logger.info("[ASL PIPELINE] chunk received")
    history: list[dict[str, Any]] = list(state.get("asl_history", []))

    def outputs(s: dict[str, Any]) -> tuple[dict[str, Any], str, str, str]:
        h = list(s.get("asl_history", []))
        return s, _accum_lines(h, "english"), _accum_lines(h, "gloss"), render_asl_html(h)

    if chunk_data is None:
        return outputs(state)

    if not (isinstance(chunk_data, (tuple, list)) and len(chunk_data) == 2):
        return outputs(state)

    src_sr, raw_samples = chunk_data
    if raw_samples is None:
        return outputs(state)

    samples = _to_16k_mono_float32(np.asarray(raw_samples), int(src_sr))
    if samples.size == 0:
        return outputs(state)
    chunk_duration = len(samples) / _SAMPLE_RATE

    # Same energy gate as live_translation.process_audio_chunk
    rms = float(np.sqrt(np.mean(samples ** 2)))
    is_speech = rms > _ENERGY_THRESHOLD

    state = dict(state)
    state["audio_buffer"] = list(state.get("audio_buffer", []))
    state["asl_history"] = list(state.get("asl_history", []))
    state["silence_seconds"] = float(state.get("silence_seconds", 0.0))

    silence_frames = int(state.get("silence_frames", 0))
    silence_seconds = float(state.get("silence_seconds", 0.0))
    speech_detected = bool(state.get("speech_detected", False))
    buf_len = len(state["audio_buffer"])
    logger.info(
        "[ASL PIPELINE] vad energy=%.6f threshold=%s silence_frames=%s/%s silence_seconds=%.2f/%.2f speech=%s buffer=%s is_speech=%s",
        rms,
        _ENERGY_THRESHOLD,
        silence_frames,
        _SILENCE_FRAMES_TO_END,
        silence_seconds,
        _SILENCE_SECONDS_TO_END,
        speech_detected,
        buf_len,
        is_speech,
    )

    lang_for_whisper: str | None = None
    ov = (source_language_override or "").strip().lower()
    if ov and ov != "auto":
        lang_for_whisper = ov

    if is_speech:
        state["speech_detected"] = True
        state["silence_frames"] = 0
        state["silence_seconds"] = 0.0
        state["audio_buffer"].append(samples)
        return outputs(state)

    if state.get("speech_detected", False):
        state["silence_frames"] = state.get("silence_frames", 0) + 1
        state["silence_seconds"] = state.get("silence_seconds", 0.0) + chunk_duration
        state["audio_buffer"].append(samples)

        if state["silence_seconds"] >= _SILENCE_SECONDS_TO_END:
            utterance = np.concatenate(state["audio_buffer"])
            duration = len(utterance) / _SAMPLE_RATE

            # Reset VAD (same order as live_translation)
            state["audio_buffer"] = []
            state["speech_detected"] = False
            state["silence_frames"] = 0
            state["silence_seconds"] = 0.0

            if duration < _MIN_UTTERANCE_SECONDS:
                logger.info(
                    "[ASL PIPELINE] end-of-utterance too short (%.2fs < %.2fs), discarding",
                    duration,
                    _MIN_UTTERANCE_SECONDS,
                )
                return outputs(state)

            logger.info(
                "[ASL PIPELINE] end-of-utterance after %.2fs silence, duration=%.2fs (%d samples), running Whisper",
                _SILENCE_SECONDS_TO_END,
                duration,
                utterance.size,
            )
            logger.info("[ASL PIPELINE] before Groq Whisper STT call")
            transcript = transcribe_with_groq_whisper(utterance, _SAMPLE_RATE, language=lang_for_whisper)
            transcript = (transcript or "").strip()
            if not transcript:
                logger.info("[ASL PIPELINE] Whisper returned empty transcript")
                return outputs(state)
            transcript_lc = transcript.lower()
            blocked_transcripts = {
                "thank you",
                "thank you.",
                "thank you for watching",
                "thank you for watching.",
                "bye",
                "bye.",
                "thanks",
                "thanks.",
            }
            if transcript_lc in blocked_transcripts:
                logger.info("[ASL PIPELINE] transcript discarded by blocklist: %s", transcript)
                return outputs(state)
            alpha_word_count = sum(1 for w in transcript.split() if re.search(r"[A-Za-z]", w))
            if alpha_word_count < 2:
                logger.info(
                    "[ASL PIPELINE] transcript discarded: fewer than 2 alphabetic words (%d): %s",
                    alpha_word_count,
                    transcript,
                )
                return outputs(state)
            logger.info("[ASL PIPELINE] transcript returned: %s", transcript)

            gloss = english_to_asl_gloss(transcript)
            logger.info("[ASL PIPELINE] gloss returned: %s", gloss)
            words = gloss_to_fingerspell_data(gloss)

            state["utterance_count"] = state.get("utterance_count", 0) + 1
            state["asl_history"].append(
                {
                    "english": transcript,
                    "gloss": gloss,
                    "words": words,
                }
            )
            if len(state["asl_history"]) > _MAX_HISTORY_UTTERANCES:
                state["asl_history"] = state["asl_history"][-_MAX_HISTORY_UTTERANCES:]

            logger.info(
                "[ASL PIPELINE] utterance #%s transcribed (%d chars)",
                state["utterance_count"],
                len(transcript),
            )
            return outputs(state)

        return outputs(state)

    return outputs(state)
