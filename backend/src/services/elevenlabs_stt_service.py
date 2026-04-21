"""
ElevenLabs Speech-to-Text (Scribe v2) for pipeline and UI transcription.

Uses the official ElevenLabs SDK. Requires ELEVENLABS_API_KEY.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("iikshana.elevenlabs_stt_service")

DEFAULT_STT_MODEL = "scribe_v2"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".flac", ".webm")


def _strip_secret_value(raw: str) -> str:
    s = str(raw).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def elevenlabs_api_key_from_env() -> str:
    """
    Resolve the ElevenLabs API key from the environment.

    Checks ``ELEVENLABS_API_KEY`` then legacy ``XI_API_KEY``. Strips whitespace and a single pair
    of surrounding quotes (a common ``.env`` mistake).
    """
    for name in ("ELEVENLABS_API_KEY", "XI_API_KEY"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        s = _strip_secret_value(str(raw))
        if s:
            logger.debug("Resolved ElevenLabs API key from environment variable: %s", name)
            return s
    logger.debug("No ElevenLabs API key found in environment")
    return ""


def _normalize_scribe_chunk(transcription: Any) -> Any:
    chunk = transcription
    if hasattr(transcription, "transcripts") and transcription.transcripts:
        logger.debug("Normalizing Scribe v2 transcription chunk from transcripts list")
        chunk = transcription.transcripts[0]
    return chunk


def _get_client(api_key: str) -> Any:
    from elevenlabs.client import ElevenLabs  # noqa: PLC0415

    logger.debug("Creating ElevenLabs client instance")
    return ElevenLabs(api_key=api_key)


def transcribe_file_scribe_v2(
    audio_path: Path | str,
    *,
    api_key: Optional[str] = None,
    model_id: str = DEFAULT_STT_MODEL,
    diarize: bool = True,
    tag_audio_events: bool = True,
    timestamps_granularity: str = "word",
) -> Any:
    """
    Run Scribe v2 on a file; returns the primary chunk (``text``, ``words``, ``language_code``).
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    key = ""
    if api_key is not None and str(api_key).strip():
        key = _strip_secret_value(str(api_key))
    if not key:
        key = elevenlabs_api_key_from_env()
    if not key:
        raise ValueError(
            "ELEVENLABS_API_KEY is not set. Set it in the environment or pass api_key."
        )

    client = _get_client(key)
    logger.info(
        "transcribe_file_scribe_v2: audio_path=%s model_id=%s diarize=%s tag_audio_events=%s timestamps=%s",
        path,
        model_id,
        diarize,
        tag_audio_events,
        timestamps_granularity,
    )
    with path.open("rb") as audio_f:
        transcription = client.speech_to_text.convert(
            file=audio_f,
            model_id=model_id,
            diarize=diarize,
            tag_audio_events=tag_audio_events,
            timestamps_granularity=timestamps_granularity,
        )
    chunk = _normalize_scribe_chunk(transcription)
    logger.debug("transcribe_file_scribe_v2: received transcription chunk %s", repr(chunk))
    return chunk


def transcribe_audio(
    audio_path: Path | str,
    *,
    model: str = DEFAULT_STT_MODEL,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> str:
    """
    Transcribe audio to plain text (batch / eval). Uses Scribe v2 without diarization extras.

    ``timeout`` is accepted for API compatibility; the ElevenLabs SDK uses its own HTTP timeouts.
    """
    _ = timeout
    logger.debug("transcribe_audio: audio_path=%s model=%s timeout=%s", audio_path, model, timeout)
    chunk = transcribe_file_scribe_v2(
        audio_path,
        api_key=api_key,
        model_id=model,
        diarize=False,
        tag_audio_events=False,
        timestamps_granularity="word",
    )
    transcript = (getattr(chunk, "text", None) or "").strip()
    logger.debug("transcribe_audio: transcript length=%d", len(transcript))
    return transcript
