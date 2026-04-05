"""
Groq Speech-to-Text (Whisper) client for transcribing audio to text.

Uses the Groq OpenAI-compatible transcriptions endpoint. Requires GROQ_API_KEY.
Supports word-level timestamps for speaker diarization alignment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import requests

TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_STT_MODEL = os.environ.get("STT_MODEL", "whisper-large-v3-turbo")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".flac", ".webm")


@dataclass
class WordTimestamp:
    """
    A single word with its start and end time in seconds.
    Used to align transcription words with diarization speaker segments.
    """
    word: str        # The transcribed word
    start: float     # When this word started (seconds)
    end: float       # When this word ended (seconds)


@dataclass
class TimestampedTranscript:
    """
    Full transcription result with word-level timestamps.
    Contains both the full text and individual word timings
    so we can align each word to a speaker segment.
    """
    text: str                    # Full transcribed text
    words: List[WordTimestamp]   # Individual words with timestamps
    language: str = "en"         # Detected language


def transcribe_audio(
    audio_path: Path | str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    language: Optional[str] = None,
    timeout: int = 60,
) -> TimestampedTranscript:
    """
    Transcribe audio file to text with word-level timestamps using Groq Whisper.
    Returns a TimestampedTranscript so words can be aligned to speakers.

    Parameters
    ----------
    audio_path : path or str
        Path to the audio file (WAV, MP3, etc.).
    model : str, optional
        Groq Whisper model id. Defaults to STT_MODEL env var or whisper-large-v3-turbo.
    api_key : str, optional
        Groq API key. If not set, uses GROQ_API_KEY env var.
    language : str, optional
        ISO-639-1 language code (e.g. "en") to improve accuracy.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    TimestampedTranscript
        Full transcription with word-level timestamps.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Set it in the environment or pass api_key."
        )

    stt_model = model or DEFAULT_STT_MODEL

    with path.open("rb") as f:
        files = {
            "file": (
                path.name,
                f,
                "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream"
            )
        }
        # Request verbose_json with word-level timestamps
        # so we can align each word to a diarization speaker segment
        data: dict = {
            "model": stt_model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        if language:
            data["language"] = language

        response = requests.post(
            TRANSCRIPTIONS_URL,
            headers={"Authorization": f"Bearer {key}"},
            files=files,
            data=data,
            timeout=timeout,
        )
    response.raise_for_status()

    result = response.json()

    # Parse word-level timestamps from the response
    words = []
    for w in result.get("words", []):
        words.append(WordTimestamp(
            word=w.get("word", ""),
            start=float(w.get("start", 0.0)),
            end=float(w.get("end", 0.0)),
        ))

    return TimestampedTranscript(
        text=(result.get("text") or "").strip(),
        words=words,
        language=result.get("language", language or "en"),
    )