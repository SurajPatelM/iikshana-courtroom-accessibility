"""
Groq Speech-to-Text (Whisper) client for transcribing audio to text.

Uses the Groq OpenAI-compatible transcriptions endpoint. Requires GROQ_API_KEY.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".flac", ".webm")


def transcribe_audio(
    audio_path: Path | str,
    *,
    model: str = DEFAULT_STT_MODEL,
    api_key: Optional[str] = None,
    language: Optional[str] = None,
    response_format: str = "text",
    timeout: int = 60,
) -> str:
    """
    Transcribe audio file to text using Groq Whisper.

    Parameters
    ----------
    audio_path : path or str
        Path to the audio file (WAV, MP3, etc.).
    model : str
        Groq Whisper model id (e.g. whisper-large-v3-turbo, whisper-large-v3).
    api_key : str, optional
        Groq API key. If not set, uses GROQ_API_KEY env var.
    language : str, optional
        ISO-639-1 language code (e.g. en) to improve accuracy.
    response_format : str
        "text" for plain text, "json" or "verbose_json" for structured.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    str
        Transcribed text (or JSON string if response_format is json).
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "GROQ_API_KEY is not set. Set it in the environment or pass api_key."
        )

    with path.open("rb") as f:
        files = {"file": (path.name, f, "audio/wav" if path.suffix.lower() == ".wav" else "application/octet-stream")}
        data: dict = {"model": model, "response_format": response_format}
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

    if response_format == "text":
        return (response.text or "").strip()
    return response.json()
