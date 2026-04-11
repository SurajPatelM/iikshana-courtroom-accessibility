"""
Visual captioning module for IIKSHANA demo.

Captures webcam frames at a configurable interval, sends them to a **local**
Ollama vision model (LLaVA) for courtroom scene description. No API keys or
outbound calls for vision — aligns with on-prem / privacy-first use.

Captions are queued and only surfaced during audio silence periods to avoid
interrupting live translation output.

Called from gradio_expo_app.py; integrates with the live translation state.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = "http://localhost:11434"
_OLLAMA_VISION_MODEL = "llava:7b"
_CAPTURE_INTERVAL_SECONDS = 8.0
_MAX_CAPTION_TOKENS = 100

# Display names for scene caption headers (align with live translation targets).
_SCENE_TARGET_LABELS = {"es": "Spanish", "fr": "French", "de": "German"}


def capture_frame_base64(video_frame: np.ndarray | Any) -> str:
    """
    Convert a numpy frame (or array-like) to base64-encoded JPEG.
    Resizes so the longest edge is at most 512px.
    """
    arr = np.asarray(video_frame)
    if arr.size == 0:
        raise ValueError("empty frame")

    if arr.ndim == 2:
        # grayscale → RGB
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)

    pil = Image.fromarray(arr, mode="RGB")
    w, h = pil.size
    longest = max(w, h)
    if longest > 512:
        scale = 512.0 / float(longest)
        pil = pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_scene_caption(frame_b64: str, previous_caption: str | None) -> str | None:
    """
    Send a frame to local Ollama LLaVA model for scene captioning.
    Ollama exposes an OpenAI-compatible API at localhost:11434.
    """
    try:
        user_text = (
            "You are describing a courtroom scene for a blind participant. "
            "Be concise (1-2 sentences). Describe who is present, their positions, "
            "and any notable actions. If nothing has changed since the previous "
            "description, respond with exactly 'NO_CHANGE'."
            f"\n\nPrevious scene description: {previous_caption or 'None yet'}. "
            "Describe the current scene only if something has changed."
        )

        response = requests.post(
            f"{_OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": _OLLAMA_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_text,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame_b64}",
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": _MAX_CAPTION_TOKENS,
                "temperature": 0.3,
            },
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        caption = str(raw).strip() if raw is not None else ""
        if not caption:
            return None

        cleaned = caption.strip(".'\"` ").upper()
        if cleaned == "NO_CHANGE" or cleaned == "NO CHANGE":
            return None

        return caption

    except requests.exceptions.ConnectionError:
        logger.warning(
            "Ollama not reachable at %s — is 'ollama serve' running?",
            _OLLAMA_BASE_URL,
        )
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Ollama vision caption failed: %s", e)
        return None


@dataclass
class CaptionState:
    last_capture_time: float = 0.0
    previous_caption: str | None = None
    pending_caption: str | None = None
    enabled: bool = False


def maybe_capture_and_caption(
    video_frame: np.ndarray | None,
    caption_state: CaptionState,
) -> CaptionState:
    if video_frame is None or not caption_state.enabled:
        return caption_state

    now = time.time()
    if now - caption_state.last_capture_time < _CAPTURE_INTERVAL_SECONDS:
        return caption_state

    try:
        frame_b64 = capture_frame_base64(video_frame)
    except Exception as e:  # noqa: BLE001
        logger.warning("capture_frame_base64 failed: %s", e)
        return replace(caption_state, last_capture_time=now)

    result = generate_scene_caption(frame_b64, caption_state.previous_caption)
    base = replace(caption_state, last_capture_time=now)
    if result is not None:
        return replace(
            base,
            pending_caption=result,
            previous_caption=result,
        )
    return base


def translate_caption(caption: str, target_language: str, config_id: str) -> str:
    """
    Translate a visual scene caption to the target language using the same
    translation backend as audio translation (Groq/Gemini/HF via config_id).

    Returns translated text, or original caption on any failure.
    """
    tl = (target_language or "").strip().lower()
    if not tl or tl == "en":
        return caption
    cid = (config_id or "").strip() or "translation_flash_v1"
    stripped = (caption or "").strip()
    if not stripped:
        return caption
    try:
        from backend.src.services.gemini_translation import translate_text  # noqa: PLC0415

        out = translate_text(
            source_text=stripped,
            source_language="en",
            target_language=tl,
            config_id=cid,
        )
        done = (out or "").strip()
        return done if done else caption
    except Exception as e:  # noqa: BLE001
        logger.warning("translate_caption failed: %s", e)
        return caption


def flush_pending_caption(
    caption_state: CaptionState,
    target_language: str | None = None,
    config_id: str | None = None,
) -> tuple[str | None, CaptionState]:
    if caption_state.pending_caption is None:
        return None, caption_state
    english = caption_state.pending_caption.strip()
    if not english:
        return None, replace(caption_state, pending_caption=None)

    tl = (target_language or "").strip().lower()
    if not tl or tl == "en":
        formatted = f"\n\n**[👁️ Scene]** {english}\n\n---\n\n"
    else:
        lang_label = _SCENE_TARGET_LABELS.get(tl, (target_language or tl).strip() or tl)
        translated = translate_caption(english, tl, config_id or "translation_flash_v1")
        formatted = (
            f"\n\n**[👁️ Scene → {lang_label}]** {translated}\n\n"
            f"*Original: {english}*\n\n---\n\n"
        )
    return formatted, replace(caption_state, pending_caption=None)
