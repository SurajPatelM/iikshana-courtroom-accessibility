"""
WebSocket handler for real-time audio streaming and transcript delivery.
Wires Aditya's frontend (useWebSocket.ts) to the RealtimeAudioSession service.

Key design decision:
    The browser's MediaRecorder sends compressed audio (webm/opus), NOT raw PCM.
    We bypass numpy conversion entirely and send raw bytes directly to ElevenLabs,
    which handles all audio format decoding server-side.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from ..models.enums import SessionState
from ..models.schemas import (
    TranscriptSegment,
    WSStatusUpdate,
    WSConfigMessage,
)
from ..services.realtime_audio_service import RealtimeAudioSession
from ..services.gemini_translation import translate_text
from ..services.elevenlabs_stt_service import transcribe_file_scribe_v2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Active session registry
# ---------------------------------------------------------------------------
_active_sessions: dict[str, RealtimeAudioSession] = {}


def get_active_session_count() -> int:
    """Returns the number of currently active WebSocket sessions."""
    return len(_active_sessions)


async def _send_status(websocket: WebSocket, state: SessionState, message: str = "") -> None:
    """Helper to send a status update to the frontend."""
    update = WSStatusUpdate(state=state, message=message)
    await websocket.send_text(update.model_dump_json())


async def _send_transcript(
    websocket: WebSocket,
    text: str,
    words: list,
    language_code: str,
    translation: Optional[str],
) -> None:
    """
    Send a transcript segment to the frontend.
    Extracts speaker_id and start_time from speech words only.
    Matches TranscriptSegment shape in Aditya's types/index.ts.
    """
    speaker_id = "speaker_0"
    start_time = 0.0

    # Filter to speech words only — exclude audio events like [static]
    speech_words = [
        w for w in words
        if getattr(w, "type", "") != "audio_event"
    ]

    if speech_words:
        first = speech_words[0]
        speaker_id = getattr(first, "speaker_id", None) or "speaker_0"
        start_time = float(getattr(first, "start", 0.0) or 0.0)

    segment = TranscriptSegment(
        speaker_id=speaker_id,
        text=text,
        translated_text=translation,
        start_time=start_time,
    )
    await websocket.send_text(segment.model_dump_json())


def _transcribe_raw_bytes(
    audio_bytes: bytes,
    api_key: Optional[str],
    diarize: bool,
) -> Optional[object]:
    """
    Write raw audio bytes (webm/opus from browser MediaRecorder) to a temp file
    and send directly to ElevenLabs. Bypasses numpy conversion entirely since
    browser audio is compressed, not raw PCM.

    Returns the ElevenLabs transcription object or None on failure.
    """
    if not audio_bytes:
        return None

    # Write to temp file — ElevenLabs accepts webm, opus, mp4, wav, etc.
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as fh:
        fh.write(audio_bytes)
        tmp_path = Path(fh.name)

    try:
        result = transcribe_file_scribe_v2(
            tmp_path,
            api_key=api_key,
            diarize=diarize,
            tag_audio_events=True,
            timestamps_granularity="word",
        )
        return result
    except Exception as exc:
        logger.error("STT error: %s", exc)
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


async def audio_websocket(websocket: WebSocket) -> None:
    """
    Main WebSocket handler registered at /ws/audio.

    Flow:
    1. Accept connection
    2. Wait for {type: "config"} JSON message from frontend
    3. Create RealtimeAudioSession for noise profile tracking
    4. Loop: receive {type: "audio", data: "<base64>"} frames
    5. Buffer raw bytes until MIN_BUFFER_BYTES threshold is reached
    6. Send buffered bytes directly to ElevenLabs (bypassing numpy)
    7. Translate result, send TranscriptSegment to frontend
    8. On disconnect: clean up session
    """
    await websocket.accept()
    session: Optional[RealtimeAudioSession] = None
    session_id = str(uuid.uuid4())

    try:
        # Step 1: Wait for config message
        await _send_status(websocket, SessionState.CREATED, "Waiting for session config")

        raw = await websocket.receive_text()
        msg = json.loads(raw)

        if msg.get("type") != "config":
            await _send_status(websocket, SessionState.ERROR, "First message must be type 'config'")
            await websocket.close()
            return

        # Parse config
        ws_config = WSConfigMessage(**msg)
        config = ws_config.config

        # Step 2: Validate ElevenLabs API key
        from ..services.elevenlabs_stt_service import elevenlabs_api_key_from_env
        api_key = elevenlabs_api_key_from_env()
        if not api_key:
            await _send_status(websocket, SessionState.ERROR, "ELEVENLABS_API_KEY is not set")
            await websocket.close()
            return

        # Step 3: Create session (used for session tracking and health reporting)
        session = RealtimeAudioSession(
            session_id=session_id,
            diarize=config.speaker_diarization,
        )
        _active_sessions[session_id] = session
        await _send_status(websocket, SessionState.STREAMING, "Session ready — send audio chunks")
        logger.info("Session %s started (diarize=%s)", session_id, config.speaker_diarization)

        # Step 4: Audio receive loop
        # Buffer raw compressed bytes from browser MediaRecorder (webm/opus).
        # MIN_BUFFER_BYTES controls how much audio to accumulate before sending to ElevenLabs.
        # Default 100000 bytes ≈ ~3-5 seconds of webm/opus audio.
        min_buffer_bytes = int(os.environ.get("MIN_BUFFER_BYTES", "100000"))
        byte_buffer: list[bytes] = []

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio":
                b64_data = msg.get("data", "")
                if not b64_data:
                    continue

                # Decode base64 → raw compressed audio bytes
                audio_bytes = base64.b64decode(b64_data)
                if not audio_bytes:
                    continue

                # Buffer raw bytes — no numpy conversion needed
                byte_buffer.append(audio_bytes)
                total_bytes = sum(len(b) for b in byte_buffer)

                # Wait until we have enough audio
                if total_bytes < min_buffer_bytes:
                    logger.debug(
                        "Session %s buffering %d / %d bytes",
                        session_id, total_bytes, min_buffer_bytes
                    )
                    continue

                # Concatenate and clear buffer
                combined_bytes = b"".join(byte_buffer)
                byte_buffer.clear()

                logger.info(
                    "Session %s sending %d bytes to ElevenLabs",
                    session_id, len(combined_bytes)
                )

                # Send directly to ElevenLabs in thread pool
                result = await asyncio.to_thread(
                    _transcribe_raw_bytes,
                    combined_bytes,
                    api_key,
                    config.speaker_diarization,
                )

                if result is None:
                    continue

                # Skip empty or pure audio event responses
                clean_text = (result.text or "").strip()
                if not clean_text or clean_text.startswith("["):
                    logger.info("Session %s skipping audio event: %s", session_id, clean_text)
                    continue

                # Debug logging
                logger.info("Session %s STT text: %s", session_id, clean_text)
                words = getattr(result, "words", []) or []
                logger.info(
                    "Session %s words sample: %s",
                    session_id, words[:3] if words else "no words"
                )

                # Step 5: Translate
                translation = None
                if clean_text and config.target_language:
                    try:
                        translation = await asyncio.to_thread(
                            translate_text,
                            clean_text,
                            config.source_language or "en",
                            config.target_language,
                            config_id=config.config_id,
                        )
                    except Exception as e:
                        logger.warning("Translation failed for session %s: %s", session_id, e)

                # Step 6: Send to frontend
                await _send_transcript(
                    websocket=websocket,
                    text=clean_text,
                    words=words,
                    language_code=getattr(result, "language_code", "eng"),
                    translation=translation,
                )

            elif msg_type == "stop":
                logger.info("Session %s received stop signal", session_id)
                break

            else:
                logger.warning(
                    "Session %s unknown message type: %s", session_id, msg_type
                )

    except WebSocketDisconnect:
        logger.info("Session %s disconnected", session_id)

    except Exception as e:
        logger.error("Session %s error: %s", session_id, e)
        try:
            await _send_status(websocket, SessionState.ERROR, str(e))
        except Exception:
            pass

    finally:
        if session:
            session.close()
        if session_id in _active_sessions:
            del _active_sessions[session_id]
        logger.info("Session %s cleaned up", session_id)