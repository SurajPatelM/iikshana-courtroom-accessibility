"""
WebSocket handler for real-time audio streaming and transcript delivery.
Wires Aditya's frontend (useWebSocket.ts) to the RealtimeAudioSession service.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import uuid
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from ..models.enums import SessionState
from ..models.schemas import (
    TranscriptSegment,
    WSStatusUpdate,
    WSConfigMessage,
)
from ..services.realtime_audio_service import RealtimeAudioSession
from ..services.gemini_translation import translate_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Active session registry
# Tracks all live WebSocket sessions so /health can report active_sessions count.
# Keys are session_id strings, values are RealtimeAudioSession objects.
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
    Helper to send a transcript segment to the frontend.
    Extracts speaker_id and start_time from the first word in the segment.
    Matches TranscriptSegment shape in Aditya's types/index.ts.
    """
    # Extract speaker_id from first word if available
    speaker_id = "speaker_0"
    start_time = 0.0
    if words:
        first = words[0]
        # ElevenLabs words have speaker_id and start attributes
        speaker_id = getattr(first, "speaker_id", None) or "speaker_0"
        start_time = float(getattr(first, "start", 0.0) or 0.0)

    segment = TranscriptSegment(
        speaker_id=speaker_id,
        text=text,
        translated_text=translation,
        start_time=start_time,
    )
    await websocket.send_text(segment.model_dump_json())


async def audio_websocket(websocket: WebSocket) -> None:
    """
    Main WebSocket handler. Registered at /ws/audio in main.py.

    Flow:
    1. Accept connection
    2. Wait for {type: "config"} message from frontend
    3. Create RealtimeAudioSession
    4. Loop: receive {type: "audio", data: "<base64>"} frames
    5. Decode base64 -> numpy array -> handle_chunk()
    6. If transcription result: optionally translate, send TranscriptSegment
    7. On disconnect: close session, remove from registry
    """
    await websocket.accept()
    session: Optional[RealtimeAudioSession] = None
    session_id = str(uuid.uuid4())
    ws_config = None

    try:
        # Step 1: Wait for config message
        await _send_status(websocket, SessionState.CREATED, "Waiting for session config")

        raw = await websocket.receive_text()
        msg = json.loads(raw)

        if msg.get("type") != "config":
            await _send_status(websocket, SessionState.ERROR, "First message must be type 'config'")
            await websocket.close()
            return

        # Parse the config
        ws_config = WSConfigMessage(**msg)
        config = ws_config.config

        # Step 2: Validate ElevenLabs API key before creating session
        from ..services.elevenlabs_stt_service import elevenlabs_api_key_from_env
        if not elevenlabs_api_key_from_env():
            await _send_status(websocket, SessionState.ERROR, "ELEVENLABS_API_KEY is not set")
            await websocket.close()
            return

        # Step 3: Create session
        session = RealtimeAudioSession(
            session_id=session_id,
            diarize=config.speaker_diarization,
        )
        _active_sessions[session_id] = session
        await _send_status(websocket, SessionState.STREAMING, "Session ready — send audio chunks")
        logger.info("Session %s started (diarize=%s)", session_id, config.speaker_diarization)

        # Step 4: Audio receive loop
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio":
                # Decode base64 audio data sent by frontend (useWebSocket.ts sendAudio)
                b64_data = msg.get("data", "")
                if not b64_data:
                    continue

                audio_bytes = base64.b64decode(b64_data)

                # Guard: int16 requires buffer size to be a multiple of 2 bytes.
                # Browser AudioContext may produce odd-length buffers — trim last byte if needed.
                if len(audio_bytes) % 2 != 0:
                    audio_bytes = audio_bytes[:-1]

                # Skip empty chunks after trimming
                if len(audio_bytes) == 0:
                    continue

                # Convert bytes to int16 numpy array
                chunk = np.frombuffer(audio_bytes, dtype=np.int16)

                # Sample rate from env var — browser default is usually 44100 or 48000
                sample_rate = int(os.environ.get("AUDIO_SAMPLE_RATE", "44100"))

                # Run handle_chunk in thread pool to avoid blocking the event loop
                result = await asyncio.to_thread(
                    session.handle_chunk, chunk, sample_rate
                )

                if result is None:
                    # VAD rejected chunk (silence/noise) — nothing to send
                    continue

                # Step 5: Optionally translate
                translation = None
                if result.text and config.target_language:
                    try:
                        source_lang = config.source_language or "en"
                        translation = await asyncio.to_thread(
                            translate_text,
                            result.text,
                            source_lang,
                            config.target_language,
                            config_id=config.config_id,
                        )
                    except Exception as e:
                        logger.warning("Translation failed for session %s: %s", session_id, e)

                # Step 6: Send transcript to frontend
                words = getattr(result, "words", []) or []
                await _send_transcript(
                    websocket=websocket,
                    text=result.text,
                    words=words,
                    language_code=getattr(result, "language_code", "eng"),
                    translation=translation,
                )

            elif msg_type == "stop":
                logger.info("Session %s received stop signal", session_id)
                break

            else:
                logger.warning(
                    "Session %s received unknown message type: %s",
                    session_id,
                    msg_type,
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
        # Always clean up session on disconnect or error
        if session:
            session.close()
        if session_id in _active_sessions:
            del _active_sessions[session_id]
        logger.info("Session %s cleaned up", session_id)