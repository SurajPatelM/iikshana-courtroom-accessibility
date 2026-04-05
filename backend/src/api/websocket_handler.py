"""
WebSocket handler for real-time audio streaming and transcript delivery.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict

import websockets
from websockets.exceptions import ConnectionClosedError

from ..models.schemas import SessionConfig, SystemStatus, TranscriptSegment
from ..models.enums import SpeakerRole, SystemState
from ..agents.orchestrator import process_session_audio


class CourtroomWebSocketHandler:
    """
    Handles WebSocket connections for real-time courtroom audio processing.

    Maintains session state and coordinates with the orchestrator agent
    to process audio chunks and stream transcript segments back to the client.
    """

    def __init__(self):
        self.session_config: Optional[SessionConfig] = None
        self.role_mapping: Optional[Dict[str, SpeakerRole]] = None

    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol) -> None:
        """
        Handle a WebSocket connection for courtroom audio streaming.

        Protocol:
        - Client sends: {"type": "config", "config": {...}} to set session config
        - Client sends: {"type": "audio", "data": "base64_audio_chunk"} for audio
        - Client sends: {"type": "roles", "mapping": {"SPEAKER_00": "judge", ...}} for role mapping
        - Server sends: SystemStatus messages for connection state
        - Server sends: TranscriptSegment messages for processed audio
        """
        try:
            # Send initial connection status
            await self._send_status(websocket, SystemState.CONNECTING, "WebSocket connected")

            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "config":
                        await self._handle_config(websocket, data)
                    elif msg_type == "audio":
                        await self._handle_audio(websocket, data)
                    elif msg_type == "roles":
                        await self._handle_roles(websocket, data)
                    else:
                        await self._send_status(websocket, SystemState.ERROR, f"Unknown message type: {msg_type}")

                except json.JSONDecodeError:
                    await self._send_status(websocket, SystemState.ERROR, "Invalid JSON message")

        except ConnectionClosedError:
            pass  # Client disconnected normally
        except Exception as e:
            await self._send_status(websocket, SystemState.ERROR, f"Connection error: {str(e)}")

    async def _handle_config(self, websocket: websockets.WebSocketServerProtocol, data: dict) -> None:
        """Handle session configuration message."""
        try:
            config_data = data.get("config", {})
            self.session_config = SessionConfig(**config_data)
            await self._send_status(websocket, SystemState.IDLE, "Session configured successfully")
        except Exception as e:
            await self._send_status(websocket, SystemState.ERROR, f"Invalid config: {str(e)}")

    async def _handle_audio(self, websocket: websockets.WebSocketServerProtocol, data: dict) -> None:
        """Handle audio chunk message."""
        if not self.session_config:
            await self._send_status(websocket, SystemState.ERROR, "Session not configured")
            return

        try:
            # Decode base64 audio data
            import base64
            audio_b64 = data.get("data", "")
            audio_bytes = base64.b64decode(audio_b64)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = Path(tmp.name)

            try:
                # Update status to processing
                await self._send_status(websocket, SystemState.PROCESSING, "Processing audio...")

                # Process through orchestrator
                segments = process_session_audio(
                    audio_path=tmp_path,
                    session_config=self.session_config,
                    role_mapping=self.role_mapping,
                    language=self.session_config.source_language,
                )

                # Send each segment to client
                for segment in segments:
                    await websocket.send(segment.model_dump_json())

                # Update status back to idle
                await self._send_status(websocket, SystemState.IDLE, "Audio processed successfully")

            finally:
                # Clean up temp file
                tmp_path.unlink(missing_ok=True)

        except Exception as e:
            await self._send_status(websocket, SystemState.ERROR, f"Audio processing failed: {str(e)}")

    async def _handle_roles(self, websocket: websockets.WebSocketServerProtocol, data: dict) -> None:
        """Handle speaker role mapping message."""
        try:
            mapping_data = data.get("mapping", {})
            self.role_mapping = {}
            for speaker_id, role_str in mapping_data.items():
                self.role_mapping[speaker_id] = SpeakerRole(role_str)
            await self._send_status(websocket, SystemState.IDLE, "Speaker roles updated")
        except Exception as e:
            await self._send_status(websocket, SystemState.ERROR, f"Invalid role mapping: {str(e)}")

    async def _send_status(self, websocket: websockets.WebSocketServerProtocol, state: SystemState, message: Optional[str] = None) -> None:
        """Send a system status message to the client."""
        status = SystemStatus(state=state, message=message)
        await websocket.send(status.model_dump_json())


# Global handler instance
_websocket_handler = CourtroomWebSocketHandler()


async def websocket_endpoint(websocket: websockets.WebSocketServerProtocol) -> None:
    """
    WebSocket endpoint for courtroom audio processing.
    Routes to the global handler instance.
    """
    await _websocket_handler.handle_connection(websocket)