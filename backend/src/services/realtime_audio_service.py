"""
Real-time audio service for courtroom streaming sessions.

Manages the full lifecycle of a preprocessing + STT session:

    Session start  → calibration window captures room noise profile
    Chunk arrival  → preprocess → VAD gate → ElevenLabs Scribe v2
    Session end    → cleanup

This is the layer that WebSocket handlers will call once implemented.
Each active courtroom connection gets its own RealtimeAudioSession so
noise profiles and state are isolated between concurrent hearings.
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Import resolution — supports running from repo root or from backend/
# ---------------------------------------------------------------------------
_BACKEND_SRC = Path(__file__).resolve().parent.parent
if str(_BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(_BACKEND_SRC))

from utils.audio_processing import CourtroomAudioPreprocessor  # noqa: E402
from services.elevenlabs_stt_service import (  # noqa: E402
    elevenlabs_api_key_from_env,
    transcribe_file_scribe_v2,
)

logger = logging.getLogger(__name__)


class RealtimeAudioSession:
    """
    Manages one active courtroom audio session.

    Each WebSocket connection should create its own instance so that noise
    profiles and calibration state are isolated between concurrent hearings.

    Typical lifecycle
    -----------------
    ::

        session = RealtimeAudioSession(session_id="hearing-2026-001")

        # Once, before anyone speaks — feed ~0.5-1 s of ambient room audio:
        session.start_calibration(silence_samples, src_sr=44100)

        # For every incoming audio chunk from the WebSocket:
        result = session.handle_chunk(raw_samples, src_sr=44100)
        if result is not None:
            # result.text           — plain transcript
            # result.words          — word-level timestamps
            # result.language_code  — detected language
            broadcast_to_frontend(result)

        # When the connection closes:
        session.close()

    Parameters
    ----------
    session_id:
        Unique identifier for this session (used in log messages).
    api_key:
        ElevenLabs API key. Resolved from ELEVENLABS_API_KEY env var if omitted.
    diarize:
        Pass diarize=True to ElevenLabs Scribe v2 so speaker labels are included
        in the returned word objects. Requires an ElevenLabs plan that supports it.
    preprocessor_kwargs:
        Optional overrides for CourtroomAudioPreprocessor constructor parameters
        (e.g. {"high_pass_hz": 100, "vad_enabled": False}).
    """

    def __init__(
        self,
        session_id: str,
        api_key: Optional[str] = None,
        diarize: bool = True,
        preprocessor_kwargs: Optional[dict] = None,
    ) -> None:
        self.session_id = session_id
        self._api_key = api_key or elevenlabs_api_key_from_env()
        self._diarize = diarize
        self._preprocessor = CourtroomAudioPreprocessor(**(preprocessor_kwargs or {}))
        self._closed = False

        logger.info("[session=%s] Created.", self.session_id)

    # ------------------------------------------------------------------
    # Session control
    # ------------------------------------------------------------------

    def start_calibration(self, silence_chunk: np.ndarray, src_sr: int) -> None:
        """
        Calibrate the noise profile from a segment of ambient room silence.

        Should be called once before the hearing starts — e.g. triggered by
        the WebSocket "session_start" message while the room is still quiet.

        Args:
            silence_chunk: ~0.5–1 s of room audio with no speech.
            src_sr:        Sample rate of the chunk.
        """
        self._preprocessor.calibrate(silence_chunk, src_sr)
        logger.info("[session=%s] Calibration complete.", self.session_id)

    def close(self) -> None:
        """Mark session as closed. Subsequent handle_chunk() calls are no-ops."""
        self._closed = True
        logger.info("[session=%s] Session closed.", self.session_id)

    # ------------------------------------------------------------------
    # Chunk processing
    # ------------------------------------------------------------------

    def handle_chunk(
        self,
        raw_chunk: np.ndarray,
        src_sr: int,
    ) -> Optional[Any]:
        """
        Preprocess one raw audio chunk and forward to STT if voice is detected.

        Accepts float32 or int16 audio. Int16 values are normalised to [-1, 1]
        automatically.

        Args:
            raw_chunk: Raw audio samples from microphone / WebSocket.
            src_sr:    Sample rate of the incoming chunk.

        Returns:
            ElevenLabs Scribe v2 transcription object with attributes:
                .text           (str)  — plain transcript
                .words          (list) — word-level timestamps + speaker labels
                .language_code  (str)  — detected language
            Returns None if no voice was detected or on STT error.
        """
        if self._closed:
            logger.warning(
                "[session=%s] handle_chunk() called on a closed session.", self.session_id
            )
            return None

        logger.debug(
            "[session=%s] handle_chunk: raw_chunk_size=%s src_sr=%s",
            self.session_id,
            raw_chunk.size if raw_chunk is not None else 0,
            src_sr,
        )
        if raw_chunk is None or raw_chunk.size == 0:
            return None

        # Normalise int16 PCM → float32 [-1, 1]
        chunk = raw_chunk.astype(np.float32)
        if chunk.max() > 1.0 or chunk.min() < -1.0:
            chunk = chunk / 32768.0

        clean = self._preprocessor.process_chunk(chunk, src_sr)
        if clean is None:
            # VAD gated this chunk — silence or noise only
            return None

        return self._transcribe(clean)

    # ------------------------------------------------------------------
    # STT dispatch
    # ------------------------------------------------------------------

    def _transcribe(self, clean_chunk: np.ndarray) -> Optional[Any]:
        """
        Write preprocessed chunk to a temporary WAV file and send to
        ElevenLabs Scribe v2. The temp file is deleted after the API call.

        Returns the Scribe v2 transcription object or None on failure.
        """
        wav_bytes = self._preprocessor.to_wav_bytes(clean_chunk)

        logger.debug(
            "[session=%s] _transcribe: writing %d bytes to temporary WAV",
            self.session_id,
            len(wav_bytes),
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            fh.write(wav_bytes)
            tmp_path = Path(fh.name)

        try:
            result = transcribe_file_scribe_v2(
                tmp_path,
                api_key=self._api_key,
                diarize=self._diarize,
                tag_audio_events=True,
                timestamps_granularity="word",
            )
            return result
        except Exception as exc:
            logger.error(
                "[session=%s] STT error: %s", self.session_id, exc
            )
            return None
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """True after start_calibration() has been called."""
        return self._preprocessor.is_calibrated

    @property
    def is_closed(self) -> bool:
        """True after close() has been called."""
        return self._closed
