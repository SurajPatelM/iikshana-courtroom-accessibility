"""
Pydantic schemas for API requests, responses, and agent outputs.
All schemas match exactly what Aditya's frontend expects in types/index.ts.
"""
from typing import Optional
from pydantic import BaseModel
from .enums import SessionState, ProcessingStage, Gender, SpeakerRole


# ---------------------------------------------------------------------------
# REST: POST /api/pipeline/trigger
# ---------------------------------------------------------------------------

# What the frontend sends (as form data, not JSON):
# audio: File, split: str, target_language: str,
# rerun_config_search: bool, manifest_tail: int

# What the frontend expects back — matches PipelineTriggerResponse in types/index.ts
class PipelineTriggerResponse(BaseModel):
    job_id: str      # Unique ID used to poll status and fetch result
    filename: str    # Saved filename on server


# ---------------------------------------------------------------------------
# REST: GET /api/pipeline/status/{job_id}
# ---------------------------------------------------------------------------

# Matches PipelineStatusResponse in types/index.ts
class PipelineStatusResponse(BaseModel):
    status: str          # "running", "completed", or "failed"
    progress: float      # 0.0 to 1.0 — shown as progress bar in frontend
    message: Optional[str] = None  # Human-readable stage description


# ---------------------------------------------------------------------------
# REST: GET /api/pipeline/result/{job_id}
# ---------------------------------------------------------------------------

# Matches PipelineResultResponse in types/index.ts
class PipelineResultResponse(BaseModel):
    translated_text: str       # Full translated transcript
    best_config: str           # config_id that was used e.g. "translation_flash_v1"
    predictions_file: str      # Path to CSV predictions file
    target_language: str       # Language translated into e.g. "es"


# ---------------------------------------------------------------------------
# WebSocket: server -> client messages
# ---------------------------------------------------------------------------

# Matches TranscriptSegment in types/index.ts
# Sent by server after each audio chunk is processed
class TranscriptSegment(BaseModel):
    speaker_id: str                          # e.g. "speaker_0" from ElevenLabs
    speaker_role: str = SpeakerRole.UNKNOWN  # Courtroom role, defaults to unknown
    text: str                                # Transcribed text for this segment
    translated_text: Optional[str] = None   # Translated text if enabled
    start_time: float                        # Segment start time in seconds


# Status update sent by server on session state changes
# Frontend checks for "state" field to distinguish from TranscriptSegment
class WSStatusUpdate(BaseModel):
    state: SessionState   # Current session state
    message: str = ""     # Human-readable description


# ---------------------------------------------------------------------------
# WebSocket: client -> server messages
# ---------------------------------------------------------------------------

# Session configuration sent by frontend on WebSocket connect
# Matches WebSocketConfig in types/index.ts
class WSConfig(BaseModel):
    speaker_diarization: bool  # Whether to identify multiple speakers
    source_language: str       # Language being spoken e.g. "en"
    target_language: str       # Language to translate into e.g. "es"
    config_id: str             # Translation model config to use


# Wrapper for the config message (type + config)
class WSConfigMessage(BaseModel):
    type: str       # Always "config"
    config: WSConfig


# Audio chunk message sent by frontend
# Audio data is base64 encoded ArrayBuffer
class WSAudioMessage(BaseModel):
    type: str       # Always "audio"
    data: str       # Base64 encoded audio bytes


# ---------------------------------------------------------------------------
# Internal: job store entry (not sent to frontend)
# ---------------------------------------------------------------------------

# Tracks the state of a background pipeline job
# Stored in memory, never serialized to frontend directly
class JobEntry(BaseModel):
    job_id: str
    filename: str
    status: str = "running"           # "running", "completed", "failed"
    progress: float = 0.0             # 0.0 to 1.0
    message: Optional[str] = None
    stage: Optional[ProcessingStage] = None
    result: Optional[PipelineResultResponse] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Health check: GET /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str               # "ok" or "degraded"
    version: str              # API version
    active_sessions: int      # Number of active WebSocket sessions
    elevenlabs_key_set: bool  # Whether ELEVENLABS_API_KEY is configured
    groq_key_set: bool        # Whether GROQ_API_KEY is configured