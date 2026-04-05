"""
Pydantic schemas for API requests, responses, and agent outputs.
"""
from typing import Optional
from pydantic import BaseModel
from .enums import SpeakerRole, Emotion, SystemState


# Raw speaker segment returned by the diarization service (pyannote).
# Contains who spoke and when, but no text yet.
# Text gets attached later after STT alignment.
class DiarizationSegment(BaseModel):
    speaker_id: str          # Raw label from pyannote e.g. "SPEAKER_00"
    start_time: float        # When this speaker started talking (seconds)
    end_time: float          # When this speaker stopped talking (seconds)


# A single unit of transcribed, translated, and speaker-labeled speech.
# This is the core data structure that flows through the entire pipeline:
# Audio Intelligence -> Orchestrator -> Translation Agent -> Frontend
class TranscriptSegment(BaseModel):
    speaker_id: str                                  # Raw diarization label e.g. "SPEAKER_00"
    speaker_role: SpeakerRole = SpeakerRole.UNKNOWN  # Courtroom role, assigned later
    text: str                                        # Original transcribed text
    translated_text: Optional[str] = None            # Translated text, filled by Translation Agent
    start_time: float                                # Start of this speech segment (seconds)
    end_time: float                                  # End of this speech segment (seconds)
    emotion: Emotion = Emotion.NEUTRAL               # Detected emotion in this segment
    confidence: float = 0.0                          # STT confidence score (0.0 to 1.0)


# Configuration for a courtroom session, set at session start.
# Determines how the pipeline behaves for the entire session.
class SessionConfig(BaseModel):
    speaker_diarization: bool = True        # Whether to identify multiple speakers
    source_language: str = "en"             # Language being spoken in court (overridable)
    target_language: str = "es"             # Language to translate into (overridable)
    config_id: str = "translation_flash_court"  # Fixed translation model per Suraj's decision


# System status message sent from backend to frontend over WebSocket.
# Keeps the frontend informed of what the system is currently doing.
class SystemStatus(BaseModel):
    state: SystemState                      # Current system state
    message: Optional[str] = None          # Optional human-readable status message