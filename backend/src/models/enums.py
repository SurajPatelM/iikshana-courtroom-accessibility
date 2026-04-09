"""
Enumerations for speaker roles, emotions, and system states.
"""
from enum import Enum


# Lifecycle states of a WebSocket audio session.
# Used by the WebSocket handler to track where each connection is
# and communicate state changes to the frontend.
class SessionState(str, Enum):
    CREATED = "created"          # Session object exists, not yet calibrated
    CALIBRATING = "calibrating"  # Receiving ambient silence for noise profile
    STREAMING = "streaming"      # Actively receiving and processing audio
    CLOSED = "closed"            # Session ended cleanly
    ERROR = "error"              # Unrecoverable error occurred


# Progress stages during batch audio analysis (REST /api/pipeline/trigger).
# Sent to the client as status updates during long-running analysis.
class ProcessingStage(str, Enum):
    UPLOADING = "uploading"        # File received, not yet processed
    NORMALIZING = "normalizing"    # Converting to 16kHz mono WAV
    TRANSCRIBING = "transcribing"  # ElevenLabs Scribe v2 STT in progress
    ANALYZING = "analyzing"        # Gender/emotion ML running
    TRANSLATING = "translating"    # On-host translation in progress
    SYNTHESIZING = "synthesizing"  # TTS generation in progress
    COMPLETE = "complete"          # Pipeline finished successfully
    FAILED = "failed"              # Pipeline encountered an error


# Courtroom participant roles.
# Initially UNKNOWN — assigned by frontend or operator
# as they identify who is speaking.
class SpeakerRole(str, Enum):
    JUDGE = "judge"
    WITNESS = "witness"
    ATTORNEY = "attorney"
    INTERPRETER = "interpreter"
    CLERK = "clerk"
    OTHER = "other"
    UNKNOWN = "unknown"


# Emotion labels from emotion2vec+ model.
# Values match what the pipeline returns (lowercase strings).
class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    UNKNOWN = "unknown"


# Estimated speaker gender from inaSpeechSegmenter.
# Values match what majority_gender() returns exactly.
class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"