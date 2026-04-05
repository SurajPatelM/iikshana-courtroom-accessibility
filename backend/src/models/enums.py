"""
Enumerations for speaker roles, emotions, and system states.
"""
from enum import Enum


# Defines the role of each speaker in the courtroom session.
# Used to map raw diarization labels (e.g. SPEAKER_00) to 
# meaningful courtroom participants.
class SpeakerRole(str, Enum):
    JUDGE = "judge"
    WITNESS = "witness"
    ATTORNEY = "attorney"
    CLERK = "clerk"
    INTERPRETER = "interpreter"
    UNKNOWN = "unknown"  # Default before role is assigned


# Defines the emotional tone detected in a speaker's audio segment.
# Passed from Audio Intelligence agent to Speech Synthesis agent
# so the TTS voice can reflect the speaker's emotion.
class Emotion(str, Enum):
    NEUTRAL = "neutral"
    ANGRY = "angry"
    SAD = "sad"
    HAPPY = "happy"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"


# Defines the current state of the system during a courtroom session.
# Used by the WebSocket handler to communicate connection and 
# processing status to the frontend.
class SystemState(str, Enum):
    IDLE = "idle"                  # System is on but not processing
    CONNECTING = "connecting"      # WebSocket connection being established
    PROCESSING = "processing"      # Actively transcribing/translating audio
    ERROR = "error"                # Something went wrong
    DISCONNECTED = "disconnected"  # Session ended or connection lost