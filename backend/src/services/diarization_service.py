"""
Diarization service using pyannote.audio.
Identifies who is speaking and when in a courtroom audio stream.
Loads the model once at startup and reuses it for the entire session.
"""
import os
from pathlib import Path
from typing import List

from pyannote.audio import Pipeline

from ..models.schemas import DiarizationSegment


# Load the model once when the module is imported.
# This avoids reloading the model on every audio segment — 
# which would be slow and wasteful during a live session.
_pipeline = None


def _get_pipeline() -> Pipeline:
    """
    Lazily loads the pyannote diarization pipeline on first call.
    Reuses the same instance for all subsequent calls in the session.
    """
    global _pipeline
    if _pipeline is None:
        hf_token = os.environ.get("HF_API_TOKEN")
        if not hf_token:
            raise ValueError("HF_API_TOKEN is not set. Please set it in the environment.")
        _pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
    return _pipeline


def diarize(audio_path: str) -> List[DiarizationSegment]:
    """
    Runs speaker diarization on an audio file.
    
    Args:
        audio_path: Path to the audio file (WAV, 16kHz mono recommended)
    
    Returns:
        List of DiarizationSegment objects, each representing one 
        continuous speech turn by a single speaker.
    """
    pipeline = _get_pipeline()
    
    # Run diarization — pyannote returns an Annotation object
    diarization = pipeline(audio_path)
    
    # Convert pyannote Annotation to our DiarizationSegment schema
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(DiarizationSegment(
            speaker_id=speaker,        # e.g. "SPEAKER_00", "SPEAKER_01"
            start_time=turn.start,     # Start time in seconds
            end_time=turn.end          # End time in seconds
        ))
    
    return segments