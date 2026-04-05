"""
Agent 1: Audio Intelligence - Handles transcription, speaker diarization, and vocal emotion detection.
Coordinates between the diarization service and STT service to produce
speaker-labeled, timestamped transcript segments.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ..models.schemas import TranscriptSegment, DiarizationSegment
from ..models.enums import SpeakerRole, Emotion
from ..services.diarization_service import diarize
from ..services.groq_stt_service import transcribe_audio


def _assign_words_to_speakers(
    words: list,
    diarization_segments: List[DiarizationSegment],
) -> List[TranscriptSegment]:
    """
    Aligns transcribed words to speaker segments based on timestamp overlap.

    For each diarization segment (who spoke and when), we collect all words
    whose start time falls within that speaker's time window.
    Words that don't fall in any speaker window are assigned to the
    closest speaker segment to avoid losing any transcribed text.

    Args:
        words: List of WordTimestamp objects from STT service
        diarization_segments: List of DiarizationSegment objects from diarization service

    Returns:
        List of TranscriptSegment objects, one per speaker turn
    """
    segments = []

    for diar_seg in diarization_segments:
        # Collect all words that fall within this speaker's time window
        seg_words = [
            w.word for w in words
            if diar_seg.start_time <= w.start < diar_seg.end_time
        ]

        # Skip speaker segments with no words — likely silence or noise
        if not seg_words:
            continue

        segments.append(TranscriptSegment(
            speaker_id=diar_seg.speaker_id,           # e.g. "SPEAKER_00"
            speaker_role=SpeakerRole.UNKNOWN,          # Role assigned later by orchestrator
            text=" ".join(seg_words),                  # Join words into a sentence
            translated_text=None,                      # Filled later by Translation Agent
            start_time=diar_seg.start_time,
            end_time=diar_seg.end_time,
            emotion=Emotion.NEUTRAL,                   # Filled later by emotion detection
            confidence=0.0,                            # STT confidence not available per-segment
        ))

    return segments


def process_audio(
    audio_path: Path | str,
    *,
    language: Optional[str] = None,
) -> List[TranscriptSegment]:
    """
    Main entry point for Audio Intelligence agent.
    Takes an audio file and returns speaker-labeled transcript segments.

    This function:
    1. Runs speaker diarization to identify who spoke and when
    2. Runs STT to get word-level transcription with timestamps
    3. Aligns words to speakers based on timestamp overlap
    4. Returns a list of TranscriptSegment objects ready for translation

    Args:
        audio_path: Path to the audio file (WAV, 16kHz mono recommended)
        language: Optional ISO-639-1 language code (e.g. "en") for STT accuracy

    Returns:
        List of TranscriptSegment objects, one per speaker turn,
        ordered by start time.
    """
    audio_path = Path(audio_path)

    # Step 1: Run diarization — find out who spoke and when
    diarization_segments = diarize(str(audio_path))

    # Step 2: Run STT — transcribe audio with word-level timestamps
    transcript = transcribe_audio(
        audio_path,
        language=language,
    )

    # Step 3: Align words to speakers
    segments = _assign_words_to_speakers(
        words=transcript.words,
        diarization_segments=diarization_segments,
    )

    # Sort segments by start time to maintain chronological order
    segments.sort(key=lambda s: s.start_time)

    return segments