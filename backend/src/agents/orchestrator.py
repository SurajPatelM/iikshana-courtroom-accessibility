"""
Central Orchestrator Agent coordinating all specialized agents and managing data flow between pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict

from ..models.schemas import TranscriptSegment, SessionConfig
from ..models.enums import SpeakerRole
from ..services.gemini_translation import translate_text
from .audio_intelligence import process_audio


def _load_best_config_id() -> str:
    """
    Read the best config ID from the config search results.
    Falls back to the fixed config if the file doesn't exist.
    """
    config_path = Path("data/processed/dev/config_search_results.json")
    if config_path.exists():
        try:
            with config_path.open("r") as f:
                data = json.load(f)
            return data.get("best_config_id", "translation_flash_court")
        except (json.JSONDecodeError, KeyError):
            pass
    return "translation_flash_court"


def _map_speaker_roles(
    segments: List[TranscriptSegment],
    role_mapping: Optional[Dict[str, SpeakerRole]] = None
) -> List[TranscriptSegment]:
    """
    Map speaker IDs to courtroom roles if role mapping is provided.
    Updates speaker_role field in place.
    """
    if not role_mapping:
        return segments

    for segment in segments:
        if segment.speaker_id in role_mapping:
            segment.speaker_role = role_mapping[segment.speaker_id]

    return segments


def process_session_audio(
    audio_path: Path | str,
    session_config: SessionConfig,
    role_mapping: Optional[Dict[str, SpeakerRole]] = None,
    language: Optional[str] = None,
) -> List[TranscriptSegment]:
    """
    Main orchestrator entry point for processing audio in a courtroom session.

    Flow:
    1. Load best translation config ID (fixed to translation_flash_court per Suraj)
    2. Process audio through Audio Intelligence (STT + diarization)
    3. Translate each segment using the configured model
    4. Apply speaker role mapping if provided
    5. Return completed segments ready for WebSocket transmission

    Args:
        audio_path: Path to audio file
        session_config: Session configuration (contains config_id)
        role_mapping: Optional mapping from speaker_id to SpeakerRole
        language: Optional language code for STT accuracy

    Returns:
        List of TranscriptSegment objects with translation and roles applied
    """
    # Step 1: Get the translation config ID (fixed per handoff doc)
    config_id = session_config.config_id

    # Step 2: Process audio through Audio Intelligence
    segments = process_audio(audio_path, language=language)

    # Step 3: Translate each segment
    for segment in segments:
        if segment.text.strip():  # Only translate non-empty text
            segment.translated_text = translate_text(
                source_text=segment.text,
                source_language=session_config.source_language,
                target_language=session_config.target_language,
                config_id=config_id,
            )

    # Step 4: Apply speaker role mapping
    segments = _map_speaker_roles(segments, role_mapping)

    return segments