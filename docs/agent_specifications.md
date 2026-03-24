# Agent Specifications

Iikshana uses seven agents, each responsible for one part of the processing pipeline. A central orchestrator coordinates them all. This document describes what each agent does, what it takes as input, and what it returns.

## Overview

| # | Agent | File | Purpose |
|---|-------|------|---------|
| 0 | Orchestrator | `orchestrator.py` | Coordinates all agents, routes input to the right pipeline |
| 1 | Audio Intelligence | `audio_intelligence.py` | Speech to text, speaker diarization, emotion detection |
| 2 | Translation | `translation_agent.py` | Multilingual translation preserving legal terms |
| 3 | Legal Glossary Guardian | `legal_glossary_guardian.py` | Validates and corrects legal terminology in translations |
| 4 | Vision Analysis | `vision_agent.py` | Generates captions for visual evidence |
| 5 | Speech Synthesis | `speech_synthesis.py` | Text to speech with per speaker voice and emotion |
| 6 | Context Manager | `context_manager.py` | Tracks conversation history and speaker profiles |

All agent files live in `backend/src/agents/`.

## Orchestrator (Agent 0)

The orchestrator is the central coordination point. It receives all input from the frontend (audio chunks, images, text commands) and decides which agents to invoke.

**Input:** Raw data from the WebSocket handler (audio bytes, image data, or control messages).

**Output:** Aggregated results back to the frontend (transcript updates, translated text, TTS audio, image captions).

**How it works:**
- For audio input, it runs the audio pipeline: Agent 1 -> Agent 2 -> Agent 3 -> Agent 5
- For image input, it routes to Agent 4
- It updates Agent 6 (Context Manager) after every interaction
- If any agent fails or returns low confidence, the orchestrator flags the output for human review

The orchestrator does not perform any processing itself. It is purely a router and coordinator.

## Audio Intelligence (Agent 1)

Handles the first stage of the audio pipeline: turning raw speech into structured text.

**Input:** Audio chunks (raw bytes from the WebSocket stream).

**Output:** Structured transcription containing:
- Transcribed text with timestamps
- Speaker identification (who said what)
- Detected vocal emotion for each segment

**Services used:** Groq Whisper (`groq_stt_service.py`) for transcription, Gemini (`gemini_service.py`) for speaker diarization and emotion analysis.

**How it works:**
1. Receives an audio chunk from the orchestrator
2. Sends it to the STT service for transcription
3. Runs diarization to identify the speaker (judge, witness, attorney, etc.)
4. Detects the vocal emotion (neutral, stressed, angry, etc.)
5. Returns all three pieces of information as a structured output

The speaker roles come from the enums defined in `models/enums.py`.

## Translation Agent (Agent 2)

Takes the transcribed text from Agent 1 and translates it into the target language.

**Input:** Transcription output from Audio Intelligence, including the source text, detected language, and speaker context.

**Output:** Translated text with metadata about which legal terms were present and how they were handled.

**Services used:** Gemini (`gemini_translation.py`) with legal context aware prompts.

**How it works:**
1. Receives the structured transcription from Agent 1
2. Constructs a translation prompt that includes legal context and glossary terms
3. Calls the Gemini translation service
4. Returns the translated text along with flags for any legal terminology detected

The translation prompts are designed to preserve legal meaning rather than produce literal word for word translations.

## Legal Glossary Guardian (Agent 3)

A post translation validation step that checks whether legal terms survived the translation correctly.

**Input:** Original text (from Agent 1) and translated text (from Agent 2), plus the legal glossary.

**Output:** Validated translation, with corrections applied if legal terms were mistranslated. Also returns a report of which terms were checked and whether any were corrected.

**Services used:** Gemini for term comparison and correction.

**How it works:**
1. Compares the translated text against the legal glossary (`data/legal_glossary/`)
2. Identifies any legal terms that were altered, omitted, or mistranslated
3. If corrections are needed, it rewrites those portions while preserving the rest
4. Returns the final validated translation and a summary of changes

This agent acts as a safety net. Even if the Translation Agent handles legal terms well in most cases, the Guardian provides a second check specifically focused on legal accuracy.

## Vision Analysis (Agent 4)

Generates accessible descriptions of visual evidence so blind users can understand images, photographs, and documents presented in court.

**Input:** Image data (uploaded from the frontend).

**Output:** A text caption describing what the image shows, written for screen reader consumption.

**Services used:** Gemini vision capabilities (`gemini_service.py`).

**How it works:**
1. Receives image data from the orchestrator
2. Sends the image to Gemini's vision endpoint
3. Generates a descriptive caption focused on factual content (what is visible, not interpretation)
4. Returns the caption for display in the ImageViewer component

This agent operates independently from the audio pipeline. It is triggered only when a user loads an image through the frontend.

## Speech Synthesis (Agent 5)

Converts translated text back into spoken audio for the listener.

**Input:** Validated translated text from Agent 3, plus speaker identity and emotion from Agent 1.

**Output:** Audio bytes (TTS output) ready to be streamed back to the frontend.

**Services used:** Google Text to Speech (`tts_service.py`) with SSML markup.

**How it works:**
1. Receives the translated text, speaker role, and detected emotion
2. Selects a voice profile based on the speaker (different voices for judge, witness, etc.)
3. Applies emotional parameters using SSML (e.g., slower and deeper for calm speech, faster for urgent speech)
4. Generates the TTS audio
5. Returns the audio bytes to the orchestrator for streaming to the frontend

Voice assignment is per speaker, so the listener can distinguish who is talking based on the voice they hear.

## Context Manager (Agent 6)

Maintains state across the entire session to keep translations and synthesis consistent.

**Input:** Updates from the orchestrator after each interaction (who spoke, what was said, current session state).

**Output:** Context data that other agents can use, including conversation history, speaker profiles, and session metadata.

**How it works:**
1. After each audio or image interaction, the orchestrator sends an update to the Context Manager
2. The Context Manager records the speaker, the original text, the translation, and the emotion
3. When subsequent agents need context (e.g., the Translation Agent needs to know what was said previously for continuity), the orchestrator pulls the relevant history from the Context Manager
4. Speaker profiles are built up over the session (e.g., "Speaker A has been identified as the judge, tends to speak slowly")

This agent does not call any external APIs. It is purely an in memory state tracker.

## How Agents Interact

Agents never talk to each other directly. All communication goes through the orchestrator:

```
Frontend  <──WebSocket──>  Orchestrator
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Audio Pipeline   Vision Pipeline   Context
         (1 → 2 → 3 → 5)     (4)            (6)
```

Each agent returns a structured Pydantic model (defined in `models/schemas.py`) to the orchestrator. The orchestrator passes the relevant fields from one agent's output as the next agent's input. This keeps the interfaces clean and makes it easy to test agents individually.
