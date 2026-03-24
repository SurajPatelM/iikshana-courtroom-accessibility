# Models Directory

This directory contains the Pydantic schemas and enumerations used across the backend.

## schemas.py

Defines the data models for API communication and internal agent outputs using Pydantic. This includes:

- Request schemas for incoming audio, text, and image data
- Response schemas for transcription results, translations, and TTS output
- Agent output schemas that standardize the data format each agent returns to the orchestrator

All schemas enforce type validation at runtime, so malformed data gets caught before it reaches agent logic.

## enums.py

Defines enumeration types used throughout the backend:

- **Speaker roles**: judge, witness, attorney, interpreter, and other courtroom participants
- **Emotions**: detected vocal emotions passed between Audio Intelligence and Speech Synthesis
- **System states**: connection status, processing stages, and error codes

These enums keep string values consistent across agents, services, and API responses instead of relying on raw strings.
