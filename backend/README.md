# Iikshana Backend

FastAPI backend that orchestrates seven AI agents to provide real time courtroom speech translation and accessibility features. The backend handles live audio streaming over WebSocket, coordinates speech to text, translation, legal term enforcement, and text to speech through a central orchestrator agent.

## Folder Structure

```text
backend/
├── src/
│   ├── agents/                # AI agent modules
│   │   ├── orchestrator.py    # Central agent coordinating all others
│   │   ├── audio_intelligence.py   # Agent 1: transcription, diarization, emotion detection
│   │   ├── translation_agent.py    # Agent 2: multilingual translation
│   │   ├── legal_glossary_guardian.py  # Agent 3: legal term validation
│   │   ├── vision_agent.py         # Agent 4: image captioning for visual evidence
│   │   ├── speech_synthesis.py     # Agent 5: voice assignment and emotional TTS
│   │   └── context_manager.py      # Agent 6: conversation history and speaker profiles
│   ├── api/
│   │   ├── routes.py          # HTTP endpoints for session management and image processing
│   │   └── websocket_handler.py  # WebSocket handler for audio streaming and transcript delivery
│   ├── models/
│   │   ├── schemas.py         # Pydantic schemas for requests, responses, agent outputs
│   │   └── enums.py           # Enumerations for speaker roles, emotions, system states
│   ├── services/
│   │   ├── gemini_service.py       # Google Gemini API client (audio, vision, text)
│   │   ├── gemini_translation.py   # Translation wrapper around Gemini
│   │   ├── groq_service.py         # Groq API client for text generation
│   │   ├── elevenlabs_stt_service.py  # ElevenLabs Scribe v2 (batch / pipeline STT)
│   │   ├── hf_service.py           # Hugging Face Inference API client
│   │   ├── tts_service.py          # Google Text to Speech with SSML support
│   │   └── websocket_service.py    # WebSocket service for bidirectional communication
│   ├── utils/
│   │   ├── audio_processing.py     # Audio format conversion and normalization
│   │   ├── config.py               # Configuration and environment variable loading
│   │   └── logger.py               # Privacy safe logging utilities
│   └── main.py               # FastAPI app entry point, middleware, route registration
├── tests/
│   ├── test_agents/
│   │   ├── test_audio_intelligence.py
│   │   └── test_orchestration.py
│   └── test_services/
│       └── test_gemini_service.py
└── Dockerfile                 # Production container (Python 3.10 slim, port 8000)
```

## Agents

The backend uses an agent based architecture where a central orchestrator delegates tasks to six specialized agents:

| Agent | Module | Role |
|-------|--------|------|
| Orchestrator | `orchestrator.py` | Coordinates all agents and manages data flow between pipelines |
| Audio Intelligence | `audio_intelligence.py` | Handles transcription, speaker diarization, and vocal emotion detection |
| Translation | `translation_agent.py` | Translates speech across languages while preserving legal terminology |
| Legal Glossary Guardian | `legal_glossary_guardian.py` | Validates legal term preservation and corrects mistranslations |
| Vision Analysis | `vision_agent.py` | Generates image captions for visual evidence and exhibits |
| Speech Synthesis | `speech_synthesis.py` | Manages voice assignment and emotional TTS generation |
| Context Manager | `context_manager.py` | Maintains conversation history and speaker profiles |

The orchestrator receives input (audio chunks, images, or text) and routes it to the appropriate agents. For the audio pipeline, the flow goes: Audio Intelligence (STT) -> Translation -> Legal Glossary Guardian -> Speech Synthesis (TTS). The Vision agent handles image captioning independently. The Context Manager runs alongside all interactions to track speaker state and conversation history.

## Services

Each service wraps an external API or communication channel:

| Service | What it does |
|---------|-------------|
| `gemini_service.py` | Google Gemini API calls for audio analysis, vision, and text processing |
| `gemini_translation.py` | Translation specific prompts built on top of the Gemini client |
| `elevenlabs_stt_service.py` | Speech to text via ElevenLabs Scribe v2 |
| `groq_service.py` | General text generation via Groq |
| `hf_service.py` | Hugging Face Inference API for text generation and translation |
| `tts_service.py` | Google Text to Speech with SSML for emotional speech output |
| `websocket_service.py` | Real time bidirectional messaging between backend and frontend |

## Local Setup

### Prerequisites

- Python 3.10 or higher
- API keys for the services you want to use (see Environment Variables below)

### Install and Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install fastapi uvicorn[standard] google-genai websockets pydantic python-dotenv pyyaml requests numpy
python src/main.py
```

The app starts on `http://localhost:8000` by default. The health check endpoint is at `/health`.

### Environment Variables

Create a `.env` file in the backend directory (or set these in your shell):

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Google Gemini API access |
| `ELEVENLABS_API_KEY` | ElevenLabs API for Scribe v2 speech-to-text (pipeline / eval) |
| `GROQ_API_KEY` | Groq API for text generation (translation configs with `provider: groq`) |
| `HF_API_TOKEN` | Hugging Face Inference API |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON (for TTS) |

Configuration files for different environments live in `config/` at the repository root (`development.yaml`, `testing.yaml`, `production.yaml`).

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

Tests cover agent logic (`test_audio_intelligence.py`, `test_orchestration.py`) and service integration (`test_gemini_service.py`).

## Docker

The included Dockerfile builds a production image based on Python 3.10 slim:

```bash
docker build -f Dockerfile -t iikshana-backend ..
docker run -p 8000:8000 --env-file .env iikshana-backend
```

Note: the Docker build context is the repository root (one level up from `backend/`), because the Dockerfile copies from `backend/src`.

The container includes a health check that pings `/health` every 30 seconds. System dependencies (gcc, libsndfile, ffmpeg) are installed for audio processing support.
