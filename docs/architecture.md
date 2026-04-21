# Iikshana Architecture

## System Overview

Iikshana has three layers that work together:

1. **Backend** (FastAPI + AI agents): handles all real time processing, from receiving audio to returning translated speech.
2. **Frontend** (React PWA): the user facing interface, built for blind participants who interact through keyboard shortcuts and screen readers.
3. **Data and Model Pipelines** (offline): evaluation, bias detection, and quality checks that run separately from the live system. These pipelines validate model performance against benchmarks and are triggered through Airflow or CI/CD.

The live system (backend + frontend) operates on premises. The evaluation pipelines run in CI or on a development machine and never touch live courtroom audio.

## How Audio Flows Through the System

The core purpose of Iikshana is turning courtroom speech in one language into assistive audio in another language. Here is how that happens, step by step:

```
Microphone / Audio Feed
        │
        ▼
┌─────────────────┐
│  Frontend        │  Captures audio via Web Audio API
│  (AudioCapture)  │  Streams raw chunks over WebSocket
└────────┬────────┘
         │ WebSocket
         ▼
┌─────────────────┐
│  Orchestrator    │  Receives audio, routes to agents in sequence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Audio           │  Transcribes speech to text (via configured STT: e.g. Gemini / cloud STT)
│  Intelligence    │  Identifies speakers (diarization)
│  (Agent 1)       │  Detects vocal emotion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Translation     │  Translates transcript between languages
│  (Agent 2)       │  Uses Gemini with legal context prompts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Legal Glossary  │  Validates that legal terms survived translation
│  Guardian        │  Corrects mistranslations of legal terminology
│  (Agent 3)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Speech          │  Converts translated text to speech
│  Synthesis       │  Assigns voice per speaker, applies detected emotion
│  (Agent 5)       │  Uses Google TTS with SSML
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Frontend        │  Receives TTS audio over WebSocket
│  (audio.ts)      │  Queues and plays back through Web Audio API
└─────────────────┘
```

Throughout this flow, the **Context Manager (Agent 6)** tracks conversation history and speaker profiles so that translation and synthesis stay consistent across turns.

## Visual Evidence Pipeline

The visual pipeline is separate from the audio flow. When a user loads an image (a photo of evidence, a document), the frontend sends it to the Orchestrator, which routes it to the **Vision Agent (Agent 4)**. The Vision Agent generates a caption using Gemini's vision capabilities, and the caption is returned to the frontend for display and screen reader announcement.

```
Image Upload (frontend)
        │
        ▼
  Orchestrator  ──►  Vision Agent (Agent 4)  ──►  Caption text
        │                                              │
        ▼                                              ▼
  Frontend displays caption with accessible ARIA structure
```

## Agent Orchestration

The Orchestrator is the only agent that talks to the frontend directly. All other agents receive input from the Orchestrator and return results back to it. This keeps the routing logic in one place and lets individual agents focus on their specific task.

The Orchestrator decides which agents to invoke based on the type of input:
- **Audio input**: routes through Agents 1 -> 2 -> 3 -> 5 (the audio pipeline)
- **Image input**: routes to Agent 4 (vision pipeline)
- **All inputs**: Context Manager (Agent 6) is updated alongside every interaction

Agents communicate through the Orchestrator, not directly with each other. Each agent returns structured output (defined by Pydantic schemas in `backend/src/models/schemas.py`) that the Orchestrator passes to the next agent in the chain.

## Frontend to Backend Connection

The frontend connects to the backend through two channels:

- **WebSocket** (socket.io): used for real time audio streaming (frontend to backend) and transcript/TTS delivery (backend to frontend). This is the primary communication channel during a live session.
- **HTTP REST**: used for session management (start, stop, configure), image uploads, and health checks. These are standard request/response calls using Axios on the frontend side.

## Where the Pipelines Fit In

The data pipeline and model pipeline are offline evaluation tools. They do not run during a live courtroom session.

- **Data pipeline** (managed by Airflow): downloads public benchmark datasets, preprocesses audio to 16 kHz mono, validates data quality, and produces evaluation splits. See `data-pipeline/README.md`.
- **Model pipeline**: takes the processed evaluation data and runs the Gemini model against it to measure translation quality (BLEU, chrF, exact match), detect bias, and search for better configurations. See `model-pipeline/README.md`.
- **CI/CD pipeline**: automates both pipelines on every push. Runs tests, triggers model evaluation when relevant files change, enforces quality gates, and blocks deployment if metrics regress. See `.github/workflows/README.md`.

## On Premises Deployment

Iikshana is designed to run entirely on premises inside the court's network. The key principle: no live courtroom audio ever leaves the local machine. The external API calls (Gemini, Groq, ElevenLabs, TTS) would be replaced with locally hosted models in a production deployment. The current implementation uses cloud APIs for development convenience, but the architecture is structured so that swapping to local inference only requires changing the service layer, not the agent logic.
