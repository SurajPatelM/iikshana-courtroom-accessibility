# API Documentation

The backend exposes two types of endpoints: standard HTTP routes for session management and configuration, and a WebSocket endpoint for real time audio streaming and transcript delivery.

All endpoints are served by FastAPI and defined in `backend/src/api/`.

## HTTP Endpoints

HTTP routes are defined in `backend/src/api/routes.py`.

### Health Check

```
GET /health
```

Returns the service status. Used by Docker's health check and CI monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

### Session Management

```
POST /session/start
```

Starts a new courtroom session. Initializes the agent orchestrator, context manager, and prepares the WebSocket channel.

**Request body:**
```json
{
  "source_language": "en",
  "target_language": "es",
  "session_config": {
    "emotion_detection": true,
    "speaker_diarization": true
  }
}
```

**Response:**
```json
{
  "session_id": "abc123",
  "status": "active",
  "websocket_url": "ws://localhost:8000/ws/abc123"
}
```

```
POST /session/stop
```

Ends the active session. Cleans up resources and closes the WebSocket connection.

**Request body:**
```json
{
  "session_id": "abc123"
}
```

### Image Upload

```
POST /image/describe
```

Uploads an image for the Vision Agent to caption. Returns an accessible text description.

**Request:** Multipart form data with the image file.

**Response:**
```json
{
  "caption": "A photograph showing a signed document with a seal in the bottom right corner.",
  "confidence": 0.92
}
```

## WebSocket Endpoint

The WebSocket handler is defined in `backend/src/api/websocket_handler.py`. This is the primary communication channel during a live session.

### Connection

```
ws://localhost:8000/ws/{session_id}
```

The frontend connects using socket.io (`socket.io-client` on the frontend, handled by `websocket_service.py` on the backend).

### Messages from Frontend to Backend

**Audio chunk:**
```json
{
  "type": "audio_chunk",
  "data": "<base64 encoded audio bytes>",
  "timestamp": 1711234567890
}
```

**Control messages:**
```json
{
  "type": "control",
  "action": "pause" | "resume" | "stop"
}
```

### Messages from Backend to Frontend

**Transcript update:**
```json
{
  "type": "transcript",
  "speaker": "judge",
  "original_text": "The court will now hear the witness.",
  "translated_text": "El tribunal escuchará ahora al testigo.",
  "emotion": "neutral",
  "confidence": 0.95,
  "timestamp": 1711234567900
}
```

**TTS audio:**
```json
{
  "type": "tts_audio",
  "data": "<base64 encoded audio bytes>",
  "speaker": "judge",
  "emotion": "neutral"
}
```

**Status update:**
```json
{
  "type": "status",
  "state": "processing" | "idle" | "error",
  "message": "Translating segment..."
}
```

**Low confidence flag:**
```json
{
  "type": "review_flag",
  "segment": "The court will now hear the witness.",
  "reason": "low_confidence",
  "confidence": 0.42
}
```

When confidence falls below the configured threshold, the system flags the segment for human interpreter review rather than silently passing through a potentially inaccurate translation.

## Data Schemas

All request and response structures are defined as Pydantic models in `backend/src/models/schemas.py`. Using Pydantic means that:

- Incoming data is validated at runtime (malformed requests get a clear 422 error)
- Response structure is guaranteed to match the schema
- FastAPI auto generates OpenAPI docs at `/docs` (Swagger UI) and `/redoc` based on these schemas

## Enumerations

Speaker roles, emotions, and system states are defined as Python enums in `backend/src/models/enums.py`. These keep values consistent across the API, so you never end up with "Judge" in one place and "judge" in another.

**Speaker roles:** judge, witness, attorney, interpreter, clerk, and others as defined per courtroom setup.

**Emotions:** neutral, stressed, angry, calm, confused, and others detected by the Audio Intelligence agent.

**System states:** idle, processing, error, connected, disconnected.

## Authentication

The current implementation does not include authentication on the API endpoints. In a production deployment, the system runs on premises within the court's local network, so network level access control is the primary security mechanism. If needed, API key or token based authentication can be added at the FastAPI middleware level.

## API Documentation UI

When the backend is running, FastAPI provides interactive documentation at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

These are auto generated from the Pydantic schemas and route definitions.
