"""
Main entry point for Iikshana backend application.
Initializes FastAPI app, configures middleware, registers routes, and starts WebSocket server.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Repo root setup
# Must happen before any local imports so demo/ and config/ are importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load .env files from repo root and airflow/.env if present
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env", override=False)
    load_dotenv(_REPO_ROOT / "airflow" / ".env", override=False)
except ImportError:
    pass  # dotenv not installed — env vars must be set manually

from .api.routes import router
from .api.websocket_handler import audio_websocket, _active_sessions

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup and shutdown logic
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup checks when server starts and cleanup when it stops.
    """
    # Startup
    elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("XI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if not elevenlabs_key:
        logger.warning("ELEVENLABS_API_KEY is not set — STT and TTS will fail")
    else:
        logger.info("ELEVENLABS_API_KEY is set")

    if not groq_key:
        logger.warning("GROQ_API_KEY is not set — translation may fail")
    else:
        logger.info("GROQ_API_KEY is set")

    logger.info("Iikshana backend started — repo root: %s", _REPO_ROOT)

    yield

    # Shutdown — close all active WebSocket sessions
    logger.info("Shutting down — closing %d active sessions", len(_active_sessions))
    for session in _active_sessions.values():
        try:
            session.close()
        except Exception:
            pass
    _active_sessions.clear()
    logger.info("Iikshana backend stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Iikshana Courtroom Accessibility API",
    description="Real-time audio transcription, speaker diarization, and translation for courtrooms.",
    version=os.environ.get("API_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware
# Allows Aditya's React frontend to call the API from a different origin.
# In production, replace allow_origins=["*"] with the actual frontend URL.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# REST endpoints from routes.py
app.include_router(router)

# WebSocket endpoint at /ws/audio — matches WS_URL in Aditya's constants.ts
app.add_api_websocket_route("/ws/audio", audio_websocket)


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.src.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )