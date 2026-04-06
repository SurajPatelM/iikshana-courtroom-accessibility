"""
Main entry point for Iikshana backend application.
Initializes FastAPI app, configures middleware, registers routes, and starts WebSocket server.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.websocket_handler import websocket_endpoint


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup: Load models, initialize services
    yield
    # Shutdown: Clean up resources


# Create FastAPI app
app = FastAPI(
    title="IIKSHANA Courtroom Accessibility Backend",
    description="Real-time speech translation and accessibility for courtrooms",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local dev only; tighten this for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


from fastapi import WebSocket

@app.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio processing."""
    await websocket_endpoint(websocket)


if __name__ == "__main__":
    # Run with uvicorn when executed directly
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )