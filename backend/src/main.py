"""
FastAPI entry point — iikshana backend.

WebSocket endpoint: ws://localhost:8000/ws
  - Client sends JSON config first, then binary Float32 PCM chunks.
  - Server returns JSON transcript/status messages.

Run from repo root:
    uvicorn backend.src.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Path resolution ────────────────────────────────────────────────────────────
# backend/src/main.py → repo root is parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Load API keys from .secrets/.env ──────────────────────────────────────────
def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        for candidate in (_REPO_ROOT / ".secrets" / ".env", _REPO_ROOT / ".env"):
            if candidate.is_file():
                load_dotenv(candidate, override=True)
                break
    except ImportError:
        pass

_load_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("iikshana")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="iikshana Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Language label → code map ──────────────────────────────────────────────────
_LANG_CODE = {
    "auto-detect": "auto",
    "english":     "en",
    "spanish":     "es",
    "mandarin":    "zh",
    "french":      "fr",
    "arabic":      "ar",
    "hindi":       "hi",
    "portuguese":  "pt",
    "russian":     "ru",
    "german":      "de",
}


def _to_lang_code(label: str) -> str:
    return _LANG_CODE.get(label.strip().lower(), label.strip().lower())


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_session(ws: WebSocket):
    await ws.accept()
    logger.info("WebSocket connection accepted.")

    # Import working pipeline (repo root is on sys.path)
    from demo.live_translation import make_initial_state, process_audio_chunk  # noqa: PLC0415

    state = make_initial_state()
    target_lang  = "es"
    source_lang  = "auto"
    config_id    = "translation_flash_v1"
    sample_rate  = 16_000
    tts_enabled  = False
    prev_utterance_count = 0

    loop = asyncio.get_event_loop()

    try:
        while True:
            msg = await ws.receive()

            # ── Text frame: config or control ──────────────────────────────────
            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "config":
                    target_lang = _to_lang_code(data.get("target_lang", "Spanish"))
                    source_lang = _to_lang_code(data.get("source_lang", "auto-detect"))
                    sample_rate = int(data.get("sample_rate", 16_000))
                    config_id   = data.get("config_id", "translation_flash_v1")
                    tts_enabled = bool(data.get("tts_enabled", False))
                    logger.info(
                        "Session config — src=%s tgt=%s sr=%d tts=%s config=%s",
                        source_lang, target_lang, sample_rate, tts_enabled, config_id,
                    )
                    await ws.send_text(json.dumps({"type": "status", "message": "Connected — start speaking"}))

                elif data.get("type") == "stop":
                    logger.info("Client requested stop.")
                    break

            # ── Binary frame: Float32 PCM chunk ────────────────────────────────
            elif "bytes" in msg and msg["bytes"]:
                raw_bytes = msg["bytes"]
                samples   = np.frombuffer(raw_bytes, dtype=np.float32)

                if samples.size == 0:
                    continue

                chunk_data = (sample_rate, samples)

                # Run blocking I/O (STT + translation + optional TTS) in a thread pool
                _state_snap = state
                _tgt        = target_lang
                _src        = None if source_lang == "auto" else source_lang
                _cfg        = config_id
                _tts        = tts_enabled

                new_state, _html, status, audio_path = await loop.run_in_executor(
                    None,
                    lambda: process_audio_chunk(
                        chunk_data,
                        _state_snap,
                        _tgt,
                        _cfg,
                        tts_enabled=_tts,
                        source_language_override=_src,
                    ),
                )
                state = new_state

                utterances    = state.get("utterances", [])
                current_count = len(utterances)

                # Push any new utterance to the frontend
                if current_count > prev_utterance_count:
                    last = utterances[-1]
                    await ws.send_text(json.dumps({
                        "type":        "transcript",
                        "speaker":     last.get("speaker", "Speaker"),
                        "original":    last.get("original", ""),
                        "translation": last.get("translation", ""),
                        "timestamp":   last.get("timestamp", ""),
                        "source_lang": last.get("source_lang", ""),
                        "target_lang": last.get("target_lang", ""),
                    }))
                    prev_utterance_count = current_count

                # If TTS produced audio, send it as base64-encoded MP3
                if audio_path:
                    try:
                        mp3_bytes = Path(audio_path).read_bytes()
                        await ws.send_text(json.dumps({
                            "type": "audio",
                            "data": base64.b64encode(mp3_bytes).decode("ascii"),
                            "mime": "audio/mpeg",
                        }))
                        logger.info("Sent TTS audio (%d bytes)", len(mp3_bytes))
                    except Exception as exc:
                        logger.warning("Failed to read TTS file: %s", exc)
                    finally:
                        try:
                            Path(audio_path).unlink(missing_ok=True)
                        except Exception:
                            pass

                # Forward status messages
                if status:
                    await ws.send_text(json.dumps({"type": "status", "message": status}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
