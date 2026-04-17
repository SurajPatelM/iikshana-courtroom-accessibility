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
from urllib.parse import quote

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

    async def _process_audio_bytes(raw_bytes: bytes) -> None:
        nonlocal state, prev_utterance_count

        logger.info("Processing audio bytes from websocket: %d bytes", len(raw_bytes))
        samples = np.frombuffer(raw_bytes, dtype=np.float32)
        logger.info("Decoded audio samples: %d samples", samples.size)
        if samples.size == 0:
            logger.info("Audio bytes decoded to empty sample buffer; skipping")
            return

        chunk_data = (sample_rate, samples)
        logger.info(
            "Running process_audio_chunk: sample_rate=%d config=%s tts=%s source_lang=%s",
            sample_rate,
            config_id,
            tts_enabled,
            source_lang,
        )

        # Run blocking I/O (STT + translation + optional TTS) in a thread pool
        _state_snap = state
        _tgt        = target_lang
        _src        = None if source_lang == "auto" else source_lang
        _cfg        = config_id
        _tts        = tts_enabled

        try:
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
        except Exception as exc:
            logger.exception("process_audio_chunk failed")
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
            return

        logger.info("process_audio_chunk completed: status=%s audio_path=%s", status, bool(audio_path))
        state = new_state

        utterances    = state.get("utterances", [])
        current_count = len(utterances)
        logger.info("Utterances after processing: %d (prev %d)", current_count, prev_utterance_count)

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
            logger.info("Sent transcript update to frontend")

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
            logger.info("Sent status message to frontend: %s", status)

    try:
        while True:
            msg = await ws.receive()
            logger.info("WebSocket message received: %s", {"type": msg.get("type"), "text": bool(msg.get("text")), "bytes": bool(msg.get("bytes"))})

            if msg.get("type") == "websocket.disconnect":
                logger.info("Client disconnected (disconnect frame).")
                break

            # ── Text frame: config or control ──────────────────────────────────
            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON text frame")
                    continue

                logger.info("Text frame payload type=%s", data.get("type"))

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

                elif data.get("type") == "audio" and isinstance(data.get("data"), str):
                    logger.info("Received base64 audio text frame, len=%d", len(data.get("data", "")))
                    try:
                        raw_bytes = base64.b64decode(data["data"])
                    except (TypeError, ValueError) as exc:
                        logger.warning("Base64 decode failed: %s", exc)
                        continue
                    await _process_audio_bytes(raw_bytes)

                elif data.get("type") == "stop":
                    logger.info("Client requested stop.")
                    break
                elif data.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                    logger.info("Responded to ping")
                else:
                    logger.info("Unhandled text frame type: %s", data.get("type"))

            # ── Binary frame: Float32 PCM chunk ────────────────────────────────
            elif "bytes" in msg and msg["bytes"]:
                logger.info("Received binary audio frame: %d bytes", len(msg["bytes"]))
                await _process_audio_bytes(msg["bytes"])

            else:
                logger.info("Received unsupported websocket frame: %s", msg)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except RuntimeError as exc:
        if "disconnect" in str(exc).lower():
            logger.info("WebSocket disconnected (runtime).")
        else:
            logger.exception("WebSocket runtime error: %s", exc)
    except Exception as exc:
        logger.exception("WebSocket error: %s", exc)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass


@app.websocket("/ws/asl")
async def ws_asl_session(ws: WebSocket):
    await ws.accept()
    logger.info("[ASL BACKEND] websocket accepted")

    from demo.asl_translation import make_initial_asl_state, process_asl_chunk  # noqa: PLC0415

    state = make_initial_asl_state()
    source_lang: str | None = None
    sample_rate = 16_000
    prev_english_block = ""
    prev_gloss_block = ""
    prev_signmt_url = ""
    chunk_count = 0
    saw_first_chunk = False

    loop = asyncio.get_event_loop()

    try:
        while True:
            msg = await ws.receive()

            if msg.get("type") == "websocket.disconnect":
                logger.info("[ASL BACKEND] client disconnected (disconnect frame).")
                break

            if "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "config":
                    source_lang = data.get("source_lang")
                    sample_rate = int(data.get("sample_rate", 16_000))
                    logger.info(
                        "[ASL BACKEND] config received src=%s sr=%d",
                        source_lang,
                        sample_rate,
                    )
                    await ws.send_text(json.dumps({"type": "status", "message": "ASL config received"}))
                elif data.get("type") == "stop":
                    logger.info("[ASL BACKEND] client requested stop")
                    break
                elif data.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

            elif "bytes" in msg and msg["bytes"]:
                raw_bytes = msg["bytes"]
                samples = np.frombuffer(raw_bytes, dtype=np.float32)
                if samples.size == 0:
                    logger.info("[ASL BACKEND] empty audio chunk received; skipping")
                    continue
                chunk_count += 1
                if not saw_first_chunk:
                    saw_first_chunk = True
                    logger.info("[ASL BACKEND] first binary audio chunk received")
                    await ws.send_text(json.dumps({"type": "status", "message": "ASL audio chunks received"}))
                elif chunk_count % 25 == 0:
                    logger.info("[ASL BACKEND] received audio chunks=%d", chunk_count)

                _state_snap = state
                _src_override = (
                    source_lang
                    if source_lang and source_lang != "Auto-detect"
                    else None
                )
                _chunk_data = (sample_rate, samples)
                try:
                    logger.info("[ASL BACKEND] before process_asl_chunk")
                    await ws.send_text(json.dumps({"type": "status", "message": "Running Groq STT / ASL pipeline"}))
                    state, english_block, gloss_block, _ = await loop.run_in_executor(
                        None,
                        lambda: process_asl_chunk(
                            _chunk_data,
                            _state_snap,
                            source_language_override=_src_override,
                        ),
                    )
                    logger.info("[ASL BACKEND] after process_asl_chunk")
                except Exception as exc:
                    logger.exception("[ASL BACKEND] process_asl_chunk failure")
                    await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
                    continue

                english_block = (english_block or "").strip()
                gloss_block = (gloss_block or "").strip()

                if english_block and english_block != prev_english_block:
                    logger.info("[ASL BACKEND] sending transcript (%d chars)", len(english_block))
                    await ws.send_text(json.dumps({
                        "type": "transcript",
                        "text": english_block,
                    }))
                    await ws.send_text(json.dumps({"type": "status", "message": "Transcript generated"}))
                    prev_english_block = english_block

                    last_line = ""
                    lines = [ln.strip() for ln in english_block.splitlines() if ln.strip()]
                    if lines:
                        last_line = lines[-1]
                    if last_line:
                        truncated = last_line[:80]
                        encoded = quote(truncated, safe="")
                        signmt_url = f"https://sign.mt/?spl=en&sl=ase&text={encoded}"
                        if signmt_url != prev_signmt_url:
                            logger.info("[ASL BACKEND] sending signmt_url")
                            await ws.send_text(json.dumps({
                                "type": "signmt_url",
                                "url": signmt_url,
                            }))
                            await ws.send_text(json.dumps({"type": "status", "message": "sign.mt URL generated"}))
                            prev_signmt_url = signmt_url

                if gloss_block and gloss_block != prev_gloss_block:
                    logger.info("[ASL BACKEND] sending gloss (%d chars)", len(gloss_block))
                    await ws.send_text(json.dumps({
                        "type": "gloss",
                        "text": gloss_block,
                    }))
                    await ws.send_text(json.dumps({"type": "status", "message": "ASL gloss generated"}))
                    prev_gloss_block = gloss_block

    except WebSocketDisconnect:
        logger.info("[ASL BACKEND] websocket disconnected")
    except RuntimeError as exc:
        if "disconnect" in str(exc).lower():
            logger.info("[ASL BACKEND] websocket disconnected (runtime).")
        else:
            logger.exception("[ASL BACKEND] websocket runtime error: %s", exc)
    except Exception as exc:
        logger.exception("[ASL BACKEND] websocket error: %s", exc)
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
