"""
HTTP API routes for pipeline job management and health check.
Matches exactly what Aditya's frontend expects in api.ts and types/index.ts.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import io

from ..models.schemas import (
    HealthResponse,
    JobEntry,
    PipelineResultResponse,
    PipelineStatusResponse,
    PipelineTriggerResponse,
)
from ..models.enums import ProcessingStage
from ..api.websocket_handler import get_active_session_count

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory job store
# Maps job_id -> JobEntry. Lives for the duration of the server process.
# For production, replace with Redis or a database.
# ---------------------------------------------------------------------------
_jobs: dict[str, JobEntry] = {}

# ---------------------------------------------------------------------------
# Repo root resolution — needed to import demo/ modules
# demo/audio_analysis_pipeline.py lives outside backend/src/
# In Docker (WORKDIR /app) the env var REPO_ROOT is set; locally we walk up.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(os.environ["REPO_ROOT"]) if "REPO_ROOT" in os.environ else Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Background pipeline runner
# Runs the full audio analysis pipeline in a thread pool.
# Updates the job entry as it progresses so the frontend can poll status.
# ---------------------------------------------------------------------------
async def _run_pipeline(
    job_id: str,
    audio_path: Path,
    target_language: str,
    config_id: str,
    skip_local_ml: bool,
) -> None:
    """
    Runs the full demo pipeline in the background:
    1. Normalize audio to 16kHz mono WAV
    2. Run ElevenLabs STT + diarization
    3. Translate transcript
    4. Update job entry with result

    All heavy work runs in asyncio.to_thread() to avoid blocking the event loop.
    """
    job = _jobs.get(job_id)
    if not job:
        return

    try:
        from demo.audio_analysis_pipeline import (
            normalize_to_wav_16k_mono,
            run_ui_audio_analysis,
            scribe_language_code_for_translation,
        )
        from backend.src.services.gemini_translation import translate_text

        # Step 1: Normalize audio
        job.stage = ProcessingStage.NORMALIZING
        job.message = "Normalizing audio to 16kHz mono..."
        job.progress = 0.1

        wav_path = audio_path.with_suffix(".wav")
        await asyncio.to_thread(normalize_to_wav_16k_mono, audio_path, wav_path)

        # Step 2: Run STT + diarization + optional gender/emotion
        job.stage = ProcessingStage.TRANSCRIBING
        job.message = "Transcribing with ElevenLabs Scribe v2..."
        job.progress = 0.3

        result = await asyncio.to_thread(
            run_ui_audio_analysis,
            wav_path,
            skip_local_ml=skip_local_ml,
        )

        if result.scribe_error:
            job.status = "failed"
            job.stage = ProcessingStage.FAILED
            job.message = result.scribe_error
            job.progress = 0.0
            return

        # Step 3: Translate
        job.stage = ProcessingStage.TRANSLATING
        job.message = f"Translating to {target_language}..."
        job.progress = 0.7

        source_language = scribe_language_code_for_translation(result.language_code)
        translated_text = await asyncio.to_thread(
            translate_text,
            result.transcript_plain,
            source_language,
            target_language,
            config_id=config_id,
        )

        # Step 4: Build predictions file path
        # Matches what the Airflow expo DAG writes to disk
        predictions_file = str(
            _REPO_ROOT / "data" / "processed" / "dev" / f"translation_predictions_{config_id}.csv"
        )

        # Step 5: Mark complete
        job.status = "completed"
        job.stage = ProcessingStage.COMPLETE
        job.message = "Pipeline complete"
        job.progress = 1.0
        job.result = PipelineResultResponse(
            translated_text=translated_text,
            best_config=config_id,
            predictions_file=predictions_file,
            target_language=target_language,
        )

    except Exception as e:
        logger.error("Pipeline job %s failed: %s", job_id, e)
        job.status = "failed"
        job.stage = ProcessingStage.FAILED
        job.message = str(e)
        job.progress = 0.0

    finally:
        # Clean up temp audio files
        for p in [audio_path, audio_path.with_suffix(".wav")]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.
    Returns server status, active WebSocket session count, and API key presence.
    """
    elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("XI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    return HealthResponse(
        status="ok",
        version=os.environ.get("API_VERSION", "0.1.0"),
        active_sessions=get_active_session_count(),
        elevenlabs_key_set=bool(elevenlabs_key),
        groq_key_set=bool(groq_key),
    )


# ---------------------------------------------------------------------------
# POST /api/pipeline/trigger
# ---------------------------------------------------------------------------
@router.post("/api/pipeline/trigger", response_model=PipelineTriggerResponse)
async def trigger_pipeline(
    audio: UploadFile = File(...),
    split: str = Form("dev"),
    target_language: str = Form("es"),
    rerun_config_search: str = Form("false"),
    manifest_tail: int = Form(1),
) -> PipelineTriggerResponse:
    """
    Accepts an audio file upload and starts the pipeline in the background.
    Returns a job_id for polling status and fetching result.

    Matches what Aditya's triggerPipeline() in api.ts sends:
    FormData with: audio, split, target_language, rerun_config_search, manifest_tail
    """
    # Read config_id from env — no hardcoding
    config_id = os.environ.get("EXPO_TRANSLATION_CONFIG_ID", "translation_flash_v1")
    skip_local_ml = os.environ.get("IIKSHANA_SKIP_LOCAL_ML", "0") == "1"

    # Save uploaded file to a temp directory
    job_id = str(uuid.uuid4())
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    filename = f"{job_id}{suffix}"

    tmp_dir = Path(tempfile.gettempdir()) / "iikshana_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_path = tmp_dir / filename

    content = await audio.read()
    audio_path.write_bytes(content)

    # Create job entry
    job = JobEntry(
        job_id=job_id,
        filename=filename,
        status="running",
        progress=0.0,
        message="Job queued",
        stage=ProcessingStage.UPLOADING,
    )
    _jobs[job_id] = job

    # Start pipeline in background
    asyncio.create_task(
        _run_pipeline(
            job_id=job_id,
            audio_path=audio_path,
            target_language=target_language,
            config_id=config_id,
            skip_local_ml=skip_local_ml,
        )
    )

    logger.info("Job %s started for file %s", job_id, filename)
    return PipelineTriggerResponse(job_id=job_id, filename=filename)


# ---------------------------------------------------------------------------
# GET /api/pipeline/status/{job_id}
# ---------------------------------------------------------------------------
@router.get("/api/pipeline/status/{job_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(job_id: str) -> PipelineStatusResponse:
    """
    Returns the current status of a pipeline job.
    Frontend polls this every POLL_INTERVAL_MS (12 seconds per constants.ts).
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return PipelineStatusResponse(
        status=job.status,
        progress=job.progress,
        message=job.message,
    )


# ---------------------------------------------------------------------------
# GET /api/pipeline/result/{job_id}
# ---------------------------------------------------------------------------
@router.get("/api/pipeline/result/{job_id}", response_model=PipelineResultResponse)
async def get_pipeline_result(job_id: str) -> PipelineResultResponse:
    """
    Returns the final result of a completed pipeline job.
    Frontend calls this after status shows 'completed'.
    Matches PipelineResultResponse in Aditya's types/index.ts.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status == "running":
        raise HTTPException(status_code=202, detail="Job still running")

    if job.status == "failed":
        raise HTTPException(status_code=500, detail=job.message or "Pipeline failed")

    if not job.result:
        raise HTTPException(status_code=500, detail="Job completed but result is missing")

    return job.result