"""
Courtroom real-time audio preprocessor.

Cleans incoming audio chunks before they reach STT (ElevenLabs Scribe v2).
Pipeline per chunk:
    mono → resample → high-pass filter → noise reduction → peak limit
    → loudness normalize → VAD gate → silence trim

Noise profile is estimated once during a calibration window at session start
(ambient room silence before the hearing begins). It is reused for the entire
session, so constant HVAC / projector hum is suppressed without recalculating
per chunk.

VAD (Voice Activity Detection) gates chunks so silence is not forwarded to STT.
Backend priority:
    1. silero-VAD   (torch — most accurate, language-agnostic)
    2. webrtcvad    (lightweight C extension, good for 16 kHz)
    3. Energy-based (numpy-only fallback, always available)

All DSP functions are adapted from data-pipeline/scripts/preprocess_audio.py.
The key difference: noise_profile is passed in (session-scoped) rather than
estimated from the first 0.5 s of the current chunk (file-scoped).
"""
from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — resolved once at module load
# ---------------------------------------------------------------------------

try:
    from scipy import signal as _scipy_signal  # noqa: F401
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import librosa as _librosa  # noqa: F401
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

_SILERO_VAD_MODEL = None   # lazy-loaded singleton
_SILERO_UTILS = None

_WEBRTCVAD_AVAILABLE = False
try:
    import webrtcvad as _webrtcvad  # noqa: F401
    _WEBRTCVAD_AVAILABLE = True
except ImportError:
    pass

_TORCH_AVAILABLE = False
try:
    import torch as _torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# DSP helpers
# Adapted from data-pipeline/scripts/preprocess_audio.py — stateless numpy /
# scipy operations that work identically on streaming chunks.
# ---------------------------------------------------------------------------

def _to_mono(data: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if data.ndim == 1:
        return data.astype(np.float32)
    return data.mean(axis=1).astype(np.float32)


def _resample(data: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample data from src_sr to target_sr."""
    logger.debug("_resample: src_sr=%d target_sr=%d length=%d", src_sr, target_sr, len(data))
    if src_sr == target_sr:
        return data
    if _LIBROSA_AVAILABLE:
        import librosa
        resampled = librosa.resample(
            data.astype(np.float64), orig_sr=src_sr, target_sr=target_sr
        )
        logger.debug("_resample: used librosa")
        return np.asarray(resampled, dtype=np.float32)
    if _SCIPY_AVAILABLE:
        from scipy import signal as scipy_signal
        num = int(len(data) * target_sr / src_sr)
        logger.debug("_resample: used scipy signal.resample")
        return np.asarray(scipy_signal.resample(data, num), dtype=np.float32)
    logger.error("_resample: no resampling backend available")
    raise RuntimeError(
        "Neither librosa nor scipy is available for resampling. "
        "Install at least one: pip install librosa  or  pip install scipy"
    )


def _high_pass_filter(data: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """
    2nd-order Butterworth high-pass filter.
    Removes low-frequency rumble (HVAC, footsteps, furniture) below cutoff_hz.
    Speech content sits mostly above ~80 Hz; the signal above cutoff is unchanged.
    """
    logger.debug("_high_pass_filter: sr=%d cutoff_hz=%s length=%d", sr, cutoff_hz, data.size)
    if not _SCIPY_AVAILABLE or cutoff_hz <= 0 or data.size == 0:
        logger.debug("_high_pass_filter: skipped (scipy unavailable, no data, or cutoff disabled)")
        return data
    from scipy import signal as scipy_signal
    nyq = sr / 2.0
    if cutoff_hz >= nyq * 0.9:
        logger.debug("_high_pass_filter: cutoff too high, skipping")
        return data
    b, a = scipy_signal.butter(2, cutoff_hz / nyq, btype="high")
    out = scipy_signal.filtfilt(b, a, data.astype(np.float64))
    logger.debug("_high_pass_filter: applied")
    return np.asarray(out, dtype=np.float32)


def _peak_limit(data: np.ndarray, limit_db: float = -3.0) -> np.ndarray:
    """
    Soft ceiling that clips transient peaks (gavel, "Objection!", door slam).
    Applied before loudness normalisation so a single loud event does not
    pull the normalisation target for the whole chunk.
    """
    if limit_db >= 0 or data.size == 0:
        return data
    ceiling = 10.0 ** (limit_db / 20.0)
    return np.clip(data, -ceiling, ceiling).astype(np.float32)


def _normalize_loudness(data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Peak-normalize then scale to target RMS dB.
    Ensures consistent input level for STT regardless of microphone distance.
    """
    logger.debug("_normalize_loudness: target_db=%s length=%d", target_db, data.size)
    if data.size == 0:
        return data
    peak = np.abs(data).max()
    if peak <= 0:
        logger.debug("_normalize_loudness: no signal present")
        return data
    data = data / peak
    rms = np.sqrt(np.mean(data ** 2))
    if rms > 0:
        target_linear = 10 ** (target_db / 20.0)
        data = data * (target_linear / rms)
    logger.debug("_normalize_loudness: peak=%s rms=%s", peak, rms)
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def _trim_silence(data: np.ndarray, sr: int, top_db: float = 25.0) -> np.ndarray:
    """Trim leading/trailing silence using energy threshold."""
    logger.debug("_trim_silence: sr=%d length=%d top_db=%s", sr, data.size, top_db)
    if data.size == 0:
        return data
    if _LIBROSA_AVAILABLE:
        import librosa
        trimmed, _ = librosa.effects.trim(data, top_db=top_db)
        logger.debug("_trim_silence: used librosa trimmed_length=%d", len(trimmed))
        return np.asarray(trimmed, dtype=np.float32)
    # Fallback: RMS windowing
    win = min(int(0.02 * sr), max(1, len(data) // 4))
    energy = np.convolve(data ** 2, np.ones(win) / win, mode="same")
    thresh = np.max(energy) * (10 ** (-top_db / 10))
    idx = np.where(energy >= thresh)[0]
    if len(idx) == 0:
        logger.debug("_trim_silence: no frames above threshold")
        return data
    trimmed = data[idx[0]: idx[-1] + 1].astype(np.float32)
    logger.debug("_trim_silence: trimmed_length=%d", len(trimmed))
    return trimmed


# ---------------------------------------------------------------------------
# Noise profile — session-scoped spectral subtraction
# ---------------------------------------------------------------------------

def build_noise_profile(data: np.ndarray, sr: int, duration_s: float = 0.5) -> np.ndarray:
    """
    Estimate stationary noise spectrum from a calibration segment (room silence).

    Returns a mean magnitude spectrum vector (rfft bins) that is stored on the
    session and passed to apply_noise_reduction() for every subsequent chunk.

    Args:
        data:       Mono float32 audio at `sr` Hz — should be room silence,
                    no speech. Typically 0.5–1 s captured before hearing starts.
        sr:         Sample rate of `data`.
        duration_s: How many seconds of `data` to use for profiling.

    Returns:
        1-D float64 array of length (frame_len // 2 + 1) = 257 bins.
        Returns a zero vector if the chunk is too short (no-op subtraction).
    """
    logger.info("build_noise_profile: sr=%d duration_s=%.2f data_length=%d", sr, duration_s, len(data))
    frame_len = 512
    hop_len = frame_len // 2
    n_noise_frames = min(20, max(1, int(duration_s * sr) // hop_len))

    if len(data) < frame_len:
        logger.warning(
            "Calibration chunk too short (%d samples). "
            "Noise profile set to zero — noise reduction disabled.",
            len(data),
        )
        return np.zeros(frame_len // 2 + 1, dtype=np.float64)

    if _SCIPY_AVAILABLE:
        from scipy import signal as scipy_signal
        window = scipy_signal.windows.hann(frame_len)
    else:
        window = np.hanning(frame_len)

    noise_fft = np.zeros(frame_len // 2 + 1, dtype=np.float64)
    frames_used = 0
    for i in range(n_noise_frames):
        start = i * hop_len
        end = start + frame_len
        if end > len(data):
            break
        frame = data[start:end].astype(np.float64) * window
        noise_fft += np.abs(np.fft.rfft(frame))
        frames_used += 1

    if frames_used > 0:
        noise_fft /= frames_used

    return noise_fft


def apply_noise_reduction(data: np.ndarray, noise_profile: np.ndarray) -> np.ndarray:
    """
    Spectral subtraction using a pre-computed noise profile.

    Unlike the data-pipeline version (which estimates noise from the first 0.5 s
    of the file being processed), this function accepts an externally supplied
    profile. The profile is estimated once at session start during calibration
    and reused for every chunk, making it suitable for streaming.

    Args:
        data:          Mono float32 audio chunk at 16 kHz.
        noise_profile: Output of build_noise_profile() — mean rfft magnitude bins.

    Returns:
        Denoised float32 array of the same length as `data`.
    """
    if data.size == 0 or noise_profile is None or not np.any(noise_profile):
        logger.debug("apply_noise_reduction: skipped because no data or empty noise profile")
        return data

    logger.debug("apply_noise_reduction: data_length=%d noise_profile_length=%d", len(data), len(noise_profile))
    frame_len = 512
    hop_len = frame_len // 2
    num_frames = (len(data) - frame_len) // hop_len + 1

    if num_frames < 2:
        return data

    if _SCIPY_AVAILABLE:
        from scipy import signal as scipy_signal
        window = scipy_signal.windows.hann(frame_len)
    else:
        window = np.hanning(frame_len)

    # Floor: 2% of noise level — prevents over-suppression / musical noise
    floor = np.maximum(noise_profile * 0.02, 1e-10)

    out = np.zeros_like(data, dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_len
        frame = data[start: start + frame_len].astype(np.float64) * window
        fft = np.fft.rfft(frame)
        mag_clean = np.maximum(np.abs(fft) - noise_profile, floor)
        phase = np.angle(fft)
        out[start: start + frame_len] += (
            np.fft.irfft(mag_clean * np.exp(1j * phase), n=frame_len) * window
        )

    # Hann 50% overlap: window sum = 0.5, scale by 2 to restore level
    return (out * 2.0).astype(np.float32)


# ---------------------------------------------------------------------------
# VAD helpers
# ---------------------------------------------------------------------------

def _load_silero_vad() -> Tuple[Optional[object], Optional[object]]:
    """
    Lazy-load silero-VAD from torch.hub. Cached as module-level singleton so
    the model is only downloaded and loaded once per process.
    """
    global _SILERO_VAD_MODEL, _SILERO_UTILS
    if _SILERO_VAD_MODEL is not None:
        return _SILERO_VAD_MODEL, _SILERO_UTILS
    if not _TORCH_AVAILABLE:
        return None, None
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        _SILERO_VAD_MODEL = model
        _SILERO_UTILS = utils
        logger.info("silero-VAD loaded.")
        return model, utils
    except Exception as exc:
        logger.warning(
            "Could not load silero-VAD (%s). "
            "Falling back to next available VAD backend.",
            exc,
        )
        return None, None


def _vad_silero(data: np.ndarray, sr: int) -> bool:
    """silero-VAD inference. Returns True if speech probability >= 0.5."""
    import torch
    model, _ = _load_silero_vad()
    if model is None:
        logger.debug("_vad_silero: silero model unavailable, fallback to energy VAD")
        return _vad_energy(data)
    try:
        tensor = torch.FloatTensor(data)
        confidence: float = model(tensor, sr).item()
        logger.debug("_vad_silero: confidence=%.3f speech=%s", confidence, confidence >= 0.5)
        return confidence >= 0.5
    except Exception as exc:
        logger.warning("_vad_silero inference failed: %s, falling back to energy VAD", exc)
        return _vad_energy(data)


def _vad_webrtc(data: np.ndarray, sr: int, aggressiveness: int = 2) -> bool:
    """
    webrtcvad-based VAD. Splits audio into 30 ms frames; returns True if at
    least 30% of frames contain speech.
    webrtcvad requires 16-bit PCM at 8/16/32/48 kHz.
    """
    import webrtcvad
    vad = webrtcvad.Vad(aggressiveness)
    frame_ms = 30
    frame_samples = int(sr * frame_ms / 1000)
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    frame_bytes = frame_samples * 2  # 2 bytes per int16 sample
    speech_frames = total_frames = 0
    for offset in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        frame = pcm[offset: offset + frame_bytes]
        try:
            if vad.is_speech(frame, sr):
                speech_frames += 1
        except Exception as exc:
            logger.debug("_vad_webrtc frame error: %s", exc)
        total_frames += 1
    if total_frames == 0:
        logger.debug("_vad_webrtc: no frames available")
        return False
    speech_fraction = speech_frames / total_frames
    logger.debug("_vad_webrtc: speech_frames=%d total_frames=%d fraction=%.3f", speech_frames, total_frames, speech_fraction)
    return speech_fraction >= 0.3


def _vad_energy(data: np.ndarray, threshold_db: float = -40.0) -> bool:
    """
    Energy-based VAD fallback. No external dependencies.
    Returns True if RMS level exceeds threshold_db.
    """
    if data.size == 0:
        return False
    rms = np.sqrt(np.mean(data ** 2))
    rms_db = 20.0 * np.log10(rms + 1e-10)
    is_voice = bool(rms_db > threshold_db)
    logger.debug("_vad_energy: rms_db=%.2f threshold_db=%.2f voice=%s", rms_db, threshold_db, is_voice)
    return is_voice


# ---------------------------------------------------------------------------
# Main preprocessor class
# ---------------------------------------------------------------------------

class CourtroomAudioPreprocessor:
    """
    Real-time audio preprocessor for courtroom settings.

    Wraps all DSP operations into a single stateful object whose session-level
    noise profile is estimated once (calibration) and reused across chunks.

    Typical usage
    -------------
    ::

        pre = CourtroomAudioPreprocessor()

        # Feed ~0.5-1 s of ambient silence before the hearing starts:
        pre.calibrate(silence_samples, src_sr=44100)

        # For every incoming audio chunk from the microphone / WebSocket:
        clean = pre.process_chunk(raw_samples, src_sr=44100)
        if clean is not None:
            send_to_stt(clean)   # Voice detected — forward to ElevenLabs

    Parameters
    ----------
    sample_rate:
        Target output sample rate. ElevenLabs Scribe v2 works best at 16 kHz.
    high_pass_hz:
        High-pass filter cutoff in Hz. 80 Hz removes HVAC rumble while leaving
        all speech fundamentals intact.
    peak_limit_db:
        Soft transient ceiling in dBFS. -3.0 clips sudden loud events without
        audibly compressing normal speech. Pass None to disable.
    target_loudness_db:
        RMS normalization target in dBFS. -20.0 matches typical STT input range.
    vad_enabled:
        Gate chunks on voice activity. Silence chunks return None from
        process_chunk() and are not forwarded to STT.
    vad_aggressiveness:
        webrtcvad aggressiveness (0–3). Ignored when silero-VAD is used.
        Higher values filter more aggressively.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        high_pass_hz: float = 80.0,
        peak_limit_db: Optional[float] = -3.0,
        target_loudness_db: float = -20.0,
        vad_enabled: bool = True,
        vad_aggressiveness: int = 2,
    ) -> None:
        self.sample_rate = sample_rate
        self.high_pass_hz = high_pass_hz
        self.peak_limit_db = peak_limit_db
        self.target_loudness_db = target_loudness_db
        self.vad_enabled = vad_enabled
        self.vad_aggressiveness = vad_aggressiveness

        self._noise_profile: Optional[np.ndarray] = None
        self._is_calibrated: bool = False
        self._vad_backend: Optional[str] = None  # resolved lazily on first VAD call
        logger.info(
            "CourtroomAudioPreprocessor initialized sample_rate=%d high_pass_hz=%s peak_limit_db=%s target_loudness_db=%s vad_enabled=%s vad_aggressiveness=%d",
            self.sample_rate,
            self.high_pass_hz,
            self.peak_limit_db,
            self.target_loudness_db,
            self.vad_enabled,
            self.vad_aggressiveness,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, audio_chunk: np.ndarray, src_sr: int) -> None:
        """
        Estimate the room's stationary noise profile from ambient silence.

        Call once before the hearing starts. Feed ~0.5–1 s of room audio with
        no one speaking (e.g. the first second after the mic opens, or a
        deliberate calibration window).

        Args:
            audio_chunk: Raw audio samples (any shape, any sample rate).
            src_sr:      Sample rate of the incoming chunk.
        """
        chunk = _to_mono(audio_chunk)
        chunk = _resample(chunk, src_sr, self.sample_rate)
        self._noise_profile = build_noise_profile(chunk, self.sample_rate)
        self._is_calibrated = True
        logger.info(
            "Noise profile calibrated from %.2f s at %d Hz.",
            len(chunk) / self.sample_rate,
            self.sample_rate,
        )

    # ------------------------------------------------------------------
    # VAD
    # ------------------------------------------------------------------

    def _get_vad_backend(self) -> str:
        """Resolve and cache the best available VAD backend."""
        if self._vad_backend is not None:
            return self._vad_backend
        model, _ = _load_silero_vad()
        if model is not None:
            self._vad_backend = "silero"
        elif _WEBRTCVAD_AVAILABLE:
            self._vad_backend = "webrtcvad"
            logger.info("VAD backend: webrtcvad.")
        else:
            self._vad_backend = "energy"
            logger.info(
                "VAD backend: energy (install torch for silero-VAD "
                "or webrtcvad for better accuracy)."
            )
        return self._vad_backend

    def is_voice(self, data: np.ndarray) -> bool:
        """
        Run VAD on a preprocessed 16 kHz mono chunk.

        Args:
            data: Float32 audio at self.sample_rate Hz.

        Returns:
            True if speech is detected, False if silence / noise.
        """
        if not self.vad_enabled:
            return True

        backend = self._get_vad_backend()
        logger.debug("is_voice: using VAD backend=%s", backend)

        if backend == "silero":
            result = _vad_silero(data, self.sample_rate)
        elif backend == "webrtcvad":
            result = _vad_webrtc(data, self.sample_rate, self.vad_aggressiveness)
        else:
            result = _vad_energy(data)
        logger.debug("is_voice: result=%s", result)
        return result

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_chunk(
        self,
        raw_chunk: np.ndarray,
        src_sr: int,
    ) -> Optional[np.ndarray]:
        """
        Run the full preprocessing pipeline on one raw audio chunk.

        Pipeline order
        --------------
        1. Convert to mono
        2. Resample to 16 kHz
        3. High-pass filter  (removes rumble < 80 Hz)
        4. Noise reduction   (spectral subtraction — only if calibrated)
        5. Peak limiting     (clips transient events)
        6. Loudness normalization
        7. VAD gate          (returns None if no voice detected)
        8. Silence trim      (strips leading/trailing quiet from speech chunk)

        Args:
            raw_chunk: Raw audio samples (float32 or int16, any shape).
            src_sr:    Sample rate of the incoming chunk.

        Returns:
            Preprocessed 16 kHz mono float32 numpy array ready for STT,
            or None if no voice was detected in this chunk.
        """
        if raw_chunk is None or raw_chunk.size == 0:
            logger.debug("process_chunk: empty raw_chunk")
            return None

        logger.info("process_chunk: src_sr=%d raw_length=%d", src_sr, raw_chunk.size)
        # --- 1. Mono ---
        data = _to_mono(raw_chunk)

        # --- 2. Resample ---
        data = _resample(data, src_sr, self.sample_rate)

        # --- 3. High-pass filter ---
        data = _high_pass_filter(data, self.sample_rate, self.high_pass_hz)

        # --- 4. Noise reduction (session-scoped profile) ---
        if self._is_calibrated and self._noise_profile is not None:
            data = apply_noise_reduction(data, self._noise_profile)

        # --- 5. Peak limit ---
        if self.peak_limit_db is not None and self.peak_limit_db < 0:
            data = _peak_limit(data, self.peak_limit_db)

        # --- 6. Loudness normalize ---
        data = _normalize_loudness(data, self.target_loudness_db)

        # --- 7. VAD gate ---
        if not self.is_voice(data):
            return None

        # --- 8. Silence trim ---
        data = _trim_silence(data, self.sample_rate)

        if data.size == 0:
            logger.debug("process_chunk: result empty after trimming")
            return None

        logger.info("process_chunk: output_length=%d", data.size)
        return data

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_wav_bytes(self, data: np.ndarray) -> bytes:
        """
        Encode a preprocessed float32 array as 16-bit PCM WAV bytes.

        Returns an in-memory WAV buffer suitable for passing to
        ElevenLabs STT (which expects a file-like object).
        """
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, data, self.sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        output = buf.read()
        logger.debug("to_wav_bytes: encoded %d bytes", len(output))
        return output

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """True after calibrate() has been called at least once."""
        return self._is_calibrated
