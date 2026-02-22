"""
Inference-style audio preprocessing: 16 kHz mono WAV, loudness normalization, silence trimming.
Same spec as when sending audio to Gemini/APIs at inference (pipeline and backend use this).
Reads from data/raw, writes to staging; stratified_split then builds evaluation sets (dev/test/holdout).
Supports .mp4 (e.g. MELD raw) by extracting audio via ffmpeg.

Optional courtroom-robust steps (config: preprocessing.courtroom_robust): high-pass filter (removes rumble),
peak limiting (prevents one loud event from dominating), and light stationary noise reduction.
"""
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Allow running as script from repo root or data-pipeline: ensure pipeline root is on path
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))

import numpy as np
import soundfile as sf

from scripts.utils import get_logger, load_config, RAW_DIR, PROCESSED_DIR

logger = get_logger("preprocess_audio")

# One-time warning for ffmpeg/video failures so we don't spam thousands of lines
_ffmpeg_fail_warned = False

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

# Used only when log_memory is True: which file index we're on (for sampling)
_memory_log_file_index = 0


def _get_rss_mb() -> Optional[float]:
    """Current process RSS in MB, or None if psutil not available."""
    if psutil is None:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _load_audio_from_video(path: Path, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Extract audio from .mp4/.mkv etc. using ffmpeg; return (samples, sample_rate)."""
    path = Path(path).resolve()
    sr = sr or 16000
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
            str(wav_path),
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except (FileNotFoundError, OSError):
            raise RuntimeError(
                "ffmpeg not found or not runnable. Install it for .mp4 (MELD), e.g.: winget install ffmpeg"
            )
        if out.returncode != 0:
            err = (out.stderr or out.stdout or "").strip()
            if err:
                err = err.split("\n")[-1].strip() or err[:120]
            logger.debug("ffmpeg failed for %s: %s", path.name, err)
            raise RuntimeError("ffmpeg extraction failed")
        data, rate = sf.read(wav_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32), int(rate)
    finally:
        wav_path.unlink(missing_ok=True)


def load_audio(path: Path, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load audio; return (samples, sample_rate). Supports .mp4 via ffmpeg; else soundfile then librosa."""
    path = Path(path).resolve()
    suf = path.suffix.lower()
    if suf in (".mp4", ".m4v", ".mkv", ".avi", ".mov"):
        return _load_audio_from_video(path, sr=sr)
    try:
        data, rate = sf.read(path)
    except Exception:
        if librosa is not None:
            data, rate = librosa.load(path, sr=sr, mono=True)
            data = np.asarray(data, dtype=np.float32)
            rate = int(rate)
            return data, rate
        raise
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr and rate != sr and librosa is not None:
        resampled = librosa.resample(data.astype(np.float64), orig_sr=rate, target_sr=sr)
        if isinstance(resampled, tuple):
            data = np.asarray(resampled[0], dtype=np.float32)
        else:
            data = np.asarray(resampled, dtype=np.float32)
        rate = sr
    elif sr and rate != sr:
        from scipy import signal as scipy_signal
        if rate != sr:
            num = int(len(data) * sr / rate)
            data = np.asarray(scipy_signal.resample(data, num), dtype=np.float32)
            rate = sr
    arr = np.asarray(data, dtype=np.float32)
    return arr, rate


def normalize_loudness(data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Peak-normalize then scale to target dB (relative to full scale)."""
    if data.size == 0:
        return data
    peak = np.abs(data).max()
    if peak <= 0:
        return data
    data = data / peak
    # RMS-based loudness approx: scale so that RMS gives ~target_db
    rms = np.sqrt(np.mean(data ** 2))
    if rms > 0:
        target_linear = 10 ** (target_db / 20.0)
        data = data * (target_linear / rms)
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def trim_silence(data: np.ndarray, sr: int, top_db: float = 25.0) -> np.ndarray:
    """Trim leading/trailing silence using energy threshold."""
    if librosa is not None:
        # librosa.effects.trim(y, top_db=...) — sr not used in recent versions
        trimmed, _ = librosa.effects.trim(data, top_db=top_db)
        return np.asarray(trimmed, dtype=np.float32)
    # Fallback: simple threshold on RMS in windows
    win = min(int(0.02 * sr), len(data) // 4)
    if win < 1:
        return data
    energy = np.convolve(data ** 2, np.ones(win) / win, mode="same")
    thresh = np.max(energy) * (10 ** (-top_db / 10))
    idx = np.where(energy >= thresh)[0]
    if len(idx) == 0:
        return data
    return data[idx[0] : idx[-1] + 1].astype(np.float32)


# -----------------------------------------------------------------------------
# Courtroom-robust processing (optional): high-pass, peak limit, noise reduction.
# Applied after load/mono and before loudness normalization so varying noise
# (HVAC, rumble, sudden peaks) affects translation/STT less.
# -----------------------------------------------------------------------------


def high_pass_filter(data: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """
    Remove low-frequency rumble (AC, footsteps, furniture) below cutoff_hz.
    Speech content is mostly above ~80 Hz; this keeps the signal unchanged above cutoff.
    Uses a 2nd-order Butterworth high-pass (scipy); no new dependency.
    """
    if cutoff_hz <= 0 or data.size == 0:
        return data
    from scipy import signal as scipy_signal
    nyq = sr / 2.0
    if cutoff_hz >= nyq * 0.9:
        return data
    b, a = scipy_signal.butter(2, cutoff_hz / nyq, btype="high")
    out = scipy_signal.filtfilt(b, a, data.astype(np.float64))
    return np.asarray(out, dtype=np.float32)


def peak_limit(data: np.ndarray, limit_db: float) -> np.ndarray:
    """
    Soft ceiling so a single loud event (door slam, objection) does not dominate.
    Values beyond 10^(limit_db/20) are clipped; applied before loudness normalization.
    """
    if limit_db >= 0 or data.size == 0:
        return data
    ceiling = 10.0 ** (limit_db / 20.0)
    return np.clip(data, -ceiling, ceiling).astype(np.float32)


def reduce_noise_stationary(data: np.ndarray, sr: int) -> np.ndarray:
    """
    Light stationary noise suppression via spectral subtraction.
    Estimates noise from the first 0.5 s (or start of file), then subtracts it
    from the magnitude spectrum per frame, with a floor to avoid over-suppression.
    Helps with constant HVAC/hum in courtroom; no extra dependency (numpy/scipy).
    """
    if data.size == 0:
        return data
    from scipy import signal as scipy_signal
    frame_len = 512
    hop_len = frame_len // 2
    n_fft = frame_len
    # Noise: first 0.5 s or 20 frames, whichever is smaller
    noise_frames = min(20, max(1, int(0.5 * sr) // hop_len))
    num_frames = (len(data) - frame_len) // hop_len + 1
    if num_frames < 2 or noise_frames >= num_frames:
        return data
    window = scipy_signal.windows.hann(frame_len)
    # Build noise magnitude spectrum (average of first few frames)
    noise_fft = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for i in range(noise_frames):
        start = i * hop_len
        frame = data[start : start + frame_len].astype(np.float64) * window
        spec = np.abs(np.fft.rfft(frame))
        noise_fft += spec
    noise_fft /= noise_frames
    # Floor: don't subtract below 0.02 * noise (avoids musical noise)
    noise_fft = np.maximum(noise_fft, 1e-10)
    floor = 0.02 * noise_fft
    # Process each frame: magnitude subtract with floor, keep phase; overlap-add with Hann (50% hop)
    out = np.zeros_like(data, dtype=np.float64)
    for i in range(num_frames):
        start = i * hop_len
        frame = data[start : start + frame_len].astype(np.float64) * window
        fft = np.fft.rfft(frame)
        mag = np.abs(fft)
        phase = np.angle(fft)
        mag_clean = np.maximum(mag - noise_fft, floor)
        out[start : start + frame_len] += np.fft.irfft(mag_clean * np.exp(1j * phase), n=frame_len) * window
    # Hann 50% overlap: sum of overlapping windows = 0.5, so scale by 2 to restore level
    out = (out * 2.0).astype(np.float32)
    return out


def process_one(
    in_path: Path,
    out_path: Path,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    trim: bool = True,
    # Courtroom-robust options (from config preprocessing.courtroom_robust); 0/None/False = off
    high_pass_hz: float = 0,
    peak_limit_db: Optional[float] = None,
    noise_reduction: bool = False,
    log_memory: bool = False,
) -> bool:
    """
    Convert one file to target format and save.
    Pipeline order: load -> mono -> [high_pass -> noise_reduction -> peak_limit] -> normalize -> trim -> WAV.
    """
    global _memory_log_file_index
    idx = _memory_log_file_index
    _memory_log_file_index += 1
    try:
        # Optional memory profiling: log RSS to find which step causes OOM (set preprocessing.log_memory: true)
        def _mem_log(step: str) -> None:
            if not log_memory:
                return
            rss = _get_rss_mb()
            if rss is None:
                return
            # First file: log after every step. Every 500 files: log only at start to see growth.
            if idx == 0:
                logger.info("memory [file_index=0] after %s: %.1f MB", step, rss)
            elif step == "start" and idx % 500 == 0:
                logger.info("memory [file_index=%d] after start: %.1f MB", idx, rss)

        _mem_log("start")
        data, sr = load_audio(in_path, sr=target_sr)
        _mem_log("load_audio")
        if mono and data.ndim > 1:
            data = data.mean(axis=1)
        # Optional courtroom-robust steps (before loudness so noise/peaks don't skew normalization)
        if high_pass_hz > 0:
            data = high_pass_filter(data, sr, high_pass_hz)
            _mem_log("high_pass_filter")
        if noise_reduction:
            data = reduce_noise_stationary(data, sr)
            _mem_log("reduce_noise_stationary")
        if peak_limit_db is not None and peak_limit_db < 0:
            data = peak_limit(data, peak_limit_db)
            _mem_log("peak_limit")
        if normalize:
            data = normalize_loudness(data)
            _mem_log("normalize_loudness")
        if trim:
            data = trim_silence(data, sr)
            _mem_log("trim_silence")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, data, sr, subtype="PCM_16")
        return True
    except Exception as e:
        global _ffmpeg_fail_warned
        try:
            if in_path.suffix.lower() in (".mp4", ".m4v", ".mkv", ".avi", ".mov"):
                if not _ffmpeg_fail_warned:
                    _ffmpeg_fail_warned = True
                    logger.warning(
                        "Video file failed (ffmpeg not found?). .mp4 (MELD) will be skipped. "
                        "Install ffmpeg and add to PATH (restart terminal after winget install ffmpeg). First file: %s",
                        in_path.name,
                    )
                    print(
                        "Video files (.mp4) will be skipped: ffmpeg not found. Install ffmpeg and restart terminal, then re-run.",
                        flush=True,
                    )
                else:
                    logger.debug("Skipped video %s: %s", in_path.name, e)
            else:
                logger.warning("Failed %s: %s", in_path, e)
        except Exception:
            pass
        return False


def collect_audio_files(root: Path, exts: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a")) -> List[Path]:
    """Collect audio files under root. MELD (.mp4) is skipped by default; add .mp4 to exts if ffmpeg is available.
    Skips macOS resource-fork files (._*) which are not real media."""
    root = Path(root).resolve()
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    # Exclude ._* (macOS resource forks / metadata) - ffmpeg fails on them and they're not real media
    files = [f.resolve() for f in files if not f.name.startswith("._")]
    return sorted(set(files))


def run_preprocessing(
    raw_subdir: Optional[Union[str, Path]] = None,
    out_subdir: Optional[Path] = None,
) -> Tuple[int, int]:
    """Run preprocessing on raw data. Returns (success_count, fail_count)."""
    cfg = load_config()
    preproc = cfg.get("preprocessing", {})
    target_sr = preproc.get("target_sr", 16000)
    mono = preproc.get("mono", True)
    normalize = preproc.get("normalize_loudness", True)
    trim = preproc.get("trim_silence", True)
    include_video = preproc.get("include_video", False)
    # Courtroom-robust options (optional); all off if courtroom_robust block is missing
    court = preproc.get("courtroom_robust", {})
    high_pass_hz = float(court.get("high_pass_hz", 0) or 0)
    peak_limit_db = court.get("peak_limit_db")
    if peak_limit_db is not None:
        peak_limit_db = float(peak_limit_db)
    noise_reduction = bool(court.get("noise_reduction", False))
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    if include_video:
        exts = exts + (".mp4", ".m4v", ".mkv", ".avi", ".mov")

    if raw_subdir:
        raw_root = Path(raw_subdir) if isinstance(raw_subdir, str) else raw_subdir
    else:
        raw_root = RAW_DIR
    if out_subdir is None:
        out_subdir = PROCESSED_DIR / "staged"
    out_subdir = Path(out_subdir)
    out_subdir.mkdir(parents=True, exist_ok=True)

    raw_root = Path(raw_root).resolve()
    files = collect_audio_files(raw_root, exts=exts)
    # Log per-folder counts so we can see if MELD/TESS/RAVDESS are found
    try:
        by_folder = {}
        for f in files:
            try:
                rel = f.relative_to(raw_root)
                top = rel.parts[0] if rel.parts else "."
                by_folder[top] = by_folder.get(top, 0) + 1
            except ValueError:
                by_folder["."] = by_folder.get(".", 0) + 1
        for folder, count in sorted(by_folder.items()):
            logger.info("  %s: %d files", folder, count)
    except Exception:
        pass
    logger.info("Found %d audio files under %s", len(files), raw_root)
    if high_pass_hz > 0 or peak_limit_db is not None or noise_reduction:
        logger.info(
            "Courtroom-robust: high_pass_hz=%s, peak_limit_db=%s, noise_reduction=%s",
            high_pass_hz or "off",
            peak_limit_db if peak_limit_db is not None else "off",
            noise_reduction,
        )
    log_memory = bool(preproc.get("log_memory", False))
    if log_memory:
        logger.info("Memory logging enabled (preprocessing.log_memory=true); install psutil for RSS stats")
    global _memory_log_file_index
    _memory_log_file_index = 0
    ok, fail = 0, 0
    total = len(files)
    ffmpeg_warned = False
    for i, fp in enumerate(files):
        if total > 100 and (i + 1) % 500 == 0:
            logger.info("  progress: %d / %d (%.0f%%)", i + 1, total, 100.0 * (i + 1) / total)
            print("  progress: %d / %d" % (i + 1, total), flush=True)
        rel = fp.relative_to(raw_root)
        out_path = out_subdir / rel.with_suffix(".wav")
        try:
            if process_one(
                fp,
                out_path,
                target_sr=target_sr,
                mono=mono,
                normalize=normalize,
                trim=trim,
                high_pass_hz=high_pass_hz,
                peak_limit_db=peak_limit_db,
                noise_reduction=noise_reduction,
                log_memory=log_memory,
            ):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            if not ffmpeg_warned and fp.suffix.lower() in (".mp4", ".m4v", ".mkv", ".avi", ".mov"):
                ffmpeg_warned = True
                msg = (
                    "Video file failed (likely ffmpeg). .mp4 (MELD) will be skipped. "
                    "If you installed ffmpeg with winget, close and reopen the terminal so PATH updates, then re-run."
                )
                logger.warning("%s First failure: %s", msg, e)
                print(msg, flush=True)
            else:
                logger.debug("Skipped %s: %s", fp.name, e)
    logger.info("Preprocessing done: %d ok, %d failed", ok, fail)
    return ok, fail


def main() -> None:
    import sys
    raw_subdir = sys.argv[1] if len(sys.argv) > 1 else None
    run_preprocessing(raw_subdir=raw_subdir)


if __name__ == "__main__":
    main()
