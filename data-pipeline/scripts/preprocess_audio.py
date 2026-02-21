"""
Audio preprocessing: 16 kHz mono WAV, loudness normalization, silence trimming.
Reads from data/raw, writes to a staging dir; use stratified_split to produce dev/test/holdout.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.utils import get_logger, load_config, RAW_DIR, PROCESSED_DIR

logger = get_logger("preprocess_audio")

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio; return (samples, sample_rate)."""
    data, rate = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr and rate != sr and librosa is not None:
        data = librosa.resample(data.astype(np.float64), orig_sr=rate, target_sr=sr)
        rate = sr
    elif sr and rate != sr:
        from scipy import signal as scipy_signal
        if rate != sr:
            num = int(len(data) * sr / rate)
            data = scipy_signal.resample(data, num).astype(np.float32)
            rate = sr
    return data.astype(np.float32), rate


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
        return trimmed.astype(np.float32)
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


def process_one(
    in_path: Path,
    out_path: Path,
    target_sr: int = 16000,
    mono: bool = True,
    normalize: bool = True,
    trim: bool = True,
) -> bool:
    """Convert one file to target format and save."""
    try:
        data, sr = load_audio(in_path, sr=target_sr)
        if mono and data.ndim > 1:
            data = data.mean(axis=1)
        if normalize:
            data = normalize_loudness(data)
        if trim:
            data = trim_silence(data, sr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, data, sr, subtype="PCM_16")
        return True
    except Exception as e:
        logger.exception("Failed %s: %s", in_path, e)
        return False


def collect_audio_files(root: Path, exts: tuple[str, ...] = (".wav", ".mp3", ".flac")) -> list[Path]:
    """Collect all audio files under root."""
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def run_preprocessing(
    raw_subdir: str | Path | None = None,
    out_subdir: Path | None = None,
) -> tuple[int, int]:
    """Run preprocessing on raw data. Returns (success_count, fail_count)."""
    cfg = load_config()
    preproc = cfg.get("preprocessing", {})
    target_sr = preproc.get("target_sr", 16000)
    mono = preproc.get("mono", True)
    normalize = preproc.get("normalize_loudness", True)
    trim = preproc.get("trim_silence", True)

    if raw_subdir:
        raw_root = Path(raw_subdir) if isinstance(raw_subdir, str) else raw_subdir
    else:
        raw_root = RAW_DIR
    if out_subdir is None:
        out_subdir = PROCESSED_DIR / "staged"
    out_subdir = Path(out_subdir)
    out_subdir.mkdir(parents=True, exist_ok=True)

    files = collect_audio_files(raw_root)
    logger.info("Found %d audio files under %s", len(files), raw_root)
    ok, fail = 0, 0
    for fp in files:
        rel = fp.relative_to(raw_root)
        out_path = out_subdir / rel.with_suffix(".wav")
        if process_one(fp, out_path, target_sr=target_sr, mono=mono, normalize=normalize, trim=trim):
            ok += 1
        else:
            fail += 1
    logger.info("Preprocessing done: %d ok, %d failed", ok, fail)
    return ok, fail


def main() -> None:
    import sys
    raw_subdir = sys.argv[1] if len(sys.argv) > 1 else None
    run_preprocessing(raw_subdir=raw_subdir)


if __name__ == "__main__":
    main()
