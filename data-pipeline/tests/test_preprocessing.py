"""Tests for audio normalization, resampling, and preprocessing."""
import numpy as np
import pytest
import soundfile as sf

# Add pipeline root for imports
import sys
from pathlib import Path
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from scripts.preprocess_audio import (
    load_audio,
    normalize_loudness,
    trim_silence,
    process_one,
    collect_audio_files,
)


def test_load_audio_mono(sample_wav_16k):
    data, sr = load_audio(sample_wav_16k, sr=16000)
    assert data.ndim == 1
    assert sr == 16000
    assert data.dtype == np.float32
    assert len(data) == 16000  # 1 sec


def test_load_audio_stereo_to_mono(sample_wav_stereo):
    data, sr = load_audio(sample_wav_stereo, sr=16000)
    assert data.ndim == 1
    assert sr == 16000


def test_normalize_loudness():
    x = np.array([0.1, -0.2, 0.15], dtype=np.float32)
    y = normalize_loudness(x)
    assert y.dtype == np.float32
    assert np.all(np.abs(y) <= 1.0 + 1e-5)
    assert len(y) == 3


def test_normalize_loudness_empty():
    x = np.array([], dtype=np.float32)
    y = normalize_loudness(x)
    assert len(y) == 0


def test_trim_silence(sample_wav_16k):
    data, sr = load_audio(sample_wav_16k)
    trimmed = trim_silence(data, sr, top_db=25.0)
    assert trimmed.dtype == np.float32
    assert len(trimmed) <= len(data)


def test_process_one(sample_wav_16k, temp_data_dir):
    out = temp_data_dir / "out.wav"
    ok = process_one(sample_wav_16k, out, target_sr=16000, mono=True, normalize=True, trim=True)
    assert ok
    assert out.exists()
    info = sf.info(out)
    assert info.samplerate == 16000
    assert info.channels == 1


def test_collect_audio_files(temp_data_dir):
    (temp_data_dir / "a.wav").write_bytes(b"x")
    (temp_data_dir / "b.txt").write_text("x")
    sub = temp_data_dir / "sub"
    sub.mkdir()
    (sub / "c.wav").write_bytes(b"y")
    files = collect_audio_files(temp_data_dir)
    assert len(files) == 2
    assert all(f.suffix == ".wav" for f in files)
