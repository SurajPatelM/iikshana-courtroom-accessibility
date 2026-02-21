"""Pytest fixtures: temp dirs, sample WAV."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def temp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_wav_16k(temp_data_dir):
    """Create a short 16 kHz mono WAV for tests."""
    path = temp_data_dir / "sample.wav"
    sr = 16000
    t = 1.0
    x = np.sin(2 * np.pi * 440 * np.linspace(0, t, int(sr * t), dtype=np.float32)) * 0.5
    sf.write(path, x, sr, subtype="PCM_16")
    return path


@pytest.fixture
def sample_wav_stereo(temp_data_dir):
    """Create a short stereo WAV (for testing mono conversion)."""
    path = temp_data_dir / "stereo.wav"
    sr = 16000
    t = 0.5
    x = np.sin(2 * np.pi * 440 * np.linspace(0, t, int(sr * t), dtype=np.float32)) * 0.5
    stereo = np.stack([x, x * 0.5], axis=1)
    sf.write(path, stereo, sr, subtype="PCM_16")
    return path
