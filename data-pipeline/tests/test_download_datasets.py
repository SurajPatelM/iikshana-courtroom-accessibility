"""Unit tests for download_datasets: checksums, unzip, and config-driven download logic."""
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

import sys
PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from scripts.download_datasets import (
    compute_sha256,
    extract_if_needed,
    unzip_if_needed,
    download_datasets,
)


def test_compute_sha256_consistent(temp_data_dir):
    """Same content produces same hash."""
    f = temp_data_dir / "a.bin"
    f.write_bytes(b"hello world")
    h1 = compute_sha256(f)
    h2 = compute_sha256(f)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_compute_sha256_different_content(temp_data_dir):
    """Different content produces different hash."""
    a = temp_data_dir / "a.bin"
    b = temp_data_dir / "b.bin"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")
    assert compute_sha256(a) != compute_sha256(b)


def test_compute_sha256_empty_file(temp_data_dir):
    """Empty file has a valid SHA256."""
    f = temp_data_dir / "empty.bin"
    f.write_bytes(b"")
    h = compute_sha256(f)
    assert len(h) == 64


def test_unzip_if_needed_non_zip(temp_data_dir):
    """Non-.zip path returns path.parent."""
    f = temp_data_dir / "data.txt"
    f.write_text("not a zip")
    out = unzip_if_needed(f)
    assert out == temp_data_dir
    assert f.exists()


def test_unzip_if_needed_zip(temp_data_dir):
    """Valid zip is extracted to out_dir (or path.stem)."""
    zip_path = temp_data_dir / "archive.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("inner.txt", "content")
        zf.writestr("nested/sub.txt", "nested")
    out = unzip_if_needed(zip_path)
    assert out == temp_data_dir / "archive"
    assert (out / "inner.txt").exists()
    assert (out / "nested" / "sub.txt").read_text() == "nested"


def test_unzip_if_needed_zip_custom_out_dir(temp_data_dir):
    """Zip with custom out_dir extracts there."""
    zip_path = temp_data_dir / "foo.zip"
    custom_out = temp_data_dir / "custom_out"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("x.txt", "x")
    out = unzip_if_needed(zip_path, out_dir=custom_out)
    assert out == custom_out
    assert (custom_out / "x.txt").read_text() == "x"


def _write_tar_gz(path: Path, members: dict[str, bytes]) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    path.write_bytes(buf.getvalue())


def test_extract_if_needed_non_archive(temp_data_dir):
    """Non-archive returns path.parent without creating extract dir."""
    f = temp_data_dir / "readme.txt"
    f.write_text("x")
    out = extract_if_needed(f)
    assert out == temp_data_dir


def test_extract_if_needed_tar_gz(temp_data_dir):
    """tar.gz extracts to default or custom out_dir."""
    tgz = temp_data_dir / "bundle.tar.gz"
    _write_tar_gz(tgz, {"a/b.txt": b"hello"})
    out = extract_if_needed(tgz)
    assert out == temp_data_dir / "bundle"
    assert (out / "a" / "b.txt").read_bytes() == b"hello"

    tgz2 = temp_data_dir / "two.tar.gz"
    _write_tar_gz(tgz2, {"x.txt": b"y"})
    custom = temp_data_dir / "extracted_here"
    out2 = extract_if_needed(tgz2, out_dir=custom)
    assert out2 == custom
    assert (custom / "x.txt").read_bytes() == b"y"


def test_extract_if_needed_tgz(temp_data_dir):
    """Short .tgz suffix is handled."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        ti = tarfile.TarInfo(name="only.txt")
        ti.size = 2
        tf.addfile(ti, io.BytesIO(b"ok"))
    path = temp_data_dir / "tiny.tgz"
    path.write_bytes(buf.getvalue())
    out = extract_if_needed(path, out_dir=temp_data_dir / "out")
    assert (out / "only.txt").read_bytes() == b"ok"


def test_download_datasets_empty_config(monkeypatch):
    """When config has no datasets with URLs, download_datasets returns without error."""
    monkeypatch.setattr(
        "scripts.download_datasets.load_config",
        lambda: {"emotion_datasets": {}, "multilingual_speech": {}},
    )
    result = download_datasets()
    assert result == {}


def test_download_datasets_skips_missing_url(monkeypatch):
    """Datasets with null/missing URL are skipped and marked False."""
    monkeypatch.setattr(
        "scripts.download_datasets.load_config",
        lambda: {
            "emotion_datasets": {
                "NoURL": {"url": None},
                "EmptyURL": {},
            },
            "multilingual_speech": {},
        },
    )
    result = download_datasets()
    assert result.get("NoURL") is False
    assert result.get("EmptyURL") is False


def test_download_datasets_filter_by_name(monkeypatch, temp_data_dir):
    """Passing datasets list filters to only those names; no request for B."""
    cfg = {
        "emotion_datasets": {
            "A": {"url": "http://example.com/a.zip"},
            "B": {"url": None},
            "C": {"url": "http://example.com/c.zip"},
        },
        "multilingual_speech": {},
    }
    monkeypatch.setattr("scripts.download_datasets.load_config", lambda: cfg)
    monkeypatch.setattr("scripts.download_datasets.RAW_DIR", temp_data_dir)
    # Mock download_file to avoid network; we only check that result keys are A and C (B filtered out)
    def mock_download(*args, **kwargs):
        return False  # simulate failed download; we only care about which datasets were attempted
    monkeypatch.setattr("scripts.download_datasets.download_file", mock_download)
    result = download_datasets(datasets=["A", "C"])
    assert "A" in result
    assert "C" in result
    assert "B" not in result
