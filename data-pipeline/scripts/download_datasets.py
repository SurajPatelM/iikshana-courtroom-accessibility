"""
Download emotion and speech datasets; validate checksums; store in data/raw/ with DVC tracking.
Uses curl (subprocess) for downloads when available to avoid worker hangs; falls back to requests.
"""
import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

from scripts.utils import get_logger, load_config, RAW_DIR

logger = get_logger("download_datasets")

# Session with headers so GitHub/Zenodo serve the file (fallback when curl not used)
DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/octet-stream, application/zip, */*",
}


def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _download_with_curl(
    url: str,
    dest: Path,
    connect_timeout: int = 30,
    max_time: int = 7200,
) -> bool:
    """Download using curl (subprocess). Reliable in Airflow workers; avoids Python requests hang."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl_cmd = [
        "curl",
        "-L",  # follow redirects
        "-f",  # fail on HTTP 4xx/5xx
        "--connect-timeout", str(connect_timeout),
        "--max-time", str(max_time),
        "-o", str(dest),
        "--",  # end options
        url,
    ]
    logger.info("Downloading with curl: %s -> %s", url, dest)
    print(f"Running curl (progress below)...", flush=True)
    try:
        result = subprocess.run(
            curl_cmd,
            timeout=max_time + 60,
        )
        if result.returncode != 0:
            logger.warning("curl exited with code %s", result.returncode)
            return False
        size = dest.stat().st_size
        logger.info("Downloaded %s (%s bytes)", dest.name, size)
        return True
    except FileNotFoundError:
        logger.info("curl not found, will try requests")
        return False
    except subprocess.TimeoutExpired:
        logger.exception("curl timed out")
        return False
    except Exception as e:
        logger.exception("curl failed: %s", e)
        return False


def _download_with_requests(
    url: str,
    dest: Path,
    connect_timeout: int = 30,
    read_timeout: int = 7200,
) -> bool:
    """Fallback: download with requests (can hang in some Airflow worker environments)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    timeout_tuple = (connect_timeout, read_timeout)
    logger.info("Downloading with requests: %s -> %s", url, dest)
    try:
        session = requests.Session()
        session.headers.update(DOWNLOAD_HEADERS)
        resp = session.get(url, stream=True, timeout=timeout_tuple, allow_redirects=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("Downloaded %s (%s bytes)", dest.name, dest.stat().st_size)
        return True
    except Exception as e:
        logger.exception("requests download failed: %s", e)
        return False


def download_file(
    url: str,
    dest: Path,
    validate_checksum: Optional[str] = None,
    connect_timeout: int = 30,
    read_timeout: int = 7200,
) -> bool:
    """Download a file. Prefers curl (reliable in workers); falls back to requests."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url} -> {dest}", flush=True)
    logger.info("Downloading %s -> %s", url, dest)

    # Prefer curl (avoids hang in Airflow worker); fall back to requests if curl fails or is missing
    ok = False
    if shutil.which("curl"):
        ok = _download_with_curl(url, dest, connect_timeout=connect_timeout, max_time=read_timeout)
    if not ok:
        ok = _download_with_requests(url, dest, connect_timeout=connect_timeout, read_timeout=read_timeout)

    if not ok:
        return False

    size = dest.stat().st_size
    if size < 100 and dest.suffix.lower() == ".zip":
        with open(dest, "rb") as f:
            head = f.read(4)
        if head != b"PK\x03\x04" and head != b"PK\x05\x06":
            logger.warning("File does not look like a zip (got %r). Server may have returned an error page.", head)
            return False
    if validate_checksum:
        actual = compute_sha256(dest)
        if actual != validate_checksum:
            logger.warning("Checksum mismatch for %s: expected %s, got %s", dest, validate_checksum, actual)
            return False
    return True


def unzip_if_needed(path: Path, out_dir: Optional[Path] = None) -> Path:
    """Unzip if path is a zip file; return directory containing contents."""
    if not path.suffix.lower() == ".zip":
        return path.parent
    out_dir = out_dir or path.parent / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(out_dir)
    logger.info("Extracted %s -> %s", path.name, out_dir)
    return out_dir


def _log_and_print(msg: str, level: str = "info") -> None:
    """Log and print so both logger and Airflow task stdout show the message."""
    getattr(logger, level)(msg)
    print(msg, flush=True)


def _download_ravdess(url: str) -> bool:
    """
    RAVDESS download + extract matching the known-working flow:
    zip at RAW_DIR/RAVDESS.zip, extract to RAW_DIR/RAVDESS/.
    """
    zip_path = RAW_DIR / "RAVDESS.zip"
    dataset_dir = RAW_DIR / "RAVDESS"
    if dataset_dir.exists():
        _log_and_print("RAVDESS already extracted. Skipping download.")
        return True
    _log_and_print(f"Downloading RAVDESS from: {url}")
    if not download_file(url, zip_path, connect_timeout=30, read_timeout=7200):
        return False
    _log_and_print("Extracting RAVDESS...")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)
    _log_and_print("RAVDESS download + extraction complete.")
    return True


def download_datasets(datasets: Optional[List[str]] = None) -> Dict[str, bool]:
    """Download configured datasets into data/raw/<name>/."""
    from scripts.utils import CONFIG_DIR

    config_path = CONFIG_DIR / "datasets.yaml"
    _log_and_print(f"[Step 1] Loading config from: {config_path}")
    cfg = load_config()
    if not cfg:
        _log_and_print("ERROR: Config is empty or file not found. No datasets will be downloaded.", "error")
        return {}

    emotion = cfg.get("emotion_datasets", {})
    speech = cfg.get("multilingual_speech", {})
    all_ds = {**emotion, **speech}
    _log_and_print(f"[Step 2] Config loaded. emotion_datasets: {list(emotion.keys())}, multilingual_speech: {list(speech.keys())}")

    if datasets:
        all_ds = {k: v for k, v in all_ds.items() if k in datasets}
        _log_and_print(f"[Step 3] Filtered to requested datasets: {list(all_ds.keys())}")
    else:
        _log_and_print(f"[Step 3] No filter (datasets=None). Using all configured datasets: {list(all_ds.keys())}")

    # Log every dataset and its URL before doing any downloads
    _log_and_print("[Step 4] Dataset URLs from config:")
    for name, meta in all_ds.items():
        if not isinstance(meta, dict):
            _log_and_print(f"  - {name}: (invalid meta, skipping)")
            continue
        url = meta.get("url")
        if url:
            _log_and_print(f"  - {name}: URL={url}")
        else:
            _log_and_print(f"  - {name}: no URL configured (will skip)")

    results = {}
    for name, meta in all_ds.items():
        if not isinstance(meta, dict):
            continue
        url = meta.get("url")
        if not url:
            _log_and_print(f"[Skip] {name}: no URL configured (skipped, pipeline continues)", "warning")
            results[name] = True  # skipped = success for pipeline; only actual download failures fail the run
            continue
        # RAVDESS: use exact flow that works when run manually (zip at raw root, extract to RAW_DIR/RAVDESS)
        if name == "RAVDESS":
            results[name] = _download_ravdess(url)
            continue
        raw_dir = RAW_DIR / name
        raw_dir.mkdir(parents=True, exist_ok=True)
        fname = url.rstrip("/").split("/")[-1].split("?")[0] or f"{name}.zip"
        dest = raw_dir / fname
        if dest.exists() and meta.get("checksum"):
            if compute_sha256(dest) == meta["checksum"]:
                _log_and_print(f"[OK] {name}: already present and valid: {dest}")
                results[name] = True
                continue
        _log_and_print(f"[Download] {name}: {url} -> {dest}")
        ok = download_file(url, dest, meta.get("checksum"))
        if ok and fname.endswith(".zip"):
            unzip_if_needed(dest, raw_dir / "extracted")
        results[name] = ok

    _log_and_print(f"[Step 5] Done. Results: {results}")
    return results


def main() -> None:
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else None
    r = download_datasets(datasets)
    failed = [k for k, v in r.items() if not v]
    if failed:
        logger.warning("Some downloads failed or skipped: %s", failed)
        sys.exit(1)
    logger.info("Download results: %s", r)


if __name__ == "__main__":
    main()
