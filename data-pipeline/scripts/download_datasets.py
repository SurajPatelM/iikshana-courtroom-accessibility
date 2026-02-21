"""
Download emotion and speech datasets; validate checksums; store in data/raw/ with DVC tracking.
"""
import hashlib
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from scripts.utils import get_logger, load_config, RAW_DIR

logger = get_logger("download_datasets")


def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, validate_checksum: str | None = None) -> bool:
    """Download a file with progress bar; optionally validate checksum."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            for chunk in tqdm(
                resp.iter_content(chunk_size=8192),
                total=total // 8192 if total else None,
                unit="kB",
                desc=dest.name,
            ):
                f.write(chunk)
        if validate_checksum:
            actual = compute_sha256(dest)
            if actual != validate_checksum:
                logger.warning("Checksum mismatch for %s: expected %s, got %s", dest, validate_checksum, actual)
                return False
        logger.info("Downloaded %s (%s bytes)", dest.name, dest.stat().st_size)
        return True
    except Exception as e:
        logger.exception("Download failed: %s", e)
        return False


def unzip_if_needed(path: Path, out_dir: Path | None = None) -> Path:
    """Unzip if path is a zip file; return directory containing contents."""
    if not path.suffix.lower() == ".zip":
        return path.parent
    out_dir = out_dir or path.parent / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(out_dir)
    logger.info("Extracted %s -> %s", path.name, out_dir)
    return out_dir


def download_datasets(datasets: list[str] | None = None) -> dict[str, bool]:
    """Download configured datasets into data/raw/<name>/."""
    cfg = load_config()
    emotion = cfg.get("emotion_datasets", {})
    speech = cfg.get("multilingual_speech", {})
    all_ds = {**emotion, **speech}
    if datasets:
        all_ds = {k: v for k, v in all_ds.items() if k in datasets}
    results = {}
    for name, meta in all_ds.items():
        if not isinstance(meta, dict):
            continue
        url = meta.get("url")
        if not url:
            logger.info("Skipping %s (no URL configured)", name)
            results[name] = False
            continue
        raw_dir = RAW_DIR / name
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Determine filename from URL
        fname = url.rstrip("/").split("/")[-1].split("?")[0] or f"{name}.zip"
        dest = raw_dir / fname
        if dest.exists() and meta.get("checksum"):
            if compute_sha256(dest) == meta["checksum"]:
                logger.info("Already present and valid: %s", dest)
                results[name] = True
                continue
        ok = download_file(url, dest, meta.get("checksum"))
        if ok and fname.endswith(".zip"):
            unzip_if_needed(dest, raw_dir / "extracted")
        results[name] = ok
    return results


def main() -> None:
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else None
    r = download_datasets(datasets)
    failed = [k for k, v in r.items() if not v]
    if failed:
        logger.warning("Some downloads failed or skipped: %s", failed)
    logger.info("Download results: %s", r)


if __name__ == "__main__":
    main()
