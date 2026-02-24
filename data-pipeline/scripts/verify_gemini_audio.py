"""
Optional smoke test: send one or two preprocessed WAVs to the Gemini API to confirm
the pipeline output format is accepted end-to-end. Used by gemini_verification_dag.
Set RUN_GEMINI_VERIFICATION=true and GEMINI_API_KEY or GOOGLE_API_KEY to enable.
"""
import os
from pathlib import Path
from typing import Any

from scripts.utils import get_logger

logger = get_logger("verify_gemini_audio")

EXIT_FAILURE = 1
EXIT_SUCCESS = 0


def _collect_wavs(data_dir: Path, max_files: int, prefer_split: str | None) -> list[Path]:
    """Collect up to max_files WAV paths from data_dir, optionally under prefer_split subdir."""
    root = Path(data_dir)
    if prefer_split and (root / prefer_split).is_dir():
        search_dir = root / prefer_split
    else:
        search_dir = root
    wavs = sorted(search_dir.rglob("*.wav"))[:max_files]
    if not wavs and search_dir != root:
        wavs = sorted(root.rglob("*.wav"))[:max_files]
    return wavs


def run_verification(
    data_dir: Path,
    max_files: int = 2,
    prefer_split: str | None = "staged",
    force_run: bool = False,
) -> dict[str, Any]:
    """
    Optionally verify preprocessed audio with the Gemini API.
    Returns a dict with: skipped, success, exit_code, results (and optionally api_response per file).
    """
    run_env = os.environ.get("RUN_GEMINI_VERIFICATION", "").strip().lower() == "true"
    if not force_run and not run_env:
        logger.info("RUN_GEMINI_VERIFICATION not set; skipping Gemini verification")
        return {"skipped": True, "success": True, "exit_code": EXIT_SUCCESS, "results": []}

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GOOGLE_API_KEY or GEMINI_API_KEY; skipping Gemini verification")
        return {"skipped": True, "success": True, "exit_code": EXIT_SUCCESS, "results": []}

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        logger.warning("Data dir %s does not exist; skipping", data_dir)
        return {"skipped": True, "success": True, "exit_code": EXIT_SUCCESS, "results": []}

    wavs = _collect_wavs(data_dir, max_files, prefer_split)
    if not wavs:
        logger.warning("No WAV files found under %s; skipping", data_dir)
        return {"skipped": True, "success": True, "exit_code": EXIT_SUCCESS, "results": []}

    try:
        from google import genai
    except ImportError as e:
        logger.error("google-genai not installed: %s", e)
        return {
            "skipped": False,
            "success": False,
            "exit_code": EXIT_FAILURE,
            "results": [{"error": "google-genai not installed", "detail": str(e)}],
        }

    client = genai.Client(api_key=api_key)
    results: list[dict[str, Any]] = []
    all_ok = True

    for wav_path in wavs:
        entry: dict[str, Any] = {"path": str(wav_path)}
        try:
            uploaded = client.files.upload(file=str(wav_path))
            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=["In one short sentence, what is in this audio?", uploaded],
            )
            text = getattr(resp, "text", None)
            if not text and getattr(resp, "candidates", None):
                c = resp.candidates[0]
                if getattr(c, "content", None) and getattr(c.content, "parts", None) and c.content.parts:
                    text = getattr(c.content.parts[0], "text", None)
            entry["api_response"] = text or "(no text)"
            entry["ok"] = True
            logger.info("Gemini accepted %s: %s", wav_path.name, (text or "")[:80])
        except Exception as e:
            logger.exception("Gemini verification failed for %s: %s", wav_path, e)
            entry["ok"] = False
            entry["error"] = str(e)
            all_ok = False
        results.append(entry)

    return {
        "skipped": False,
        "success": all_ok,
        "exit_code": EXIT_SUCCESS if all_ok else EXIT_FAILURE,
        "results": results,
    }


def main() -> int:
    """CLI entrypoint: run verification and exit with exit_code."""
    import argparse
    parser = argparse.ArgumentParser(description="Verify preprocessed audio with Gemini API")
    parser.add_argument("data_dir", nargs="?", default=None, help="Processed data dir (default: from utils.PROCESSED_DIR)")
    parser.add_argument("--max-files", type=int, default=2, help="Max WAV files to send (default: 2)")
    parser.add_argument("--force", action="store_true", help="Run even if RUN_GEMINI_VERIFICATION is not set")
    parser.add_argument("--prefer-split", default="staged", help="Prefer this subdir for WAVs (default: staged)")
    args = parser.parse_args()

    from scripts.utils import PROCESSED_DIR
    data_dir = Path(args.data_dir) if args.data_dir else PROCESSED_DIR
    result = run_verification(
        data_dir=data_dir,
        max_files=args.max_files,
        prefer_split=args.prefer_split or None,
        force_run=args.force,
    )
    if result.get("skipped"):
        return EXIT_SUCCESS
    return result.get("exit_code", EXIT_FAILURE)


if __name__ == "__main__":
    raise SystemExit(main())
