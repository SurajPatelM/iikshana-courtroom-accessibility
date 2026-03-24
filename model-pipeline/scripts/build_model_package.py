"""
Build a model package (API model artifact) for registry push (Task 2.6).

The package fully specifies how to use the API model:
- Model provider and name (e.g. groq:llama-3.1-8b-instant)
- Prompt templates (copied into package)
- Config YAML (parsing, postprocessing, thresholds)
- Code and schema version for reproducibility

Output: a directory (and optional .tar.gz) under model-pipeline/artifacts/
that can be pushed to GCP Artifact Registry as a Docker image or tarball.

Usage (PowerShell):
  $env:PYTHONPATH = "."; python model-pipeline/scripts/build_model_package.py --config-id translation_flash_v1
  $env:PYTHONPATH = "."; python model-pipeline/scripts/build_model_package.py --config-id translation_flash_v1 --version 1.0.0 --tarball
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "config" / "models"
PROMPTS_DIR = REPO_ROOT / "prompts"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "model-pipeline" / "artifacts"
MANIFEST_FILENAME = "model_package_manifest.json"
SCHEMA_VERSION = "1.0"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build API model package for registry (2.6).")
    p.add_argument("--config-id", type=str, required=True, help="Model config id (e.g. translation_flash_v1)")
    p.add_argument("--version", type=str, default="", help="Package version (default: git describe or 'dev')")
    p.add_argument("--output-dir", type=str, default="", help=f"Package output directory (default: {DEFAULT_ARTIFACTS_DIR}/<config_id>_<version>)")
    p.add_argument("--tarball", action="store_true", help="Also create a .tar.gz of the package")
    return p.parse_args()


def _get_code_version() -> str:
    """Return git describe or 'dev'."""
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()
    except Exception:
        pass
    return "dev"


def _load_config_yaml(config_id: str) -> dict:
    import yaml
    config_path = CONFIG_DIR / f"{config_id}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prompt_ids_from_config(raw: dict) -> list[str]:
    """Collect prompt template IDs referenced in config."""
    ids = []
    if raw.get("prompt_template_id"):
        ids.append(raw["prompt_template_id"])
    if raw.get("system_prompt_id"):
        ids.append(raw["system_prompt_id"])
    return list(dict.fromkeys(ids))


def main() -> None:
    args = _parse_args()
    import yaml

    config_id = args.config_id
    version = args.version or _get_code_version()
    version_safe = version.replace("/", "_").strip() or "dev"
    pkg_name = f"{config_id}_{version_safe}"
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_ARTIFACTS_DIR / pkg_name
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    raw = _load_config_yaml(config_id)
    prompt_ids = _prompt_ids_from_config(raw)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config").mkdir(exist_ok=True)
    (out_dir / "prompts").mkdir(exist_ok=True)

    # Copy config YAML
    shutil.copy2(CONFIG_DIR / f"{config_id}.yaml", out_dir / "config" / f"{config_id}.yaml")

    # Copy referenced prompt files
    for pid in prompt_ids:
        src = PROMPTS_DIR / f"{pid}.txt"
        if src.exists():
            shutil.copy2(src, out_dir / "prompts" / f"{pid}.txt")
        else:
            print(f"[WARN] Prompt template not found: {src}", file=sys.stderr)

    # Build manifest (what deployment needs to call the API)
    provider = str(raw.get("provider", "vertex-ai"))
    model_name = str(raw.get("model_name", ""))
    manifest = {
        "config_id": config_id,
        "provider": provider,
        "model_name": model_name,
        "model_ref": f"{provider}:{model_name}",
        "task_type": str(raw.get("task_type", "translation")),
        "prompt_template_id": raw.get("prompt_template_id"),
        "system_prompt_id": raw.get("system_prompt_id"),
        "temperature": float(raw.get("temperature", 0.0)),
        "top_p": float(raw.get("top_p", 1.0)),
        "max_output_tokens": int(raw.get("max_output_tokens", 256)),
        "input_fields": raw.get("input_fields", []),
        "output_schema": raw.get("output_schema", {}),
        "parsing": raw.get("parsing", {}),
        "postprocessing": raw.get("postprocessing", {}),
        "code_version": _get_code_version(),
        "schema_version": SCHEMA_VERSION,
        "package_version": version or _get_code_version(),
    }
    manifest_path = out_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Built package: {out_dir}")
    print(f"  manifest: {MANIFEST_FILENAME}")
    print(f"  config: config/{config_id}.yaml")
    print(f"  prompts: {[f'prompts/{p}.txt' for p in prompt_ids]}")

    if args.tarball:
        tarball_path = out_dir.parent / f"{pkg_name}.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tf:
            tf.add(out_dir, arcname=pkg_name)
        print(f"  tarball: {tarball_path}")
    print("Next: push_model_to_registry.py to push to GCP Artifact Registry (Docker image or tarball).")


if __name__ == "__main__":
    main()
