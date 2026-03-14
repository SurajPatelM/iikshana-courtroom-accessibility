"""
Push API model package to GCP Artifact Registry (Task 2.6).

You cannot push underlying weights for an API model; you push a *model package*
that fully specifies how to use the API (provider, model name, prompts,
parsing, thresholds, code/schema version).

Supports two storage forms:
- Docker image: package is baked into an image; deployment pulls the image
  and reads the package from a known path (e.g. /app/model_package).
- Tarball: package .tar.gz is uploaded as a generic artifact; deployment
  pulls and extracts the bundle.

Requires: gcloud CLI, Docker (for --method docker), and an Artifact Registry
repository. Set REGION, PROJECT, REPO via args or env (e.g. ARTIFACT_REGISTRY_REPO).

Usage (PowerShell):
  $env:PYTHONPATH = "."; python model-pipeline/scripts/build_model_package.py --config-id translation_flash_v1 --tarball
  $env:PYTHONPATH = "."; python model-pipeline/scripts/push_model_to_registry.py --config-id translation_flash_v1 --method tarball --project my-project --location us-central1 --repository model-packages
  $env:PYTHONPATH = "."; python model-pipeline/scripts/push_model_to_registry.py --config-id translation_flash_v1 --method docker --project my-project --location us-central1 --repository model-packages
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "model-pipeline" / "artifacts"
PACKAGE_DIRNAME = "model_package"  # inside Docker image


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Push API model package to GCP Artifact Registry (2.6).")
    p.add_argument("--config-id", type=str, required=True)
    p.add_argument("--version", type=str, default="", help="Package version (default: from build or 'dev')")
    p.add_argument("--package-dir", type=str, default="", help="Path to built package dir (default: artifacts/<config_id>_<version>)")
    p.add_argument("--method", type=str, choices=["docker", "tarball"], default="tarball",
                   help="Push as Docker image or tarball to Artifact Registry")
    p.add_argument("--project", type=str, default=os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    p.add_argument("--location", type=str, default=os.environ.get("ARTIFACT_REGISTRY_LOCATION", "us-central1"))
    p.add_argument("--repository", type=str, default=os.environ.get("ARTIFACT_REGISTRY_REPO", "model-packages"))
    p.add_argument("--build-if-missing", action="store_true", help="Run build_model_package if package not found")
    return p.parse_args()


def _get_code_version() -> str:
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


def _find_package_dir(config_id: str, version: str) -> Path | None:
    version_safe = (version or _get_code_version()).replace("/", "_").strip() or "dev"
    pkg_name = f"{config_id}_{version_safe}"
    cand = DEFAULT_ARTIFACTS_DIR / pkg_name
    if cand.is_dir():
        return cand
    # try without version
    for d in DEFAULT_ARTIFACTS_DIR.iterdir():
        if d.is_dir() and d.name.startswith(config_id + "_"):
            return d
    return None


def main() -> None:
    args = _parse_args()
    config_id = args.config_id
    version = args.version or _get_code_version()
    version_safe = version.replace("/", "_").strip() or "dev"
    pkg_name = f"{config_id}_{version_safe}"

    if args.package_dir:
        pkg_dir = Path(args.package_dir)
        if not pkg_dir.is_absolute():
            pkg_dir = REPO_ROOT / pkg_dir
    else:
        pkg_dir = _find_package_dir(config_id, version)

    if not pkg_dir or not pkg_dir.is_dir():
        if args.build_if_missing:
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "model-pipeline" / "scripts" / "build_model_package.py"),
                 "--config-id", config_id, "--version", version, "--tarball"],
                cwd=REPO_ROOT,
                check=True,
            )
            pkg_dir = _find_package_dir(config_id, version)
        if not pkg_dir or not pkg_dir.is_dir():
            print("[ERROR] Package directory not found. Run build_model_package.py first or use --build-if-missing.", file=sys.stderr)
            sys.exit(1)

    if not args.project:
        print("[ERROR] Set --project or GOOGLE_CLOUD_PROJECT.", file=sys.stderr)
        sys.exit(1)

    registry_host = f"{args.location}-docker.pkg.dev"
    full_repo = f"{registry_host}/{args.project}/{args.repository}"

    if args.method == "tarball":
        tarball = pkg_dir.parent / f"{pkg_name}.tar.gz"
        if not tarball.exists():
            print(f"[ERROR] Tarball not found: {tarball}. Run build_model_package.py with --tarball.", file=sys.stderr)
            sys.exit(1)
        # Generic artifact upload: repository must be a generic repo (not Docker).
        # If the repo is Docker-only, we'd use a separate repo for generic artifacts.
        # Repository must be generic format (gcloud artifacts repositories create --repository-format=generic).
        cmd = [
            "gcloud", "artifacts", "generic", "upload",
            f"--repository={args.repository}",
            f"--location={args.location}",
            f"--project={args.project}",
            f"--source={tarball}",
            f"--package=api-model-{config_id}",
            f"--version={version_safe}",
        ]
        print("Run:", " ".join(cmd))
        rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
        if rc != 0:
            print("[INFO] Ensure the repository is generic format (--repository-format=generic) or use --method docker.", file=sys.stderr)
        sys.exit(rc)

    # Docker: build image containing the package, tag, push
    dockerfile_dir = REPO_ROOT / "model-pipeline" / "scripts"
    # Write a one-off Dockerfile that copies the package
    dockerfile = dockerfile_dir / "Dockerfile.model_package"
    with open(dockerfile, "w", encoding="utf-8") as f:
        f.write(f"""# API model package image for Artifact Registry (2.6)
FROM scratch
COPY package /{PACKAGE_DIRNAME}
""")
    # Copy package into a context dir so Docker COPY works
    import shutil
    ctx_dir = dockerfile_dir / ".docker_ctx_model_package"
    if ctx_dir.exists():
        shutil.rmtree(ctx_dir)
    (ctx_dir / "package").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pkg_dir, ctx_dir / "package")
    # Rewrite Dockerfile to use ctx-relative path
    with open(dockerfile, "w", encoding="utf-8") as f:
        f.write(f"""# API model package image for Artifact Registry (2.6)
FROM scratch
COPY package /{PACKAGE_DIRNAME}
""")
    image_tag = f"{full_repo}/api-model-{config_id}:{version_safe}"
    build_cmd = ["docker", "build", "-f", str(dockerfile), "-t", image_tag, str(ctx_dir)]
    print("Build:", " ".join(build_cmd))
    rc = subprocess.run(build_cmd, cwd=REPO_ROOT).returncode
    if rc != 0:
        sys.exit(rc)
    subprocess.run(["gcloud", "auth", "configure-docker", registry_host, "--quiet"], check=True)
    print("Push:", image_tag)
    rc = subprocess.run(["docker", "push", image_tag]).returncode
    if ctx_dir.exists():
        shutil.rmtree(ctx_dir)
    if (dockerfile_dir / "Dockerfile.model_package").exists():
        (dockerfile_dir / "Dockerfile.model_package").unlink()
    sys.exit(rc)


if __name__ == "__main__":
    main()
