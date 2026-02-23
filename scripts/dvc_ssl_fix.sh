#!/usr/bin/env bash
# Fix SSL certificate verification for DVC + GCS on macOS (python.org installer).
# Usage: source scripts/dvc_ssl_fix.sh   then run  dvc push  or  dvc pull
#    or: scripts/dvc_ssl_fix.sh push
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if command -v python3 &>/dev/null; then
  CERT_PATH=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null || true)
fi
if [ -z "$CERT_PATH" ] && command -v python &>/dev/null; then
  CERT_PATH=$(python -c "import certifi; print(certifi.where())" 2>/dev/null || true)
fi
if [ -n "$CERT_PATH" ] && [ -f "$CERT_PATH" ]; then
  export SSL_CERT_FILE="$CERT_PATH"
  export REQUESTS_CA_BUNDLE="$CERT_PATH"
fi
if [ $# -gt 0 ]; then
  exec dvc "$@"
fi
