#!/bin/sh
set -e
cd /app

# Emit demo/config.js so the static page can reach the correct Cloud Run backend
# without baking URLs into index.html. BACKEND_PUBLIC_URL is set by Cloud Run / CI.
python3 << 'PY'
import os

url = (os.environ.get("BACKEND_PUBLIC_URL") or "").strip().rstrip("/")
path = "demo/config.js"
if url:
    escaped = url.replace("\\", "\\\\").replace("'", "\\'")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"window.__IIKSHANA_BACKEND_BASE__ = '{escaped}';\n")
else:
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.__IIKSHANA_BACKEND_BASE__ = '';\n")
PY

exec python -m http.server 8080 --directory demo
