#!/usr/bin/env bash
# One-time setup for Airflow in Docker (run from repo root or airflow/).
# DAGs live here; pipeline scripts/config/data live in ../data-pipeline.
# Then start with: docker compose up
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Airflow Docker one-time setup ==="

# Ensure data-pipeline dirs exist for mounts
PIPELINE="../data-pipeline"
if [ ! -d "$PIPELINE" ]; then
  echo "Error: $PIPELINE not found. Run from repo root or ensure data-pipeline exists."
  exit 1
fi
mkdir -p "$PIPELINE/logs" "$PIPELINE/data/raw" "$PIPELINE/data/processed" "$PIPELINE/plugins"

# AIRFLOW_UID (official requirement)
touch .env
grep -q '^AIRFLOW_UID=' .env 2>/dev/null || echo "AIRFLOW_UID=$(id -u)" >> .env

# Shared credentials
if ! grep -q "_AIRFLOW_WWW_USER_USERNAME" .env 2>/dev/null; then
  echo "_AIRFLOW_WWW_USER_USERNAME=airflow" >> .env
  echo "_AIRFLOW_WWW_USER_PASSWORD=airflow" >> .env
  echo "Added default credentials to .env (airflow/airflow)"
fi

echo "Created/updated .env. Initializing Airflow DB and admin user..."
docker compose up airflow-init

echo ""
echo "Done. Start Airflow with: docker compose up"
echo "Then open http://localhost:8080 and log in with credentials in .env"
