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

# DVC + GCS (optional): full_pipeline_dag runs dvc_pull/dvc_push when these are set
if ! grep -q "DVC_GCS_BUCKET" .env 2>/dev/null; then
  echo "# DVC: set your GCS bucket name to enable dvc pull/push in full_pipeline_dag" >> .env
  echo "# DVC_GCS_BUCKET=your-bucket-name" >> .env
fi
if ! grep -q "GOOGLE_APPLICATION_CREDENTIALS" .env 2>/dev/null; then
  echo "# GCS credentials path (repo root .secrets/; create from GCP service account key)" >> .env
  echo "# GOOGLE_APPLICATION_CREDENTIALS=/workspace/.secrets/gcp-dvc-key.json" >> .env
fi

# Alerts (optional): email + Slack on task failure
if ! grep -q "ALERT_EMAIL" .env 2>/dev/null; then
  echo "# Comma-separated emails for failure alerts (e.g. anomaly_detection_dag)" >> .env
  echo "# ALERT_EMAIL=you@example.com" >> .env
fi
if ! grep -q "SLACK_WEBHOOK_URL" .env 2>/dev/null; then
  echo "# Slack Incoming Webhook URL for failure alerts" >> .env
  echo "# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/..." >> .env
fi
if ! grep -q "AIRFLOW__SMTP__SMTP_HOST" .env 2>/dev/null; then
  echo "# SMTP for email_on_failure (see .env.example for full list)" >> .env
  echo "# AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com" >> .env
  echo "# AIRFLOW__SMTP__SMTP_PORT=587" >> .env
  echo "# AIRFLOW__SMTP__SMTP_USER=..." >> .env
  echo "# AIRFLOW__SMTP__SMTP_PASSWORD=..." >> .env
  echo "# AIRFLOW__SMTP__SMTP_MAIL_FROM=..." >> .env
fi

# Ensure repo root has .secrets for GCP key (optional)
mkdir -p "$SCRIPT_DIR/../.secrets"
if [ ! -f "$SCRIPT_DIR/../.secrets/gcp-dvc-key.json" ]; then
  echo "Note: For DVC+GCS, place your GCP service account key at repo root: .secrets/gcp-dvc-key.json"
fi

echo "Created/updated .env. Initializing Airflow DB and admin user..."
docker compose up airflow-init

echo ""
echo "Done. Start Airflow with: docker compose up"
echo "Then open http://localhost:8080 and log in with credentials in .env"
