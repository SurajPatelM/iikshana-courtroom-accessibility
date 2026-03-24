# Deployment Guide

This guide covers how to deploy the Iikshana system, from local development to production on court provided devices.

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend and pipeline scripts |
| Node.js | 18+ | Frontend build and development |
| npm | comes with Node | Frontend package management |
| Docker | 20+ | Container builds and Airflow |
| Docker Compose | v2+ | Multi container orchestration (Airflow) |
| Git | any recent | Version control |

You also need API keys for the external services the backend calls (see Environment Variables below).

## Environment Variables

### Backend

| Variable | Purpose |
|----------|---------|
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Google Gemini for translation, vision, and audio analysis |
| `GROQ_API_KEY` | Groq Whisper for speech to text |
| `HF_API_TOKEN` | Hugging Face Inference API |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON for Text to Speech |

### CI/CD (GitHub Secrets)

| Secret | Purpose |
|--------|---------|
| `GCP_SA_KEY` | Base64 encoded GCP service account key |
| `GCP_PROJECT_ID` | GCP project ID |
| `ARTIFACT_REGISTRY_LOCATION` | GCP region (e.g., `us-central1`) |
| `ARTIFACT_REGISTRY_REPO` | Docker repository name in Artifact Registry |
| `GROQ_API_KEY` | For model evaluation in CI |
| `SMTP_SERVER` | Email server for CI notifications |
| `SMTP_PORT` | Email server port |
| `SMTP_USERNAME` | Email account username |
| `SMTP_PASSWORD` | Email account password |
| `NOTIFICATION_EMAIL` | Where to send CI status emails |

For details on setting up GitHub Secrets, see `.github/workflows/README.md`.

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn[standard] google-genai websockets pydantic python-dotenv pyyaml requests numpy
python src/main.py
```

Runs on `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm start
```

Runs on `http://localhost:3000`. Connects to the backend at `localhost:8000` by default.

### Data Pipeline (Airflow)

```bash
cd airflow
bash setup.sh
docker compose up
```

Airflow UI at `http://localhost:8080` (default credentials: `airflow` / `airflow`).

See `airflow/README.md` for DAG details and troubleshooting.

## Docker Deployment

### Backend Image

The backend has a Dockerfile at `backend/Dockerfile` that builds a production image:

```bash
docker build -f backend/Dockerfile -t iikshana-backend .
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_key \
  -e GROQ_API_KEY=your_key \
  iikshana-backend
```

The build context is the repository root. The image is based on Python 3.10 slim and includes system dependencies for audio processing (ffmpeg, libsndfile). A health check pings `/health` every 30 seconds.

### Frontend Build

```bash
cd frontend
npm run build
```

This produces a static build in `frontend/build/` that can be served by any web server (nginx, Apache, a CDN, or a simple static file server).

## CI/CD Automated Deployment

The GitHub Actions pipeline handles deployment automatically:

1. **CI workflow** (`ci.yml`): runs on every push and pull request. Tests backend, frontend, and data pipeline. If model related files changed, runs model evaluation and quality gates. If anything fails, deployment is blocked.

2. **Backend deploy** (`deploy-backend.yml`): triggers automatically after CI passes on the `main` branch. Builds the Docker image, pushes it to GCP Artifact Registry.

3. **Frontend deploy** (`deploy-frontend.yml`): triggers automatically after CI passes on `main`. Builds the frontend and uploads the artifact.

4. **Model package push**: if model files changed and all quality gates passed, the CI workflow pushes a model configuration package to a generic Artifact Registry repository.

The full CI/CD pipeline is documented in `.github/workflows/README.md`.

## Production: On Premises Deployment

Iikshana is designed for deployment on court provided hardware within the court's local network. The core principle is that no live courtroom audio leaves the premises.

### What this means in practice

- The backend, frontend, and any required model inference run on local machines
- In a production setup, the external API calls (Gemini, Groq, Google TTS) would be replaced with locally hosted model endpoints. The agent and service layer is structured so that swapping from a cloud API to a local endpoint only requires changing the service configuration, not the agent logic
- No courtroom audio is stored, logged, or transmitted outside the local network
- Public datasets used for offline evaluation (in the data and model pipelines) are the only external data involved, and those never contain real courtroom recordings

### Hardware considerations

- Sufficient CPU/RAM for running inference models locally
- Low latency audio I/O (microphones and speakers for the courtroom)
- Restricted network egress (block outbound connections to enforce data sovereignty)
- Standard web server to host the frontend static build

### Network architecture

```
Court Local Network
├── Backend server (port 8000)
├── Frontend (served via web server, port 443 or 80)
├── Local model inference endpoints (replacing cloud APIs)
└── Microphone / speaker hardware
```

All communication stays within the local network. The frontend connects to the backend via WebSocket and HTTP over the LAN.

## Monitoring

- The backend exposes `/health` for uptime checks
- Docker includes a built in health check in the Dockerfile
- CI/CD sends email notifications on pipeline success or failure
- Model evaluation metrics are tracked as CI artifacts and compared against baselines in the rollback check
