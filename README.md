## Iikshana: On-Prem Courtroom Language Access System

### ADA-Aligned Audio Accessibility (Assistive, Not Authoritative)

Iikshana is an **on-prem, AI-assisted courtroom accessibility system** that provides real-time spoken-language support for participants who are blind but can hear.

The system offers **live speech recognition, translation, and audio playback**, while preserving interpreter authority and court data sovereignty.

> **Assistive only** — does **not** replace certified interpreters, does **not** produce official court records, and may require human oversight.

---

### Team — Group 16 (IE7374 MLOps)

- Aditya Vasisht
- Akshata Kumble
- Amit Karanth Gurpur
- Rohit Abhijit Kulkarni
- Shridhar Sunilkumar Pol
- Suraj Patel Muthe Gowda

---

## Project Overview

### What the system does

1. Captures live courtroom audio (e.g., judge and witness channels).
2. Converts speech → text (speech-to-text).
3. Translates text between languages.
4. Converts translated text → speech (text-to-speech).
5. Streams assistive audio back to the listener in near real time.

---

## Data & Deployment Principles

### On-Prem Data Policy

The system is designed for **local, on-premises deployment only**:

- 100% of processing occurs on-prem.
- No outbound API calls with live courtroom audio.
- No storage of live courtroom recordings.
- No reuse of operational audio for training.
- Public research datasets are used **only** for offline benchmarking.

### Input Pipeline (Deterministic Preprocessing)

Audio undergoes minimal, inference-style preprocessing before model or API calls:

- Conversion to 16 kHz mono WAV.
- Loudness normalization.
- Optional silence trimming.
- Format and basic-quality validation.

No enhancement, denoising, or content modification is applied.  
**Goal:** stable, predictable input for models and APIs across devices and microphones.

---

## Architecture Overview

The system is organized into three main layers:

- **Backend**: FastAPI-based backend orchestrating AI agents, Gemini calls, and WebSocket streaming.
- **Frontend**: React PWA with WCAG 2.1 Level AAA–oriented design for blind or low-vision participants.
- **Data Pipeline**: Inference-style data pipeline for evaluation, bias analysis, and quality checks using Airflow, DVC, and Great Expectations–style validation.

### High-Level Flow

1. **Audio Capture** → microphone / courtroom audio feed.
2. **Preprocessing** → convert to standard 16 kHz mono, normalize, validate.
3. **Speech-to-Text** → streaming transcription with confidence scoring.
4. **Translation** → multilingual translation with legal glossary enforcement.
5. **Text-to-Speech** → low-latency audio output back to the listener.
6. **Human-in-the-loop** → interpreters may override or correct any output.

### Human-in-the-Loop Safeguards

- Human interpreters can override outputs at any time.
- Low-confidence segments are flagged for review.
- System never claims official or authoritative translation.
- AI output is always labeled as **assistive**.

---

## Repository Structure

```text
iikshana-courtroom-accessibility/
├── backend/                 # FastAPI backend + AI agents
│   └── src/
│       ├── agents/          # Agentic orchestration (Gemini, translation, etc.)
│       ├── api/             # HTTP + WebSocket handlers
│       ├── models/          # Schemas, enums, backend README
│       ├── services/        # Gemini, TTS, WebSocket, utilities
│       └── main.py          # Backend entry point (FastAPI app)
├── frontend/                # React PWA (WCAG-oriented UI)
│   ├── src/
│   │   ├── components/      # Accessible components and layouts
│   │   ├── services/        # API clients, WebSocket client
│   │   ├── hooks/           # Custom React hooks
│   │   └── App.tsx
│   └── README.md            # Frontend-specific instructions
├── data/                    # Evaluation and glossary data (no live court data)
│   ├── legal_glossary/      # Legal terms and glossaries
│   └── README.md
├── data-pipeline/           # Data pipeline (scripts, tests, config)
│   ├── scripts/             # Acquisition, preprocessing, validation, evaluation
│   ├── tests/               # Preprocessing / validation / split tests
│   ├── config/              # Dataset configuration
│   └── README.md
├── model-pipeline/          # Model pipeline (runs after data pipeline; manual trigger)
│   ├── scripts/             # Apply model + prompts to processed data
│   └── README.md            # Task 1: Clarify your setup + how to run
├── airflow/                 # Airflow DAGs + Docker setup (data + model pipelines)
│   ├── dags/                # full_pipeline_dag and stage DAGs
│   ├── docker-compose.yaml
│   └── README.md
├── docs/                    # Design and usage documentation
│   ├── architecture.md
│   ├── api_documentation.md
│   ├── user_manual.md
│   ├── deployment_guide.md
│   └── agent_specifications.md
├── .github/workflows/       # CI/CD pipeline (GitHub Actions)
│   ├── ci.yml               # Main CI workflow (tests, evaluation, gates)
│   ├── deploy-backend.yml   # Backend Docker deploy (main only)
│   ├── deploy-frontend.yml  # Frontend build deploy (main only)
│   └── README.md            # CI/CD pipeline documentation
├── scripts/                 # CI/CD quality gate scripts
│   ├── quality_gate.py      # Enforces BLEU, chrF, exact match thresholds
│   ├── bias_gate.py         # Enforces bias detection thresholds
│   ├── config_search_gate.py # Validates config search results
│   └── rollback_check.py    # Compares metrics against baseline to prevent regression
├── config/                  # Environment-specific app configuration
│   ├── development.yaml
│   ├── testing.yaml
│   └── production.yaml
└── requirements.txt         # Data pipeline / tooling deps (Python 3.10+)
```

---

## Getting Started

### Prerequisites

- **Python**: 3.10+
- **Node.js**: 18+ (for the React frontend)
- **npm**: comes with Node
- **Docker + Docker Compose** (for Airflow-based pipeline orchestration; optional but recommended)

> For production or realistic evaluation, ensure a machine with sufficient CPU/RAM, low-latency audio IO, and restricted network egress.

---

### 1. Clone the repository

```bash
git clone https://github.com/SurajPatelM/iikshana-courtroom-accessibility.git
cd iikshana-courtroom-accessibility
```

---

### 2. Set up the Python environment (data pipeline and tooling)

From the **repository root**:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The root `requirements.txt` contains dependencies used by the **data pipeline**, Airflow integration, and related tooling (e.g., numpy, pandas, DVC, Fairlearn, Gemini client).

For detailed pipeline usage, see `data-pipeline/README.md`.

---

### 3. Run the backend API (FastAPI prototype)

From the **repository root**, in a separate terminal:

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install backend dependencies (example; adjust as needed)
pip install fastapi uvicorn[standard] google-genai websockets pydantic python-dotenv
```

Then start the backend:

```bash
python src/main.py
# or, once the FastAPI app is wired:
# uvicorn src.main:app --reload
```

By default this will expose a FastAPI app (and WebSocket endpoints) on `http://localhost:<port>` as configured in `src/main.py` and `config/*.yaml`.

For backend internals (agents, services, Docker), see `backend/README.md`.

---

### 4. Run the frontend (React PWA)

From the **repository root**, in another terminal:

```bash
cd frontend
npm install
npm start
```

This will start the React app (Create React App / React Scripts) on `http://localhost:3000` by default.

**Features:**

- High-contrast themes and keyboard-friendly navigation.
- Screen-reader-first flows for blind participants.
- Real-time status indicators for connection and translation state.

For more details, see `frontend/README.md`.

---

### 5. Airflow + Data Pipeline (optional, for evaluation and bias analysis)

The data pipeline uses **Apache Airflow**, **DVC**, and validation tooling to:

- Download and preprocess public speech / emotion datasets.
- Enforce inference-style preprocessing (16 kHz mono, normalization).
- Validate API-input quality (schema + statistics).
- Run evaluation against APIs (STT, translation, emotion).
- Detect bias and anomalies across slices (emotion, speaker, etc.).

#### 5.1 Set up pipeline dependencies

You already installed the root `requirements.txt` in step 2, which covers the data-pipeline scripts and most tools.

#### 5.2 Start Airflow via Docker

From the **repository root**:

```bash
cd airflow
bash setup.sh          # one-time: .env + airflow-init
docker compose up      # Airflow web UI at http://localhost:8080
```

- Default login: **username** `airflow`, **password** `airflow` (configurable in `.env`).
- DAGs live in `airflow/dags/` and orchestrate the data pipeline defined in `data-pipeline/`.

For details on DAGs, data layout, and troubleshooting, see `airflow/README.md` and `data-pipeline/README.md`.

#### 5.3 Model pipeline (manual, after data pipeline)

The **model pipeline** runs separately from the data pipeline. After the data pipeline has produced processed splits (e.g. under `data/processed/emotions/`), trigger the **model pipeline** manually to apply the Gemini model and prompts to that data:

- **Via Airflow**: Run the **`model_pipeline_dag`** from the Airflow UI (manual trigger only; it is not part of `full_pipeline_dag`).
- **Via CLI**: From repo root, `PYTHONPATH=. python model-pipeline/scripts/model_setup.py --split dev --config-id translation_flash_v1` (uses existing splits under `data/processed/emotions/`; add `data/processed/emotions/<split>/translation_inputs.csv` to run translation).

See `model-pipeline/README.md` for Task 1 setup and usage.

---

## Evaluation, Monitoring, and Bias

### Core Metrics (example targets)

| Metric                    | Target      |
| ------------------------- | ----------- |
| End-to-end latency        | ≤ 2 seconds |
| Translation latency       | ≤ 1 second  |
| WER (benchmark audio)     | < 10%       |
| Glossary enforcement      | ≥ 95%       |
| Interpreter override rate | < 20%       |

### Bias Detection

The pipeline can produce reports such as `data/processed/bias_report.json` with:

- Per-emotion and per-speaker counts.
- Fairlearn-based per-group metrics.
- Detected disparities and mitigation recommendations.

The **data pipeline** scripts for offline evaluation and bias detection live in `data-pipeline/scripts/`:

- `data-pipeline/scripts/detect_bias.py`
- `data-pipeline/scripts/anomaly_check.py`

See `data-pipeline/README.md` for how to run these.

The **CI/CD quality gate** scripts in the root `scripts/` folder enforce thresholds automatically during the pipeline:

- `scripts/quality_gate.py` (translation metric thresholds)
- `scripts/bias_gate.py` (bias detection thresholds)
- `scripts/config_search_gate.py` (config search validation)
- `scripts/rollback_check.py` (regression check against baseline)

See `.github/workflows/README.md` for how these gates fit into the CI pipeline.

---

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment. The pipeline runs tests for all three layers (data pipeline, backend, frontend), evaluates the model when relevant files change, enforces quality gates on translation metrics and bias, and blocks deployment if anything regresses.

For a full breakdown of every job, the required GitHub Secrets, and how the pipeline maps to the project requirements, see [`.github/workflows/README.md`](.github/workflows/README.md).

---

## Standards Alignment

The project is designed with reference to:

- ADA Title II (Effective Communication).
- US DOJ Language Access Guidance.
  +- NCSC Court Technology Guidance.
- NIST AI Risk Management Framework.

---

## Important Disclaimer

Iikshana is a **research prototype** developed for the Northeastern course IE7374 (MLOps).

It is:

- **Assistive**, not authoritative.
- **Not** a certified assistive technology device.
- **Not** legally certified for courtroom use.
- Always subject to **human review and oversight**.

---

## License / Intended Use

This repository is intended for **academic and research use only**.
