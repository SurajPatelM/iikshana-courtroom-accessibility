# Iikshana: On-Prem Courtroom Language Access System  
### ADA-Aligned Audio Accessibility (Assistive, Not Authoritative)

Iikshana is an **on-prem, AI-assisted courtroom accessibility system** designed to improve real-time spoken-language access for participants who are blind but can hear.  

The system provides **live speech recognition, translation, and audio playback**, while preserving interpreter authority and court data sovereignty.

> This system is assistive only.  
> It does NOT replace certified interpreters.  
> It does NOT produce official court records.  
> All outputs require human oversight.

---

# Team — Group 16 (IE7374 MLOps)

- Aditya Vasisht  
- Akshata Kumble  
- Amit Karanth Gurpur  
- Rohit Abhijit Kulkarni  
- Shridhar Sunilkumar Pol  
- Suraj Patel Muthe Gowda  

---

# System Scope

## What the system does

1. Capture live courtroom audio (Judge / Witness channels)
2. Convert speech → text (Google Speech-to-Text)
3. Translate text (Google Translation / Gemini)
4. Convert translated text → speech (Google Text-to-Speech)
5. Stream audio back to listener in real time

## What the system does NOT do

- No visual accessibility features (no OCR, no braille)
- No transcript storage for record-keeping
- No automated legal summarization
- No certification of interpretation
- No external data transmission

---

# On-Prem Data Policy

This system is designed for **local deployment only**.

- 100% of processing occurs on-prem
- No outbound API calls with courtroom audio
- No storage of live courtroom recordings
- No reuse of operational audio for training

Public research datasets are used only for offline benchmarking.

---

# Architecture Overview

## Input Pipeline (Deterministic Preprocessing)

Audio undergoes minimal signal hygiene before model inference:

- Convert to 16 kHz mono WAV
- Loudness normalization
- Silence trimming (optional)
- Format validation

No enhancement, denoising, or content modification is applied.

Purpose: Ensure stable model input across devices and microphones.

---

## Model Inference

### 1. Speech-to-Text
- Streaming transcription
- Confidence scoring
- Optional speaker attribution

### 2. Translation
- Automatic language detection
- Legal glossary enforcement
- Confidence preservation
- Timestamp alignment

### 3. Text-to-Speech
- Immediate audio playback
- No post-processing on generated audio

---

## Human-in-the-Loop Safeguards

- Human interpreters may override outputs at any time
- Low-confidence segments are flagged
- System never claims "official" translation
- AI output clearly labeled as assistive

---

## Repository Structure
```
iikshana-courtroom-accessibility/
├── backend/
│   └── src/
│       ├── agents/          # 6 AI agents + orchestrator
│       ├── services/        # Gemini, TTS, WebSocket
│       ├── api/             # Routes, handlers
│       └── main.py
├── frontend/
│   └── src/
│       ├── components/      # React UI (WCAG AAA)
│       ├── services/        # API clients
│       ├── hooks/           # Custom hooks
│       └── App.tsx
├── data/
│   ├── legal_glossary/      # 500+ legal terms
│   └── raw/                 # Evaluation datasets
├── docs/                    # Architecture, API docs
└── config/                  # Environment configs
```

**Data versioning:** Large data under `data/` is versioned with **DVC**; remote storage is **Google Cloud Storage (GCS)**. See [`docs/DVC_GCS_SETUP.md`](docs/DVC_GCS_SETUP.md) for `dvc push` / `dvc pull` setup.

---

# Quick Start (Development)

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload
```
---

## Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## Evaluation & Monitoring

The system is evaluated using public datasets and synthetic courtroom-style audio.

### Core Metrics

| Metric                    | Target        |
|---------------------------|--------------|
| End-to-end latency        | ≤ 2 seconds  |
| Translation latency       | ≤ 1 second   |
| WER (benchmark audio)     | < 10%        |
| Glossary enforcement      | ≥ 95%        |
| Interpreter override rate | < 20%        |

---

## Pipeline Vulnerability Assessment

| Stage              | Risk                      | Mitigation                     |
|-------------------|---------------------------|---------------------------------|
| Audio Capture      | Background noise          | Confidence gating               |
| Speech-to-Text     | Accent / emotion drift    | Flag low-confidence segments    |
| Translation        | Legal term distortion     | Legal glossary enforcement      |
| Language Detection | Incorrect language inference | Human confirmation fallback |
| TTS Routing        | Output routed to wrong party | Strict channel separation   |
| Deployment         | Data leakage              | On-prem firewall, no egress     |

---

## Standards Alignment

This project aligns with:

- ADA Title II (Effective Communication)
- DOJ Language Access Guidance
- NCSC Court Technology Guidance
- NIST AI Risk Management Framework

---

## Data Usage Statement

The system uses:

- Public research datasets (offline benchmarking only)
- No operational courtroom audio stored
- No training on live court data

---

## Important Disclaimer

Iikshana is a research prototype developed for IE7374 (MLOps).

It is:

- Assistive
- Non-authoritative
- Not legally certified
- Subject to human review

---

## License

Academic project use only.
