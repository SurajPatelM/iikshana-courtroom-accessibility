FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    websockets \
    pyyaml \
    requests \
    numpy \
    google-genai>=1.0.0

# Copy backend source code
COPY backend/src ./src

# demo/ — audio analysis pipeline used by routes.py
COPY demo ./demo

# config/ — translation model YAML configs used by gemini_translation.py
COPY config ./config

# prompts/ — prompt templates used by gemini_translation.py
COPY prompts ./prompts

COPY scripts ./scripts

COPY data-pipeline ./data-pipeline

COPY model-pipeline ./model-pipeline

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]