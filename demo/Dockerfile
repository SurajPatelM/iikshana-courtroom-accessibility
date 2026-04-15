FROM python:3.11-slim

WORKDIR /app

# System dependencies (ffmpeg for inaSpeechSegmenter, libsndfile for soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps "inaSpeechSegmenter==0.8.0"

# Copy only the source needed by the demo app
COPY demo/ ./demo/
COPY backend/src/services/ ./backend/src/services/

# Config files (loaded via translate_text config_id)
COPY config/ ./config/
COPY prompts/ ./prompts/

EXPOSE 8080

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8080

CMD ["python", "-m", "http.server", "8080", "--directory", "demo"]
