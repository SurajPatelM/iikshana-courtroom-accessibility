FROM python:3.11-slim

WORKDIR /app

# System dependencies (ffmpeg for inaSpeechSegmenter, libsndfile for soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements-demo-ui.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps "inaSpeechSegmenter==0.8.0"

# Copy only the source needed by the demo app
COPY demo/ ./demo/
COPY backend/src/services/ ./backend/src/services/

# Config files (loaded via translate_text config_id)
COPY config/ ./config/
COPY prompts/ ./prompts/

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "demo/gradio_expo_app.py"]
