"""
Courtroom Translation Device - Streamlit Frontend
Requirements:
    pip install streamlit streamlit-webrtc av requests pydub numpy
Run:
    streamlit run app.py
"""

import streamlit as st
import requests
import tempfile
import os
import wave
import numpy as np
import io
import queue
import time
from pathlib import Path

# --- Configuration ---
BACKEND_API_URL = st.sidebar.text_input(
    "Backend API URL",
    value="http://localhost:8000/analyze",
    help="The endpoint your ML backend exposes for audio analysis",
)
SAMPLE_RATE = 16000
CHANNELS = 1

st.set_page_config(page_title="Courtroom Translator", page_icon="⚖️", layout="wide")

st.title("⚖️ Courtroom Translation Device")
st.caption("Record courtroom audio, save as WAV, and send to the ML backend for analysis.")

# ─────────────────────────────────────────────
# Option 1: Browser-based recording via streamlit-webrtc
# ─────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

if WEBRTC_AVAILABLE:
    st.header("🎙️ Live Recording")

    if "audio_frames" not in st.session_state:
        st.session_state.audio_frames = []

    audio_queue: queue.Queue = queue.Queue()

    def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
        """Collect raw audio frames during recording."""
        audio_queue.put(frame)
        return frame

    webrtc_ctx = webrtc_streamer(
        key="courtroom-recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
        audio_frame_callback=audio_frame_callback,
    )

    if webrtc_ctx.state.playing:
        st.info("🔴 **Recording...** Click Stop to finish.")
        status_placeholder = st.empty()
        while webrtc_ctx.state.playing:
            try:
                frame = audio_queue.get(timeout=1.0)
                st.session_state.audio_frames.append(
                    frame.to_ndarray().flatten().astype(np.float32)
                )
            except queue.Empty:
                pass

    if st.session_state.audio_frames and not (
            webrtc_ctx.state.playing if webrtc_ctx else False
    ):
        st.success(
            f"Captured {len(st.session_state.audio_frames)} audio chunks."
        )

        if st.button("💾 Save & Send to Backend", key="webrtc_send"):
            # Combine frames into a single numpy array
            audio_data = np.concatenate(st.session_state.audio_frames)
            # Normalize to int16 for WAV
            audio_int16 = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

            # Write WAV to a temp file
            tmp_path = os.path.join(tempfile.gettempdir(), "courtroom_recording.wav")
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())

            st.audio(tmp_path, format="audio/wav")
            st.write(f"Saved to `{tmp_path}`")

            # Send to backend
            with st.spinner("Sending to backend for analysis..."):
                try:
                    with open(tmp_path, "rb") as f:
                        files = {"file": ("recording.wav", f, "audio/wav")}
                        response = requests.post(
                            BACKEND_API_URL, files=files, timeout=120
                        )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Analysis complete!")
                        st.json(result)
                    else:
                        st.error(
                            f"Backend returned status {response.status_code}: "
                            f"{response.text}"
                        )
                except requests.ConnectionError:
                    st.error(
                        f"Could not connect to backend at `{BACKEND_API_URL}`. "
                        "Is the server running?"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

            # Clear frames for next recording
            st.session_state.audio_frames = []

    st.divider()

# ─────────────────────────────────────────────
# Option 2: Upload a pre-recorded WAV file
# ─────────────────────────────────────────────
st.header("📁 Upload Audio File")

uploaded_file = st.file_uploader(
    "Upload a WAV file for analysis",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    help="Drag and drop or browse for an audio file.",
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")
    st.write(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.1f} KB")

    if st.button("🚀 Send to Backend for Analysis", key="upload_send"):
        with st.spinner("Sending to backend for analysis..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "audio/wav",
                    )
                }
                response = requests.post(BACKEND_API_URL, files=files, timeout=120)

                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Analysis complete!")

                    # --- Display results nicely ---
                    if "transcription" in result:
                        st.subheader("📝 Transcription")
                        st.write(result["transcription"])

                    if "translation" in result:
                        st.subheader("🌐 Translation")
                        st.write(result["translation"])

                    if "language" in result:
                        st.subheader("🗣️ Detected Language")
                        st.write(result["language"])

                    # Show full JSON in an expander
                    with st.expander("Raw API Response"):
                        st.json(result)
                else:
                    st.error(
                        f"Backend returned status {response.status_code}: "
                        f"{response.text}"
                    )

            except requests.ConnectionError:
                st.error(
                    f"Could not connect to backend at `{BACKEND_API_URL}`. "
                    "Is the server running?"
                )
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# Sidebar info
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This frontend captures courtroom audio and sends it
        to a backend ML pipeline for:

        - **Speech-to-Text** transcription
        - **Language Detection**
        - **Translation** to the target language

        **Expected backend API contract:**

        ```
        POST /analyze
        Content-Type: multipart/form-data

        Body: file=<audio.wav>

        Response (JSON):
        {
          "transcription": "...",
          "language": "es",
          "translation": "..."
        }
        ```
        """
    )
    st.divider()
    st.markdown("**Tech Stack:** Streamlit · WebRTC · FastAPI")