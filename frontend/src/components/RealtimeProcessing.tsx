import { useCallback, useEffect, useRef, useState } from "react";
import { useWebSocket } from "../hooks/useWebSocket";
import TranscriptDisplay from "./TranscriptDisplay";
import { WS_URL } from "../utils/constants";

export default function RealtimeProcessing() {
  const {
    connected,
    statusMessage,
    transcripts,
    connect,
    sendAudio,
  } = useWebSocket();

  const [isRecording, setIsRecording] = useState(false);
  const [audioSent, setAudioSent] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Auto-connect on mount
  useEffect(() => {
    connect();
  }, [connect]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" });
        blob.arrayBuffer().then((buf) => {
          sendAudio(buf);
          setAudioSent(true);
        });
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setIsRecording(true);
      setAudioSent(false);
    } catch {
      // mic access denied
    }
  }, [sendAudio]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  }, []);

  return (
    <div>
      {/* Connection status */}
      <div className="form-row" style={{ alignItems: "flex-start" }}>
        <div className="input-card" style={{ flex: 3 }}>
          <h3>🔗 Backend Connection</h3>
          {connected ? (
            <div className="success-box">✅ Connected to backend at {WS_URL}</div>
          ) : (
            <div className="error-box">❌ Not connected to backend at {WS_URL}</div>
          )}
          {statusMessage && <p className="caption">Status: {statusMessage}</p>}
        </div>
        <div style={{ flex: 0 }}>
          <button className="btn-secondary" onClick={connect}>
            🔄 Reconnect
          </button>
        </div>
      </div>

      {/* Audio input */}
      <div className="input-card">
        <h3>🎙️ Real-Time Audio Input</h3>
        <div className="audio-controls">
          {isRecording ? (
            <button className="btn-record recording" onClick={stopRecording}>
              ⏹ Stop Recording
            </button>
          ) : (
            <button
              className="btn-record"
              onClick={startRecording}
              disabled={!connected}
            >
              🎙️ Record courtroom audio for real-time processing
            </button>
          )}
        </div>
        {audioSent && <div className="success-box">Audio sent for processing</div>}
      </div>

      {/* Live transcripts */}
      <div className="input-card">
        <h3>📝 Live Transcripts</h3>
        <TranscriptDisplay transcripts={transcripts} />
      </div>
    </div>
  );
}
