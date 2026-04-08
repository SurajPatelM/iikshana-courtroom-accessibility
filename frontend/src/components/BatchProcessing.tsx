import { useCallback, useRef, useState } from "react";
import {
  triggerPipeline,
  getPipelineStatus,
  getPipelineResult,
} from "../services/api";
import { POLL_INTERVAL_MS, MAX_WAIT_MS } from "../utils/constants";
import type {
  Split,
  TargetLanguage,
  PipelineResultResponse,
} from "../types";
import { LANGUAGE_LABELS } from "../types";

export default function BatchProcessing() {
  const [split, setSplit] = useState<Split>("dev");
  const [targetLanguage, setTargetLanguage] = useState<TargetLanguage>("es");
  const [rerunConfigSearch, setRerunConfigSearch] = useState(false);
  const [manifestTail, setManifestTail] = useState(200);

  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<PipelineResultResponse | null>(null);
  const [error, setError] = useState("");
  const [filename, setFilename] = useState("");

  const hasAudio = recordedBlob !== null || uploadedFile !== null;

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
        setRecordedBlob(blob);
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setIsRecording(true);
    } catch {
      setError("Microphone access denied.");
    }
  }, []);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    setUploadedFile(file);
  };

  const handleTrigger = async () => {
    const audio = uploadedFile ?? recordedBlob;
    if (!audio) return;

    setLoading(true);
    setError("");
    setResult(null);
    setProgress(0);
    setStatusText("Preprocessing audio & triggering pipeline…");

    try {
      const { job_id, filename: fn } = await triggerPipeline(audio, {
        split,
        targetLanguage,
        rerunConfigSearch,
        manifestTail,
      });
      setFilename(fn);
      setStatusText("Pipeline triggered — waiting for translation…");

      // Poll for completion
      const deadline = Date.now() + MAX_WAIT_MS;
      while (Date.now() < deadline) {
        const status = await getPipelineStatus(job_id);
        const elapsed = MAX_WAIT_MS - (deadline - Date.now());
        setProgress(Math.min(elapsed / MAX_WAIT_MS, 0.99));

        if (status.status === "completed") {
          const res = await getPipelineResult(job_id);
          setResult(res);
          setProgress(1);
          setStatusText("");
          speakText(res.translated_text, targetLanguage);
          break;
        }
        if (status.status === "failed") {
          setError(status.message ?? "Pipeline failed. Check Airflow UI.");
          break;
        }

        const left = deadline - Date.now();
        const mins = Math.floor(left / 60_000);
        const secs = Math.floor((left % 60_000) / 1000);
        setStatusText(
          `Polling every ${POLL_INTERVAL_MS / 1000}s — ${fn} · ~${mins}m ${secs}s remaining`
        );

        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }

      if (!result && !error) {
        setError(
          "Timed out waiting for predictions. The DAG may still be running — check Airflow UI."
        );
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Pipeline trigger failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Pipeline config card */}
      <div className="input-card">
        <h3>⚙️ Pipeline Configuration</h3>
        <div className="form-row">
          <label>
            Split
            <select
              value={split}
              onChange={(e) => setSplit(e.target.value as Split)}
            >
              <option value="dev">dev</option>
              <option value="test">test</option>
              <option value="holdout">holdout</option>
            </select>
          </label>

          <label>
            Translate to
            <select
              value={targetLanguage}
              onChange={(e) =>
                setTargetLanguage(e.target.value as TargetLanguage)
              }
            >
              <option value="es">Spanish 🇪🇸</option>
              <option value="fr">French 🇫🇷</option>
              <option value="de">German 🇩🇪</option>
            </select>
          </label>
        </div>

        <div className="form-row">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={rerunConfigSearch}
              onChange={(e) => setRerunConfigSearch(e.target.checked)}
            />
            Re-run config search (slow)
          </label>

          <label>
            Manifest tail rows (STT cap)
            <input
              type="number"
              min={1}
              max={500}
              value={manifestTail}
              onChange={(e) => setManifestTail(Number(e.target.value))}
            />
          </label>
        </div>
      </div>

      {/* Audio input card */}
      <div className="input-card">
        <h3>🎙️ Audio Input</h3>

        <div className="audio-controls">
          {isRecording ? (
            <button className="btn-record recording" onClick={stopRecording}>
              ⏹ Stop Recording
            </button>
          ) : (
            <button className="btn-record" onClick={startRecording}>
              🎙️ Record courtroom audio
            </button>
          )}
          {recordedBlob && !isRecording && (
            <span className="file-info">Recording captured ✓</span>
          )}
        </div>

        <div className="upload-section">
          <label className="upload-label">
            Or upload an audio file
            <input
              type="file"
              accept=".wav,.mp3,.webm,.m4a,.ogg"
              onChange={handleFileChange}
            />
          </label>
          {uploadedFile && (
            <span className="file-info">{uploadedFile.name}</span>
          )}
        </div>
      </div>

      {/* Trigger button */}
      <button
        className="btn-primary"
        disabled={!hasAudio || loading}
        onClick={handleTrigger}
      >
        ⚖️ Save Clip &amp; Trigger Airflow Pipeline
      </button>

      {/* Status area */}
      {!loading && !result && !error && (
        <div className="info-box">
          <strong>Getting started:</strong> run{" "}
          <code>cd airflow &amp;&amp; docker compose up</code> from the repo
          root, then unpause <strong>model_pipeline_dag</strong> in the Airflow
          UI if needed.
        </div>
      )}

      {loading && (
        <div className="status-area">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
          <p className="status-caption">{statusText}</p>
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      {/* Result card */}
      {result && (
        <>
          <hr className="divider" />
          <div className="input-card">
            <h3>
              🌐 Translation Result ({LANGUAGE_LABELS[targetLanguage]})
            </h3>
          </div>
          <div className="result-text">{result.translated_text}</div>
          {filename && (
            <p className="caption">
              💡 The filename <code>{filename}</code> contains the UTC save
              time. Use it to locate the row in the predictions CSV.
            </p>
          )}
          <details className="pipeline-details">
            <summary>Pipeline details</summary>
            <p>
              <strong>Best config:</strong> <code>{result.best_config}</code>
            </p>
            <p>
              <strong>Predictions file:</strong>{" "}
              <code>{result.predictions_file}</code>
            </p>
          </details>
        </>
      )}
    </div>
  );
}

function speakText(text: string, lang: string) {
  if (!window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = lang;
  window.speechSynthesis.speak(utter);
}
