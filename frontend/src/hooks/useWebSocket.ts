import { useCallback, useEffect, useRef, useState } from "react";
import { WS_URL } from "../utils/constants";
import type { TranscriptSegment, WebSocketConfig } from "../types";

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [transcripts, setTranscripts] = useState<TranscriptSegment[]>([]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        setConnected(true);
        setStatusMessage("Connected to backend");
        const config: { type: string; config: WebSocketConfig } = {
          type: "config",
          config: {
            speaker_diarization: true,
            source_language: "en",
            target_language: "es",
            config_id: "translation_flash_court",
          },
        };
        ws.send(JSON.stringify(config));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if ("state" in data) {
            setStatusMessage(data.message ?? "");
            return;
          }
          setTranscripts((prev) => {
            const next = [...prev, data as TranscriptSegment];
            return next.length > 50 ? next.slice(-50) : next;
          });
        } catch {
          // ignore non-JSON
        }
      };

      ws.onclose = () => {
        setConnected(false);
        setStatusMessage("Connection lost");
      };

      ws.onerror = () => {
        setConnected(false);
        setStatusMessage("Connection error");
      };

      wsRef.current = ws;
    } catch (err) {
      setConnected(false);
      setStatusMessage(`Connection failed: ${err}`);
    }
  }, []);

  const sendAudio = useCallback((audioData: ArrayBuffer) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const b64 = btoa(
      new Uint8Array(audioData).reduce(
        (acc, byte) => acc + String.fromCharCode(byte),
        ""
      )
    );
    ws.send(JSON.stringify({ type: "audio", data: b64 }));
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  return { connected, statusMessage, transcripts, connect, sendAudio, disconnect };
}