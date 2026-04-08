import type { TranscriptSegment } from "../types";

interface Props {
  transcripts: TranscriptSegment[];
}

export default function TranscriptDisplay({ transcripts }: Props) {
  if (transcripts.length === 0) {
    return (
      <div className="info-box">
        No transcripts received yet. Start recording to see real-time
        transcription and translation.
      </div>
    );
  }

  return (
    <>
      {transcripts.map((t, i) => (
        <div className="transcript-card" key={i}>
          <div className="speaker-label">
            {t.speaker_id} ({t.speaker_role}) – {t.start_time.toFixed(1)}s
          </div>
          <div className="original-text">{t.text}</div>
          {t.translated_text && (
            <div className="translated-text">{t.translated_text}</div>
          )}
        </div>
      ))}
    </>
  );
}