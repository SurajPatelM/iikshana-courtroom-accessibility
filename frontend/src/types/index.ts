export type Split = "dev" | "test" | "holdout";
export type TargetLanguage = "es" | "fr" | "de";

export interface PipelineConfig {
  split: Split;
  targetLanguage: TargetLanguage;
  rerunConfigSearch: boolean;
  manifestTail: number;
}

export interface PipelineTriggerResponse {
  job_id: string;
  filename: string;
}

export interface PipelineStatusResponse {
  status: "running" | "completed" | "failed";
  progress: number;
  message?: string;
}

export interface PipelineResultResponse {
  translated_text: string;
  best_config: string;
  predictions_file: string;
  target_language: string;
}

export interface TranscriptSegment {
  speaker_id: string;
  speaker_role: string;
  text: string;
  translated_text?: string;
  start_time: number;
}

export interface WebSocketConfig {
  speaker_diarization: boolean;
  source_language: string;
  target_language: string;
  config_id: string;
}

export const LANGUAGE_LABELS: Record<TargetLanguage, string> = {
  es: "Spanish 🇪🇸",
  fr: "French 🇫🇷",
  de: "German 🇩🇪",
};