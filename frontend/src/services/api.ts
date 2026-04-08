import axios from "axios";
import { API_BASE_URL } from "../utils/constants";
import type {
  PipelineConfig,
  PipelineTriggerResponse,
  PipelineStatusResponse,
  PipelineResultResponse,
} from "../types";

const client = axios.create({ baseURL: API_BASE_URL });

export async function triggerPipeline(
  audioFile: File | Blob,
  config: PipelineConfig
): Promise<PipelineTriggerResponse> {
  const form = new FormData();
  form.append("audio", audioFile);
  form.append("split", config.split);
  form.append("target_language", config.targetLanguage);
  form.append("rerun_config_search", String(config.rerunConfigSearch));
  form.append("manifest_tail", String(config.manifestTail));

  const { data } = await client.post<PipelineTriggerResponse>(
    "/api/pipeline/trigger",
    form
  );
  return data;
}

export async function getPipelineStatus(
  jobId: string
): Promise<PipelineStatusResponse> {
  const { data } = await client.get<PipelineStatusResponse>(
    `/api/pipeline/status/${encodeURIComponent(jobId)}`
  );
  return data;
}

export async function getPipelineResult(
  jobId: string
): Promise<PipelineResultResponse> {
  const { data } = await client.get<PipelineResultResponse>(
    `/api/pipeline/result/${encodeURIComponent(jobId)}`
  );
  return data;
}