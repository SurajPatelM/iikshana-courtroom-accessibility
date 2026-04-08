export const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:8000";

export const WS_URL =
  process.env.REACT_APP_WS_URL || "ws://localhost:8000/ws/audio";

export const VALID_SPLITS = ["dev", "test", "holdout"] as const;
export const TARGET_LANGUAGES = ["es", "fr", "de"] as const;

export const POLL_INTERVAL_MS = 12_000;
export const MAX_WAIT_MS = 45 * 60 * 1000;