# Expo demo (Gradio)

**ElevenLabs** covers **Scribe** (STT) and **TTS**. **Gender** and **emotion** in the UI come from **local** models (**inaSpeechSegmenter**, **emotion2vec+** — see `requirements-demo-ui.txt`), not from ElevenLabs.

**Default: fast translate on** — after full local analysis (unless you check **Skip local gender/emotion**), **on-host** `translate_text` + **translation TTS** (no Airflow CSV wait). Optionally **ingest + trigger Airflow** in the background.

**Batch translate:** Uncheck **Fast translate** (or **`IIKSHANA_REALTIME_MODE=0`**) to use **Airflow** CSV translation (**`expo_translation_dag`** by default). **Wait for DAG translation** uses **`EXPO_POLL_SEC`** (default **1** s). **`IIKSHANA_SKIP_LOCAL_ML=1`** defaults the skip-ML checkbox on for speed-only runs.

That Docker path uses its **own** Scribe step in the worker and needs **`ELEVENLABS_API_KEY`** in **`airflow/.env`** for batch STT when applicable.

## What “attached to the pipeline” means

Each **Run** does this **automatically** (no extra button):

1. **`data/raw/expo_ui/recording_<timestamp>…`** — raw copy of the clip (audit trail; same tree as other `data/raw/...` imports).
2. **`process_one`** from **`data-pipeline/scripts/preprocess_audio.py`** — same preprocessing as batch (target SR, mono, loudness, trim, optional courtroom-robust options from `data-pipeline/config`).
3. **`data/processed/<split>/EXPO_<timestamp>.wav`** — normalized WAV next to your dataset splits.
4. **`manifest.json` append** — one row `{ file, dataset: "EXPO", speaker_id: "live_ui", emotion: "neutral" }` so tooling that expects manifest + WAVs (e.g. **`build_translation_inputs_from_audio.py`**, Airflow **`build_translation_inputs_from_audio`**) sees this file in the chosen split (**dev** / **test** / **holdout**).

Then STT and translation run on the **processed** WAV so the demo matches what the model stack would read from disk.

### Airflow (automatic model stages)

With **Docker Compose** running in `airflow/` (`docker compose up`), each **Run** ingests and triggers the DAG (default **`expo_translation_dag`**). The app uses `docker compose exec` like:

`docker compose exec -T airflow-scheduler airflow dags trigger expo_translation_dag --conf '{"split":"dev","refresh_inputs":true,"manifest_tail":1,"translate_delay":0,"config_id":"translation_flash_v1"}'`

(Adjust split, manifest tail, and translation config in the UI.)

1. **Unpause** **`expo_translation_dag`** (or **`model_pipeline_dag`** if you switched) once in http://localhost:8080 (new DAGs often start paused).
2. **`refresh_inputs`** removes stale `translation_inputs.csv` so **build_translation_inputs_from_audio** runs again. **`expo_translation_dag`** does **not** run config search. For **`model_pipeline_dag`**, optional **`refresh_config_search`** clears `config_search_results.json`.

**Low-latency env (optional, on the Gradio host):** `EXPO_POLL_SEC` (CSV poll interval, default 3), `EXPO_TRANSLATE_DELAY` / `EXPO_STT_DELAY` (passed into DAG conf; default 0), `EXPO_TRANSLATION_CONFIG_ID` (default dropdown if unset).

Override paths via env: `AIRFLOW_COMPOSE_DIR`, `AIRFLOW_SCHEDULER_SERVICE`, `AIRFLOW_MODEL_DAG_ID` (default `expo_translation_dag`).

Set **`ELEVENLABS_API_KEY`** (and **`GROQ_API_KEY`**, etc.) in **`airflow/.env`** and/or **`.secrets/.env`** at the repo root — `docker-compose.yaml` loads both via **`env_file`** so values are not overwritten by empty `${VAR:-}` entries. After editing, run **`docker compose down && docker compose up`** from `airflow/`. See **`airflow/.env.example`**.

Each Gradio ingest also writes **`EXPO_*.wav.scribe.txt`** with the **local** Scribe transcript; **`build_translation_inputs_from_audio`** uses that sidecar when container STT returns empty, so batch translation can still run without container Scribe.

## Run

From the **repository root**:

```bash
source .venv/bin/activate   # optional
pip install -r requirements.txt   # includes gradio + elevenlabs
export PYTHONPATH=.
export ELEVENLABS_API_KEY=...   # STT + TTS on the main demo; also set in airflow/.env for DAG STT
export GROQ_API_KEY=...         # still required for Groq-backed *translation* configs in the DAG
# or: set -a && source .env && set +a

python demo/gradio_expo_app.py
```

Open the URL Gradio prints (default **http://127.0.0.1:7860**). Allow **microphone** access in the browser.

Optional: `GRADIO_SERVER_PORT=7861 python demo/gradio_expo_app.py`

The app defaults to **`127.0.0.1`** so the **browser microphone** works (open **http://127.0.0.1:7860**, not `http://0.0.0.0:…`). For access from other machines on the LAN, run `GRADIO_SERVER_NAME=0.0.0.0 python demo/gradio_expo_app.py` and prefer **upload** or HTTPS for mic capture.

For **gender/emotion** models, also: `pip install -r requirements-demo-ui.txt` and the `inaSpeechSegmenter` line documented in that file.

## Notes

- **EXPO** clips do not use RAVDESS phrase IDs; gold **reference_translation** for those rows may be empty unless you extend the mapping. The UI still gives **live transcript + translation**.
- If the mic widget is flaky on Safari, try **Chrome** or **file upload**.
- **Airflow** does not start from the UI; reruns that need the new file should **see** it on the shared volume after ingest (same paths as above).
