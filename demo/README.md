# Expo demo (Gradio)

**Single Gradio flow:** **microphone or file** → ElevenLabs **Scribe v2** → gender/emotion models (host-only, **not** in Docker) → **ElevenLabs TTS** (spoken summary) → **ingest + `model_pipeline_dag`** (batch translation to predictions CSV) → **ElevenLabs TTS** (translated text). **Translation in the UI comes only from the DAG** (CSV poll when **Wait for DAG translation** is on). The DAG still uses Groq (or your configured backends) for the actual translate step.

That Docker path uses its **own** Scribe step in the worker and needs **`ELEVENLABS_API_KEY`** in **`airflow/.env`** for batch STT when applicable.

## What “attached to the pipeline” means

Each **Run** does this **automatically** (no extra button):

1. **`data/raw/expo_ui/recording_<timestamp>…`** — raw copy of the clip (audit trail; same tree as other `data/raw/...` imports).
2. **`process_one`** from **`data-pipeline/scripts/preprocess_audio.py`** — same preprocessing as batch (target SR, mono, loudness, trim, optional courtroom-robust options from `data-pipeline/config`).
3. **`data/processed/<split>/EXPO_<timestamp>.wav`** — normalized WAV next to your dataset splits.
4. **`manifest.json` append** — one row `{ file, dataset: "EXPO", speaker_id: "live_ui", emotion: "neutral" }` so tooling that expects manifest + WAVs (e.g. **`build_translation_inputs_from_audio.py`**, Airflow **`build_translation_inputs_from_audio`**) sees this file in the chosen split (**dev** / **test** / **holdout**).

Then STT and translation run on the **processed** WAV so the demo matches what the model stack would read from disk.

### Airflow (automatic model stages)

With **Docker Compose** running in `airflow/` (`docker compose up`), each **Run** on the main button ingests and triggers the DAG. The app runs:

`docker compose exec -T airflow-scheduler airflow dags trigger model_pipeline_dag --conf '{"split":"dev","refresh_inputs":true}'`

(Adjust split via the UI selector.)

1. **Unpause** `model_pipeline_dag` once in http://localhost:8080 (new DAGs often start paused).
2. **`refresh_inputs`** deletes existing `translation_inputs` / `config_search_results` for that split (under `data/model_runs` or `data/processed`) so **build_translation_inputs_from_audio** and **config search** run again.

Override paths via env: `AIRFLOW_COMPOSE_DIR`, `AIRFLOW_SCHEDULER_SERVICE`, `AIRFLOW_MODEL_DAG_ID`.

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
