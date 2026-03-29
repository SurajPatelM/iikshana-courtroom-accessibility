# Expo demo (Streamlit)

Live **microphone or file** → **pipeline ingest** → **Groq STT** → **translation**.

## What “attached to the pipeline” means

Each **Run** does this **automatically** (no extra button):

1. **`data/raw/expo_ui/recording_<timestamp>…`** — raw copy of the clip (audit trail; same tree as other `data/raw/...` imports).
2. **`process_one`** from **`data-pipeline/scripts/preprocess_audio.py`** — same preprocessing as batch (target SR, mono, loudness, trim, optional courtroom-robust options from `data-pipeline/config`).
3. **`data/processed/<split>/EXPO_<timestamp>.wav`** — normalized WAV next to your dataset splits.
4. **`manifest.json` append** — one row `{ file, dataset: "EXPO", speaker_id: "live_ui", emotion: "neutral" }` so tooling that expects manifest + WAVs (e.g. **`build_translation_inputs_from_audio.py`**, Airflow **`build_translation_inputs_from_audio`**) sees this file in the chosen split (**dev** / **test** / **holdout**).

Then STT and translation run on the **processed** WAV so the demo matches what the model stack would read from disk.

### Airflow (automatic model stages)

With **Docker Compose** running in `airflow/` (`docker compose up`), enable **“trigger model_pipeline_dag”** in the sidebar. After each ingest, the app runs:

`docker compose exec -T airflow-scheduler airflow dags trigger model_pipeline_dag --conf '{"split":"dev","refresh_inputs":true}'`

(Adjust split via the UI selector.)

1. **Unpause** `model_pipeline_dag` once in http://localhost:8080 (new DAGs often start paused).
2. **`refresh_inputs`** deletes existing `translation_inputs` / `config_search_results` for that split (under `data/model_runs` or `data/processed`) so **build_translation_inputs_from_audio** and **config search** run again.

Override paths via env: `AIRFLOW_COMPOSE_DIR`, `AIRFLOW_SCHEDULER_SERVICE`, `AIRFLOW_MODEL_DAG_ID`.

## Run

From the **repository root**:

```bash
source .venv/bin/activate   # optional
pip install -r requirements.txt   # includes streamlit>=1.33.0
export PYTHONPATH=.
export GROQ_API_KEY=...     # or: set -a && source .env && set +a

streamlit run demo/streamlit_expo_app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). Allow **microphone** access in the browser.

## Notes

- **EXPO** clips do not use RAVDESS phrase IDs; gold **reference_translation** for those rows may be empty unless you extend the mapping. The UI still gives **live transcript + translation**.
- If the mic widget is flaky on Safari, try **Chrome** or **file upload**.
- **Airflow** does not start from the UI; reruns that need the new file should **see** it on the shared volume after ingest (same paths as above).
