# Model Pipeline (Clean Runbook)

This folder contains the model-development flow.
It runs on top of `data/processed/<split>/` produced by the data pipeline.

Core idea: we do **not** train external API model weights. We tune and evaluate
**model configuration** (provider/model + prompts + decoding params), then
package that configuration and push it to registry.

---

## Current Status 

- The model pipeline covers end-to-end on top of `data/processed/<split>/`.
- The flow is **config-driven** (YAML + prompts), not hardcoded.
- It runs translation experiments, ranks configs, validates a selected config, performs fairness slicing, and packages selected configs for registry.

### End-to-end flow currently in use

1. **1 Input prep**
   - Build eval tables from audio/manifests (`translation_inputs`), court phrases (`court_translation_inputs`), and optional merged tables (`combined_translation_inputs`).
2. **2 Config search**
   - Run multiple configs from `config/models/*.yaml`.
   - Compare BLEU/chrF (+ glossary when available).
   - Save ranked results and pick a best config.
3. **3 Validation**
   - Run selected config on split.
   - Compute BLEU, chrF, exact-match (+ optional glossary).
   - Save JSON/CSV and plots under `data/processed/<split>/`.
4. **Fairness slices**
   - Compute per-group exact-match on sensitive columns (e.g. `emotion`, `speaker_id`).
   - Save `translation_bias_metrics_<config>.json`.
5. **Registry packaging**
   - Build model package tarball (manifest + config + prompts).
   - Push to GCP Artifact Registry (generic repo).

### Primary scripts used

- `build_translation_inputs_from_audio.py`
- `build_translation_inputs_from_court_phrases.py`
- `build_combined_translation_inputs.py`
- `run_config_search.py`
- `run_validation.py`
- `run_translation_bias_analysis.py`
- `build_model_package.py`
- `push_model_to_registry.py`

### Main output artifacts

- `config_search_results*.json` (ranking + winner)
- `validation_metrics.json` / `validation_metrics.csv` (+ plots) 
- `translation_bias_metrics_<config>.json` 
- `model-pipeline/artifacts/*.tar.gz` and corresponding Artifact Registry package entries 

### Airflow status

- `full_pipeline_dag` can run data stages followed by model stages.
- CLI remains the fastest path for local iteration/debugging.

---

## Scope 

- Build/prepare evaluation inputs (`translation_inputs`, `court_translation_inputs`, `combined_translation_inputs`) from pipeline outputs and courtroom phrases.
- Run prompt/config search and rank candidate configs by BLEU/chrF (+ optional glossary checks).
- Validate selected configs and save metrics report files (JSON/CSV + plots).
- Package selected model configs (manifest + prompts + config YAML) and push to GCP Artifact Registry.
- Added **modeling-side translation bias detection** (separate from `data-pipeline/scripts/detect_bias.py`):
  - `model-pipeline/scripts/run_model_bias_detection.py`
  - `model-pipeline/scripts/model_bias_detection_core.py`
  - API mode on `translation_inputs` or no-extra-call mode from `translation_predictions_<config>` via `--from-predictions`
  - Per-slice metrics (exact match, mean sentence BLEU when available), disparity flags, mitigation text
  - Default read/write under `data/model_runs/<split>/` with fallback to `data/processed/<split>/`
  - JSON report + optional dataset bar chart; output filename suffix derived from `--group-cols`
  - Docs: `model-pipeline/docs/model_bias_detection.md`
  - Tests: `model-pipeline/tests/test_model_bias_detection_core.py`, `model-pipeline/tests/test_run_model_bias_detection_cli.py`
- Related: `run_translation_bias_analysis.py` remains available as the earlier per-group fairness script and now follows the same model-runs vs processed path behavior.
  

---

## Quick flow (PowerShell)

From repo root:

```powershell
$env:PYTHONPATH = "."
```

### Step 1 - Prepare eval inputs 

```powershell
# Audio-derived table
python model-pipeline/scripts/build_translation_inputs_from_audio.py --split dev

# Court-phrases table
python model-pipeline/scripts/build_translation_inputs_from_court_phrases.py --split dev

# Optional merged table
python model-pipeline/scripts/build_combined_translation_inputs.py --split dev
```

### Step 2 - Config search 

```powershell
python model-pipeline/scripts/run_config_search.py --split dev --inputs-basename translation_inputs
```

Optional:

```powershell
python model-pipeline/scripts/run_config_search.py --split dev --inputs-basename court_translation_inputs
```

### Step 3 - Validation report 

```powershell
python model-pipeline/scripts/run_validation.py --split dev --config-id translation_flash_v1 --task translation
```

### Step 4 - Fairness slices 

```powershell
python model-pipeline/scripts/run_translation_bias_analysis.py --split dev --config-id translation_flash_v1 --group-cols emotion,speaker_id
```

### Step 5 - Package + registry 

```powershell
python model-pipeline/scripts/build_model_package.py --config-id translation_flash_v1 --tarball
```

Then upload tarball to Artifact Registry (generic repo):

```powershell
gcloud artifacts generic upload --repository=model-packages --location=us-central1 --project=YOUR_PROJECT_ID --source="PATH_TO_TARBALL" --package=api-model-translation_flash_v1 --version=YOUR_VERSION
```

---

## Minimal outputs to share with team

- `config_search_results*.json` 
- `validation_metrics.json` / `validation_metrics.csv` 
- `translation_bias_metrics_<config>.json` 
- Artifact Registry package entries for selected configs 
