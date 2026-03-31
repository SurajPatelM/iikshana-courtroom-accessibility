# Model Pipeline (Clean Runbook)

This folder contains the model-development flow.
It runs on top of `data/processed/emotions/<split>/` produced by the data pipeline (override with `PIPELINE_DATA_DIR` for other roots, e.g. `data/processed/stt`).

Core idea: we do **not** train external API model weights. We tune and evaluate
**model configuration** (provider/model + prompts + decoding params), then
package that configuration and push it to registry.

---

## Current Status 

- The model pipeline covers end-to-end on top of `data/processed/emotions/<split>/` by default.
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
   - Save JSON/CSV and plots under `data/model_runs/<split>/` by default (legacy single root: `data/processed/<split>/`).
4. **Fairness slices**
   - **`run_translation_bias_analysis.py`:** per-group exact-match via Fairlearn; writes `translation_bias_metrics_<config>.json`.
   - **`run_model_bias_detection.py`:** slice metrics, disparity flags vs threshold, mitigation text; writes `model_bias_report_<config>__<group_suffix>.json` (+ optional dataset bar chart). See **Modeling bias detection** below.
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
- `run_model_bias_detection.py`
- `model_setup.py` (writes `translation_predictions_<config>.csv` for bias-from-predictions flows)
- `build_model_package.py`
- `push_model_to_registry.py`

### Main output artifacts

- `config_search_results*.json` (ranking + winner)
- `validation_metrics.json` / `validation_metrics.csv` (+ plots) under `data/model_runs/<split>/` by default
- `translation_bias_metrics_<config>.json`
- `model_bias_report_<config>__<group_suffix>.json` (+ optional `model_bias_by_dataset_<config>__<group_suffix>.png`)
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
  - JSON report + optional dataset bar chart; output filenames include a sanitized suffix derived from `--group-cols` (e.g. `__dataset_emotion`).
  - Full usage: see **Modeling bias detection** in this README.
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

Modeling bias (disparities + mitigation report; often run after `model_setup.py` so you can use `--from-predictions`):

```powershell
python model-pipeline/scripts/run_model_bias_detection.py --split dev --config-id translation_flash_v1 --from-predictions --group-cols dataset,emotion
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

## Modeling bias detection (`run_model_bias_detection.py`)

**Scope:** Translation **model/API** fairness — metrics and disparities **per slice** (e.g. `dataset`, `emotion`, `speaker_id`, language codes). Separate from **data-pipeline** representation bias (`data-pipeline/scripts/detect_bias.py`).

### Prerequisites

- Repo root on `PYTHONPATH` (examples below use `$env:PYTHONPATH = "."` from repo root).
- Eval table (`translation_inputs`) or predictions file (`translation_predictions_<config_id>.csv`) resolved under **`data/model_runs/<split>/` first**, then `data/processed/<split>/` (legacy `--data-dir` uses one root for both).
- `fairlearn` (root `requirements.txt`). Mean sentence BLEU in the report needs `sacrebleu` when available.

### Required columns

| Column | Role |
|--------|------|
| `source_text` | Source utterance |
| `source_language` | Source language code |
| `target_language` | Target language code |
| `reference_translation` | Gold translation for metrics |

**API mode** (default): reads `translation_inputs` (basename configurable via `--inputs-basename`); calls `translate_text` per row.

**Predictions mode** (`--from-predictions`): reads `translation_predictions_<config_id>.csv` (or `--predictions-path`) from `model_setup.py`; must include `reference_translation` and `translated_text_model` (override with `--pred-col` / `--ref-col` if needed).

**Slice columns** (`--group-cols`, default `dataset,emotion`) must exist on the table (e.g. from `build_translation_inputs_from_audio.py`).

### Outputs

Written to **`data/model_runs/<split>/`** by default:

| File | Description |
|------|-------------|
| `model_bias_report_<config_id>__<group_suffix>.json` | Overall exact-match, Fairlearn overall + `by_group`, disparities vs `--disparity-threshold`, mitigation strings |
| `model_bias_by_dataset_<config_id>__<group_suffix>.png` | Bar chart of mean exact-match by `dataset` (only if `dataset` is in `--group-cols` and plotting succeeds) |

The `<group_suffix>` is derived from the final `--group-cols` list so different slice runs with the same `--config-id` do not overwrite each other.

### Examples (PowerShell, repo root)

```powershell
$env:PYTHONPATH = "."

# Live API on translation_inputs
python model-pipeline/scripts/run_model_bias_detection.py `
  --split dev `
  --config-id translation_flash_v1 `
  --group-cols dataset,emotion `
  --max-rows 20

# Reuse predictions only (no extra Gemini calls for bias)
python model-pipeline/scripts/run_model_bias_detection.py `
  --split dev `
  --config-id translation_flash_v1 `
  --from-predictions `
  --group-cols dataset,emotion
```

### Other slices (reuse predictions)

Change only `--group-cols`:

```powershell
# Emotion
python model-pipeline/scripts/run_model_bias_detection.py --split dev --config-id translation_flash_v1 --from-predictions --group-cols emotion

# Language pairs (meaningful only if multiple values appear in the table)
python model-pipeline/scripts/run_model_bias_detection.py --split dev --config-id translation_flash_v1 --from-predictions --group-cols source_language,target_language

# Speaker proxy
python model-pipeline/scripts/run_model_bias_detection.py --split dev --config-id translation_flash_v1 --from-predictions --group-cols speaker_id

# Corpus only
python model-pipeline/scripts/run_model_bias_detection.py --split dev --config-id translation_flash_v1 --from-predictions --group-cols dataset
```

### Reusing one predictions file under different report names

You can pass a unique `--config-id` together with `--predictions-path` pointing at the same CSV if you want separate report names without changing slice columns:

```powershell
python model-pipeline/scripts/run_model_bias_detection.py `
  --split dev `
  --config-id translation_flash_v1_bias_emotion `
  --from-predictions `
  --predictions-path data/model_runs/dev/translation_predictions_translation_flash_v1.csv `
  --group-cols emotion
```

### Implementation

- `scripts/model_bias_detection_core.py` — Fairlearn metrics, disparities, plot helpers.
- `scripts/run_model_bias_detection.py` — CLI entry.

Tests: `tests/test_model_bias_detection_core.py`, `tests/test_run_model_bias_detection_cli.py`.

---

## Minimal outputs to share with team

- `config_search_results*.json` 
- `validation_metrics.json` / `validation_metrics.csv` 
- `translation_bias_metrics_<config>.json`
- `model_bias_report_<config>__<group_suffix>.json` (modeling bias / disparities)
- Artifact Registry package entries for selected configs 
