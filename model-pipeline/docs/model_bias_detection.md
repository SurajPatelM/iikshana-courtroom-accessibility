# Modeling bias detection (`run_model_bias_detection.py`)

**Scope:** Translation **API** fairness — metrics and disparities **per slice** (e.g. corpus `dataset` from the data pipeline, `emotion`, `speaker_id`). This is separate from **data-pipeline** representation bias (`data-pipeline/scripts/detect_bias.py`).

## Prerequisites

- Repo root on `PYTHONPATH` (and `model-pipeline/scripts` is added automatically by the runner).
- Eval table or predictions file under `data/processed/<split>/`.
- `fairlearn` (see root `requirements.txt`).

## Input tables

**Columns required for every run**

| Column | Role |
|--------|------|
| `source_text` | Source utterance |
| `source_language` | Source language code |
| `target_language` | Target language code |
| `reference_translation` | Gold translation for metrics |

**API mode** (`default`): reads `translation_inputs.csv` or `.parquet` (basename configurable). The script calls `translate_text` for each row.

**Predictions mode** (`--from-predictions`): reads `translation_predictions_<config_id>.csv` or `.parquet` produced by `model_setup.py` (must still include `reference_translation` plus `translated_text_model` or override with `--pred-col`).

**Slice columns** (`--group-cols`, default `dataset,emotion`) must exist on that table. The `dataset` field is populated when building inputs from manifests (e.g. `build_translation_inputs_from_audio.py`).

## Outputs

Written to `data/model_runs/<split>/` by default (legacy `--data-dir` falls back to `data/processed/<split>/`):

| File | Description |
|------|-------------|
| `model_bias_report_<config_id>.json` | Overall exact-match, Fairlearn overall + `by_group`, disparities vs threshold, mitigation strings |
| `model_bias_by_dataset_<config_id>.png` | Bar chart of mean exact-match by `dataset` (if `dataset` is in `--group-cols` and plot succeeds) |

## Examples

From repository root:

```bash
export PYTHONPATH="${PWD}"

# Live API calls on translation_inputs
python model-pipeline/scripts/run_model_bias_detection.py \
  --split dev \
  --config-id translation_flash_v1 \
  --group-cols dataset,emotion \
  --max-rows 20

# Reuse predictions only (no API)
python model-pipeline/scripts/run_model_bias_detection.py \
  --split dev \
  --config-id translation_flash_v1 \
  --from-predictions \
  --group-cols dataset,emotion
```

## Other slices for translation bias (reuse predictions)
You can reuse your existing predictions with `--from-predictions` (no extra Gemini calls), and change only `--group-cols` to see disparities across different slice dimensions.

Emotion slices:
```bash
python model-pipeline/scripts/run_model_bias_detection.py \
  --split dev \
  --config-id translation_flash_v1 \
  --from-predictions \
  --group-cols emotion
```

Language slices (may be limited if your split only contains `en`/`es`):
```bash
python model-pipeline/scripts/run_model_bias_detection.py \
  --split dev \
  --config-id translation_flash_v1 \
  --from-predictions \
  --group-cols source_language,target_language
```

Speaker proxy slices:
```bash
python model-pipeline/scripts/run_model_bias_detection.py \
  --split dev \
  --config-id translation_flash_v1 \
  --from-predictions \
  --group-cols speaker_id
```

Dataset/corpus slices:
```bash
python model-pipeline/scripts/run_model_bias_detection.py \
  --split dev \
  --config-id translation_flash_v1 \
  --from-predictions \
  --group-cols dataset
```

### Output naming: does it overwrite?
The output filenames are keyed by `--config-id` and the slice columns (`--group-cols`):
- `model_bias_report_<config_id>.json`
- `model_bias_by_dataset_<config_id>.png` (only when `dataset` is included in `--group-cols`)

So if you run multiple `--group-cols ...` variants with the same `--config-id`, you will no longer overwrite as long as the `--group-cols` set differs.

If you want to keep multiple reports while reusing the same predictions file, you can either:
- pass different `--group-cols`, or
- pass a unique `--config-id` (useful when comparing configs rather than slices).

Example (explicit predictions path + slice-specific config-id):
```bash
python model-pipeline/scripts/run_model_bias_detection.py \
  --split dev \
  --config-id translation_flash_v1_bias_emotion \
  --from-predictions \
  --predictions-path data/model_runs/dev/translation_predictions_translation_flash_v1.csv \
  --group-cols emotion
```

## Implementation files (model-pipeline only)

- `scripts/model_bias_detection_core.py` — Fairlearn metrics, disparities, plot helpers (importable / testable).
- `scripts/run_model_bias_detection.py` — CLI entry.

Tests: `model-pipeline/tests/test_model_bias_detection_core.py`.
