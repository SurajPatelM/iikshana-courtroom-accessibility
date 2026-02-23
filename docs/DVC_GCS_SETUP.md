# DVC and GCS Setup (Iikshana)

This project uses **DVC** for data versioning with **Google Cloud Storage (GCS)** as the remote, so you can `dvc push` and `dvc pull` to share datasets without storing them in Git.

## Prerequisites

- **DVC** and **dvc-gs** (GCS support):
  ```bash
  pip install dvc dvc-gs
  ```
  Or use the project’s `requirements.txt`, which already includes them.

- **Google Cloud**: a GCP project and a GCS bucket.
- **Authentication** (one of):
  - **Application Default Credentials** (recommended for local use):
    ```bash
    gcloud auth application-default login
    ```
  - **Service account**: set `GOOGLE_APPLICATION_CREDENTIALS` to the path of your service account JSON key.

## One-time setup

### 1. Create a GCS bucket (if you don’t have one)

```bash
# Set your project and bucket name
export GCP_PROJECT=your-gcp-project-id
export GCS_BUCKET=your-bucket-name   # e.g. iikshana-dvc-data

gcloud storage buckets create gs://$GCS_BUCKET/dvc --project=$GCP_PROJECT
# Or create the bucket first, then use a prefix:
# gs://$GCS_BUCKET/dvc
```

### 2. Configure DVC to use your GCS bucket

From the **repository root**:

**Option A – Override in config (recommended so the bucket name is not in Git)**

```bash
dvc remote add -d storage gs://YOUR_BUCKET_NAME/dvc --local
```

This writes to `.dvc/config.local` (which is gitignored), so each developer or CI can set their own bucket.

**Option B – Set in the main config (bucket name will be in Git)**

```bash
dvc remote modify storage url gs://YOUR_BUCKET_NAME/dvc
```

Replace `YOUR_BUCKET_NAME` with your actual GCS bucket (and path if you use one, e.g. `gs://my-bucket/iikshana/dvc`).

### 3. Verify

```bash
dvc config -l
# or
dvc remote list
```

## Daily workflow

### Track data with DVC

From **`data-pipeline/`** (where `dvc.yaml` lives), track outputs and large inputs:

```bash
cd data-pipeline

# Examples (paths are relative to repo root)
dvc add ../data/raw/RAVDESS ../data/raw/IEMOCAP
dvc add ../data/processed/dev ../data/processed/test ../data/processed/holdout
dvc add ../data/legal_glossary
```

Then commit the `.dvc` pointer files and `.gitignore` updates:

```bash
git add *.dvc .gitignore
git commit -m "Track pipeline data with DVC"
```

### Push data to GCS

From the **repository root** (or from `data-pipeline/`; DVC will use the repo root):

```bash
dvc push
```

This uploads DVC-tracked data to the default remote (`storage` → GCS).

### Pull data from GCS

On another clone or machine, after cloning the repo and configuring the remote (step 2 above):

```bash
dvc pull
```

This downloads all data referenced by the current `.dvc` files.

### Reproduce the pipeline

From **`data-pipeline/`**:

```bash
cd data-pipeline
dvc repro
```

## What is versioned with DVC

- **Raw datasets** – e.g. under `data/raw/` (RAVDESS, MELD, etc.)
- **Processed outputs** – e.g. `data/processed/dev`, `data/processed/test`, `data/processed/holdout`
- **Other large assets** – e.g. `data/legal_glossary`
- **Pipeline definition** – `data-pipeline/dvc.yaml` (in Git); DVC tracks which data versions go with it via `.dvc` files

Git stores only small `.dvc` pointer files; the actual files are in the DVC cache and on GCS.

## Linking runs to data versions

When you run evaluations or training, you can record the current DVC state (e.g. dataset version) in your experiment tracker (e.g. MLflow):

- Use the commit hash (or the hash from the relevant `.dvc` file) as `dataset_version`.
- Later you can `git checkout <commit>` and `dvc pull` to get the exact same data.

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **SSL: CERTIFICATE_VERIFY_FAILED** (macOS) | Use the project SSL fix so GCS can be reached: `source scripts/dvc_ssl_fix.sh` then `dvc push`, or run `./scripts/dvc_ssl_fix.sh push`. Ensure `certifi` is installed: `pip install certifi`. |
| `dvc push` / `dvc pull` fails with auth error | Run `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS`. |
| "Bucket not found" or 403 | Check bucket name and IAM: your identity needs `Storage Object Admin` (or equivalent) on the bucket. |
| No default remote | Run `dvc remote add -d storage gs://YOUR_BUCKET/dvc` (add `--local` to keep it out of Git). |
| Running from `data-pipeline/` | `dvc push` and `dvc pull` work from any subdirectory; DVC uses the repo root. |

## Summary

1. Install: `pip install dvc dvc-gs`
2. One-time: create a GCS bucket and run `dvc remote add -d storage gs://YOUR_BUCKET/dvc --local`
3. Track: `dvc add ...` then `git add *.dvc && git commit`
4. Share: `dvc push` (upload), `dvc pull` (download)
