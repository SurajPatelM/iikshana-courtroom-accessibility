# Extend Airflow image with pipeline + DVC deps so full_pipeline_dag can run dvc pull / dvc commit / dvc push.
FROM apache/airflow:2.7.0-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
USER airflow

RUN pip install --no-cache-dir \
    pandas numpy scipy soundfile librosa pyyaml great-expectations requests tqdm checksumdir \
    certifi dvc dvc-gs
