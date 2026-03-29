"""
Expo: attach recording → **Docker Airflow** ``model_pipeline_dag`` → read translation from pipeline CSV.

Always uses ``docker compose exec airflow-scheduler`` (see ``demo/airflow_trigger.py``).
Repo root is mounted at ``/workspace`` in compose; poller reads ``data/model_runs/<split>/`` on the host.

Run: ``PYTHONPATH=. streamlit run demo/streamlit_expo_app.py``
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for p in (REPO_ROOT / ".secrets" / ".env", REPO_ROOT / ".env"):
        if p.is_file():
            load_dotenv(p, override=False)


_load_repo_dotenv()

from demo.airflow_trigger import trigger_model_pipeline_dag
from demo.local_model_pipeline import try_read_pipeline_translation
from demo.pipeline_ingest import VALID_SPLITS, ingest_expo_recording

POLL_SEC = 12
MAX_WAIT_SEC = 45 * 60


def main() -> None:
    st.set_page_config(page_title="Iikshana — Live demo", page_icon="⚖️", layout="centered")
    st.title("Courtroom phrase → translation")
    st.caption(
        "Uses **Docker** (``airflow/docker-compose.yaml``): saves your clip, **unpauses + triggers** "
        f"**model_pipeline_dag**, then polls for the translation (up to {MAX_WAIT_SEC // 60} min)."
    )

    split = st.selectbox("Split", VALID_SPLITS, index=0)
    target_language = st.selectbox("Translate to", ("es", "fr", "de"), index=0)
    target_label = {"es": "Spanish", "fr": "French", "de": "German"}.get(target_language, target_language)
    rerun_config_search = st.checkbox(
        "Re-run **config search** (slow; compares many models). Leave off to reuse the last best config.",
        value=False,
    )
    manifest_tail = st.number_input(
        "Rows from **end** of manifest in each DAG run (STT cap)",
        min_value=1,
        max_value=500,
        value=200,
        help=(
            "Each run **rewrites** `translation_inputs.csv` / predictions (no append). "
            "Only the last N manifest WAVs are included — line count stays near N+1 header, not “growing” per clip."
        ),
    )

    audio_bytes = st.audio_input("Record")
    uploaded = st.file_uploader("Or upload audio", type=("wav", "mp3", "webm", "m4a", "ogg"))

    go = st.button("Save clip & trigger Airflow pipeline", type="primary", disabled=not audio_bytes and not uploaded)

    if not go:
        st.info(
            "Start stack from repo: `cd airflow && docker compose up` — then unpause **model_pipeline_dag** once in "
            "the UI if needed (the app also runs `airflow dags unpause` before each trigger)."
        )
        return

    suffix, raw = ".wav", None
    if uploaded is not None:
        raw = uploaded.getvalue()
        name = (uploaded.name or "upload").lower()
        if "." in name:
            suffix = name[name.rfind(".") :]
    elif audio_bytes is not None:
        raw = audio_bytes.getvalue()
    if not raw:
        st.warning("No audio.")
        return

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        with st.spinner("Saving into data/processed (preprocess + manifest)…"):
            try:
                _, manifest_row = ingest_expo_recording(tmp_path, split=split, source_suffix=suffix)
            except Exception as e:  # noqa: BLE001
                st.error(f"Ingest failed: {e}")
                return
    finally:
        tmp_path.unlink(missing_ok=True)

    st.success(f"Attached **`{manifest_row['file']}`** to split `{split}`.")
    if str(manifest_row.get("file", "")).startswith("EXPO_"):
        st.caption(
            "Tip: the middle of the filename is the **UTC save time** (e.g. `EXPO_20260329T183745Z.wav` → "
            "`2026-03-29` 18:37:45Z). Use it to match a row in `translation_predictions_*.csv`."
        )

    with st.spinner("Docker: unpause + trigger model_pipeline_dag…"):
        code, log, _ = trigger_model_pipeline_dag(
            split=split,
            refresh_inputs=True,
            refresh_config_search=rerun_config_search,
            manifest_tail=int(manifest_tail),
            target_language=target_language,
        )

    if code != 0:
        st.error(
            "**Airflow trigger failed.** From repo root, check: `cd airflow && docker compose ps` "
            f"(scheduler must be running).\n\nExit `{code}`\n\n```\n{log or '(no output)'}\n```"
        )
        return

    st.success("**model_pipeline_dag** triggered via Docker — [Airflow UI](http://localhost:8080).")
    with st.expander("Docker / Airflow CLI output"):
        st.code(log or "(ok)", language="text")

    status = st.status("Waiting for pipeline to write your translation…", state="running")
    bar = st.progress(0.0)
    caption = st.empty()
    deadline = time.monotonic() + MAX_WAIT_SEC
    got = None
    while time.monotonic() < deadline:
        got = try_read_pipeline_translation(REPO_ROOT, split, manifest_row["file"])
        if got is not None:
            break
        left = max(0.0, deadline - time.monotonic())
        bar.progress(1.0 - (left / MAX_WAIT_SEC))
        caption.caption(
            f"Polling every {POLL_SEC}s — `{manifest_row['file']}` · ~{int(left // 60)}m {int(left % 60)}s left"
        )
        time.sleep(POLL_SEC)

    if got is None:
        status.update(label="Timed out waiting for predictions CSV", state="error")
        st.warning(
            "DAG may still be running or failed — check task logs in the Airflow UI. "
            f"Outputs: `data/model_runs/{split}/translation_predictions_*.csv`."
        )
        return

    text, pred_path, best_cfg = got
    status.update(label="Translation ready", state="complete")
    st.subheader(f"Translation ({target_label})")
    st.write(text)
    with st.expander("Details"):
        st.write(f"**Best config:** `{best_cfg}`")
        st.write(f"**Predictions:** `{pred_path}`")


if __name__ == "__main__":
    main()
