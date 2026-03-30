"""
IIKSHANA COURTROOM ACCESSIBILITY
Audio accessibility for the visually impaired — courtroom speech → translation.

Expo: attach recording → **Docker Airflow** ``model_pipeline_dag`` → read translation from pipeline CSV.
Always uses ``docker compose exec airflow-scheduler`` (see ``demo/airflow_trigger.py``).
Repo root is mounted at ``/workspace`` in compose; poller reads ``data/model_runs/<split>/`` on the host.

Run: ``PYTHONPATH=. streamlit run demo/streamlit_expo_app.py``
"""

from __future__ import annotations

import base64
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

# ── Themed CSS ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global background ─── */
.stApp {
    background: linear-gradient(160deg, #0a0f1a 0%, #111d35 40%, #1a1a2e 70%, #16213e 100%);
    color: #e0e0e0;
}

/* ── Decorative SVG overlay (scales of justice + sound waves) ─── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        /* Scales of Justice — top right */
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' opacity='0.04'%3E%3Cline x1='100' y1='30' x2='100' y2='110' stroke='%23c9a84c' stroke-width='3'/%3E%3Cline x1='55' y1='70' x2='145' y2='70' stroke='%23c9a84c' stroke-width='3'/%3E%3Ccircle cx='100' cy='30' r='8' fill='%23c9a84c'/%3E%3Cpath d='M55 70 L40 110 A25 8 0 0 0 70 110 Z' fill='none' stroke='%23c9a84c' stroke-width='2'/%3E%3Cpath d='M145 70 L130 110 A25 8 0 0 0 160 110 Z' fill='none' stroke='%23c9a84c' stroke-width='2'/%3E%3Crect x='90' y='110' width='20' height='40' rx='3' fill='%23c9a84c' opacity='0.5'/%3E%3Crect x='75' y='148' width='50' height='8' rx='4' fill='%23c9a84c' opacity='0.5'/%3E%3C/svg%3E"),
        /* Sound wave rings — bottom left */
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' opacity='0.035'%3E%3Ccircle cx='100' cy='100' r='20' fill='none' stroke='%234ea8de' stroke-width='2'/%3E%3Ccircle cx='100' cy='100' r='40' fill='none' stroke='%234ea8de' stroke-width='1.5'/%3E%3Ccircle cx='100' cy='100' r='60' fill='none' stroke='%234ea8de' stroke-width='1'/%3E%3Ccircle cx='100' cy='100' r='80' fill='none' stroke='%234ea8de' stroke-width='0.7'/%3E%3Ccircle cx='100' cy='100' r='95' fill='none' stroke='%234ea8de' stroke-width='0.4'/%3E%3C/svg%3E"),
        /* Accessibility figure — center faint */
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 160' opacity='0.025'%3E%3Ccircle cx='50' cy='18' r='12' fill='%23ffffff'/%3E%3Cpath d='M50 30 L50 85' stroke='%23ffffff' stroke-width='4' stroke-linecap='round'/%3E%3Cpath d='M25 50 L75 50' stroke='%23ffffff' stroke-width='4' stroke-linecap='round'/%3E%3Cpath d='M50 85 L30 130' stroke='%23ffffff' stroke-width='4' stroke-linecap='round'/%3E%3Cpath d='M50 85 L70 130' stroke='%23ffffff' stroke-width='4' stroke-linecap='round'/%3E%3Cline x1='25' y1='33' x2='10' y2='60' stroke='%23ffffff' stroke-width='3' stroke-linecap='round' opacity='0.6'/%3E%3C/svg%3E");
    background-position: top 40px right 30px, bottom 60px left 30px, center center;
    background-size: 320px, 350px, 200px;
    background-repeat: no-repeat;
}

/* ── Hero banner ─── */
.hero-banner {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 2.5rem 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 0 0 24px 24px;
    background: linear-gradient(135deg, rgba(26,26,46,0.95) 0%, rgba(22,33,62,0.9) 50%, rgba(10,15,26,0.95) 100%);
    border-bottom: 2px solid rgba(201,168,76,0.3);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.hero-banner .project-label {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #4ea8de;
    margin-bottom: 0.4rem;
}
.hero-banner h1 {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #c9a84c, #f0d078, #c9a84c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0.3rem 0 0.2rem;
    line-height: 1.15;
}
.hero-banner .tagline {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 1.05rem;
    color: #a0b4c8;
    margin-top: 0.5rem;
}
.hero-banner .divider-icons {
    margin-top: 1rem;
    font-size: 1.3rem;
    letter-spacing: 12px;
    opacity: 0.5;
}

/* ── Cards for input sections ─── */
.input-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(4px);
    position: relative;
    z-index: 1;
}
.input-card h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4ea8de;
    margin-bottom: 0.8rem;
}

/* ── Streamlit widget overrides ─── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(201,168,76,0.25) !important;
    color: #e0e0e0 !important;
    border-radius: 10px !important;
}
.stSelectbox label, .stNumberInput label, .stCheckbox label, .stFileUploader label {
    color: #c0c8d4 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #c9a84c, #a8893a) !important;
    color: #0a0f1a !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 16px rgba(201,168,76,0.3) !important;
    transition: all 0.3s ease !important;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(201,168,76,0.5) !important;
    transform: translateY(-1px);
}

/* ── Info / success / error boxes ─── */
.stAlert {
    border-radius: 12px !important;
    position: relative;
    z-index: 1;
}

/* ── Footer ─── */
.footer {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 2rem 0 1rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.footer p {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #5a6a7a;
    letter-spacing: 0.5px;
}
</style>
"""


# ── Hero HTML ─────────────────────────────────────────────────────────────────

HERO_HTML = """
<div class="hero-banner">
    <div class="project-label">Accessible Justice Initiative</div>
    <h1>IIKSHANA</h1>
    <div class="tagline">
        Courtroom Audio Accessibility for the Visually Impaired<br>
        <em>Speech → Transcription → Translation</em>
    </div>
    <div class="divider-icons">⚖️ &nbsp; 🔊 &nbsp; ♿ &nbsp; 🌐</div>
</div>
"""

FOOTER_HTML = """
<div class="footer">
    <p>⚖️ IIKSHANA COURTROOM ACCESSIBILITY &nbsp;·&nbsp; Empowering equal access to justice through audio AI</p>
</div>
"""


def main() -> None:
    st.set_page_config(
        page_title="IIKSHANA — Courtroom Accessibility",
        page_icon="⚖️",
        layout="centered",
    )

    # ── Inject CSS + Hero ─────────────────────────────────────────────────
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(HERO_HTML, unsafe_allow_html=True)

    # ── Pipeline config card ──────────────────────────────────────────────
    st.markdown('<div class="input-card"><h3>⚙️ Pipeline Configuration</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        split = st.selectbox("Split", VALID_SPLITS, index=0)
    with col2:
        target_language = st.selectbox("Translate to", ("es", "fr", "de"), index=0)

    target_label = {"es": "Spanish 🇪🇸", "fr": "French 🇫🇷", "de": "German 🇩🇪"}.get(
        target_language, target_language
    )

    col3, col4 = st.columns([1, 1])
    with col3:
        rerun_config_search = st.checkbox(
            "Re-run config search (slow)",
            value=False,
            help="Compares many models. Leave off to reuse the last best config.",
        )
    with col4:
        manifest_tail = st.number_input(
            "Manifest tail rows (STT cap)",
            min_value=1,
            max_value=500,
            value=200,
            help=(
                "Each run rewrites translation_inputs.csv / predictions (no append). "
                "Only the last N manifest WAVs are included."
            ),
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Audio input card ──────────────────────────────────────────────────
    st.markdown('<div class="input-card"><h3>🎙️ Audio Input</h3>', unsafe_allow_html=True)

    audio_bytes = st.audio_input("Record courtroom audio")
    uploaded = st.file_uploader(
        "Or upload an audio file",
        type=("wav", "mp3", "webm", "m4a", "ogg"),
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Trigger button ────────────────────────────────────────────────────
    go = st.button(
        "⚖️  Save Clip & Trigger Airflow Pipeline",
        type="primary",
        disabled=not audio_bytes and not uploaded,
        use_container_width=True,
    )

    if not go:
        st.info(
            "**Getting started:** run `cd airflow && docker compose up` from the repo root, "
            "then unpause **model_pipeline_dag** in the Airflow UI if needed. "
            "The app also runs `airflow dags unpause` automatically before each trigger."
        )
        st.markdown(FOOTER_HTML, unsafe_allow_html=True)
        return

    # ── Process audio ─────────────────────────────────────────────────────
    suffix, raw = ".wav", None
    if uploaded is not None:
        raw = uploaded.getvalue()
        name = (uploaded.name or "upload").lower()
        if "." in name:
            suffix = name[name.rfind(".") :]
    elif audio_bytes is not None:
        raw = audio_bytes.getvalue()
    if not raw:
        st.warning("No audio detected — please record or upload a clip.")
        return

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        with st.spinner("Preprocessing audio & updating manifest…"):
            try:
                _, manifest_row = ingest_expo_recording(tmp_path, split=split, source_suffix=suffix)
            except Exception as e:  # noqa: BLE001
                st.error(f"Ingest failed: {e}")
                return
    finally:
        tmp_path.unlink(missing_ok=True)

    st.success(f"✅ Attached **`{manifest_row['file']}`** to split `{split}`.")
    if str(manifest_row.get("file", "")).startswith("EXPO_"):
        st.caption(
            "💡 The filename contains the **UTC save time** "
            "(e.g. `EXPO_20260329T183745Z.wav` → 2026-03-29 18:37:45 UTC). "
            "Use it to locate the row in `translation_predictions_*.csv`."
        )

    # ── Trigger Airflow ───────────────────────────────────────────────────
    with st.spinner("Connecting to Docker Airflow — unpausing + triggering pipeline…"):
        code, log, _ = trigger_model_pipeline_dag(
            split=split,
            refresh_inputs=True,
            refresh_config_search=rerun_config_search,
            manifest_tail=int(manifest_tail),
            target_language=target_language,
        )

    if code != 0:
        st.error(
            "**Airflow trigger failed.** Check: `cd airflow && docker compose ps` "
            f"(scheduler must be running).\n\nExit `{code}`\n\n```\n{log or '(no output)'}\n```"
        )
        return

    st.success("🚀 **model_pipeline_dag** triggered — [Open Airflow UI](http://localhost:8080)")
    with st.expander("Docker / Airflow CLI output"):
        st.code(log or "(ok)", language="text")

    # ── Poll for translation ──────────────────────────────────────────────
    status = st.status("Waiting for pipeline to produce your translation…", state="running")
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
            f"Polling every {POLL_SEC}s — `{manifest_row['file']}` · "
            f"~{int(left // 60)}m {int(left % 60)}s remaining"
        )
        time.sleep(POLL_SEC)

    if got is None:
        status.update(label="⏱️ Timed out waiting for predictions", state="error")
        st.warning(
            "The DAG may still be running — check task logs in the Airflow UI. "
            f"Look for outputs in `data/model_runs/{split}/translation_predictions_*.csv`."
        )
        st.markdown(FOOTER_HTML, unsafe_allow_html=True)
        return

    text, pred_path, best_cfg = got
    status.update(label="✅ Translation ready!", state="complete")

    # ── Result card ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div class="input-card"><h3>🌐 Translation Result ({target_label})</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {text}")

    with st.expander("Pipeline details"):
        st.write(f"**Best config:** `{best_cfg}`")
        st.write(f"**Predictions file:** `{pred_path}`")

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


if __name__ == "__main__":
    main()