"""
IIKSHANA — courtroom accessibility expo UI (Gradio).
Clean, minimal UI matching the mockup designs.
"""

from __future__ import annotations

import html
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
EXPO_MIC_SVG = DEMO_DIR / "assets" / "mic_glyph.svg"
EXPO_LOGO = DEMO_DIR / "assets" / "iikshana_logo.png"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for p in (REPO_ROOT / ".env", REPO_ROOT / ".secrets" / ".env"):
        if p.is_file():
            load_dotenv(p, override=True)

_load_repo_dotenv()

from demo.asl_translation import make_initial_asl_state, process_asl_chunk
from demo.live_translation import make_initial_state, process_audio_chunk

DEFAULT_TRANSLATION_CONFIG_ID = (
    os.environ.get("EXPO_TRANSLATION_CONFIG_ID", "translation_flash_v1").strip() or "translation_flash_v1"
)
TRANSCRIPT_TRANSLATION_CONFIG_ID = (
    os.environ.get("EXPO_TRANSCRIPT_TRANSLATION_CONFIG_ID", "translation_flash_transcript_strict").strip()
    or "translation_flash_transcript_strict"
)


# Mitigate: RuntimeError: Response content longer than Content-Length (Gradio Brotli + uvicorn httptools).
# Set IIKSHANA_GRADIO_BROTLI=1 to keep Brotli. Set IIKSHANA_UVICORN_HTTP=httptools to skip h11 fix.
_HTTP_WORKAROUNDS_APPLIED = False


def _apply_gradio_uvicorn_http_workarounds() -> None:
    global _HTTP_WORKAROUNDS_APPLIED
    if _HTTP_WORKAROUNDS_APPLIED:
        return
    _HTTP_WORKAROUNDS_APPLIED = True

    if os.environ.get("IIKSHANA_UVICORN_HTTP", "h11").strip().lower() != "httptools":
        import uvicorn

        _RealConfig = uvicorn.Config

        class _UvicornConfigUseH11(_RealConfig):
            def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
                kwargs.setdefault("http", "h11")
                super().__init__(*args, **kwargs)

        uvicorn.Config = _UvicornConfigUseH11  # type: ignore[misc, assignment]

    if os.environ.get("IIKSHANA_GRADIO_BROTLI", "").strip().lower() not in ("1", "true", "yes", "on"):
        import gradio.brotli_middleware as brotli_mw

        async def _brotli_passthrough(self, scope, receive, send):  # noqa: ANN001, ANN201
            return await self.app(scope, receive, send)

        brotli_mw.BrotliMiddleware.__call__ = _brotli_passthrough  # type: ignore[method-assign]

# Dropdown choices: (label, value). Live "Listening in" uses auto + fixed codes for translate source.
LANG_LISTEN_CHOICES: list[tuple[str, str]] = [
    ("Auto-detect", "auto"),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
]
LANG_SPEAK_CHOICES: list[tuple[str, str]] = [
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
]

# Match expo canvas so Gradio's theme engine does not paint a white body behind our UI.
_EXPO_CANVAS_HEX = "#090d14"
_EXPO_PANEL_HEX = "#161f2e"
EXPO_THEME = gr.themes.Base().set(
    body_background_fill=_EXPO_CANVAS_HEX,
    body_background_fill_dark=_EXPO_CANVAS_HEX,
    panel_background_fill=_EXPO_PANEL_HEX,
    panel_background_fill_dark=_EXPO_PANEL_HEX,
)

# =============================================================================
# AGGRESSIVE CSS - Force override Gradio defaults
# =============================================================================
CUSTOM_CSS = """
/* Formal dark UI: cool blue-slate (neutral, institutional — not green-tinted) */
:root {
    --expo-canvas: #090d14;
    --expo-panel: #161f2e;
    --expo-muted: #94a3b8;
}

/* Full-bleed canvas: remove default white gutters (Gradio shell + browser) */
html {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    min-height: 100% !important;
    background-color: var(--expo-canvas) !important;
    background: var(--expo-canvas) !important;
}

html, body, #root, #root > div {
    background-color: var(--expo-canvas) !important;
    background: var(--expo-canvas) !important;
}

body {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    min-height: 100vh !important;
}

#root {
    max-width: none !important;
    width: 100% !important;
    min-height: 100vh !important;
}

/* Brand header: logo + title */
.gradio-container .expo-header-wrap {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 8px 0 0 0 !important;
}

.gradio-container .expo-header-wrap .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-container .expo-brand-logo,
.gradio-container .expo-brand-logo > .wrap,
.gradio-container .expo-brand-logo .image-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-container .expo-brand-logo img {
    max-height: clamp(72px, 20vw, 120px) !important;
    width: auto !important;
    max-width: min(340px, 92vw) !important;
    object-fit: contain !important;
    margin: 0 auto !important;
    display: block !important;
}

@media (min-width: 1024px) {
    .gradio-container .expo-brand-logo img {
        max-width: min(400px, 42vw) !important;
        max-height: 100px !important;
    }
}

/* Force dark background everywhere */
.gradio-container, .main, .contain, .wrap, .gr-panel, .gr-box, .gr-form,
.gr-group, .block, .form, .gap, [class*="block"], [class*="panel"] {
    background: var(--expo-canvas) !important;
    border: none !important;
    box-shadow: none !important;
}

/* Root: avoid horizontal scroll on narrow phones */
#root, .gradio-container {
    box-sizing: border-box !important;
    background-color: var(--expo-canvas) !important;
}

/* Main content column: nearly full width on laptop; small side padding only */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 12px clamp(12px, 3vw, 24px) !important;
    padding-left: max(12px, env(safe-area-inset-left, 0px)) !important;
    padding-right: max(12px, env(safe-area-inset-right, 0px)) !important;
    padding-bottom: max(12px, env(safe-area-inset-bottom, 0px)) !important;
    overflow-x: clip !important;
}

@media (min-width: 480px) {
    .gradio-container {
        max-width: min(640px, 100%) !important;
        padding-left: max(16px, env(safe-area-inset-left, 0px)) !important;
        padding-right: max(16px, env(safe-area-inset-right, 0px)) !important;
    }
}

@media (min-width: 768px) {
    .gradio-container {
        max-width: min(1200px, calc(100vw - 32px)) !important;
        padding: 16px clamp(16px, 2.5vw, 36px) !important;
    }
}

@media (min-width: 1024px) {
    .gradio-container {
        max-width: min(1600px, calc(100vw - 48px)) !important;
        padding: 20px clamp(20px, 2vw, 48px) !important;
    }
}

@media (min-width: 1600px) {
    .gradio-container {
        max-width: min(1760px, calc(100vw - 64px)) !important;
    }
}

/* Live tab: vertical stack — mic + options centered, transcript full width below */
.gradio-container .expo-live-layout {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 16px !important;
    width: 100% !important;
}

.gradio-container .expo-live-sidebar {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    width: 100% !important;
    max-width: 36rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Gradio nests mic / options inside .wrap — center that column */
.gradio-container .expo-live-sidebar > .wrap,
.gradio-container .expo-live-sidebar .wrap.overflow-visible,
.gradio-container .expo-live-sidebar .form {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    width: 100% !important;
}

.gradio-container .expo-live-sidebar .live-options-row {
    justify-content: center !important;
    width: 100% !important;
    max-width: 100% !important;
}

.gradio-container .expo-live-main {
    width: 100% !important;
    max-width: min(56rem, 100%) !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

@media (min-width: 768px) {
    .gradio-container .expo-live-layout {
        gap: 24px !important;
    }

    .gradio-container .expo-live-sidebar {
        max-width: 40rem !important;
    }

    .gradio-container .expo-live-feed {
        min-height: min(420px, 55vh) !important;
    }
}

/* Title block: scale up on larger screens */
@media (min-width: 1024px) {
    .gradio-container .expo-title-block h1 {
        font-size: 1.75rem !important;
    }
}

/* Tighter vertical rhythm between Gradio blocks */
.gradio-container .form,
.gradio-container .wrap {
    gap: 0.5rem !important;
}

.gradio-container .block,
.gradio-container [class*="block-label"] {
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
}

/* Hide footer */
footer, .footer, [class*="footer"] {
    display: none !important;
}

/* Tab styling */
.tabs, .tabitem, [class*="tab"] {
    background: transparent !important;
    border: none !important;
}

.tab-nav {
    background: transparent !important;
    border: none !important;
    gap: 8px !important;
    padding: 0 !important;
    margin-bottom: 8px !important;
    display: flex !important;
    flex-wrap: wrap !important;
}

@media (min-width: 1024px) {
    .tab-nav {
        gap: 12px !important;
        margin-bottom: 14px !important;
    }
}

.tab-nav button {
    background: rgba(255,255,255,0.06) !important;
    border: none !important;
    border-radius: 8px !important;
    color: var(--expo-muted) !important;
    padding: 10px 20px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

.tab-nav button.selected {
    background: rgba(255,255,255,0.12) !important;
    color: #fff !important;
}

/* Audio component */
.gr-audio, [data-testid="audio"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Checkbox - minimal. Never style input[data-testid=checkbox]: Gradio uses
   background/background-image for the checked state; forcing background here clears it. */
.gr-checkbox,
.checkbox-wrap {
    background: transparent !important;
}

.gr-checkbox label, .checkbox-wrap label {
    display: flex !important;
    color: #999 !important;
    font-size: 13px !important;
}

/* Dropdown - hidden by default, shown when needed */
.gr-dropdown {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #ddd !important;
}

/* Language cards: dropdowns inside paired columns */
.gradio-container .lang-pair-row {
    align-items: stretch !important;
    gap: 10px !important;
    margin: 6px 0 10px 0 !important;
}

.gradio-container .lang-strip-card {
    background: var(--expo-panel) !important;
    border-radius: 10px !important;
    padding: 10px 12px 12px !important;
    min-height: 76px !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
    border: none !important;
    box-shadow: none !important;
    justify-content: center !important;
}

.gradio-container .lang-strip-card .wrap,
.gradio-container .lang-strip-card .gr-dropdown {
    width: 100% !important;
}

.gradio-container .lang-arrow-col {
    flex: 0 0 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    align-self: stretch !important;
    background: transparent !important;
}

.gradio-container .lang-arrow-col .gr-html {
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
}

/* Dropdown / combo: readable text; 16px+ on small screens reduces iOS input zoom */
.gradio-container .lang-strip-card .wrap label,
.gradio-container .lang-strip-card .gr-dropdown button,
.gradio-container .lang-strip-card [class*="dropdown"] button {
    font-size: clamp(14px, 3.8vw, 16px) !important;
    min-height: 44px !important;
    align-items: center !important;
}

/* Laptop / tablet: language row stays horizontal */
@media (min-width: 641px) {
    .gradio-container .lang-strip-card .wrap label,
    .gradio-container .lang-strip-card .gr-dropdown button,
    .gradio-container .lang-strip-card [class*="dropdown"] button {
        min-height: 40px !important;
    }
}

/* Mobile: stack language cards; arrow points down */
@media (max-width: 640px) {
    .gradio-container .lang-pair-row {
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 8px !important;
    }

    .gradio-container .lang-arrow-col {
        flex: 0 0 auto !important;
        min-width: 100% !important;
        max-width: 100% !important;
        padding: 2px 0 !important;
    }

    .gradio-container .lang-arrow-col .gr-html div {
        transform: rotate(90deg);
        line-height: 1 !important;
    }

    .gradio-container .lang-strip-card {
        min-height: 0 !important;
    }

    .tab-nav {
        width: 100% !important;
        display: flex !important;
    }

    .tab-nav button {
        flex: 1 1 0 !important;
        min-height: 48px !important;
        padding: 12px 10px !important;
        font-size: 15px !important;
    }

    .gr-button-primary, button.primary {
        min-height: 48px !important;
        padding: 16px 20px !important;
        font-size: 16px !important;
    }

    .live-options-row,
    .record-options-row {
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 10px !important;
    }

    .live-options-row .gr-checkbox,
    .record-options-row .gr-checkbox {
        width: 100% !important;
    }

    .live-options-row .gr-checkbox label,
    .record-options-row .gr-checkbox label {
        min-height: 44px !important;
        align-items: center !important;
        font-size: 15px !important;
    }
}

/* Accordion */
.gr-accordion {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
}

.gr-accordion > .label-wrap {
    display: flex !important;
    color: #888 !important;
    font-size: 13px !important;
    padding: 12px 16px !important;
}

/* Buttons */
.gr-button-primary, button.primary {
    background: #10B981 !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 24px !important;
    width: 100% !important;
    cursor: pointer !important;
}

.gr-button-primary:hover {
    background: #0d9668 !important;
}

.gr-button-secondary, button.secondary {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #999 !important;
}

/* Stop button */
.gr-button-stop, button.stop {
    background: #ef4444 !important;
    border: none !important;
    color: white !important;
}

/* Markdown text */
.gr-markdown, .gr-markdown p {
    color: #999 !important;
    font-size: 14px !important;
}

.gr-markdown h3 {
    color: #ddd !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    margin-top: 0 !important;
}

/* Textbox */
.gr-textbox textarea, .gr-text-input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #eee !important;
    font-size: 14px !important;
}

/* Row layout */
.gr-row, .row {
    gap: 12px !important;
}

@media (min-width: 1024px) {
    .gr-row, .row {
        gap: 16px !important;
    }
}

/* Live: circular mic (real Button + SVG icon file — avoids clipped data-URIs) */
.gradio-container .expo-mic-stack {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    padding: 10px 0 4px 0 !important;
    max-width: 100% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-container .expo-mic-stack .block {
    background: transparent !important;
}

.gradio-container button.expo-mic-btn {
    width: clamp(88px, 28vw, 120px) !important;
    height: clamp(88px, 28vw, 120px) !important;
    min-width: clamp(88px, 28vw, 120px) !important;
    min-height: clamp(88px, 28vw, 120px) !important;
    border-radius: 50% !important;
    background-color: #10b981 !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 auto !important;
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
}

.gradio-container button.expo-mic-btn:hover {
    background-color: #0d9668 !important;
}

.gradio-container button.expo-mic-btn .button-icon,
.gradio-container button.expo-mic-btn .button-icon img {
    width: clamp(40px, 11vw, 52px) !important;
    height: clamp(40px, 11vw, 52px) !important;
    min-width: clamp(40px, 11vw, 52px) !important;
    min-height: clamp(40px, 11vw, 52px) !important;
    object-fit: contain !important;
    flex-shrink: 0 !important;
}

.gradio-container button.expo-mic-btn--live {
    box-shadow:
        0 0 0 8px rgba(16, 185, 129, 0.2),
        0 0 0 16px rgba(16, 185, 129, 0.1) !important;
}

/* Focus states */
*:focus-visible {
    outline: 2px solid #fbbf24 !important;
    outline-offset: 2px !important;
}

/* Animations */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

@keyframes dots {
    0%, 20% { opacity: 0.2; }
    50% { opacity: 1; }
    80%, 100% { opacity: 0.2; }
}

"""

EXPO_HEAD = (
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />'
    '<meta name="theme-color" content="#090d14" />'
    '<meta name="description" content="IIKSHANA courtroom audio assistive demo." />'
    "<style>html,body{background:#090d14!important;margin:0;min-height:100%;}"
    "#root{min-height:100vh;background:#090d14!important;max-width:none!important;width:100%!important;}"
    "</style>"
)


def _default_fast_translate() -> bool:
    return os.environ.get("IIKSHANA_REALTIME_MODE", "1").lower() not in ("0", "false", "no")


def _default_skip_local_ml() -> bool:
    return os.environ.get("IIKSHANA_SKIP_LOCAL_ML", "0").lower() in ("1", "true", "yes")


# =============================================================================
# CLEAN HTML COMPONENTS
# =============================================================================

def header_html(*, show_wordmark: bool = True) -> str:
    """Title under optional logo. When logo asset is present, hide small IIKSHANA wordmark (logo already includes it)."""
    wordmark = ""
    if show_wordmark:
        wordmark = """
        <p style="margin:0 0 4px 0; font-size:clamp(10px, 2.8vw, 11px); letter-spacing:2px; color:#666; text-transform:uppercase;">IIKSHANA</p>
        """
    return f"""
    <div class="expo-title-block" style="text-align:center; padding:10px 0 8px 0; padding-left:max(0px, env(safe-area-inset-left)); padding-right:max(0px, env(safe-area-inset-right));">
        {wordmark}
        <h1 style="margin:0; font-size:clamp(1.15rem, 5vw, 1.75rem); font-weight:600; color:#f5f5f5; line-height:1.2;">Courtroom audio assistant</h1>
    </div>
    """


def mic_status_html(
    active: bool = False,
    notice: str | None = None,
    *,
    connecting: bool = False,
) -> str:
    """Caption under the real mic button (class name avoids `[class*=\"block\"]` theme overrides)."""
    if connecting:
        status = "Starting…"
        hint = "Allow microphone access if your browser asks."
        color = "#94a3b8"
    elif active:
        status = "Listening..."
        hint = "Tap the microphone to stop"
        color = "#10B981"
    else:
        status = "Ready to listen"
        hint = "Tap the microphone to start"
        color = "#10B981"

    notice_html = ""
    if notice:
        safe = html.escape(notice)
        notice_html = (
            f'<p role="alert" style="margin:10px 0 0 0;font-size:clamp(12px,3.4vw,13px);'
            f"color:#fbbf24;text-align:center;padding:0 14px;max-width:100%;line-height:1.45;"
            f'font-weight:500;">{safe}</p>'
        )

    return f"""
    <div class="expo-mic-caption" style="display:flex;flex-direction:column;align-items:center;padding:4px 0 0 0;max-width:100%;">
        <p style="margin:10px 0 2px 0;font-size:clamp(15px,4.2vw,17px);font-weight:500;color:{color};text-align:center;padding:0 8px;">{status}</p>
        <p style="margin:0;font-size:clamp(12px,3.5vw,13px);color:#777;text-align:center;padding:0 12px;max-width:100%;">{hint}</p>
        {notice_html}
    </div>
    """


def live_session_header_html(is_live: bool = True) -> str:
    dot = '<div style="width:8px; height:8px; border-radius:50%; background:#10B981; animation:pulse 2s infinite;"></div>' if is_live else ''
    label = '<span style="font-size:13px; font-weight:500; color:#10B981;">Live</span>' if is_live else ''
    
    return f"""
    <style>@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}</style>
    <div style="display:flex; justify-content:space-between; align-items:center; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:16px;">
        <div>
            <h2 style="margin:0 0 2px 0; font-size:18px; font-weight:600; color:#f0f0f0;">Courtroom 3A</h2>
            <p style="margin:0; font-size:13px; color:#888;">English → Spanish</p>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
            {dot}
            {label}
        </div>
    </div>
    """


def speaker_card_html(speaker: str, text: str, translation: str, time: str = "", active: bool = False) -> str:
    colors = {"Judge": "#8B5CF6", "Witness": "#10B981", "Attorney": "#F59E0B", "Defendant": "#EF4444"}
    color = colors.get(speaker, "#888")
    border = "border-left:3px solid #10B981; padding-left:12px; margin-left:-15px;" if active else ""
    time_html = f'<span style="font-size:11px; color:#666;">{time}</span>' if time else ""
    
    dots = ""
    if active:
        dots = """<span style="display:inline-flex; gap:3px; margin-left:6px;">
            <span style="width:4px; height:4px; border-radius:50%; background:#888; animation:dots 1.4s infinite;"></span>
            <span style="width:4px; height:4px; border-radius:50%; background:#888; animation:dots 1.4s infinite 0.2s;"></span>
            <span style="width:4px; height:4px; border-radius:50%; background:#888; animation:dots 1.4s infinite 0.4s;"></span>
        </span>"""
    
    return f"""
    <style>@keyframes dots {{ 0%,20% {{ opacity:0.2; }} 50% {{ opacity:1; }} 80%,100% {{ opacity:0.2; }} }}</style>
    <div style="background:rgba(255,255,255,0.03); border-radius:10px; padding:12px 14px; margin-bottom:8px; {border}">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:12px; font-weight:600; color:{color};">{speaker}{dots}</span>
            {time_html}
        </div>
        <p style="margin:0 0 4px 0; font-size:14px; color:#eee; line-height:1.45;">{text}</p>
        <p style="margin:0; font-size:13px; color:#999; font-style:italic;">{translation}</p>
    </div>
    """


def listening_indicator_html() -> str:
    return """
    <style>@keyframes dots { 0%,20% { opacity:0.2; } 50% { opacity:1; } 80%,100% { opacity:0.2; } }</style>
    <div style="background:rgba(16,185,129,0.06); border-left:3px solid #10B981; border-radius:10px; padding:12px 14px; margin-left:-3px;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:12px; font-weight:600; color:#10B981;">Listening</span>
            <span style="display:inline-flex; gap:4px;">
                <span style="width:5px; height:5px; border-radius:50%; background:#10B981; animation:dots 1.4s infinite;"></span>
                <span style="width:5px; height:5px; border-radius:50%; background:#10B981; animation:dots 1.4s infinite 0.2s;"></span>
                <span style="width:5px; height:5px; border-radius:50%; background:#10B981; animation:dots 1.4s infinite 0.4s;"></span>
            </span>
        </div>
    </div>
    """


def bottom_actions_html() -> str:
    return """
    <div style="display:flex; justify-content:space-between; align-items:center; padding-top:16px; border-top:1px solid rgba(255,255,255,0.08); margin-top:20px;">
        <span style="font-size:13px; color:#777; cursor:pointer;">Settings</span>
        <button style="background:#ef4444; color:white; border:none; border-radius:8px; padding:10px 20px; font-size:13px; font-weight:500; cursor:pointer;">End session</button>
    </div>
    """


def footer_html() -> str:
    return """
    <div style="text-align:center; padding:12px 0 max(6px, env(safe-area-inset-bottom)); margin-top:12px; border-top:1px solid rgba(255,255,255,0.06); padding-left:max(0px, env(safe-area-inset-left)); padding-right:max(0px, env(safe-area-inset-right));">
        <p style="margin:0; font-size:clamp(10px, 2.9vw, 11px); color:#666; line-height:1.45; max-width:36rem; margin-left:auto; margin-right:auto;">Assistive tool only · Human interpreters have final authority · Not for official records</p>
    </div>
    """


# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def _gradio_audio_to_temp_wav(audio: Any) -> tuple[Path | None, list[Path]]:
    cleanup: list[Path] = []
    if audio is None:
        return None, cleanup
    if isinstance(audio, dict):
        p = audio.get("path")
        if isinstance(p, str) and Path(p).is_file():
            return Path(p), cleanup
        return None, cleanup
    if isinstance(audio, (str, Path)):
        p = Path(audio)
        return (p, cleanup) if p.is_file() else (None, cleanup)
    if not isinstance(audio, tuple) or len(audio) != 2:
        return None, cleanup
    sr_raw, y = audio
    if y is None:
        return None, cleanup
    y = np.asarray(y)
    if y.size == 0:
        return None, cleanup
    if np.issubdtype(y.dtype, np.integer):
        y = np.clip(y.astype(np.float32) / 32768.0, -1.0, 1.0)
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    sr = int(sr_raw) if sr_raw else 48_000
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    p = Path(name)
    sf.write(str(p), y, sr, subtype="PCM_16")
    cleanup.append(p)
    return p, cleanup


def _transcript_rich_html_from_lines(lines: list[str], plain: str) -> str:
    """Build HTML blocks for diarized lines (already escaped-safe via html.escape)."""
    blocks: list[str] = []
    for line in lines:
        escaped = html.escape(line.strip())
        if not escaped:
            continue
        blocks.append(
            '<div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:12px 16px; '
            'margin-bottom:8px; font-size:clamp(14px,3.8vw,15px); color:#eee; line-height:1.5; '
            f'word-break:break-word;">{escaped}</div>'
        )
    if blocks:
        return "".join(blocks)
    return (
        f'<div style="background:rgba(255,255,255,0.04); border-radius:10px; padding:16px; '
        f'font-size:clamp(14px,3.8vw,15px); color:#eee; line-height:1.5; word-break:break-word;">'
        f"{html.escape(plain)}</div>"
    )


def _split_for_translation(text: str, max_chars: int = 1200) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    chunks: list[str] = []
    start = 0
    n = len(t)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            window = t[start:end]
            br = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
            if br >= max_chars // 4:
                end = start + br + 1
        piece = t[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end
    return chunks


def _looks_like_meta_translation_response(text: str) -> bool:
    """Detect assistant/refusal style responses that are not direct translations."""
    t = (text or "").strip().lower()
    if not t:
        return False
    markers = (
        "you provided",
        "please provide",
        "i can",
        "i cannot",
        "i can't",
        "i am unable",
        "it appears",
        "if you could",
        "i'd be happy to help",
        "as an ai",
    )
    return any(m in t for m in markers)


def _run_session_transcript(audio, fast_mode, skip_analysis) -> tuple[str, str, str]:
    """ElevenLabs transcription + Groq text translation into English."""
    src, to_clean = _gradio_audio_to_temp_wav(audio)
    if src is None:
        return "Upload an audio file first", "", ""

    skip_local_ml = skip_analysis if fast_mode else False

    try:
        from demo.audio_analysis_pipeline import (
            normalize_to_wav_16k_mono,
            run_ui_audio_analysis,
            scribe_language_code_for_translation,
        )
        from backend.src.services.gemini_translation import (
            translate_text,
            translation_skipped_no_source,
        )

        fd, wav_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp_wav = Path(wav_name)

        try:
            normalize_to_wav_16k_mono(src, tmp_wav)
            result = run_ui_audio_analysis(tmp_wav, status=lambda x: None, skip_local_ml=skip_local_ml)
        finally:
            tmp_wav.unlink(missing_ok=True)

        if result.scribe_error:
            return f"Error: {result.scribe_error}", "", ""

        plain = (result.transcript_plain or "").strip()
        if not plain:
            return "No speech detected in this file.", "", ""

        src_lang = scribe_language_code_for_translation(result.language_code)

        def tr_en(text: str, max_out: int) -> str:
            t = (text or "").strip()
            if not t or translation_skipped_no_source(t):
                return t
            parts = _split_for_translation(t, max_chars=1200)
            if not parts:
                return t
            out: list[str] = []
            for part in parts:
                translated = translate_text(
                    source_text=part,
                    source_language=src_lang or "auto",
                    target_language="en",
                    config_id=TRANSCRIPT_TRANSLATION_CONFIG_ID,
                    max_output_tokens=max_out,
                ).strip()
                if _looks_like_meta_translation_response(translated):
                    # Retry once with explicit constraints embedded in source text for stubborn cases.
                    translated = translate_text(
                        source_text=(
                            "Direct courtroom translation required. Output only the translated utterance text in "
                            "English. Do not explain, refuse, or mention language/script.\n\nUtterance:\n" + part
                        ),
                        source_language=src_lang or "auto",
                        target_language="en",
                        config_id=TRANSCRIPT_TRANSLATION_CONFIG_ID,
                        max_output_tokens=max_out,
                    ).strip()
                out.append(translated)
            return "\n\n".join(x for x in out if x)

        try:
            plain_en = tr_en(plain, 4096)
        except Exception as te:  # noqa: BLE001
            return f"Error: translation to English failed: {te}", "", ""

        lines: list[str] = []
        speaker_role_map = {
            "Speaker 1": "Judge",
            "Speaker 2": "Defendant",
            "Speaker 3": "Witness",
            "Speaker 4": "Prosecutor",
        }
        for seg in result.segments or []:
            if seg.get("is_audio_event"):
                tag = (seg.get("audio_event_tag") or seg.get("text") or "event").strip()
                lines.append(f"[{tag}]")
                continue
            label_raw = (seg.get("speaker_label") or "Speaker").strip()
            label = speaker_role_map.get(label_raw, label_raw)
            text_src = (seg.get("text") or "").strip()
            if not text_src:
                continue
            try:
                text_en = tr_en(text_src, 1024)
            except Exception as te:  # noqa: BLE001
                return f"Error: translation to English failed: {te}", "", ""
            if text_en:
                lines.append(f"[{label}]: {text_en}")
        if not lines:
            lines = [plain_en or plain]

        rich_html = _transcript_rich_html_from_lines(lines, plain_en or plain)

        lang = (result.language_display or "Unknown").strip()
        return f"✓ Transcript ready · English (audio detected: {lang})", rich_html, (plain_en or plain)

    except Exception as e:
        return f"Error: {e}", "", ""
    finally:
        for p in to_clean:
            p.unlink(missing_ok=True)


# =============================================================================
# BUILD APP
# =============================================================================

def build_demo() -> gr.Blocks:
    
    with gr.Blocks(
        title="IIKSHANA",
        css=CUSTOM_CSS,
        head=EXPO_HEAD,
        theme=EXPO_THEME,
        fill_width=True,
        fill_height=True,
    ) as demo:
        
        # Header: brand logo (PNG) + page title
        with gr.Column(elem_classes=["expo-header-wrap"], scale=0):
            if EXPO_LOGO.is_file():
                gr.Image(
                    value=str(EXPO_LOGO),
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    container=False,
                    interactive=False,
                    scale=0,
                    elem_classes=["expo-brand-logo"],
                )
            gr.HTML(header_html(show_wordmark=not EXPO_LOGO.is_file()))

        # Listening / speaking language (dropdowns inside paired cards)
        with gr.Row(equal_height=True, elem_classes=["lang-pair-row"]):
            with gr.Column(scale=1, elem_classes=["lang-strip-card"]):
                gr.HTML(
                    '<p style="margin:0 0 6px 0;font-size:10px;line-height:1.2;color:#777;text-transform:uppercase;letter-spacing:0.5px;text-align:center;">Listening in</p>'
                )
                source_lang_dd = gr.Dropdown(
                    choices=LANG_LISTEN_CHOICES,
                    value="auto",
                    label="",
                    show_label=False,
                )
            with gr.Column(scale=0, elem_classes=["lang-arrow-col"]):
                gr.HTML('<div style="color:#555;font-size:16px;line-height:1;">→</div>')
            with gr.Column(scale=1, elem_classes=["lang-strip-card"]):
                gr.HTML(
                    '<p style="margin:0 0 6px 0;font-size:10px;line-height:1.2;color:#777;text-transform:uppercase;letter-spacing:0.5px;text-align:center;">Speaking in</p>'
                )
                target_lang_dd = gr.Dropdown(
                    choices=LANG_SPEAK_CHOICES,
                    value="es",
                    label="",
                    show_label=False,
                )

        with gr.Tabs():
            
            # ===================== LIVE TAB =====================
            with gr.Tab("Live"):
                with gr.Row(equal_height=False, elem_classes=["expo-live-layout"]):
                    with gr.Column(scale=0, elem_classes=["expo-live-sidebar"]):
                        with gr.Column(elem_classes=["expo-mic-stack"]):
                            mic_toggle_btn = gr.Button(
                                "\u200b",
                                variant="secondary",
                                elem_classes=["expo-mic-btn"],
                                scale=0,
                                icon=str(EXPO_MIC_SVG),
                            )
                            mic_status = gr.HTML(value=mic_status_html(False))

                        with gr.Row(elem_classes=["live-options-row"]):
                            headphones = gr.Checkbox(label="🎧 Headphones", value=False)
                            text_only = gr.Checkbox(label="📝 Text only", value=False)

                        # Microphone stream (shown when live). Do NOT set recording=True from Python: Gradio's
                        # streaming UI only creates the MediaRecorder inside record(), so recording=True without
                        # a prior record() leaves capture never started and start_recording never fires.
                        live_audio = gr.Audio(
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                            visible=False,
                            recording=False,
                            show_label=False,
                            label="",
                        )

                    with gr.Column(scale=1, elem_classes=["expo-live-main"]):
                        # Output area (wider on laptop — sits beside mic column)
                        live_output = gr.HTML(value="", visible=False, elem_classes=["expo-live-feed"])
                        # TTS playback
                        live_tts = gr.Audio(type="filepath", autoplay=True, visible=False)
                
                # State
                live_state = gr.State(value=make_initial_state())
                is_live = gr.State(value=False)

                def live_idle_outputs():
                    """Reset live tab when recording stops (mic button, Audio stop, or clear)."""
                    return (
                        False,
                        mic_status_html(False),
                        gr.update(elem_classes=["expo-mic-btn"]),
                        gr.update(visible=False, recording=False, value=None),
                        gr.update(value="", visible=False),
                        gr.update(visible=False, value=None),
                        make_initial_state(),
                    )

                def toggle_live(is_on, hp, txt, state):
                    idle_btn = gr.update(elem_classes=["expo-mic-btn"])
                    live_btn = gr.update(elem_classes=["expo-mic-btn", "expo-mic-btn--live"])
                    if not is_on:
                        if not hp and not txt:
                            return (
                                False,
                                mic_status_html(
                                    False,
                                    notice="Please select Headphones or Text only before starting.",
                                ),
                                idle_btn,
                                gr.update(visible=False, recording=False, value=None),
                                gr.update(visible=False),
                                gr.update(visible=False),
                                state,
                            )
                        return (
                            True,
                            mic_status_html(
                                False,
                                notice="Tap Record below to allow the microphone.",
                            ),
                            live_btn,
                            gr.update(visible=True, recording=False, value=None),
                            gr.update(visible=True, value=""),
                            gr.update(visible=hp and not txt),
                            make_initial_state(),
                        )
                    return live_idle_outputs()

                mic_toggle_btn.click(
                    toggle_live,
                    [is_live, headphones, text_only, live_state],
                    [is_live, mic_status, mic_toggle_btn, live_audio, live_output, live_tts, live_state],
                )

                def on_audio_recording_started():
                    """User tapped Gradio's Record and getUserMedia ran — show Listening."""
                    return mic_status_html(True)

                live_audio.start_recording(
                    on_audio_recording_started,
                    inputs=None,
                    outputs=[mic_status],
                )

                live_audio.stop_recording(
                    live_idle_outputs,
                    inputs=None,
                    outputs=[
                        is_live,
                        mic_status,
                        mic_toggle_btn,
                        live_audio,
                        live_output,
                        live_tts,
                        live_state,
                    ],
                )
                
                def process_chunk(chunk, state, tgt, src_listen, hp):
                    src_override = None if not src_listen or str(src_listen).lower() == "auto" else str(src_listen)
                    state, display_html, _status, audio_path = process_audio_chunk(
                        chunk,
                        state,
                        tgt,
                        DEFAULT_TRANSLATION_CONFIG_ID,
                        bool(hp),
                        source_language_override=src_override,
                    )
                    # display_html is already safe markup from live_translation (translations escaped there)
                    return state, gr.update(value=display_html or ""), audio_path if audio_path else gr.update()
                
                live_audio.stream(
                    process_chunk,
                    [live_audio, live_state, target_lang_dd, source_lang_dd, headphones],
                    [live_state, live_output, live_tts],
                )

                def on_live_audio_cleared():
                    """User clicked the Audio control X — full idle (same as stopping)."""
                    return live_idle_outputs()

                live_audio.clear(
                    on_live_audio_cleared,
                    inputs=None,
                    outputs=[is_live, mic_status, mic_toggle_btn, live_audio, live_output, live_tts, live_state],
                )

            # ===================== TRANSCRIPT TAB =====================
            with gr.Tab("Transcript"):
                gr.HTML(
                    '<p style="margin:0 0 12px 0; font-size:13px; color:#94a3b8; line-height:1.45;">'
                    "Upload a courtroom session recording. The app transcribes with ElevenLabs and then "
                    "translates each turn into <strong>English</strong> using your configured translation model "
                    "(default: Groq config)."
                    "</p>"
                )
                audio_input = gr.Audio(
                    sources=["upload"],
                    type="filepath",
                    label="Courtroom session audio",
                )

                with gr.Row(elem_classes=["record-options-row"]):
                    fast_mode = gr.Checkbox(
                        label="Fast path",
                        value=_default_fast_translate(),
                        info="When off, runs full speaker analysis (slower).",
                    )
                    skip_analysis = gr.Checkbox(
                        label="Skip detailed analysis",
                        value=_default_skip_local_ml(),
                        info="Skip gender/emotion models when Fast path is on.",
                    )

                process_btn = gr.Button("Generate transcript", variant="primary")

                status_out = gr.HTML(value="")
                transcript_out = gr.HTML(value="")
                # Hidden: keeps a plain-text copy in the backend without cluttering the UI.
                transcript_plain = gr.Textbox(
                    label="Plain text (copy)",
                    lines=12,
                    max_lines=24,
                    interactive=True,
                    show_copy_button=True,
                    visible=False,
                )

                def process_transcript_and_format(audio, fast, skip):
                    status, rich_html, plain = _run_session_transcript(audio, fast, skip)

                    if status.startswith("✓"):
                        status_html = (
                            f'<div style="background:rgba(16,185,129,0.1); color:#10B981; padding:10px 16px; '
                            f'border-radius:8px; font-size:14px; font-weight:500; display:inline-block;">{html.escape(status)}</div>'
                        )
                    elif status.startswith("Error"):
                        status_html = (
                            f'<div style="background:rgba(239,68,68,0.1); color:#ef4444; padding:10px 16px; '
                            f'border-radius:8px; font-size:14px;">{html.escape(status)}</div>'
                        )
                    else:
                        status_html = f'<div style="color:#888; font-size:14px;">{html.escape(status)}</div>'

                    body_html = (
                        f'<div style="margin-top:12px;">{rich_html}</div>' if rich_html else ""
                    )
                    return status_html, body_html, plain or ""

                process_btn.click(
                    process_transcript_and_format,
                    [audio_input, fast_mode, skip_analysis],
                    [status_out, transcript_out, transcript_plain],
                )

            # ===================== SIGN LANGUAGE TAB =====================
            with gr.Tab("Sign Language"):
                gr.Markdown(
                    """
                    <div style="background:rgba(0,200,150,0.08); padding:12px 16px; border-radius:10px; border-left:4px solid #00c896;">
                    <h3 style="margin:0; color:#00c896;">🤟 ASL for Deaf Courtroom Participants</h3>
                    <p style="margin:4px 0 0 0; font-size:14px; color:#ccc;">
                    Courtroom speech is automatically transcribed and signed by the ASL avatar in real-time.
                    </p>
                    </div>
                    """
                )

                asl_source_lang = gr.Dropdown(
                    choices=LANG_LISTEN_CHOICES,
                    value="auto",
                    label="Listening in",
                )

                asl_audio = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Microphone",
                )

                asl_state = gr.State(value=make_initial_asl_state())

                signmt_iframe = gr.HTML(
                    value="""
                    <div style="display:flex; flex-direction:column; align-items:center; padding:0;">
                        <iframe
                            id="signmt-frame"
                            name="signmt-frame"
                            src="https://sign.mt/?spl=en&sl=ase"
                            width="100%"
                            height="650"
                            style="border:2px solid rgba(0,200,150,0.3); border-radius:12px; max-width:900px;"
                            allow="camera; microphone; autoplay"
                            loading="lazy"
                        ></iframe>
                    </div>
                    """,
                    label="ASL Avatar",
                )

                asl_transcript = gr.Textbox(
                    label="Transcript",
                    interactive=False,
                    lines=2,
                    placeholder="Speak to see transcript here...",
                )

                asl_transcript.change(
                    fn=None,
                    inputs=[asl_transcript],
                    outputs=[],
                    js="""
                    (text) => {
                        if (!text || !String(text).trim()) return;
                        const lines = String(text).trim().split(/\\r?\\n|\\r/).filter((l) => l.trim());
                        if (lines.length === 0) return;
                        const lastLine = lines[lines.length - 1].trim();
                        if (!lastLine || lastLine.length < 3) return;
                        if (window._lastSignMtText === lastLine) return;
                        window._lastSignMtText = lastLine;
                        const truncated = lastLine.substring(0, 80);
                        const encoded = encodeURIComponent(truncated);
                        const newSrc = "https://sign.mt/?spl=en&sl=ase&text=" + encoded;
                        const iframe = document.getElementById("signmt-frame");
                        if (!iframe) {
                            const fallback = document.querySelector('iframe[name="signmt-frame"]')
                                || document.querySelector('iframe[src*="sign.mt"]');
                            if (fallback) fallback.src = newSrc;
                            return;
                        }
                        iframe.src = newSrc;
                    }
                    """,
                )

                def process_asl(chunk, state, src_lang):
                    src_override = (
                        None if not src_lang or str(src_lang).lower() == "auto" else str(src_lang)
                    )
                    state, english, _gloss, _asl_html = process_asl_chunk(
                        chunk, state, source_language_override=src_override
                    )
                    return state, english or ""

                asl_audio.stream(
                    process_asl,
                    [asl_audio, asl_state, asl_source_lang],
                    [asl_state, asl_transcript],
                )

        # Footer
        gr.HTML(footer_html())
    
    return demo


def main() -> None:
    _apply_gradio_uvicorn_http_workarounds()
    demo = build_demo()
    try:
        demo.queue(default_concurrency_limit=1)
    except TypeError:
        demo.queue()
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=host, server_port=port, ssr_mode=False)


if __name__ == "__main__":
    main()