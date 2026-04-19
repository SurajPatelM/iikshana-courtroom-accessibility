// Default for local `python -m http.server` / Docker without BACKEND_PUBLIC_URL.
// Cloud Run: overwritten at container start by docker-entrypoint.sh.
window.__IIKSHANA_BACKEND_BASE__ = '';
