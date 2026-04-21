"""
UI-path batch audio analysis: ElevenLabs Scribe v2 → per-turn splits →
inaSpeechSegmenter (gender) → FunASR emotion2vec+ → summary + ElevenLabs TTS.

Heavy models are loaded lazily once per process (singletons).
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import librosa
import numpy as np
import soundfile as sf

# --- Lazy singletons for heavyweight models ---------------------------------

_gender_segmenter: Any = None
_emotion_model: Any = None


def get_gender_segmenter() -> Any:
    global _gender_segmenter
    if _gender_segmenter is None:
        from inaSpeechSegmenter import Segmenter  # noqa: PLC0415

        _gender_segmenter = Segmenter()
    return _gender_segmenter


def get_emotion_model() -> Any:
    global _emotion_model
    if _emotion_model is None:
        from funasr import AutoModel  # noqa: PLC0415

        _emotion_model = AutoModel(model="iic/emotion2vec_plus_large")
    return _emotion_model


# --- Language display (common ISO 639-3 from Scribe) -------------------------

_ISO639_3 = {
    "eng": ("English", "en"),
    "spa": ("Spanish", "es"),
    "fra": ("French", "fr"),
    "deu": ("German", "de"),
    "ita": ("Italian", "it"),
    "por": ("Portuguese", "pt"),
    "hin": ("Hindi", "hi"),
    "mar": ("Marathi", "mr"),
    "tel": ("Telugu", "te"),
    "tam": ("Tamil", "ta"),
    "ben": ("Bengali", "bn"),
    "urd": ("Urdu", "ur"),
    "ara": ("Arabic", "ar"),
    "zho": ("Chinese", "zh"),
    "jpn": ("Japanese", "ja"),
    "kor": ("Korean", "ko"),
    "rus": ("Russian", "ru"),
    "nld": ("Dutch", "nl"),
    "pol": ("Polish", "pl"),
    "tur": ("Turkish", "tr"),
    "vie": ("Vietnamese", "vi"),
}


def language_display(code: str | None) -> str:
    if not code or not str(code).strip():
        return "Unknown"
    c = str(code).strip().lower()
    if len(c) == 2:
        for name, short in _ISO639_3.values():
            if short == c:
                return f"{name} ({short})"
        return f"Unknown ({c})"
    info = _ISO639_3.get(c[:3])
    if info:
        name, short = info
        return f"{name} ({short})"
    return f"Unknown ({c[:3]})"


def scribe_language_code_for_translation(code: str | None) -> str:
    """Map Scribe ``language_code`` (often ISO 639-3) to a short tag for ``translate_text`` (en, es, …)."""
    if not code or not str(code).strip():
        return "en"
    c = str(code).strip().lower()
    if len(c) == 2 and c.isalpha():
        return c
    info = _ISO639_3.get(c[:3])
    return info[1] if info else "en"


# --- Audio I/O ---------------------------------------------------------------

def normalize_to_wav_16k_mono(src_path: Path, dst_path: Path) -> None:
    y, _ = librosa.load(str(src_path), sr=16_000, mono=True)
    sf.write(str(dst_path), y, 16_000, subtype="PCM_16")


def _word_is_audio_event(w: Any) -> bool:
    t = getattr(w, "type", None)
    v = getattr(t, "value", t)
    return str(v).lower() == "audio_event"


def _scribe_to_segments(words: list[Any]) -> list[dict[str, Any]]:
    """Group ElevenLabs word list into segments (speech turns + audio events)."""
    none_sentinel = "__none__"
    order: list[str] = []
    seen: set[str] = set()
    for w in words:
        if _word_is_audio_event(w):
            continue
        sid = none_sentinel if w.speaker_id is None else str(w.speaker_id)
        if sid not in seen:
            seen.add(sid)
            order.append(sid)
    if not order:
        order = [none_sentinel]
    sid_to_label = {sid: f"Speaker {i + 1}" for i, sid in enumerate(order)}

    segments: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal cur
        if cur is None:
            return
        out = cur
        cur = None
        out.pop("_sk", None)
        if out.get("is_audio_event") or (out.get("text") or "").strip():
            segments.append(out)

    for w in words:
        start = w.start
        end = w.end
        text = (w.text or "").strip()
        if _word_is_audio_event(w):
            flush()
            segments.append(
                {
                    "speaker_id": None,
                    "speaker_label": "",
                    "start": float(start or 0.0),
                    "end": float(end or start or 0.0),
                    "text": text,
                    "is_audio_event": True,
                    "audio_event_tag": text,
                }
            )
            continue

        sid = w.speaker_id
        sk = none_sentinel if sid is None else str(sid)
        label = sid_to_label[sk]

        if cur is None or cur.get("is_audio_event"):
            cur = {
                "speaker_id": sid,
                "_sk": sk,
                "speaker_label": label,
                "start": float(start or 0.0),
                "end": float(end or start or 0.0),
                "text": text,
                "is_audio_event": False,
                "audio_event_tag": None,
            }
        elif sk == cur.get("_sk"):
            cur["end"] = float(end or cur["end"])
            if text:
                cur["text"] = (cur["text"] + " " + text).strip()
        else:
            flush()
            cur = {
                "speaker_id": sid,
                "_sk": sk,
                "speaker_label": label,
                "start": float(start or 0.0),
                "end": float(end or start or 0.0),
                "text": text,
                "is_audio_event": False,
                "audio_event_tag": None,
            }


    flush()
    return segments


def majority_gender(
    ina_segments: list[tuple[str, float, float]],
    t0: float,
    t1: float,
) -> str:
    male_t = female_t = 0.0
    for lab, s, e in ina_segments:
        if lab not in ("male", "female"):
            continue
        overlap = max(0.0, min(e, t1) - max(s, t0))
        if overlap <= 0:
            continue
        if lab == "male":
            male_t += overlap
        else:
            female_t += overlap
    if male_t == 0.0 and female_t == 0.0:
        return "unknown"
    return "male" if male_t >= female_t else "female"


def _parse_emotion_output(res: Any) -> tuple[str, float]:
    if res is None:
        return "unknown", 0.0
    item: Any = res[0] if isinstance(res, list) and res else res
    if not isinstance(item, dict):
        return "unknown", 0.0
    if "labels" in item:
        labs = item["labels"]
        if labs and isinstance(labs[0], (list, tuple)) and len(labs[0]) >= 2:
            return str(labs[0][0]).lower(), float(labs[0][1])
        if isinstance(labs, list) and labs and isinstance(labs[0], str):
            return str(labs[0]).lower(), float(item.get("scores", [1.0])[0]) if item.get("scores") else 1.0
    if "emotion" in item:
        return str(item["emotion"]).lower(), float(item.get("score", 1.0))
    for k in ("text", "pred", "predict"):
        if k in item:
            v = item[k]
            if isinstance(v, str):
                return v.lower(), float(item.get("score", item.get("confidence", 1.0)))
    return "unknown", 0.0


def _emotion_for_wav(model: Any, wav_path: Path) -> tuple[str, float]:
    try:
        try:
            res = model.generate(
                input=str(wav_path),
                granularity="utterance",
                extract_embedding=False,
            )
        except TypeError:
            res = model.generate(str(wav_path), granularity="utterance", extract_embedding=False)
        return _parse_emotion_output(res)
    except Exception:
        return "unknown", 0.0


# Built-in premade voices (ElevenLabs defaults — not Voice Library community voices).
# Free API plans return 402 for ``category=library`` voices; ``voices.get_all()`` often lists those first,
# so we filter by category and fall through this list.
_PREMADE_TTS_VOICE_IDS: tuple[str, ...] = (
    "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "AZnzlk1XvdvUeBnXmlld",  # Domi
    "EXAVITQu4vr4xnSDxMaL",  # Bella
    "ErXwobaYiN019PkySvjV",  # Antoni
    "MF3mGyEYCl7XYWbV9V6O",  # Elli
    "TxGEqnHWrfWFTfGW9XjX",  # Josh
    "VR6AewLTigWG4xSOukaG",  # Arnold
    "pNInz6obpgDQGcFmaJgB",  # Adam
)


def _voice_category(v: Any) -> str:
    raw = getattr(v, "category", None)
    return str(raw or "").strip().lower()


def _voice_id_attr(v: Any) -> str:
    return str(getattr(v, "voice_id", v))


def _voices_from_get_all(client: Any) -> list[Any]:
    try:
        all_v = client.voices.get_all()
        voices = getattr(all_v, "voices", None)
        if voices is None and isinstance(all_v, list):
            voices = all_v
        return list(voices or [])
    except Exception:
        return []


def _resolve_tts_voice_candidates(client: Any) -> list[str]:
    """
    Ordered voice IDs to try. Respects ``ELEVENLABS_TTS_VOICE_ID`` / ``ELEVENLABS_VOICE_ID`` when set.

    Skips ``category=library`` (Voice Library) entries — those trigger 402 ``paid_plan_required`` on free tiers.
    """
    env = (os.environ.get("ELEVENLABS_TTS_VOICE_ID") or os.environ.get("ELEVENLABS_VOICE_ID") or "").strip()
    if env:
        return [env]

    voices = _voices_from_get_all(client)
    non_lib = [v for v in voices if _voice_category(v) != "library"]
    premade_vs = [v for v in non_lib if _voice_category(v) == "premade"]
    other_vs = [v for v in non_lib if _voice_category(v) != "premade"]

    ordered: list[str] = []
    seen: set[str] = set()
    for bucket in (premade_vs, other_vs):
        for v in bucket:
            vid = _voice_id_attr(v)
            if vid and vid not in seen:
                seen.add(vid)
                ordered.append(vid)

    for vid in _PREMADE_TTS_VOICE_IDS:
        if vid not in seen:
            seen.add(vid)
            ordered.append(vid)

    return ordered if ordered else list(_PREMADE_TTS_VOICE_IDS)


def _tts_model_ids_to_try() -> list[str]:
    """Flash / Turbo models are typically included on free tiers; multilingual v2 is tried last."""
    raw = (os.environ.get("ELEVENLABS_TTS_MODEL_ID") or "").strip()
    if raw:
        return [raw]
    return [
        "eleven_flash_v2_5",
        "eleven_flash_v2",
        "eleven_turbo_v2_5",
        "eleven_multilingual_v2",
    ]


def _is_paid_or_library_voice_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        "402" in s
        or "payment_required" in s
        or "paid_plan" in s
        or "library voice" in s
        or "paid users" in s
    )


def _tts_convert_once(client: Any, voice_id: str, text: str, model_id: str) -> bytes:
    try:
        stream: Iterator[bytes] = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            output_format="mp3_44100_128",
        )
    except Exception:
        stream = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
        )
    return b"".join(stream)


def _tts_bytes(client: Any, text: str) -> bytes:
    voice_ids = _resolve_tts_voice_candidates(client)
    models = _tts_model_ids_to_try()
    last_exc: BaseException | None = None
    for model_id in models:
        for vid in voice_ids:
            try:
                return _tts_convert_once(client, vid, text, model_id)
            except Exception as e:
                last_exc = e
                if _is_paid_or_library_voice_error(e):
                    continue
                raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("ElevenLabs TTS: no voice/model combination succeeded")


def synthesize_speech_mp3(text: str) -> tuple[bytes | None, str | None]:
    """
    ElevenLabs multilingual TTS for arbitrary text (e.g. translated transcript).

    Returns ``(mp3_bytes, error_message)``. Requires ``ELEVENLABS_API_KEY``.
    """
    stripped = (text or "").strip()
    if not stripped:
        return None, "No text to synthesize."
    if "[EMPTY_TRANSCRIPT]" in stripped:
        return None, None
    from backend.src.services.elevenlabs_stt_service import elevenlabs_api_key_from_env

    api_key = elevenlabs_api_key_from_env()
    if not api_key:
        return None, "ELEVENLABS_API_KEY is not set (or only empty in the shell — restart after fixing .env)."
    try:
        from elevenlabs.client import ElevenLabs  # noqa: PLC0415

        client = ElevenLabs(api_key=api_key)
        return _tts_bytes(client, stripped), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def build_spoken_summary(
    language_display_str: str,
    speaker_summaries: list[dict[str, Any]],
    num_speakers: int,
) -> str:
    parts = [f"The conversation was in {language_display_str}."]
    parts.append(f"{num_speakers} speaker{' was' if num_speakers == 1 else 's were'} detected.")
    for s in speaker_summaries:
        label = s["speaker_label"]
        g = s["estimated_gender"]
        dom = s["dominant_emotion"]
        secs = int(round(s["total_seconds"]))
        parts.append(
            f"{label}, estimated {g}, spoke for about {secs} seconds and appeared mostly {dom}."
        )
    return " ".join(parts)


@dataclass
class PipelineResult:
    language_code: str
    language_display: str
    transcript_plain: str
    transcript_rich_lines: list[str]
    segments: list[dict[str, Any]]
    speaker_summaries: list[dict[str, Any]]
    tts_audio_mp3: bytes | None
    scribe_error: str | None = None
    tts_error: str | None = None


def run_ui_audio_analysis(
    wav_16k_path: Path,
    *,
    status: Callable[[str], None] | None = None,
    skip_local_ml: bool = False,
) -> PipelineResult:
    """
    Run full pipeline on a 16 kHz mono WAV. Requires ELEVENLABS_API_KEY.

    When ``skip_local_ml`` is True, skips inaSpeechSegmenter and emotion2vec+ (no heavy downloads
    or CPU inference) — suitable for low-latency / real-time demo; speaker gender and emotion
    show as unknown unless Scribe diarization labels are enough for display.
    """
    from elevenlabs.client import ElevenLabs  # noqa: PLC0415

    from backend.src.services.elevenlabs_stt_service import (
        elevenlabs_api_key_from_env,
        transcribe_file_scribe_v2,
    )

    def log(msg: str) -> None:
        if status:
            status(msg)

    api_key = elevenlabs_api_key_from_env()
    if not api_key:
        return PipelineResult(
            language_code="",
            language_display="Unknown",
            transcript_plain="",
            transcript_rich_lines=[],
            segments=[],
            speaker_summaries=[],
            tts_audio_mp3=None,
            scribe_error=(
                "ELEVENLABS_API_KEY is not set. Add it to repo root `.env` or `.secrets/.env`, "
                "then restart the app. You can also use legacy env name `XI_API_KEY`."
            ),
        )

    log("Transcribing with ElevenLabs Scribe v2…")
    try:
        chunk = transcribe_file_scribe_v2(
            wav_16k_path,
            api_key=api_key,
            model_id="scribe_v2",
            diarize=True,
            tag_audio_events=True,
            timestamps_granularity="word",
        )
    except Exception as e:  # noqa: BLE001
        err = str(e).lower()
        hint = ""
        if "401" in str(e) or "invalid_api_key" in err:
            hint = (
                " **Tip:** The key in the environment is rejected. Open a new key at "
                "https://elevenlabs.io (Profile → API keys), set `ELEVENLABS_API_KEY=...` in `.env` "
                "with no extra spaces, restart the app, and ensure you did not export an empty "
                "`ELEVENLABS_API_KEY` in the terminal (that used to block `.env` loading)."
            )
        return PipelineResult(
            language_code="",
            language_display="Unknown",
            transcript_plain="",
            transcript_rich_lines=[],
            segments=[],
            speaker_summaries=[],
            tts_audio_mp3=None,
            scribe_error=f"ElevenLabs Speech-to-Text failed: {e}.{hint}",
        )

    words = list(getattr(chunk, "words", []) or [])
    language_code = getattr(chunk, "language_code", None) or ""
    lang_disp = language_display(language_code)
    transcript_plain = (getattr(chunk, "text", None) or "").strip()

    segments = _scribe_to_segments(words)
    if not segments and transcript_plain:
        dur = float(librosa.get_duration(path=str(wav_16k_path)))
        segments = [
            {
                "speaker_id": None,
                "speaker_label": "Speaker 1",
                "start": 0.0,
                "end": max(dur, 0.01),
                "text": transcript_plain,
                "is_audio_event": False,
                "audio_event_tag": None,
            }
        ]

    ina_full: list[tuple[str, float, float]] = []
    emo_model: Any = None
    if skip_local_ml:
        log("Skipping local gender/emotion models (fast path)…")
    else:
        log("Running speaker gender model (inaSpeechSegmenter)…")
        try:
            segmenter = get_gender_segmenter()
            ina_full = segmenter(str(wav_16k_path))
        except Exception as e:  # noqa: BLE001
            ina_full = []
            log(f"Gender model warning: {e}")

        log("Running emotion model (emotion2vec+)…")
        try:
            emo_model = get_emotion_model()
        except Exception as e:  # noqa: BLE001
            emo_model = None
            log(f"Emotion model unavailable: {e}")

    y_full = np.array([], dtype=np.float32)
    sr = 16_000
    if emo_model is not None:
        y_full, sr = sf.read(str(wav_16k_path))
        if y_full.ndim > 1:
            y_full = np.mean(y_full, axis=1)
        y_full = np.asarray(y_full, dtype=np.float32)
        if sr != 16_000:
            y_full = librosa.resample(y_full, orig_sr=sr, target_sr=16_000)
            sr = 16_000

    rich_lines: list[str] = []
    per_speaker_emotions: dict[str, list[tuple[str, float]]] = {}
    per_speaker_time: dict[str, float] = {}

    for seg in segments:
        if seg["is_audio_event"]:
            tag = seg.get("audio_event_tag") or seg.get("text") or "event"
            rich_lines.append(f"[{tag}]")
            continue

        t0, t1 = seg["start"], seg["end"]
        label = seg["speaker_label"]
        dur = max(0.0, t1 - t0)
        per_speaker_time[label] = per_speaker_time.get(label, 0.0) + dur

        gender = majority_gender(ina_full, t0, t1) if ina_full else "unknown"

        emotion, conf = ("unknown", 0.0)
        if emo_model is not None and dur >= 0.05:
            i0 = max(0, int(t0 * sr))
            i1 = min(len(y_full), int(t1 * sr))
            chunk_audio = y_full[i0:i1]
            if len(chunk_audio) >= 256:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                try:
                    sf.write(str(tmp_path), chunk_audio, sr, subtype="PCM_16")
                    emotion, conf = _emotion_for_wav(emo_model, tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)

        per_speaker_emotions.setdefault(label, []).append((emotion, conf))

        g_disp = "unknown" if gender == "unknown" else ("male" if gender == "male" else "female")
        seg["estimated_gender"] = g_disp
        seg["emotion"] = emotion
        seg["emotion_confidence"] = conf

        rich_lines.append(
            f"[{label} — estimated {g_disp}, {emotion}]: {seg.get('text', '').strip()}"
        )

    # Per-speaker summaries
    summaries: list[dict[str, Any]] = []
    labels_ordered = sorted(per_speaker_time.keys(), key=lambda x: (-per_speaker_time[x], x))
    for lab in labels_ordered:
        emos = [e for e, _ in per_speaker_emotions.get(lab, [])]
        dom = Counter(emos).most_common(1)[0][0] if emos else "unknown"
        # Estimated gender: majority across segment-level guesses
        genders = [
            s.get("estimated_gender")
            for s in segments
            if not s["is_audio_event"] and s["speaker_label"] == lab and s.get("estimated_gender")
        ]
        g_counts = Counter(genders)
        top_g = g_counts.most_common(1)[0][0] if g_counts else "unknown"

        summaries.append(
            {
                "speaker_label": lab,
                "estimated_gender": top_g,
                "dominant_emotion": dom,
                "total_seconds": per_speaker_time.get(lab, 0.0),
            }
        )

    num_speakers = len(labels_ordered)
    summary_text = build_spoken_summary(lang_disp, summaries, num_speakers)

    client = ElevenLabs(api_key=api_key)

    tts_bytes_out: bytes | None = None
    tts_err: str | None = None
    log("Generating spoken summary (ElevenLabs TTS)…")
    try:
        tts_bytes_out = _tts_bytes(client, summary_text)
    except Exception as e:  # noqa: BLE001
        tts_err = f"ElevenLabs TTS failed: {e}"

    return PipelineResult(
        language_code=language_code,
        language_display=lang_disp,
        transcript_plain=transcript_plain,
        transcript_rich_lines=rich_lines,
        segments=segments,
        speaker_summaries=summaries,
        tts_audio_mp3=tts_bytes_out,
        scribe_error=None,
        tts_error=tts_err,
    )
