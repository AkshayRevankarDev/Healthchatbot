"""
FastAPI REST backend for the React native chat frontend.

Endpoints:
  GET  /api/health          — liveness check
  POST /api/chat            — send one message, get agent reply + session state
  POST /api/transcribe      — upload audio blob → Whisper transcript
  POST /api/reset           — clear a session
  GET  /api/session/{id}    — get raw session state (debug)

Run with:
  uvicorn server:app --port 8502 --reload
"""

from __future__ import annotations
import os, sys, uuid, traceback, io, tempfile
from pathlib import Path
from typing import Optional

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Indic Mental Health Chat API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # open for local dev
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy backend imports ──────────────────────────────────────────────────────
# Import here so the server starts quickly; heavy models load on first request.
sys.path.insert(0, str(Path(__file__).parent))

_graph = None
def _get_graph():
    global _graph
    if _graph is None:
        from dialogue_manager import build_dialogue_graph
        _graph = build_dialogue_graph()
    return _graph

# ── Flores-200 → ISO 639-1 mapping ───────────────────────────────────────────
FLORES_TO_ISO: dict[str, str] = {
    "eng_Latn": "en", "hin_Deva": "hi", "ben_Beng": "bn", "tam_Taml": "ta",
    "tel_Telu": "te", "mal_Mlym": "ml", "kan_Knda": "kn", "mar_Deva": "mr",
    "guj_Gujr": "gu", "pan_Guru": "pa", "urd_Arab": "ur", "ory_Orya": "or",
    "asm_Beng": "as", "npi_Deva": "ne", "mai_Deva": "hi", "sat_Olck": "en",
    "kas_Arab": "ur", "kas_Deva": "hi", "mni_Beng": "bn", "mni_Mtei": "en",
    "doi_Deva": "hi", "brx_Deva": "hi", "gom_Deva": "mr", "san_Deva": "hi",
    "snd_Arab": "ur", "snd_Deva": "hi",
}

# ── In-memory sessions ────────────────────────────────────────────────────────
# { session_id: { "state": DialogueState, "lang": ISO code, "history": [...] } }
_sessions: dict[str, dict] = {}

def _new_session(lang_iso: str = "en") -> dict:
    from dialogue_manager import create_initial_state
    return {
        "state":   create_initial_state(),
        "lang":    lang_iso,
        "history": [],               # [{"role": "user"|"agent", "text": str}]
        "safety_alert":    False,
        "quota_error":     False,
    }

def _get_or_create(session_id: str, lang_iso: str = "en") -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = _new_session(lang_iso)
    return _sessions[session_id]

# ── Helpers ───────────────────────────────────────────────────────────────────
DOMAINS = ["sleep", "mood", "energy", "appetite", "concentration"]

PHASE_LABEL_MAP = {
    "rapport":       "RAPPORT",
    "sleep":         "SLEEP",
    "mood":          "MOOD",
    "energy":        "ENERGY",
    "appetite":      "APPETITE",
    "concentration": "CONCENTRATION",
}

def _state_to_frontend(sess: dict) -> dict:
    """Convert internal dialogue state → JSON the React frontend expects."""
    state   = sess["state"]
    scores  = state.get("scores", {})
    conf_raw = state.get("confidence", {d: 0.0 for d in DOMAINS})

    # Confidence 0–100 for the progress bars
    confidence = {}
    for d in DOMAINS:
        raw_val = conf_raw.get(d, 0.0)
        if d in scores:
            # Domain fully scored: show percentage of max (3)
            confidence[d] = round((scores[d] / 3) * 100)
        else:
            # In-progress: use internal confidence float (0–1 → 0–100)
            confidence[d] = round(float(raw_val) * 100)

    domains_assessed = list(scores.keys())
    phase_raw = state.get("phase", "rapport")
    phase_label = PHASE_LABEL_MAP.get(phase_raw, phase_raw.upper())

    return {
        "phase":            phase_label,
        "turn":             state.get("turn_count", 0),
        "confidence":       confidence,
        "domains_assessed": domains_assessed,
        "safety_alert":     sess.get("safety_alert", False),
        "session_complete": state.get("session_complete", False),
        "quota_error":      sess.get("quota_error", False),
    }

def _is_indic(iso: str) -> bool:
    return iso not in ("en", "auto", "")

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: str
    lang_code:  str = "eng_Latn"   # Flores-200 code

class ResetRequest(BaseModel):
    session_id: str
    lang_code:  str = "eng_Latn"

class GreetingRequest(BaseModel):
    session_id: str
    lang_code:  str  = "eng_Latn"
    mood:       str  = ""

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/greeting")
def greeting(req: GreetingRequest):
    """Generate the opening greeting for a new session."""
    lang_iso = FLORES_TO_ISO.get(req.lang_code, "en")
    sess = _get_or_create(req.session_id, lang_iso)
    sess["lang"] = lang_iso

    _mood_greetings = {
        "very low":   "I'm really glad you reached out — it takes courage. What's been going on for you?",
        "struggling": "I'm glad you're here. It sounds like things have been tough — want to tell me what's been happening?",
        "neutral":    "Hi there! How have things been going for you lately?",
        "okay":       "Hey, glad you stopped by! What's been on your mind recently?",
        "good":       "Hi! You mentioned things are going well — tell me more about how you've been!",
    }

    from llm_client import call_llm
    try:
        mood_hint = f"The person has indicated they are feeling '{req.mood}' today. " if req.mood else ""
        greeting_en = call_llm(
            f"You are starting a warm mental health check-in. {mood_hint}"
            "Write ONE short friendly opening sentence (max 20 words) acknowledging their mood "
            "and gently inviting them to share more. Plain text only, no asterisks or markdown.",
            temperature=0.8, max_tokens=60,
        )
        if not greeting_en or len(greeting_en) < 10:
            greeting_en = _mood_greetings.get(req.mood, "Hi, I'm glad you're here. How have things been going for you lately?")
    except Exception:
        greeting_en = _mood_greetings.get(req.mood, "Hi, I'm glad you're here. How have things been going for you lately?")

    # Translate if needed
    if _is_indic(lang_iso):
        try:
            from translator import translate_from_english
            text = translate_from_english(greeting_en, lang_iso)
        except Exception:
            text = greeting_en
    else:
        text = greeting_en

    sess["history"].append({"role": "agent", "text": text})
    return {"reply": text, **_state_to_frontend(sess)}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Process one user message and return the agent reply."""
    lang_iso = FLORES_TO_ISO.get(req.lang_code, "en")
    sess = _get_or_create(req.session_id, lang_iso)
    sess["lang"] = lang_iso   # may change mid-session

    user_text_original = req.message.strip()
    if not user_text_original:
        raise HTTPException(400, "Empty message")

    # ── 1. Language detection / override ─────────────────────────────────────
    try:
        from translator import detect_language
        detected = detect_language(user_text_original)
        if detected not in ("en", "auto") and lang_iso == "en":
            lang_iso = detected
            sess["lang"] = detected
    except Exception:
        pass

    # ── 2. Translate → English ────────────────────────────────────────────────
    user_text_en = user_text_original
    if _is_indic(lang_iso):
        try:
            from translator import translate_to_english
            user_text_en = translate_to_english(user_text_original, lang_iso)
        except Exception:
            pass
    else:
        # Force-translate romanized Hindi/Urdu that slipped past langdetect
        try:
            from deep_translator import GoogleTranslator as _GT
            _auto = _GT(source="auto", target="en").translate(user_text_original)
            if (
                _auto and _auto.strip()
                and "Error" not in _auto[:30]
                and _auto.strip().lower() != user_text_original.strip().lower()
                and len(set(_auto.lower().split()) - set(user_text_original.lower().split())) > 1
                and any(w in _auto.lower().split()
                        for w in {"i","am","is","the","a","my","me","to","want","feel","have","can","not"})
            ):
                user_text_en = _auto.strip()
        except Exception:
            pass

    # ── 3. Safety check ───────────────────────────────────────────────────────
    try:
        from safety_monitor import check_safety
        safety_en   = check_safety(user_text_en)
        safety_orig = check_safety(user_text_original)
        if safety_en["triggered"] or safety_orig["triggered"]:
            sess["safety_alert"] = True
            sess["state"]["safety_triggered"] = True
    except Exception:
        pass

    # ── 4. Add user message to history ───────────────────────────────────────
    sess["history"].append({"role": "user", "text": user_text_original})

    # ── 5. Run dialogue manager ───────────────────────────────────────────────
    try:
        from dialogue_manager import run_interactive_turn
        new_state, bot_en = run_interactive_turn(
            user_text_en,
            sess["state"],
            _get_graph(),
        )
        sess["state"] = new_state
    except Exception as exc:
        traceback.print_exc()
        bot_en = "I appreciate you sharing that. Could you tell me more?"

    if new_state.get("llm_failed"):
        sess["quota_error"] = True

    # ── 6. Translate response → user language ─────────────────────────────────
    if _is_indic(lang_iso) and bot_en:
        try:
            from translator import translate_from_english
            bot_text = translate_from_english(bot_en, lang_iso)
        except Exception:
            bot_text = bot_en
    else:
        bot_text = bot_en or "Could you tell me a little more about that?"

    sess["history"].append({"role": "agent", "text": bot_text})

    return {"reply": bot_text, **_state_to_frontend(sess)}


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    lang_code: str    = Form("eng_Latn"),
):
    """Transcribe uploaded audio with Whisper."""
    lang_iso = FLORES_TO_ISO.get(lang_code, "en")
    data = await audio.read()

    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        from translator import transcribe_audio
        text = transcribe_audio(tmp_path, user_lang=lang_iso)
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return {"text": text, "lang": lang_iso}


@app.post("/api/reset")
def reset(req: ResetRequest):
    """Clear session and start fresh."""
    lang_iso = FLORES_TO_ISO.get(req.lang_code, "en")
    _sessions[req.session_id] = _new_session(lang_iso)
    return {"ok": True}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    """Return raw session state (debug endpoint)."""
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")
    sess = _sessions[session_id]
    return {
        "lang":    sess["lang"],
        "history": sess["history"],
        **_state_to_frontend(sess),
    }
