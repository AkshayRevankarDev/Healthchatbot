"""
Streamlit UI for Mental Health Screening Conversational Agent
Two-panel interface: chat on left, live confidence bars + scores on right.
Supports multilingual input/output and Whisper voice-to-text.
"""

import streamlit as st
import json
import os
import random
import tempfile
from pathlib import Path

# Page config must be first
st.set_page_config(
    page_title="Mental Health Screening Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Lazy resource loaders ──────────────────────────────────────────────────────

@st.cache_resource
def load_graph():
    from dialogue_manager import build_dialogue_graph
    return build_dialogue_graph()

@st.cache_resource
def load_kb():
    with open("dsm_kb.json") as f:
        return json.load(f)

@st.cache_resource
def load_whisper_model():
    """Load Whisper 'base' model (~74 MB) — cached after first load."""
    try:
        from translator import load_whisper_model as _load
        return _load("base")
    except Exception:
        return None

from dialogue_manager import create_initial_state, run_interactive_turn, DOMAINS, RAPPORT_TURNS
from safety_monitor import check_safety
from translator import (
    LANGUAGE_OPTIONS, get_language_name,
    translate_to_english, translate_from_english,
    detect_language, detect_script, transcribe_audio,
    is_indic, are_indic_models_loaded, load_all_indic_models,
    get_indictrans2_error,
)

# ─── Session State Init ────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "dialogue_state":  create_initial_state(),
        "chat_history":    [],
        "quota_error":     False,
        "graph":           None,
        "safety_alert":    False,
        "session_started": False,
        "user_lang":       "en",          # ISO code of user's preferred language
        "whisper_model":   None,          # lazy Whisper load
        "voice_transcript": "",           # last Whisper output
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.dialogue_state  = create_initial_state()
    st.session_state.chat_history    = []
    st.session_state.safety_alert    = False
    st.session_state.session_started = False
    st.session_state.quota_error     = False
    st.session_state.voice_transcript = ""


# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .safety-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    .domain-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: inherit;
        opacity: 0.9;
        margin-bottom: 2px;
    }
    .score-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .score-0 { background: #d4edda; color: #155724; }
    .score-1 { background: #fff3cd; color: #7a5c00; }
    .score-2 { background: #ffe5b4; color: #7d4e00; }
    .score-3 { background: #f8d7da; color: #721c24; }
    .current-domain-box {
        background: rgba(31, 119, 180, 0.15);
        border-left: 4px solid #1f77b4;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 8px 0;
        color: inherit;
    }
    .lang-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        background: rgba(31, 119, 180, 0.18);
        font-size: 0.8rem;
        font-weight: 600;
        color: inherit;
        margin-left: 6px;
    }
    .translation-note {
        font-size: 0.75rem;
        opacity: 0.6;
        font-style: italic;
        margin-top: 2px;
    }
    .voice-section {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar: Language & Settings ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("🌐 Language")
        lang_names = list(LANGUAGE_OPTIONS.keys())
        current_lang_name = next(
            (name for name, code in LANGUAGE_OPTIONS.items()
             if code == st.session_state.user_lang),
            "English"
        )
        selected = st.selectbox(
            "Respond in:",
            lang_names,
            index=lang_names.index(current_lang_name),
            help="The agent will understand your input and respond in this language."
        )
        new_code = LANGUAGE_OPTIONS[selected]
        if new_code != st.session_state.user_lang:
            st.session_state.user_lang = new_code
            st.rerun()

        if st.session_state.user_lang != "en":
            st.info(
                f"🌐 **{selected}** mode active\n\n"
                "Your messages are auto-translated to English for processing, "
                "then responses are translated back to your language."
            )

        st.divider()
        st.subheader("🎙️ Voice Input")
        st.caption(
            "Uses OpenAI Whisper (runs locally). "
            "Supports Hindi, Urdu, English and 90+ languages."
        )
        # Robust Whisper availability check
        _whisper_ok = False
        _whisper_err = ""
        try:
            import whisper as _wmod
            _wmod.available_models()   # exercises the package, catches bad installs
            _whisper_ok = True
        except Exception as _e:
            _whisper_err = str(_e)

        if _whisper_ok:
            st.caption("✅ Whisper available — use the mic button in the chat panel.")
        else:
            import sys as _sys
            st.warning(
                "⚠️ Whisper not available.\n\n"
                f"**Error:** `{_whisper_err or 'import failed'}`\n\n"
                f"**Python:** `{_sys.executable}`\n\n"
                "**Fix:** `pip install openai-whisper`  \n"
                "Then restart the Streamlit app."
            )

        st.divider()
        st.subheader("ℹ️ About")
        st.caption(
            "Mental health screening powered by:\n"
            "- **Gemini 2.5 Flash** (LLM)\n"
            "- **LangGraph** (dialogue flow)\n"
            "- **SBERT** (safety monitor)\n"
            "- **IndicTrans2** (Indian language translation)\n"
            "- **Google Translate** (other languages)\n"
            "- **Whisper** (voice-to-text)"
        )
        # IndicTrans2 model status panel
        if is_indic(st.session_state.get("user_lang", "en")):
            models_ready = are_indic_models_loaded()
            with st.expander("🔧 IndicTrans2 Status", expanded=not models_ready):
                st.caption("200M-parameter distilled models (ai4bharat).")
                if models_ready:
                    st.caption("✅ Both models loaded — using IndicTrans2 exclusively.")
                else:
                    st.caption("⏳ Models will load automatically when you start chatting.")
        st.caption("⚠️ This is a screening tool only — not a clinical diagnosis.")

        if st.button("🔄 Reset Session", type="secondary", use_container_width=True):
            reset_session()
            st.rerun()


# ─── Helper: process one user turn ────────────────────────────────────────────

def process_user_turn(user_input_original: str, english_override: str = ""):
    """
    Handle safety check, translation, LLM processing, and response translation.
    Updates st.session_state in place.

    Args:
        user_input_original: Text in user's language (shown in chat).
        english_override:    Pre-translated English text (from Whisper's translate
                             task) — skips translation step when provided.
    """
    user_lang = st.session_state.user_lang

    # 1. Auto-detect language from text/script when user hasn't picked one manually
    if not english_override:   # skip for voice (Whisper already detected language)
        detected = detect_language(user_input_original)
        if detected not in ("en",) and user_lang == "en":
            user_lang = detected
            st.session_state.user_lang = detected
            st.toast(
                f"🌐 Detected **{get_language_name(detected)}** — "
                f"I'll respond in {get_language_name(detected)}.",
                icon="🌐"
            )

    # 1b. Ensure IndicTrans2 models are loaded before any translation attempt.
    #     This handles language being auto-detected or switched mid-session
    #     (the loading gate in main() only catches language chosen before page load).
    if is_indic(user_lang) and not are_indic_models_loaded():
        with st.spinner(
            f"⏳ Loading IndicTrans2 models for {get_language_name(user_lang)} "
            "(first time: 2–5 min — cached locally after that)…"
        ):
            try:
                load_all_indic_models()
            except Exception as _load_err:
                st.warning(
                    f"Could not load translation models: {_load_err}\n\n"
                    "Responding in English for now."
                )
                user_lang = "en"
                st.session_state.user_lang = "en"

    # 2. Translate input → English
    if english_override:
        # Voice path: Whisper's own translate task already gave us English
        user_input_en = english_override
    elif user_lang != "en":
        with st.spinner(f"Translating your message to English…"):
            user_input_en = translate_to_english(user_input_original, user_lang)
    else:
        user_input_en = user_input_original

    # 3. Safety check on English text
    safety = check_safety(user_input_en)
    # Also check original (catches script-specific keywords)
    safety_orig = check_safety(user_input_original)
    if safety["triggered"] or safety_orig["triggered"]:
        st.session_state.safety_alert = True

    # 4. Show user message in original language
    st.session_state.chat_history.append({"role": "user", "content": user_input_original})

    # 5. Process through dialogue manager (always English internally)
    with st.spinner("Thinking..." if user_lang == "en" else "Processing..."):
        new_state, bot_response_en = run_interactive_turn(
            user_input_en,
            st.session_state.dialogue_state,
            st.session_state.graph
        )

    st.session_state.dialogue_state = new_state

    # 6. Detect quota / LLM failure
    if new_state.get("llm_failed"):
        st.session_state.quota_error = True

    if not bot_response_en or bot_response_en.strip() == "":
        st.session_state.quota_error = True
        return

    # 7. Translate bot response → user's language
    if user_lang != "en":
        lang_name = get_language_name(user_lang)
        with st.spinner(f"Translating response to {lang_name}…"):
            bot_response = translate_from_english(bot_response_en, user_lang)
        note = f"Responded in {lang_name} · original English available on request"
    else:
        bot_response = bot_response_en
        note = ""

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": bot_response,
        "note": note,
    })

    if new_state.get("safety_triggered"):
        st.session_state.safety_alert = True


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    init_session()
    render_sidebar()

    user_lang = st.session_state.user_lang
    lang_label = "" if user_lang == "en" else f"<span class='lang-badge'>🌐 {get_language_name(user_lang)}</span>"

    st.markdown(f"## 🧠 Mental Health Screening Agent {lang_label}", unsafe_allow_html=True)
    st.caption("Adaptive conversational screening powered by Gemini + LangGraph")

    # Load resources
    kb = load_kb()
    if st.session_state.graph is None:
        with st.spinner("Loading dialogue graph..."):
            st.session_state.graph = load_graph()

    # ── IndicTrans2 HuggingFace auth warning ───────────────────────────────────
    _it2_err = get_indictrans2_error()
    if _it2_err:
        st.warning(
            "⚠️ **IndicTrans2 models are gated on HuggingFace** — "
            "using Google Translate as fallback.\n\n"
            "**To enable IndicTrans2 (better quality for Indian languages):**\n"
            "1. Accept model terms: https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M  \n"
            "2. Get your token: https://huggingface.co/settings/tokens  \n"
            "3. In your terminal: `export HF_TOKEN=your_token_here`  \n"
            "4. Restart the app.",
            icon="🔑"
        )

    # ── IndicTrans2 mandatory loading gate ──────────────────────────────────────
    # When the user has chosen an Indic language and models aren't loaded yet,
    # attempt to load them. Skip if we already know they're gated (no auth).
    if is_indic(user_lang) and not are_indic_models_loaded() and not _it2_err:
        st.info(
            "⏳ **Loading IndicTrans2 translation models** for "
            f"{get_language_name(user_lang)}…\n\n"
            "First run downloads ~200 MB from HuggingFace (takes 2-5 min).  \n"
            "Subsequent starts load from local cache in ~10 seconds."
        )
        with st.spinner("Downloading & loading IndicTrans2 models — please wait…"):
            try:
                load_all_indic_models()
            except Exception as _load_err:
                st.warning(
                    f"Could not load IndicTrans2: `{_load_err}`\n\n"
                    "Falling back to Google Translate."
                )
        st.rerun()
        return

    # ── Quota error banner ──
    if st.session_state.get("quota_error"):
        st.error(
            "⚠️ **Gemini API credits depleted.**\n\n"
            "**Quick fix (free, no credit card):**\n"
            "1. Go to https://aistudio.google.com/app/apikey\n"
            "2. Click **Create API key** → copy the new key\n"
            "3. Open `.env` in the project folder and replace `GEMINI_API_KEY=...` with the new key\n"
            "4. Restart the app\n\n"
            "The AI Studio free tier gives 1,500 requests/day on Gemini 2.0 Flash — plenty for a session.",
            icon="🚫"
        )

    # ── Safety alert banner ──
    if st.session_state.safety_alert:
        st.markdown("""
        <div class="safety-alert">
            🚨 SAFETY ALERT — If you are in crisis, call or text <b>988</b> (Suicide &amp; Crisis Lifeline) right now.
        </div>
        """, unsafe_allow_html=True)

    # ── Two-column layout ──
    col_chat, col_monitor = st.columns([3, 2])

    # ════════════════════════════════════════════
    # LEFT: Chat Panel
    # ════════════════════════════════════════════
    with col_chat:
        st.subheader("Conversation")

        # Chat display area
        chat_container = st.container(height=430)
        with chat_container:
            if not st.session_state.chat_history:
                st.info("Start the conversation by typing or recording your message below.")

            for msg in st.session_state.chat_history:
                role    = msg["role"]
                content = msg["content"]
                note    = msg.get("note", "")
                if role == "assistant":
                    with st.chat_message("assistant", avatar="🧠"):
                        st.write(content)
                        if note:
                            st.markdown(f"<span class='translation-note'>{note}</span>", unsafe_allow_html=True)
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.write(content)
                        if note:
                            st.markdown(f"<span class='translation-note'>{note}</span>", unsafe_allow_html=True)

        # ── Input area ──
        session_done = (
            st.session_state.dialogue_state.get("session_complete")
            or st.session_state.safety_alert
        )

        if session_done:
            if st.session_state.dialogue_state.get("session_complete"):
                st.success("✅ Session complete. See your scores in the right panel.")
            if st.button("🔄 Start New Session", type="primary"):
                reset_session()
                st.rerun()

        else:
            # ── Opening greeting (first load) ──
            if not st.session_state.session_started:
                greetings = [
                    "Hi, I'm glad you're here. How have things been going for you lately?",
                    "Hello! Thanks for coming in. How have you been doing recently?",
                    "Welcome — I appreciate you taking the time. How have things been for you?",
                    "Hi there, good to see you. What's been on your mind lately?",
                ]
                from llm_client import call_llm
                try:
                    greeting_en = call_llm(
                        "You are starting a warm mental health check-in. "
                        "Write ONE short friendly opening sentence (max 18 words) asking how the person has been doing. "
                        "Plain text only, no asterisks or markdown.",
                        temperature=0.7,
                        max_tokens=50,
                    )
                    if not greeting_en or len(greeting_en) < 10 or greeting_en.count(" ") < 3:
                        greeting_en = random.choice(greetings)
                except Exception:
                    greeting_en = random.choice(greetings)

                # Translate greeting to user language
                if user_lang != "en":
                    greeting = translate_from_english(greeting_en, user_lang)
                else:
                    greeting = greeting_en

                st.session_state.chat_history.append({"role": "assistant", "content": greeting})
                st.session_state.session_started = True
                st.rerun()

            # ── Voice input ──
            try:
                import whisper as _wc
                _wc.available_models()
                whisper_available = True
            except Exception:
                whisper_available = False

            if whisper_available:
                with st.expander("🎙️ Voice Input", expanded=True):
                    st.caption("Record your message — Whisper detects your language automatically and the agent replies in the same language.")
                    audio_input = st.audio_input("Record your voice message")

                    if audio_input is not None:
                        audio_bytes = audio_input.read()
                        if audio_bytes and len(audio_bytes) > 1000:
                            with st.spinner("🎙️ Transcribing with Whisper…"):
                                result = transcribe_audio(audio_bytes, user_lang)

                            if result["success"] and result["text"]:
                                transcript    = result["text"]          # original language text
                                eng_text      = result["english_text"]  # Whisper's English translation
                                detected_lang = result["language"]

                                # Always trust Whisper's language detection
                                # (it hears what was actually spoken, regardless of sidebar selection)
                                if detected_lang and detected_lang != "en":
                                    st.session_state.user_lang = detected_lang
                                    user_lang = detected_lang

                                lang_display = get_language_name(detected_lang) if detected_lang else "English"
                                st.success(f"🎤 Heard ({lang_display}): **{transcript}**")
                                if detected_lang != "en" and eng_text and eng_text != transcript:
                                    st.caption(f"🔤 English: _{eng_text}_")

                                # Pre-load IndicTrans2 models while user reads the transcript,
                                # so the Send button response is fast.
                                if is_indic(user_lang) and not are_indic_models_loaded():
                                    with st.spinner(
                                        f"⏳ Loading IndicTrans2 for {lang_display} "
                                        "(one-time, 2-5 min)…"
                                    ):
                                        try:
                                            load_all_indic_models()
                                        except Exception as _le:
                                            st.warning(f"Model load failed: {_le}")

                                # Send button — passes Whisper's English directly so
                                # translate_to_english is bypassed (already done by Whisper)
                                if st.button("✅ Send this message", key="send_voice"):
                                    process_user_turn(
                                        transcript,
                                        english_override=eng_text if detected_lang != "en" else ""
                                    )
                                    st.rerun()
                            else:
                                st.error(f"Transcription failed: {result.get('error', 'Unknown error')}")

            # ── Text input ──
            placeholder = {
                "en": "Type your message here...",
                "hi": "यहाँ लिखें / Type here in Hindi...",
                "ur": "یہاں لکھیں / Type here in Urdu...",
                "bn": "এখানে লিখুন...",
            }.get(user_lang, "Type your message here...")

            user_input = st.chat_input(placeholder)
            if user_input and user_input.strip():
                process_user_turn(user_input.strip())
                st.rerun()

    # ════════════════════════════════════════════
    # RIGHT: Monitor Panel
    # ════════════════════════════════════════════
    with col_monitor:
        st.subheader("Live Session Monitor")

        state          = st.session_state.dialogue_state
        confidence     = state.get("confidence", {d: 0.0 for d in DOMAINS})
        scores         = state.get("scores", {})
        current_domain = state.get("current_domain", "")
        phase          = state.get("phase", "rapport")
        turn_count     = state.get("turn_count", 0)
        session_complete = state.get("session_complete", False)

        # ── Phase / progress ──
        if phase == "rapport":
            rapport_done = min(turn_count, RAPPORT_TURNS)
            phase_label  = f"RAPPORT ({rapport_done}/{RAPPORT_TURNS} turns)"
        else:
            phase_label = "SCREENING"

        # Language badge
        lang_str = "" if user_lang == "en" else f" 🌐 {get_language_name(user_lang)}"
        st.markdown(f"**Phase:** `{phase_label}` | **Turn:** `{turn_count}`{lang_str}")

        if phase == "rapport" and turn_count < RAPPORT_TURNS:
            st.progress(turn_count / RAPPORT_TURNS,
                        text=f"Building rapport ({turn_count}/{RAPPORT_TURNS})")
        elif current_domain and not session_complete:
            domain_display_map = {
                "sleep": "Sleep", "mood": "Mood / Anhedonia",
                "energy": "Energy", "appetite": "Appetite",
                "concentration": "Concentration",
            }
            st.markdown(f"""
            <div class="current-domain-box">
                Currently screening: <b>{domain_display_map.get(current_domain, current_domain.replace('_',' ').title())}</b>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Domain confidence bars ──
        st.markdown("**Domain Confidence**")
        domain_display = {
            "sleep":         "Sleep",
            "mood":          "Mood / Anhedonia",
            "energy":        "Energy",
            "appetite":      "Appetite",
            "concentration": "Concentration",
        }

        for domain in DOMAINS:
            conf   = confidence.get(domain, 0.0)
            scored = domain in scores

            label_col, bar_col = st.columns([2, 3])
            with label_col:
                if scored:
                    score_val   = scores[domain]
                    score_class = f"score-{score_val}"
                    st.markdown(
                        f"<span class='domain-label'>{domain_display[domain]}</span><br>"
                        f"<span class='score-badge {score_class}'>Score: {score_val}/3</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<span class='domain-label'>{domain_display[domain]}</span>",
                        unsafe_allow_html=True
                    )
            with bar_col:
                st.progress(min(1.0, conf), text=f"{conf:.0%}")

        st.divider()

        # ── Final results (session complete) ──
        if session_complete and scores:
            total = sum(scores.values())

            if total <= 2:
                severity_label, sev_emoji, sev_color, sev_bg = "Minimal", "🟢", "#155724", "#d4edda"
                recommendation = "No significant symptoms detected. Continue healthy habits and check in periodically."
            elif total <= 5:
                severity_label, sev_emoji, sev_color, sev_bg = "Mild", "🟡", "#856404", "#fff3cd"
                recommendation = "Mild symptoms present. Consider speaking with a counselor or GP."
            elif total <= 9:
                severity_label, sev_emoji, sev_color, sev_bg = "Moderate", "🟠", "#7d4e00", "#ffe5b4"
                recommendation = "Moderate symptoms. Speaking with a mental health professional is recommended."
            else:
                severity_label, sev_emoji, sev_color, sev_bg = "Severe", "🔴", "#721c24", "#f8d7da"
                recommendation = "Significant symptoms detected. Please reach out to a mental health professional promptly."

            # Translate recommendation if needed
            if user_lang != "en":
                recommendation = translate_from_english(recommendation, user_lang)

            st.markdown(
                f"<div style='background:{sev_bg};border-radius:8px;padding:12px 16px;margin-bottom:12px;'>"
                f"<div style='font-size:1.3rem;font-weight:bold;color:{sev_color};'>{sev_emoji} {severity_label} — {total}/15</div>"
                f"<div style='font-size:0.85rem;color:{sev_color};margin-top:4px;'>{recommendation}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown("**Domain Breakdown**")
            score_labels      = ["None", "Mild", "Moderate", "Severe"]
            score_colors      = ["#d4edda", "#fff3cd", "#ffe5b4", "#f8d7da"]
            score_text_colors = ["#155724", "#856404", "#7d4e00", "#721c24"]

            for d in DOMAINS:
                s = scores.get(d, 0)
                col_a, col_b = st.columns([3, 2])
                with col_a:
                    st.markdown(
                        f"<span style='font-size:0.9rem;'>{domain_display[d]}</span>",
                        unsafe_allow_html=True
                    )
                with col_b:
                    st.markdown(
                        f"<span style='background:{score_colors[s]};color:{score_text_colors[s]};"
                        f"padding:2px 10px;border-radius:10px;font-size:0.85rem;font-weight:bold;'>"
                        f"{score_labels[s]} ({s}/3)</span>",
                        unsafe_allow_html=True
                    )

            st.divider()
            st.caption(
                "⚠️ This is a screening tool only, covering 5 of 9 PHQ-9 domains. "
                "Scores are not a clinical diagnosis. Please consult a qualified mental health professional."
            )
            if total >= 6:
                st.info("🆘 If you are in crisis, call or text **988** (Suicide & Crisis Lifeline) anytime.")

        elif not session_complete:
            domains_scored = len(scores)
            st.markdown(f"**Progress:** {domains_scored}/{len(DOMAINS)} domains assessed")
            st.progress(domains_scored / len(DOMAINS))
            if domains_scored < len(DOMAINS):
                remaining = [d for d in DOMAINS if d not in scores]
                st.caption(f"Remaining: {', '.join(domain_display[d] for d in remaining)}")


if __name__ == "__main__":
    main()
