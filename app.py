"""
Streamlit UI for Mental Health Screening Conversational Agent
Two-panel interface: chat on left, live confidence bars + scores on right.
"""

import streamlit as st
import json
import time
from pathlib import Path

# Page config must be first
st.set_page_config(
    page_title="Mental Health Screening Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Lazy imports to avoid blocking startup
@st.cache_resource
def load_graph():
    from dialogue_manager import build_dialogue_graph
    return build_dialogue_graph()

@st.cache_resource
def load_kb():
    with open("dsm_kb.json") as f:
        return json.load(f)

from dialogue_manager import create_initial_state, run_interactive_turn, DOMAINS
from safety_monitor import check_safety

# ─── Session State Init ────────────────────────────────────────────────────────

def init_session():
    if "dialogue_state" not in st.session_state:
        st.session_state.dialogue_state = create_initial_state()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "safety_alert" not in st.session_state:
        st.session_state.safety_alert = False
    if "session_started" not in st.session_state:
        st.session_state.session_started = False


def reset_session():
    st.session_state.dialogue_state = create_initial_state()
    st.session_state.chat_history = []
    st.session_state.safety_alert = False
    st.session_state.session_started = False


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
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 2px;
    }
    .score-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .score-0 { background: #d4edda; color: #155724; }
    .score-1 { background: #fff3cd; color: #856404; }
    .score-2 { background: #ffe5b4; color: #7d4e00; }
    .score-3 { background: #f8d7da; color: #721c24; }
    .current-domain-box {
        background: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .chat-message {
        padding: 8px 12px;
        margin: 6px 0;
        border-radius: 8px;
    }
    .user-msg {
        background: #e3f2fd;
        margin-left: 20%;
        text-align: right;
    }
    .bot-msg {
        background: #f5f5f5;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    init_session()

    st.title("Mental Health Screening Agent")
    st.caption("Adaptive conversational screening powered by Ollama llama3 + LangGraph")

    # Load resources
    kb = load_kb()
    if st.session_state.graph is None:
        with st.spinner("Loading dialogue graph..."):
            st.session_state.graph = load_graph()

    # Safety alert banner
    if st.session_state.safety_alert:
        st.markdown("""
        <div class="safety-alert">
            SAFETY ALERT — Crisis resources detected in message.
            If you are in crisis, call or text 988 (Suicide & Crisis Lifeline).
        </div>
        """, unsafe_allow_html=True)

    # ── Two-column layout ──
    col_chat, col_monitor = st.columns([3, 2])

    # ── LEFT: Chat Panel ──
    with col_chat:
        st.subheader("Conversation")

        # Chat display area
        chat_container = st.container(height=450)
        with chat_container:
            if not st.session_state.chat_history:
                st.info("Start the conversation by typing a message below.")

            for msg in st.session_state.chat_history:
                role = msg["role"]
                content = msg["content"]
                if role == "assistant":
                    with st.chat_message("assistant", avatar="🧠"):
                        st.write(content)
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.write(content)

        # Input area
        if st.session_state.dialogue_state.get("session_complete") or st.session_state.safety_alert:
            if st.session_state.dialogue_state.get("session_complete"):
                st.success("Session complete. See your scores in the right panel.")
            if st.button("Start New Session", type="primary"):
                reset_session()
                st.rerun()
        else:
            # Show welcome message if session hasn't started
            if not st.session_state.session_started:
                with st.spinner("Starting session..."):
                    state = st.session_state.dialogue_state
                    # Generate initial greeting
                    import ollama as _ollama
                    try:
                        resp = _ollama.chat(
                            model="llama3",
                            messages=[{
                                "role": "user",
                                "content": "Generate a warm, brief opening for a mental health check-in session. 2 sentences max. Ask how they've been doing."
                            }],
                            options={"temperature": 0.7, "num_predict": 100}
                        )
                        greeting = resp["message"]["content"].strip()
                    except Exception:
                        greeting = "Hi, welcome. I'm here to chat and check in with you. How have you been doing lately?"

                    st.session_state.chat_history.append({"role": "assistant", "content": greeting})
                    st.session_state.session_started = True
                    st.rerun()

            user_input = st.chat_input("Type your message here...")
            if user_input and user_input.strip():
                # Safety check first
                safety = check_safety(user_input)
                if safety["triggered"]:
                    st.session_state.safety_alert = True

                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})

                # Process through dialogue manager
                with st.spinner("Thinking..."):
                    new_state, bot_response = run_interactive_turn(
                        user_input,
                        st.session_state.dialogue_state,
                        st.session_state.graph
                    )

                st.session_state.dialogue_state = new_state
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

                if new_state.get("safety_triggered"):
                    st.session_state.safety_alert = True

                st.rerun()

    # ── RIGHT: Monitor Panel ──
    with col_monitor:
        st.subheader("Live Session Monitor")

        state = st.session_state.dialogue_state
        confidence = state.get("confidence", {d: 0.0 for d in DOMAINS})
        scores = state.get("scores", {})
        current_domain = state.get("current_domain", "")
        phase = state.get("phase", "rapport")
        turn_count = state.get("turn_count", 0)
        session_complete = state.get("session_complete", False)

        # Session info
        st.markdown(f"**Phase:** `{phase.upper()}` | **Turn:** `{turn_count}`")

        if current_domain:
            st.markdown(f"""
            <div class="current-domain-box">
                Currently screening: <b>{current_domain.replace('_', ' ').title()}</b>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Confidence bars
        st.markdown("**Domain Confidence**")
        domain_display = {
            "sleep": "Sleep",
            "mood": "Mood / Anhedonia",
            "energy": "Energy",
            "concentration": "Concentration",
            "self_worth": "Self-Worth"
        }

        for domain in DOMAINS:
            conf = confidence.get(domain, 0.0)
            scored = domain in scores

            label_col, bar_col = st.columns([2, 3])
            with label_col:
                if scored:
                    score_val = scores[domain]
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
                # Color the bar based on threshold
                if scored:
                    bar_color = "#28a745"
                elif conf >= 0.75:
                    bar_color = "#17a2b8"
                else:
                    bar_color = "#6c757d"
                st.progress(
                    min(1.0, conf),
                    text=f"{conf:.0%}"
                )

        st.divider()

        # Final scores table (shown when complete)
        if session_complete and scores:
            st.markdown("**Final PHQ-9 Scores**")
            total = sum(scores.values())

            severity_labels = {
                (0, 4): ("Minimal", "green"),
                (5, 9): ("Mild", "orange"),
                (10, 14): ("Moderate", "darkorange"),
                (15, 27): ("Severe", "red"),
            }
            severity_label = "Unknown"
            severity_color = "gray"
            for (lo, hi), (label, color) in severity_labels.items():
                if lo <= total <= hi:
                    severity_label = label
                    severity_color = color
                    break

            # Scores table
            score_data = {
                "Domain": [domain_display[d] for d in DOMAINS],
                "Score": [scores.get(d, "-") for d in DOMAINS],
                "Severity": [
                    ["None", "Mild", "Moderate", "Severe"][scores.get(d, 0)] if d in scores else "-"
                    for d in DOMAINS
                ]
            }
            st.table(score_data)

            st.markdown(f"**Total PHQ-9: {total}/15** — "
                       f"<span style='color:{severity_color};font-weight:bold'>{severity_label}</span>",
                       unsafe_allow_html=True)

            st.caption("Note: This is a screening tool only. "
                      "Please consult a qualified mental health professional for diagnosis and treatment.")

        elif not session_complete:
            domains_scored = len(scores)
            domains_remaining = len(DOMAINS) - domains_scored
            st.markdown(f"**Progress:** {domains_scored}/{len(DOMAINS)} domains assessed")
            if domains_remaining > 0:
                remaining = [d for d in DOMAINS if d not in scores]
                st.caption(f"Remaining: {', '.join(domain_display[d] for d in remaining)}")


if __name__ == "__main__":
    main()
