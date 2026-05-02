"""
LangGraph Dialogue Manager
Manages adaptive mental health screening conversation flow.
"""

import json
import time
import random
from typing import TypedDict, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END
from llm_client import call_llm

from inference_engine import score_domain, estimate_confidence_increment
from safety_monitor import check_safety

# Load DSM-5 KB
with open("dsm_kb.json") as f:
    DSM_KB = json.load(f)

DOMAINS = ["sleep", "mood", "energy", "appetite", "concentration"]
CONFIDENCE_THRESHOLD = 0.75
MIN_PROBE_TURNS = 1   # Minimum turns per domain before scoring
MAX_DOMAIN_TURNS = 3  # Force score after this many turns regardless of confidence
RAPPORT_TURNS = 4     # 4 open-ended turns as per M1 presentation

# Varied rapport fallbacks so repeated failures aren't robotic
_RAPPORT_FALLBACKS = [
    "I hear you — that sounds like a lot to carry. What's been the hardest part lately?",
    "Thank you for sharing that. How long have things been feeling this way?",
    "That must be really tough. Can you tell me a bit more about what's been going on?",
    "I appreciate you opening up. How has this been affecting your day-to-day life?",
    "It takes courage to talk about this. What feels most overwhelming right now?",
]


# ─── State Schema ──────────────────────────────────────────────────────────────

class DialogueState(TypedDict):
    messages: list                  # Full conversation {role, content}
    confidence: dict                # {domain: float 0.0-1.0}
    scores: dict                    # {domain: int 0-3} — set when domain scored
    cot_chains: dict                # {domain: cot_result dict}
    domain_turn_counts: dict        # {domain: int} turns spent on this domain
    current_domain: str             # Currently active domain
    turn_count: int                 # Total turn count
    phase: str                      # "rapport" | "screening"
    safety_triggered: bool
    safety_info: dict
    session_complete: bool
    domain_order: list              # Order to screen domains
    confidence_history: dict        # {domain: [float per turn]}
    response_latencies: list        # [float seconds per bot turn]
    last_response: str              # Most recent bot message
    llm_failed: bool                # True if LLM call failed (quota/network)


# ─── Keyword Extractor for Rapport Phase ──────────────────────────────────────

def extract_domain_signals(text: str) -> dict:
    """Return confidence boosts per domain based on keyword presence."""
    text_lower = text.lower()
    boosts = {}
    for domain, entry in DSM_KB.items():
        keywords = entry.get("keywords", [])
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            boosts[domain] = min(0.15 * hits, 0.3)
    return boosts


# ─── Node: Rapport ─────────────────────────────────────────────────────────────

def rapport_node(state: DialogueState) -> DialogueState:
    """First 4 turns: casual open-ended conversation + keyword scanning."""
    t0 = time.time()

    messages = state["messages"]
    turn_count = state["turn_count"]

    # Generate rapport question
    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-6:])
    system = "You are a warm, empathetic mental health counselor having a genuine conversation. Sound like a real human, not a script."
    prompt = f"""You are in the early part of a mental health check-in (turn {turn_count + 1} of 4 rapport turns).

Recent exchange:
{conv_text}

Write a short, natural response (1-2 sentences) that:
- First briefly acknowledges or responds to what the patient just said
- Then asks ONE gentle open-ended question about their life, feelings, or what's been going on
- Does NOT use clinical language or feel like a questionnaire
- Sounds warm and genuine

Return ONLY your response, no preamble."""

    response = call_llm(prompt, system, max_tokens=300)
    llm_failed = not response
    if not response:
        # Pick a varied fallback based on turn so it doesn't repeat
        response = _RAPPORT_FALLBACKS[turn_count % len(_RAPPORT_FALLBACKS)]

    latency = time.time() - t0

    # Update confidence from any patient messages scanned
    new_confidence = dict(state["confidence"])
    if messages:
        last_patient_msgs = [m for m in messages if m["role"] == "patient"]
        if last_patient_msgs:
            boosts = extract_domain_signals(last_patient_msgs[-1]["content"])
            for domain, boost in boosts.items():
                new_confidence[domain] = min(1.0, new_confidence[domain] + boost)

    # Update confidence history
    new_conf_history = {d: list(v) for d, v in state["confidence_history"].items()}
    for domain in DOMAINS:
        new_conf_history[domain].append(new_confidence[domain])

    # Determine if rapport phase should end
    new_phase = "rapport"
    new_domain_order = list(state["domain_order"])

    if turn_count >= RAPPORT_TURNS - 1:  # After 4 rapport turns, move to screening
        new_phase = "screening"
        # Order domains by pre-activated confidence (highest first)
        remaining = [d for d in DOMAINS if d not in state["scores"]]
        new_domain_order = sorted(remaining, key=lambda d: new_confidence[d], reverse=True)

    new_latencies = list(state["response_latencies"]) + [latency]

    return {
        **state,
        "messages": messages + [{"role": "assistant", "content": response}],
        "confidence": new_confidence,
        "confidence_history": new_conf_history,
        "turn_count": turn_count + 1,
        "phase": new_phase,
        "domain_order": new_domain_order,
        "response_latencies": new_latencies,
        "last_response": response,
        "llm_failed": llm_failed,
    }


# ─── Node: Domain Screener (generic) ──────────────────────────────────────────

def make_domain_node(domain: str):
    """Factory: creates a screening node for a given domain."""

    def domain_node(state: DialogueState) -> DialogueState:
        t0 = time.time()
        messages = state["messages"]
        confidence = state["confidence"]
        scores = dict(state["scores"])
        cot_chains = dict(state["cot_chains"])
        domain_turn_counts = dict(state["domain_turn_counts"])
        turn_count = state["turn_count"]

        current_turns = domain_turn_counts.get(domain, 0)
        kb_entry = DSM_KB[domain]

        # Update confidence from most recent patient response
        new_confidence = dict(confidence)
        patient_msgs = [m for m in messages if m["role"] == "patient"]
        last_patient = ""
        if patient_msgs:
            last_patient = patient_msgs[-1]["content"]
            # Current domain update
            increment = estimate_confidence_increment(domain, last_patient, kb_entry)
            new_confidence[domain] = min(1.0, new_confidence[domain] + increment)
            # Cross-domain update: boost other domains from same response
            cross_boosts = extract_domain_signals(last_patient)
            for d, boost in cross_boosts.items():
                if d != domain:
                    new_confidence[d] = min(1.0, new_confidence[d] + boost * 0.6)

        conf = new_confidence[domain]

        # Re-sort domain_order by updated confidence so most-signaled domains come next
        current_order = list(state["domain_order"]) if state["domain_order"] else [d for d in DOMAINS if d not in scores]
        unscored = [d for d in current_order if d not in scores]
        new_domain_order = sorted(unscored, key=lambda d: new_confidence[d], reverse=True)

        # Decide: probe or score
        should_score = (
            (conf >= CONFIDENCE_THRESHOLD and current_turns >= MIN_PROBE_TURNS)
            or current_turns >= MAX_DOMAIN_TURNS  # force score after ceiling
        )

        if should_score and domain not in scores:
            # Score the domain via CoT
            cot_result = score_domain(domain, messages, kb_entry)
            scores[domain] = cot_result["score"]
            cot_chains[domain] = cot_result

            # Generate a transitional acknowledgment
            system = "You are a warm, empathetic mental health counselor."
            conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-4:])
            prompt = f"""The patient has been discussing their {domain} experiences.

Recent exchange:
{conv_text}

Write 1-2 warm sentences that:
1. Acknowledge or validate what they just shared
2. Naturally signal you're moving on to something else

Do NOT mention a score or diagnosis. Return ONLY the response text."""
            response = call_llm(prompt, system, max_tokens=200)
            llm_failed = not response
            if not response:
                response = "Thank you for sharing that. I'd like to hear a bit more about how things have been for you overall."

            domain_turn_counts[domain] = current_turns + 1

        else:
            # Generate a contextual follow-up — never recycle the same question
            probe_questions = kb_entry.get("probe_questions", [])
            conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-8:])

            # Collect what the bot has already asked in this domain
            bot_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
            already_asked = bot_msgs[-current_turns:] if current_turns > 0 else []

            system = "You are a warm, empathetic mental health counselor having a real conversation."
            prompt = f"""You are gently exploring the patient's {domain} ({kb_entry['item_name']}: {kb_entry['description']}).

Recent conversation:
{conv_text}

Example probe directions (do NOT copy verbatim):
{chr(10).join(f'- {q}' for q in probe_questions)}

Questions you have ALREADY asked (do not repeat these):
{chr(10).join(f'- {q}' for q in already_asked) if already_asked else '- none yet'}

Write ONE short, natural follow-up question that:
- Directly responds to what the patient just said
- Explores {domain} without repeating anything above
- Sounds like a real human conversation, not a questionnaire
- Is 1 sentence only

Return ONLY the question, no preamble."""

            response = call_llm(prompt, system, max_tokens=150)
            llm_failed = not response
            if not response:
                # Fallback to a fresh probe not yet used
                used = set(already_asked)
                response = next((q for q in probe_questions if q not in used), probe_questions[0])

            domain_turn_counts[domain] = current_turns + 1

        latency = time.time() - t0

        # Update confidence history
        new_conf_history = {d: list(v) for d, v in state["confidence_history"].items()}
        for d in DOMAINS:
            new_conf_history[d].append(new_confidence[d])

        # Check session complete
        session_complete = len(scores) == len(DOMAINS)

        if session_complete:
            # Generate closing message
            total = sum(scores.values())
            system = "You are a compassionate mental health counselor wrapping up a screening session."
            prompt = f"""The screening session is complete. Generate a warm, empathetic closing statement.
Thank the patient for sharing, briefly acknowledge their courage, and let them know next steps will be discussed.
Keep it to 2-3 sentences. Do NOT mention specific scores or diagnoses. Return ONLY the closing statement."""
            closing = call_llm(prompt, system, max_tokens=300)
            if not closing:
                closing = "Thank you so much for sharing all of this with me today. It takes real courage to talk about these things. We'll review what we've discussed and talk about the best path forward for you."
            response = closing

        new_latencies = list(state["response_latencies"]) + [latency]

        return {
            **state,
            "messages": messages + [{"role": "assistant", "content": response}],
            "confidence": new_confidence,
            "confidence_history": new_conf_history,
            "scores": scores,
            "cot_chains": cot_chains,
            "domain_turn_counts": domain_turn_counts,
            "current_domain": domain,
            "turn_count": turn_count + 1,
            "session_complete": session_complete,
            "response_latencies": new_latencies,
            "last_response": response,
            "domain_order": new_domain_order,
            "llm_failed": llm_failed,
        }

    domain_node.__name__ = f"{domain}_node"
    return domain_node


# ─── Node: Safety ──────────────────────────────────────────────────────────────

def safety_node(state: DialogueState) -> DialogueState:
    """Check most recent patient message for safety concerns."""
    messages = state["messages"]
    patient_msgs = [m for m in messages if m["role"] == "patient"]

    if not patient_msgs:
        return state

    last_patient = patient_msgs[-1]["content"]
    safety_result = check_safety(last_patient)

    if safety_result["triggered"]:
        system = "You are a crisis counselor. Respond with immediate compassion and safety resources."
        prompt = f"""A patient has said something concerning: "{last_patient}"

Generate an immediate, compassionate response that:
1. Acknowledges their pain with empathy
2. Expresses genuine concern
3. Provides the crisis line (988 Suicide & Crisis Lifeline)
4. Asks if they are safe right now

Keep it warm and non-clinical. Return ONLY the response."""

        response = call_llm(prompt, system, temperature=0.3, max_tokens=300)
        if not response:
            response = ("I hear you, and I'm really glad you shared that with me. What you're feeling matters deeply. "
                       "If you're in crisis, please reach out to the 988 Suicide & Crisis Lifeline — call or text 988. "
                       "Are you safe right now?")

        return {
            **state,
            "messages": messages + [{"role": "assistant", "content": response}],
            "safety_triggered": True,
            "safety_info": safety_result,
            "last_response": response,
        }

    return {**state, "safety_info": safety_result}


# ─── Router ────────────────────────────────────────────────────────────────────

def route_next(state: DialogueState) -> str:
    """Determine which node to visit next."""
    if state["session_complete"]:
        return END

    if state["safety_triggered"]:
        return END

    phase = state["phase"]
    turn_count = state["turn_count"]

    if phase == "rapport" and turn_count < RAPPORT_TURNS:
        return "rapport"

    # Move to screening
    scores = state["scores"]
    domain_order = state["domain_order"]

    # If domain_order is empty (edge case), build it
    if not domain_order:
        remaining = [d for d in DOMAINS if d not in scores]
        if not remaining:
            return END
        return remaining[0]

    # Find next unscored domain
    for domain in domain_order:
        if domain not in scores:
            return domain

    return END


def route_after_rapport(state: DialogueState) -> str:
    """Route after rapport node."""
    if state["phase"] == "rapport":
        return "rapport"
    # Move to first domain
    domain_order = state["domain_order"]
    if domain_order:
        return domain_order[0]
    return DOMAINS[0]


# ─── Build Graph ───────────────────────────────────────────────────────────────

def build_dialogue_graph():
    """Build and compile the LangGraph StateGraph."""
    graph = StateGraph(DialogueState)

    # Add nodes
    graph.add_node("safety", safety_node)
    graph.add_node("rapport", rapport_node)
    for domain in DOMAINS:
        graph.add_node(domain, make_domain_node(domain))

    # Entry point: always check safety first
    graph.set_entry_point("safety")

    # After safety: route to rapport or domain
    graph.add_conditional_edges("safety", route_next, {
        "rapport": "rapport",
        **{d: d for d in DOMAINS},
        END: END,
    })

    # After rapport: route to next phase/domain
    graph.add_conditional_edges("rapport", route_after_rapport, {
        "rapport": "rapport",
        **{d: d for d in DOMAINS},
    })

    # After each domain: route to next domain or end
    for domain in DOMAINS:
        graph.add_conditional_edges(domain, route_next, {
            **{d: d for d in DOMAINS},
            END: END,
        })

    return graph.compile()


# ─── Session Runner ────────────────────────────────────────────────────────────

def create_initial_state() -> DialogueState:
    """Create a fresh dialogue state."""
    return {
        "messages": [],
        "confidence": {d: 0.0 for d in DOMAINS},
        "scores": {},
        "cot_chains": {},
        "domain_turn_counts": {d: 0 for d in DOMAINS},
        "current_domain": "",
        "turn_count": 0,
        "phase": "rapport",
        "safety_triggered": False,
        "safety_info": {},
        "session_complete": False,
        "domain_order": [],
        "confidence_history": {d: [] for d in DOMAINS},
        "response_latencies": [],
        "last_response": "",
        "llm_failed": False,
    }


def run_session(conversation: list, graph=None) -> dict:
    """
    Run a full pre-generated conversation through the dialogue manager.

    Args:
        conversation: List of {role, content} turns (patient turns only fed in)
        graph: Compiled LangGraph (built if None)

    Returns:
        Final state dict with scores, confidence, cot_chains, etc.
    """
    if graph is None:
        graph = build_dialogue_graph()

    state = create_initial_state()

    # Feed patient turns through the graph one by one
    patient_turns = [t for t in conversation if t["role"] == "patient"]

    for i, patient_turn in enumerate(patient_turns):
        if state["session_complete"] or state["safety_triggered"]:
            break

        # Add patient message to state
        state = {**state, "messages": state["messages"] + [{"role": "patient", "content": patient_turn["content"]}]}

        # Run one step through the graph
        try:
            result = graph.invoke(state)
            state = result
        except Exception as e:
            print(f"[DialogueManager ERROR] Graph step failed: {e}")
            break

    return state


def run_interactive_turn(user_message: str, state: DialogueState, graph=None) -> tuple:
    """
    Process ONE patient turn and return a single bot response.

    Directly dispatches to the correct node rather than calling graph.invoke(),
    which would run the entire graph to completion in one shot (wrong for interactive use).

    Returns (updated_state, bot_response).
    """
    # Add patient message
    state = {**state, "messages": state["messages"] + [{"role": "patient", "content": user_message}]}

    # Safety check always runs first
    try:
        new_state = safety_node(state)
    except Exception as e:
        print(f"[DialogueManager ERROR] safety_node: {e}")
        new_state = state

    if new_state.get("safety_triggered"):
        return new_state, new_state.get("last_response", "")

    state = new_state  # carry safety_info forward

    # Dispatch to exactly ONE node
    try:
        phase        = state["phase"]
        turn_count   = state["turn_count"]
        scores       = state["scores"]
        domain_order = state["domain_order"]

        if phase == "rapport" and turn_count < RAPPORT_TURNS:
            new_state = rapport_node(state)
        else:
            # Find the next unscored domain
            ordered = domain_order if domain_order else [d for d in DOMAINS if d not in scores]
            domain  = next((d for d in ordered if d not in scores), None)

            if domain is None:
                # All domains scored — session complete
                return {**state, "session_complete": True}, state.get("last_response", "Thank you for completing the screening.")

            node_fn   = make_domain_node(domain)
            new_state = node_fn(state)

    except Exception as e:
        print(f"[DialogueManager ERROR] node dispatch: {e}")
        return state, "I appreciate you sharing that. Could you tell me more?"

    return new_state, new_state.get("last_response", "I appreciate you sharing that.")


if __name__ == "__main__":
    print("Testing dialogue manager with a short conversation...")
    graph = build_dialogue_graph()

    test_conv = [
        {"role": "patient", "content": "Hi, I've been feeling really off lately."},
        {"role": "patient", "content": "Yeah, I just don't sleep well. Lying awake most nights."},
        {"role": "patient", "content": "I haven't really been enjoying my hobbies either. Everything feels kind of flat."},
        {"role": "patient", "content": "I feel drained all the time, even after a full night of tossing and turning."},
        {"role": "patient", "content": "I keep forgetting things at work. My boss mentioned it."},
        {"role": "patient", "content": "I feel like I'm letting everyone down. Like I'm not good enough."},
    ]

    state = run_session(test_conv, graph)
    print(f"\nScores: {state['scores']}")
    print(f"Confidence: {state['confidence']}")
    print(f"Session complete: {state['session_complete']}")
