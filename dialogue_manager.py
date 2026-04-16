"""
LangGraph Dialogue Manager
Manages adaptive mental health screening conversation flow.
"""

import json
import time
import random
from typing import TypedDict, Optional
from pathlib import Path

import ollama
from langgraph.graph import StateGraph, END

from inference_engine import score_domain, estimate_confidence_increment
from safety_monitor import check_safety

# Load DSM-5 KB
with open("dsm_kb.json") as f:
    DSM_KB = json.load(f)

DOMAINS = ["sleep", "mood", "energy", "concentration", "self_worth"]
CONFIDENCE_THRESHOLD = 0.75
MIN_PROBE_TURNS = 2  # Minimum turns per domain before scoring


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


# ─── Ollama Helper ─────────────────────────────────────────────────────────────

def call_ollama(prompt: str, system: str = "", temperature: float = 0.7) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = ollama.chat(
            model="llama3",
            messages=messages,
            options={"temperature": temperature, "num_predict": 300}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[DialogueManager WARN] Ollama failed: {e}")
        return "I appreciate you sharing that. Could you tell me a bit more?"


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
    """First 3 turns: casual open-ended conversation + keyword scanning."""
    t0 = time.time()

    messages = state["messages"]
    turn_count = state["turn_count"]

    # Generate rapport question
    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-6:])
    system = "You are a warm, empathetic mental health counselor starting a conversation. Be casual, caring, and open-ended."
    prompt = f"""This is turn {turn_count + 1} of a mental health screening session (rapport phase, first 3 turns).
Previous exchange:
{conv_text}

Generate ONE open-ended, warm question to build rapport and learn about the patient's recent life.
Keep it conversational and natural. Do not ask about symptoms directly.
Return ONLY the question, no other text."""

    response = call_ollama(prompt, system)
    if not response:
        response = "Thanks for being here today. How have things been going for you lately — what's life been like?"

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

    if turn_count >= 2:  # After turn 3, move to screening
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
        if patient_msgs:
            last_patient = patient_msgs[-1]["content"]
            increment = estimate_confidence_increment(domain, last_patient, kb_entry)
            new_confidence[domain] = min(1.0, new_confidence[domain] + increment)

        conf = new_confidence[domain]

        # Decide: probe or score
        should_score = (conf >= CONFIDENCE_THRESHOLD and current_turns >= MIN_PROBE_TURNS)

        if should_score and domain not in scores:
            # Score the domain via CoT
            cot_result = score_domain(domain, messages, kb_entry)
            scores[domain] = cot_result["score"]
            cot_chains[domain] = cot_result

            # Generate a transitional response
            system = "You are a warm mental health counselor. Acknowledge what the patient said and smoothly transition."
            prompt = f"""The patient just discussed their {domain} experiences.
Generate a brief empathetic acknowledgment (1-2 sentences) and naturally transition to the next topic.
Return ONLY the response text."""
            response = call_ollama(prompt, system)
            if not response:
                response = f"Thank you for sharing that with me. I'd like to understand a bit more about how you've been feeling overall."

            domain_turn_counts[domain] = current_turns + 1

        else:
            # Generate a probe question
            probe_questions = kb_entry.get("probe_questions", [])
            # Pick probe question based on turn count to avoid repetition
            probe_idx = current_turns % len(probe_questions)

            system = "You are a warm, empathetic mental health counselor. Ask natural, conversational follow-up questions."
            conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-6:])
            prompt = f"""You are screening for: {kb_entry['item_name']} — {kb_entry['description']}

Recent conversation:
{conv_text}

Suggested probe: "{probe_questions[probe_idx]}"

Generate a natural, empathetic question to explore the patient's {domain} symptoms further.
Build on what they've already said. Don't repeat earlier questions.
Keep it conversational. Return ONLY the question."""

            response = call_ollama(prompt, system)
            if not response:
                response = probe_questions[probe_idx]

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
            closing = call_ollama(prompt, system)
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

        response = call_ollama(prompt, system, temperature=0.3)
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

    if phase == "rapport" and turn_count < 3:
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
    Process a single user turn interactively.
    Returns (updated_state, bot_response).
    """
    if graph is None:
        graph = build_dialogue_graph()

    # Add user message
    state = {**state, "messages": state["messages"] + [{"role": "patient", "content": user_message}]}

    try:
        result = graph.invoke(state)
        return result, result.get("last_response", "I appreciate you sharing that.")
    except Exception as e:
        print(f"[DialogueManager ERROR] {e}")
        return state, "I appreciate you sharing that. Could you tell me more?"


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
