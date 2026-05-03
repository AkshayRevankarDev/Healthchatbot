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

# ── Master system prompt used across all nodes ────────────────────────────────
# One unified persona so every response sounds like the same person.
COUNSELOR_SYSTEM = """You are a warm, empathetic mental health counselor conducting a gentle check-in conversation.

Your personality:
- Calm, caring, and non-judgmental — like a trusted friend who happens to know a lot about mental health
- You NEVER sound like a questionnaire or a robot
- You ALWAYS acknowledge what the person just said before asking anything new
- You ask ONE question at a time — never two
- Short responses: 1-3 sentences max
- You mirror the person's energy: if they're brief, be brief; if they open up, gently follow

Hard rules:
- NEVER say "Thank you for sharing" or "I appreciate you opening up" — these are robotic
- NEVER repeat a question you already asked
- NEVER use clinical terms like "PHQ-9", "screening", "domain", "diagnose"
- NEVER say "I understand" as your first word — it feels hollow
- DO use natural acknowledgments: "That's really tough", "Yeah, that makes sense", "Oh wow", "That sounds exhausting"
- If someone is being casual or even a bit rude, stay warm — don't get formal"""

# Varied fallbacks that sound human (used only if LLM fails)
_RAPPORT_FALLBACKS = [
    "That sounds really tough. What's been hitting you hardest lately?",
    "Yeah, that makes sense. How long have things been feeling this way?",
    "I can hear that. What's been going on day-to-day?",
    "Oh wow, that's a lot. What does a typical day look like for you right now?",
    "That's heavy. What's been weighing on you the most?",
]

_TRANSITION_FALLBACKS = [
    "Got it. I want to make sure I'm getting the full picture — can I ask about something else?",
    "That helps me understand. Let me ask you about something different.",
    "Okay, I hear you. I want to check in on a few other things too.",
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

    # Generate rapport response — genuinely conversational, not scripted
    conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-8:])
    prompt = f"""The conversation so far:
{conv_text}

It's turn {turn_count + 1}. Write a single natural response (1-2 sentences).

Step 1 — ACKNOWLEDGE: Respond directly to what they just said. Reference specific words they used.
Step 2 — INVITE: Ask ONE open, gentle follow-up about how they're doing or what's been going on.

Examples of what good looks like:
- "Yeah, not sleeping well sounds exhausting. What else has been going on for you lately?"
- "That's a lot on your plate. How long has it been feeling this heavy?"
- "Ugh, that sounds really rough. What does a normal day look like for you right now?"

Return ONLY the response text."""

    response = call_llm(prompt, COUNSELOR_SYSTEM, max_tokens=200)
    llm_failed = not response
    if not response:
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

        # --- Score or probe ---
        did_score = False
        if should_score and domain not in scores:
            cot_result = score_domain(domain, messages, kb_entry)

            # If LLM failed (empty evidence + empty reasoning = default 0),
            # do NOT commit a fake score — fall through to probe instead.
            llm_returned_nothing = (
                not cot_result.get("evidence")
                and not cot_result.get("reasoning", "").strip()
            )
            if not llm_returned_nothing:
                scores[domain] = cot_result["score"]
                cot_chains[domain] = cot_result
                did_score = True

        if did_score:
            # Generate a natural transition to the next topic
            conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-4:])
            prompt = f"""Conversation so far:
{conv_text}

Write 1-2 sentences. First acknowledge exactly what they just said about their {domain} — use their own words.
Then NATURALLY shift to something else WITHOUT saying "let's move on" or "I'd like to ask about".
Just pivot conversationally, like a real person would.

Good examples:
- "Yeah, not sleeping well sounds exhausting. How's your energy been holding up through all this?"
- "That makes sense given everything. How about your mood overall — been up and down too?"

Return ONLY the response text."""
            response = call_llm(prompt, COUNSELOR_SYSTEM, max_tokens=200)
            llm_failed = not response
            if not response:
                idx = len(scores) % len(_TRANSITION_FALLBACKS)
                response = _TRANSITION_FALLBACKS[idx]

            domain_turn_counts[domain] = current_turns + 1

        else:
            # Probe — respond to what they said, dig one level deeper
            probe_questions = kb_entry.get("probe_questions", [])
            conv_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages[-8:])

            bot_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
            already_said = bot_msgs[-(current_turns + 2):] if current_turns > 0 else bot_msgs[-2:]
            already_said_text = "\n".join(f"- {s[:100]}" for s in already_said) if already_said else "- nothing yet"

            patient_msgs_list = [m["content"] for m in messages if m["role"] == "patient"]
            patient_context = "\n".join(f"- {p[:120]}" for p in patient_msgs_list[-3:]) if patient_msgs_list else ""

            prompt = f"""Full conversation:
{conv_text}

You're gently exploring {kb_entry['item_name']} ({kb_entry['description']}).
The person has told you: {patient_context}

What YOU already said (do NOT repeat these):
{already_said_text}

Write ONE sentence:
1. Short natural reaction to their last message ("Got it", "That makes sense", "Oh wow", "Yeah, that sounds rough", "Ugh")
2. ONE new follow-up question about {domain} not yet asked

Under 25 words. Human, not clinical. Return ONLY the sentence."""

            response = call_llm(prompt, COUNSELOR_SYSTEM, max_tokens=120)
            llm_failed = not response
            if not response:
                used = set(already_said)
                fresh = [q for q in probe_questions if q not in used]
                response = fresh[0] if fresh else probe_questions[0]

            domain_turn_counts[domain] = current_turns + 1

        latency = time.time() - t0

        # Update confidence history
        new_conf_history = {d: list(v) for d, v in state["confidence_history"].items()}
        for d in DOMAINS:
            new_conf_history[d].append(new_confidence[d])

        # Check session complete
        session_complete = len(scores) == len(DOMAINS)

        if session_complete:
            # Generate closing — warm, specific, not generic
            total = sum(scores.values())
            severity = "a lot" if total >= 8 else ("quite a bit" if total >= 4 else "some things")
            conv_summary = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in messages[-6:]
                if m["role"] == "patient"
            )
            prompt = f"""The check-in is complete. The person has shared {severity} with you.
Their last few messages: {conv_summary}

Write 2-3 warm, closing sentences that:
- Feel like a real conversation ending, not a formal sign-off
- Acknowledge the specific things they struggled with (sleep, mood, energy etc.) without using those exact clinical words
- Are honest and caring without being over-the-top

Do NOT say "It takes courage", do NOT say "Thank you for sharing".
Return ONLY the closing statement."""
            closing = call_llm(prompt, COUNSELOR_SYSTEM, max_tokens=300)
            if not closing:
                closing = "I'm really glad you talked to me today — I can hear how much you've been carrying. Let's figure out the best way to support you from here."
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
        prompt = f"""Someone just said: "{last_patient}"

This is serious. Write a response that:
- Opens with genuine shock/concern, not a formal preamble ("Hey, I need to stop and check on you" / "Wait — what you just said worries me")
- Tells them directly that you care and they're not alone
- Gives the 988 Suicide & Crisis Lifeline (call or text 988) in a natural way, not as a bullet point
- Ends with "Are you safe right now?"

2-4 sentences. Warm. Direct. Human. Return ONLY the response."""

        response = call_llm(prompt, COUNSELOR_SYSTEM, temperature=0.3, max_tokens=300)
        if not response:
            response = ("Hey — I need to stop and check on you. What you just said has me genuinely worried, "
                        "and I want you to know you're not alone in this. "
                        "Please reach out to the 988 Suicide & Crisis Lifeline right now — just call or text 988, "
                        "they're there 24/7. Are you safe right now?")

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
