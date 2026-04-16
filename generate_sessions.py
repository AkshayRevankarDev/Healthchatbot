"""
Synthetic Patient Session Generator
Generates 10 realistic patient sessions with gold PHQ-9 scores using Ollama llama3.
"""

import json
import os
import random
import time
from pathlib import Path

import ollama

SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

DOMAINS = ["sleep", "mood", "energy", "concentration", "self_worth"]

# Severity buckets: (total_min, total_max, count)
SEVERITY_PROFILES = [
    {"label": "minimal", "score_range": (0, 1), "count": 3},   # total 0-4
    {"label": "mild",    "score_range": (1, 2), "count": 3},   # total 5-9
    {"label": "moderate","score_range": (2, 3), "count": 2},   # total 10-14
    {"label": "severe",  "score_range": (2, 3), "count": 2},   # total 15-24
]


def call_ollama(prompt: str, system: str = "") -> str:
    """Call Ollama llama3 with error handling and fallback."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = ollama.chat(
            model="llama3",
            messages=messages,
            options={"temperature": 0.8, "num_predict": 1500}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[WARN] Ollama call failed: {e}. Using fallback.")
        return "I've been doing okay, just a bit stressed lately."


def generate_gold_scores(severity_label: str, score_range: tuple) -> dict:
    """Generate realistic PHQ-9 gold scores for a severity level."""
    lo, hi = score_range
    scores = {}
    if severity_label == "minimal":
        for d in DOMAINS:
            scores[d] = random.choice([0, 0, 0, 1])
    elif severity_label == "mild":
        for d in DOMAINS:
            scores[d] = random.choice([0, 1, 1, 2])
    elif severity_label == "moderate":
        for d in DOMAINS:
            scores[d] = random.choice([1, 2, 2, 3])
    else:  # severe
        for d in DOMAINS:
            scores[d] = random.choice([2, 3, 3, 3])
    return scores


def generate_patient_profile(session_id: int, severity: str, gold_scores: dict) -> str:
    """Generate a patient backstory consistent with their PHQ-9 scores."""
    score_desc = ", ".join(f"{d}: {s}/3" for d, s in gold_scores.items())
    prompt = f"""Create a brief patient profile for a mental health screening session.
Severity level: {severity}
PHQ-9 domain scores (0=none, 3=severe): {score_desc}

Write 3-4 sentences describing this patient's life situation, job/student status, and what's been going on lately.
Make it realistic and specific. The patient should NOT say they are depressed — they express it indirectly through daily life.
Do NOT include any diagnosis. Just describe the person and their recent life circumstances."""
    return call_ollama(prompt)


def generate_conversation(patient_profile: str, gold_scores: dict, severity: str) -> list:
    """Generate a realistic 15-20 turn conversation."""
    score_desc = ", ".join(f"{d}: {s}/3" for d, s in gold_scores.items())

    system_prompt = """You are simulating a realistic patient in a mental health screening conversation.
You speak naturally, indirectly, and through daily life details — never saying "I am depressed" or clinical terms.
Express your symptoms through stories, metaphors, and everyday experiences.
Be authentic, sometimes deflecting, sometimes opening up gradually."""

    prompt = f"""Patient profile: {patient_profile}
PHQ-9 scores to express naturally (0=not at all, 1=several days, 2=more than half, 3=nearly every day):
{score_desc}

Generate a realistic 16-turn mental health screening conversation (8 patient turns, 8 therapist turns, alternating).
Start with the therapist greeting. The patient gradually reveals symptoms through natural conversation about daily life.
Never use clinical terms. Express scores through specific, realistic details.

Format as JSON array:
[
  {{"role": "therapist", "content": "..."}},
  {{"role": "patient", "content": "..."}},
  ...
]

Generate exactly 16 turns total (alternating therapist/patient). Return ONLY the JSON array, no other text."""

    raw = call_ollama(prompt, system_prompt)

    # Parse the JSON
    try:
        # Try to extract JSON array from response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            conversation = json.loads(raw[start:end])
            return conversation
    except json.JSONDecodeError:
        pass

    # Fallback: generate a minimal conversation
    print("[WARN] Could not parse conversation JSON, using fallback.")
    return generate_fallback_conversation(gold_scores, severity)


def generate_fallback_conversation(gold_scores: dict, severity: str) -> list:
    """Minimal fallback conversation if Ollama fails."""
    mood_score = gold_scores.get("mood", 1)
    sleep_score = gold_scores.get("sleep", 1)
    energy_score = gold_scores.get("energy", 1)

    sleep_phrases = [
        "I sleep okay usually.",
        "I've been waking up a couple of times a week.",
        "Most nights I toss and turn for a while.",
        "I can barely sleep anymore, I'm up until 3am every night."
    ]
    mood_phrases = [
        "I'm doing alright, keeping busy.",
        "Some days feel a bit flat, not sure why.",
        "I don't really look forward to things the way I used to.",
        "Nothing really feels worth doing anymore."
    ]
    energy_phrases = [
        "My energy's been fine.",
        "Some days I feel a bit drained.",
        "I've been dragging myself through most days.",
        "I can barely get out of bed most mornings."
    ]

    return [
        {"role": "therapist", "content": "Hi, thanks for coming in today. How have things been going for you lately?"},
        {"role": "patient", "content": mood_phrases[min(mood_score, 3)]},
        {"role": "therapist", "content": "I hear you. Can you tell me a bit more about your day-to-day life right now?"},
        {"role": "patient", "content": energy_phrases[min(energy_score, 3)]},
        {"role": "therapist", "content": "That sounds tough. How about your sleep — are you getting enough rest?"},
        {"role": "patient", "content": sleep_phrases[min(sleep_score, 3)]},
        {"role": "therapist", "content": "Thanks for sharing that. How are things going at work or school?"},
        {"role": "patient", "content": "It's been hard to keep up. I keep losing track of things."},
        {"role": "therapist", "content": "How do you feel about yourself these days?"},
        {"role": "patient", "content": "Sometimes I wonder if I'm doing enough, you know? Like I'm letting people down."},
        {"role": "therapist", "content": "That's really understandable. Is there anything specific that's been weighing on you?"},
        {"role": "patient", "content": "Just this general heaviness, I guess. Like everything takes more effort than it should."},
        {"role": "therapist", "content": "Do you still find enjoyment in the things you used to like?"},
        {"role": "patient", "content": "Not as much. I used to love going out with friends but lately I just... don't bother."},
        {"role": "therapist", "content": "I appreciate you opening up. Is there anything else you'd like me to know?"},
        {"role": "patient", "content": "I think that covers it. I just feel like I'm going through the motions most of the time."}
    ]


def main():
    print("=" * 60)
    print("Generating 10 synthetic patient sessions...")
    print("=" * 60)

    sessions = []
    session_id = 1

    for profile in SEVERITY_PROFILES:
        for i in range(profile["count"]):
            print(f"\n[Session {session_id}] Severity: {profile['label']}")

            # Generate gold scores
            gold_scores = generate_gold_scores(profile["label"], profile["score_range"])
            total = sum(gold_scores.values())
            print(f"  Gold scores: {gold_scores} (total={total})")

            # Generate patient profile
            print("  Generating patient profile...")
            patient_profile = generate_patient_profile(session_id, profile["label"], gold_scores)

            # Generate conversation
            print("  Generating conversation...")
            conversation = generate_conversation(patient_profile, gold_scores, profile["label"])

            session = {
                "session_id": f"session_{session_id:02d}",
                "severity": profile["label"],
                "total_phq9": total,
                "gold_scores": gold_scores,
                "patient_profile": patient_profile,
                "conversation": conversation
            }

            # Save individual session
            session_path = SESSIONS_DIR / f"session_{session_id:02d}.json"
            with open(session_path, "w") as f:
                json.dump(session, f, indent=2)
            print(f"  Saved to {session_path}")

            sessions.append(session)
            session_id += 1
            time.sleep(0.5)  # Brief pause between calls

    # Save all sessions index
    with open(SESSIONS_DIR / "sessions_index.json", "w") as f:
        summary = [
            {
                "session_id": s["session_id"],
                "severity": s["severity"],
                "total_phq9": s["total_phq9"],
                "gold_scores": s["gold_scores"]
            }
            for s in sessions
        ]
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Generated {len(sessions)} sessions successfully.")
    print(f"Sessions saved to: {SESSIONS_DIR}/")
    print("=" * 60)

    # Print summary table
    print("\nSession Summary:")
    print(f"{'ID':<12} {'Severity':<12} {'Total':>6}  {'Sleep':>5} {'Mood':>5} {'Energy':>7} {'Conc':>5} {'Worth':>6}")
    print("-" * 65)
    for s in sessions:
        g = s["gold_scores"]
        print(f"{s['session_id']:<12} {s['severity']:<12} {s['total_phq9']:>6}  "
              f"{g['sleep']:>5} {g['mood']:>5} {g['energy']:>7} "
              f"{g['concentration']:>5} {g['self_worth']:>6}")


if __name__ == "__main__":
    main()
