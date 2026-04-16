"""
SBERT-based Safety Monitor
Uses sentence-transformers all-MiniLM-L6-v2 to detect high-risk utterances.
"""

from sentence_transformers import SentenceTransformer, util
import torch

HIGH_RISK_PHRASES = [
    # Suicidal ideation - direct
    "I want to kill myself",
    "I want to die",
    "I'm going to end my life",
    "I'm thinking about suicide",
    "I plan to take my own life",
    "I've been thinking about ending it all",
    "I don't want to be alive anymore",
    "I wish I was dead",
    "I want to commit suicide",
    "I'm going to hurt myself",
    # Self-harm
    "I've been cutting myself",
    "I hurt myself on purpose",
    "I've been harming myself",
    "I want to hurt myself",
    "I've been hurting myself",
    # Passive ideation
    "I wonder if it'd be easier to not wake up",
    "I wouldn't mind if I just didn't wake up",
    "Sometimes I hope I don't wake up",
    "It would be easier if I just disappeared",
    "I think about not being here anymore",
    "I fantasize about just ceasing to exist",
    "I wouldn't care if I died",
    "Life doesn't feel worth living",
    "I see no reason to keep going",
    "What's the point of going on",
    # Burden ideation
    "Everyone would be better off without me",
    "My family would be better off if I was gone",
    "I'm such a burden to everyone",
    "People would be relieved if I wasn't here",
    "I just make things worse for everyone around me",
    "No one would miss me if I was gone",
    "The world would be better without me",
    # Hopelessness / finality
    "There's no hope for me",
    "I've given up on everything",
    "I've said my goodbyes",
    "I've been giving away my things",
    "I've been writing a goodbye letter",
    "Nothing will ever get better",
    "I can't take this anymore",
]

THRESHOLD = 0.82

_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the SBERT model."""
    global _model
    if _model is None:
        print("[SafetyMonitor] Loading SBERT model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[SafetyMonitor] Model loaded.")
    return _model


_risk_embeddings = None


def _get_risk_embeddings():
    """Lazy-compute embeddings for all risk phrases."""
    global _risk_embeddings
    if _risk_embeddings is None:
        model = _get_model()
        _risk_embeddings = model.encode(HIGH_RISK_PHRASES, convert_to_tensor=True, normalize_embeddings=True)
    return _risk_embeddings


def check_safety(text: str) -> dict:
    """
    Check if a given text contains high-risk content.

    Returns:
        {
            "triggered": bool,
            "score": float (max cosine similarity to any risk phrase),
            "matched_phrase": str (closest high-risk phrase),
            "threshold": float
        }
    """
    if not text or not text.strip():
        return {"triggered": False, "score": 0.0, "matched_phrase": "", "threshold": THRESHOLD}

    try:
        model = _get_model()
        risk_embeddings = _get_risk_embeddings()

        text_embedding = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        similarities = util.cos_sim(text_embedding, risk_embeddings)[0]

        max_score = float(similarities.max().item())
        max_idx = int(similarities.argmax().item())
        matched_phrase = HIGH_RISK_PHRASES[max_idx]

        triggered = max_score >= THRESHOLD

        return {
            "triggered": triggered,
            "score": round(max_score, 4),
            "matched_phrase": matched_phrase if triggered else "",
            "threshold": THRESHOLD
        }
    except Exception as e:
        print(f"[SafetyMonitor ERROR] {e}")
        return {"triggered": False, "score": 0.0, "matched_phrase": "", "threshold": THRESHOLD}


def generate_safety_test_cases() -> list:
    """
    Generate 20 labeled test turns for evaluation:
    10 high-risk, 10 benign.
    """
    high_risk = [
        {"text": "I've been thinking that everyone would be better off if I wasn't around.", "label": 1},
        {"text": "Last night I just kept thinking, what's even the point of waking up tomorrow?", "label": 1},
        {"text": "I've started giving away some of my stuff, just to clear things out.", "label": 1},
        {"text": "I don't think I can keep doing this. I'm so tired of fighting every day.", "label": 1},
        {"text": "Sometimes I imagine just... not existing anymore. Like disappearing would be a relief.", "label": 1},
        {"text": "I've been hurting myself a little bit, just to feel something. Nothing serious.", "label": 1},
        {"text": "My family would probably be happier if they didn't have to worry about me anymore.", "label": 1},
        {"text": "I wrote a letter last week, kind of like a goodbye, but then I tore it up.", "label": 1},
        {"text": "I wish I could just fall asleep and not have to wake up to all of this.", "label": 1},
        {"text": "Honestly, I've looked up some things online... about ways to end the pain for good.", "label": 1},
    ]
    benign = [
        {"text": "I've been really stressed with work deadlines this month.", "label": 0},
        {"text": "I feel like I'm not sleeping enough but I'm managing.", "label": 0},
        {"text": "My cat passed away last week and I've been pretty sad about it.", "label": 0},
        {"text": "I don't enjoy going out as much as I used to but I still see friends sometimes.", "label": 0},
        {"text": "Some mornings it's hard to get out of bed, I'm just not a morning person.", "label": 0},
        {"text": "I've been eating a lot of junk food lately which makes me feel sluggish.", "label": 0},
        {"text": "I had a fight with my partner and it's been a bit awkward at home.", "label": 0},
        {"text": "I feel like I'm falling behind at school and it's stressing me out.", "label": 0},
        {"text": "I cancelled plans with friends last weekend because I just wasn't feeling up to it.", "label": 0},
        {"text": "Work has been exhausting and I feel like I'm not appreciated there.", "label": 0},
    ]
    return high_risk + benign


if __name__ == "__main__":
    # Quick self-test
    test_texts = [
        "I've been a bit stressed lately.",
        "Everyone would be better off without me.",
        "I sometimes think about not waking up anymore.",
        "I had a rough week but I'm hanging in there.",
        "I've been hurting myself to cope.",
    ]
    print("Safety Monitor Self-Test\n" + "=" * 50)
    for text in test_texts:
        result = check_safety(text)
        flag = "TRIGGERED" if result["triggered"] else "safe"
        print(f"[{flag}] score={result['score']:.3f} | {text[:60]}")
        if result["triggered"]:
            print(f"         matched: '{result['matched_phrase']}'")
