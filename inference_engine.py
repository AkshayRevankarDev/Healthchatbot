"""
Chain-of-Thought Inference Engine
Scores PHQ-9 domains using Gemini with strict 3-step CoT reasoning.
"""

import json
import re

from llm_client import call_llm


def score_domain(domain: str, conversation_history: list, kb_entry: dict) -> dict:
    """
    Score a PHQ-9 domain using 3-step chain-of-thought reasoning.

    Args:
        domain: Domain name (sleep, mood, energy, concentration, self_worth)
        conversation_history: List of {role, content} dicts
        kb_entry: DSM-5-TR knowledge base entry for this domain

    Returns:
        {
            "domain": str,
            "evidence": list[str],
            "reasoning": str,
            "score": int (0-3),
            "justification": str,
            "raw_cot": str
        }
    """
    # Format conversation for the prompt
    conv_text = "\n".join(
        f"{turn['role'].upper()}: {turn['content']}"
        for turn in conversation_history
    )

    # Build DSM-5-TR severity reference
    severity_ref = "\n".join([
        f"  Score {i} ({kb_entry[f'score_{i}']['label']}): {', '.join(kb_entry[f'score_{i}']['indicators'][:3])}"
        for i in range(4)
    ])

    system_prompt = """You are a clinical psychologist scoring PHQ-9 items from patient conversations.
You extract evidence from the full conversation and score it accurately against DSM-5-TR severity criteria.
Always return valid JSON."""

    prompt = f"""Score the PHQ-9 domain "{domain.upper()}" ({kb_entry['item_name']}) from this patient conversation.

DSM-5-TR Definition: {kb_entry['description']}

Severity Indicators:
{severity_ref}

SCORING GUIDE:
  Score 0 — Patient says there is NO problem, or gives zero relevant information about this domain.
             e.g. "I sleep fine", "appetite is normal", "I feel okay", or topic never mentioned.
  Score 1 — Problem exists but is mild/occasional (several days in the past 2 weeks).
             e.g. "sometimes", "a few times", "a bit", "here and there", "occasionally"
  Score 2 — Problem is frequent (more than half the days).
             e.g. "most days", "usually", "often", "hard most of the time", "frequently"
  Score 3 — Problem is constant or nearly every day.
             e.g. "every day", "all the time", "I can't at all", "never", "0", "completely gone",
             "all the ime", "absolutely 0", "not anymore at all", "existing is hard"

INFORMAL LANGUAGE RULES (patients rarely use clinical wording):
- "0 sleep / 0 energy / 0 appetite" → Score 3 (means none at all)
- "all the time" / "every time" → Score 3
- "not anymore" with strong finality → Score 2-3
- "existing is hard" / "can't imagine looking forward to anything" → Score 3 for mood/anhedonia
- "can't concentrate at all" / "simply not able to" → Score 3 for concentration
- Answering "yeah" / "yes" to a direct probe about severity confirms that severity
- "sometimes" alone without severity context → Score 1

IMPORTANT: Score 0 only if the patient clearly reports NO problem. Do NOT score 0 just because
the answer is short. If the patient confirms a problem exists, score at least 1.

Conversation:
{conv_text}

Follow these EXACT 3 steps and return ONLY valid JSON:

Step 1 EXTRACT: Quote ALL patient utterances across the full conversation relevant to {domain}.
                Include indirect evidence (e.g. "I just stare at a blank screen" → concentration).
Step 2 REASON: For each quote, map it to a severity level using the guide above.
               Consider the pattern across all quotes, not just the last one.
Step 3 SCORE: Assign the single score (0-3) that best fits the overall pattern.

Return this exact JSON structure:
{{
  "evidence": ["verbatim patient quote 1", "verbatim patient quote 2"],
  "reasoning": "Step-by-step mapping of each quote to severity level, then overall pattern",
  "score": 0,
  "justification": "One sentence citing specific frequency evidence for the chosen score"
}}

Return ONLY the JSON object, no other text."""

    raw_cot = call_llm(prompt, system_prompt, temperature=0.1, max_tokens=2048, thinking=True)

    # Parse JSON from response
    result = _parse_cot_response(raw_cot, domain)
    result["raw_cot"] = raw_cot
    return result


def _parse_cot_response(raw: str, domain: str) -> dict:
    """Parse and validate the CoT JSON response."""
    default = {
        "domain": domain,
        "evidence": [],
        "reasoning": "Unable to parse reasoning.",
        "score": 0,
        "justification": "Parsing failed; defaulting to score 0."
    }

    if not raw:
        return default

    try:
        # Try direct JSON parse first
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            obj = json.loads(raw[start:end])
            score = obj.get("score", 0)
            # Validate score
            if not isinstance(score, int) or score not in [0, 1, 2, 3]:
                # Try to coerce
                try:
                    score = int(str(score).strip())
                    score = max(0, min(3, score))
                except Exception:
                    score = 0
            return {
                "domain": domain,
                "evidence": obj.get("evidence", []) if isinstance(obj.get("evidence"), list) else [],
                "reasoning": str(obj.get("reasoning", "")),
                "score": score,
                "justification": str(obj.get("justification", ""))
            }
    except json.JSONDecodeError:
        pass

    # Fallback: try regex extraction
    score_match = re.search(r'"score"\s*:\s*([0-3])', raw)
    just_match = re.search(r'"justification"\s*:\s*"([^"]+)"', raw)
    evidence_match = re.findall(r'"([^"]{20,})"', raw)

    score = int(score_match.group(1)) if score_match else 0
    justification = just_match.group(1) if just_match else "Could not extract justification."

    return {
        "domain": domain,
        "evidence": evidence_match[:3] if evidence_match else [],
        "reasoning": raw[:500] if raw else "No reasoning extracted.",
        "score": score,
        "justification": justification
    }


def estimate_confidence_increment(domain: str, patient_response: str, kb_entry: dict) -> float:
    """
    Estimate confidence increment from keyword matching alone.
    Fast, free, and accurate enough for routing decisions.
    Returns 0.05–0.40.
    """
    keywords = kb_entry.get("keywords", [])
    text_lower = patient_response.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)

    if hits == 0:
        return 0.05
    elif hits == 1:
        return 0.15
    elif hits == 2:
        return 0.25
    elif hits == 3:
        return 0.32
    else:
        return 0.40


def compute_ragas_faithfulness(cot_result: dict) -> float:
    """
    Compute a simple RAGAS faithfulness score.
    Fraction of reasoning sentences that reference or contain words from the evidence.
    """
    evidence = cot_result.get("evidence", [])
    reasoning = cot_result.get("reasoning", "")

    if not reasoning or not evidence:
        return 0.0

    # Extract key words from evidence
    evidence_words = set()
    for quote in evidence:
        words = [w.lower().strip('.,!?";') for w in quote.split() if len(w) > 4]
        evidence_words.update(words)

    if not evidence_words:
        return 0.0

    # Split reasoning into sentences
    sentences = [s.strip() for s in re.split(r'[.!?]+', reasoning) if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    supported = 0
    for sentence in sentences:
        sentence_words = set(w.lower().strip('.,!?";') for w in sentence.split())
        if sentence_words & evidence_words:  # intersection
            supported += 1

    return round(supported / len(sentences), 4)


if __name__ == "__main__":
    # Quick test
    import json
    with open("dsm_kb.json") as f:
        kb = json.load(f)

    test_conv = [
        {"role": "therapist", "content": "How has your sleep been lately?"},
        {"role": "patient", "content": "Honestly not great. I lie awake most nights just staring at the ceiling. By the time I fall asleep it's like 3am."},
        {"role": "therapist", "content": "That sounds exhausting. Does this happen often?"},
        {"role": "patient", "content": "Yeah, almost every night for the past few weeks. I'm completely wiped out in the morning."},
    ]

    print("Testing CoT inference for sleep domain...")
    result = score_domain("sleep", test_conv, kb["sleep"])
    print(json.dumps({k: v for k, v in result.items() if k != "raw_cot"}, indent=2))
    faith = compute_ragas_faithfulness(result)
    print(f"\nRAGAS Faithfulness: {faith}")
