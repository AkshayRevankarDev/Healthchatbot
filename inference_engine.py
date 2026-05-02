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
You follow a strict chain-of-thought protocol. Be precise, conservative, and evidence-based.
Always return valid JSON."""

    prompt = f"""Score the PHQ-9 domain "{domain.upper()}" ({kb_entry['item_name']}) from this patient conversation.

DSM-5-TR Definition: {kb_entry['description']}

Severity Indicators:
{severity_ref}

CALIBRATION (use as reference — be conservative):
  Score 0: Patient clearly has NO issue. e.g. "I sleep fine", "my appetite is normal", "I feel good"
  Score 1: Issue present SEVERAL DAYS only. e.g. "sometimes", "a few times", "occasionally", "a bit"
  Score 2: Issue present MORE THAN HALF THE DAYS. e.g. "most days", "usually", "often", "frequently"
  Score 3: Issue present NEARLY EVERY DAY. e.g. "every night", "constantly", "always", "I can't at all"

RULES:
- When in doubt between two scores, choose the LOWER one
- Short or vague answers (e.g. "yes", "no", "okay") should score 0 or 1, not higher
- Score 0 if the patient explicitly says there is no problem in this domain

Conversation:
{conv_text}

Follow these EXACT 3 steps and return ONLY valid JSON:

Step 1 EXTRACT: Find and quote all patient utterances relevant to {domain}.
Step 2 REASON: Compare each quote to the severity indicators. Apply the calibration rules.
Step 3 SCORE: Assign 0, 1, 2, or 3.

Return this exact JSON structure:
{{
  "evidence": ["verbatim patient quote 1", "verbatim patient quote 2"],
  "reasoning": "Step-by-step comparison to severity indicators",
  "score": 0,
  "justification": "One sentence explaining the score and frequency"
}}

Return ONLY the JSON object, no other text."""

    raw_cot = call_llm(prompt, system_prompt, temperature=0.1, max_tokens=1500, thinking=True)

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
