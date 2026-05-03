"""
Centralized LLM client — OpenAI GPT.

Usage:
    from llm_client import call_llm
    response = call_llm("Your prompt here", system="Optional system instruction")

Set your API key in .env or environment:
    OPENAI_API_KEY=sk-...

Models (override via env vars):
    OPENAI_MODEL         = gpt-4o-mini   (dialogue — fast, cheap)
    OPENAI_SCORING_MODEL = gpt-4o        (CoT scoring — better reasoning)
"""

import os
import time
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL         = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")   # dialogue
OPENAI_SCORING_MODEL = os.environ.get("OPENAI_SCORING_MODEL", "gpt-4o") # CoT scoring

# Keep these for any code that still references GEMINI_MODEL
GEMINI_MODEL = OPENAI_MODEL  # alias so old imports don't break

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "\n\n[llm_client] OPENAI_API_KEY is not set.\n"
        "Add it to your .env file:\n"
        "    OPENAI_API_KEY=sk-...\n"
        "Or export it in your shell:\n"
        "    export OPENAI_API_KEY=sk-...\n"
        "Get a key: https://platform.openai.com/api-keys\n"
    )

try:
    from openai import OpenAI
    _client = OpenAI(api_key=OPENAI_API_KEY)
except ImportError:
    raise ImportError(
        "[llm_client] openai is not installed.\n"
        "Run:  pip install openai"
    )


def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1500,
    retries: int = 2,
    thinking: bool = False,
) -> str:
    """
    Call OpenAI and return the text response.

    Args:
        prompt:      The user message / instruction.
        system:      Optional system-level instruction.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens:  Maximum output tokens.
        retries:     Number of retry attempts on transient errors.
        thinking:    If True, uses the higher-quality scoring model (gpt-4o).

    Returns:
        Stripped text response, or "" on failure.
    """
    model = OPENAI_SCORING_MODEL if thinking else OPENAI_MODEL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, retries + 1):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e)
            is_quota = any(k in err for k in ("429", "rate_limit", "quota", "insufficient_quota"))
            is_last  = attempt == retries

            if is_quota and is_last:
                print(
                    f"\n[LLM ERROR] OpenAI quota/rate limit hit on {model}.\n"
                    f"  Add credits: https://platform.openai.com/account/billing\n"
                )
            elif not is_last:
                wait = min(2 ** attempt, 30)
                print(f"[LLM WARN] {model} attempt {attempt} failed: {err[:80]}. Retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"[LLM ERROR] {model} all {retries} attempts failed: {err[:120]}")

    return ""
