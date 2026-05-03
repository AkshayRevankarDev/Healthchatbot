"""
Centralized LLM client — Google Gemini.

Usage:
    from llm_client import call_llm

    response = call_llm("Your prompt here", system="Optional system instruction")

Set your API key:
    export GEMINI_API_KEY="your-key-here"
    # or create a .env file with GEMINI_API_KEY=your-key-here

Model can be overridden via:
    export GEMINI_MODEL="gemini-2.0-flash"   # default
"""

import os
import time
from pathlib import Path

# Load .env file if present (no extra deps needed)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Fallback model cascade — tried in order when the primary model hits quota
_FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "\n\n[llm_client] GEMINI_API_KEY is not set.\n"
        "Create a .env file in the project root with:\n"
        "    GEMINI_API_KEY=your-key-here\n"
        "Or export it in your shell:\n"
        "    export GEMINI_API_KEY=your-key-here\n"
    )

try:
    from google import genai
    from google.genai import types as _gtypes
    _client = genai.Client(api_key=GEMINI_API_KEY)
except ImportError:
    raise ImportError(
        "[llm_client] google-genai is not installed.\n"
        "Run:  pip install google-genai"
    )


def _call_model(model: str, prompt: str, config) -> str:
    """Call a single model and return the text. Raises on any error."""
    # Only gemini-2.5-x and gemini-3.x models support thinking_config.
    # All others (2.0-flash, 1.5-flash, etc.) must have it stripped.
    _supports_thinking = ("gemini-2.5", "gemini-3")
    if not any(x in model for x in _supports_thinking):
        from google.genai import types as _gt
        safe_config = _gt.GenerateContentConfig(
            system_instruction=config.system_instruction,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )
    else:
        safe_config = config
    response = _client.models.generate_content(
        model=model,
        contents=prompt,
        config=safe_config,
    )
    return response.text.strip() if response.text else ""


def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1500,
    retries: int = 2,
    thinking: bool = False,
) -> str:
    """
    Call Google Gemini and return the text response.
    Automatically falls back through model cascade on quota errors.

    Args:
        prompt:      The user message / instruction.
        system:      Optional system-level instruction.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens:  Maximum output tokens.
        retries:     Number of retry attempts per model on transient errors.

    Returns:
        Stripped text response, or "" on failure.
    """
    # Disable thinking for simple dialogue turns (it eats the token budget).
    # Only enable for deep CoT scoring where reasoning quality matters.
    thinking_config = (
        _gtypes.ThinkingConfig(thinking_budget=1024)
        if thinking
        else _gtypes.ThinkingConfig(thinking_budget=0)
    )

    config = _gtypes.GenerateContentConfig(
        system_instruction=system if system else None,
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=thinking_config,
    )

    # Build ordered model list: primary model first, then fallbacks (deduplicated)
    models_to_try = [GEMINI_MODEL] + [m for m in _FALLBACK_MODELS if m != GEMINI_MODEL]

    for model in models_to_try:
        for attempt in range(1, retries + 1):
            try:
                text = _call_model(model, prompt, config)
                if model != GEMINI_MODEL:
                    print(f"[LLM] Using fallback model: {model}")
                return text
            except Exception as e:
                err = str(e)
                is_quota = "429" in err or "RESOURCE_EXHAUSTED" in err
                is_last_attempt = attempt == retries

                if is_quota:
                    print(f"[LLM WARN] {model} quota exceeded — trying next model…")
                    break  # skip remaining retries for this model, try next
                elif not is_last_attempt:
                    wait = min(2 ** attempt, 30)
                    print(f"[LLM WARN] {model} attempt {attempt} failed: {err[:80]}. Retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    print(f"[LLM ERROR] {model} all {retries} attempts failed: {err[:120]}")

    print(
        f"\n[LLM ERROR] All Gemini models exhausted.\n"
        f"  If you see 'prepayment credits are depleted':\n"
        f"    → Your Google Cloud prepaid credits are EMPTY (not a rate limit).\n"
        f"    → Get a FREE AI Studio key (no credit card): https://aistudio.google.com/app/apikey\n"
        f"    → Replace GEMINI_API_KEY in .env with the new key and restart.\n"
        f"\n"
        f"  If you see '429 RESOURCE_EXHAUSTED' (rate limit):\n"
        f"    → Wait for daily quota reset (midnight Pacific time)\n"
        f"    → Or enable billing: https://console.cloud.google.com/billing\n"
    )
    return ""
