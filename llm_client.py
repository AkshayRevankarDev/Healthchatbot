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
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

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


def call_llm(
    prompt: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1500,
    retries: int = 3,
    thinking: bool = False,
) -> str:
    """
    Call Google Gemini and return the text response.

    Args:
        prompt:      The user message / instruction.
        system:      Optional system-level instruction.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens:  Maximum output tokens.
        retries:     Number of retry attempts on transient errors.

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

    for attempt in range(1, retries + 1):
        try:
            response = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            text = response.text.strip() if response.text else ""
            return text
        except Exception as e:
            err = str(e)
            is_quota = "429" in err or "RESOURCE_EXHAUSTED" in err
            is_last   = attempt == retries

            if is_quota and is_last:
                print(
                    f"\n[LLM ERROR] Gemini quota exceeded.\n"
                    f"  Free tier for {GEMINI_MODEL} is very limited.\n"
                    f"  Fix: enable billing at https://console.cloud.google.com/billing\n"
                    f"  Cost is ~$0.15 per million tokens (pennies per session).\n"
                )
            elif not is_last:
                wait = min(2 ** attempt, 60)   # cap at 60 s
                print(f"[LLM WARN] Attempt {attempt} failed. Retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"[LLM ERROR] All {retries} attempts failed: {err}")
    return ""
