"""
Multilingual Support Module
Handles language detection, text translation (Google Translate via deep-translator),
and Whisper-based speech-to-text transcription.

Supports all Indian languages + 100+ others out of the box.
"""

import os
import re
import tempfile
from typing import Optional

# ── Language registry ──────────────────────────────────────────────────────────

LANGUAGE_OPTIONS = {
    "English":    "en",
    "Hindi (हिन्दी)":       "hi",
    "Urdu (اردو)":         "ur",
    "Bengali (বাংলা)":     "bn",
    "Marathi (मराठी)":     "mr",
    "Tamil (தமிழ்)":       "ta",
    "Telugu (తెలుగు)":     "te",
    "Gujarati (ગુજરાતી)":  "gu",
    "Kannada (ಕನ್ನಡ)":    "kn",
    "Malayalam (മലയാളം)":  "ml",
    "Punjabi (ਪੰਜਾਬੀ)":   "pa",
    "Odia (ଓଡ଼ିଆ)":       "or",
    "Arabic (العربية)":    "ar",
    "Spanish (Español)":   "es",
    "French (Français)":   "fr",
    "Portuguese":          "pt",
    "German (Deutsch)":    "de",
    "Japanese (日本語)":    "ja",
    "Korean (한국어)":      "ko",
    "Chinese (中文)":       "zh-CN",
}

# ISO code → display name (reverse map)
CODE_TO_LANG = {v: k for k, v in LANGUAGE_OPTIONS.items()}

# ── Script-based detection (fast, no network) ──────────────────────────────────

_SCRIPT_RANGES = [
    ("ऀ", "ॿ", "hi"),   # Devanagari → Hindi/Marathi
    ("؀", "ۿ", "ur"),   # Arabic → Urdu/Arabic
    ("ঀ", "৿", "bn"),   # Bengali
    ("଀", "୿", "or"),   # Odia
    ("஀", "௿", "ta"),   # Tamil
    ("ఀ", "౿", "te"),   # Telugu
    ("઀", "૿", "gu"),   # Gujarati
    ("ಀ", "೿", "kn"),   # Kannada
    ("ഀ", "ൿ", "ml"),   # Malayalam
    ("਀", "੿", "pa"),   # Punjabi (Gurmukhi)
    ("一", "鿿", "zh-CN"),# CJK → Chinese
    ("぀", "ヿ", "ja"),   # Hiragana/Katakana → Japanese
    ("가", "힯", "ko"),   # Hangul → Korean
]


def detect_script(text: str) -> Optional[str]:
    """Detect language from Unicode script ranges. Returns ISO code or None."""
    for char in text:
        for start, end, lang in _SCRIPT_RANGES:
            if start <= char <= end:
                return lang
    return None


def detect_language(text: str, fallback: str = "en") -> str:
    """
    Detect language of text.
    1. Check for non-Latin scripts (reliable, instant)
    2. Try langdetect for Latin-script languages
    3. Fall back to `fallback` (default "en")
    """
    if not text or not text.strip():
        return fallback

    # Script detection is instant and reliable for Indic/CJK scripts
    script_lang = detect_script(text)
    if script_lang:
        return script_lang

    # For Latin-script input: langdetect, but only trust it for longer text
    # Short romanized text (e.g. "mai udaas hu") is unreliable
    if len(text.split()) >= 5:
        try:
            from langdetect import detect
            # False-positive blocklist: short text often misdetected as these
            _FP = {"af", "so", "sw", "yo", "ig", "ha", "tl", "id", "ms"}
            detected = detect(text)
            if detected and detected not in _FP:
                return detected
        except Exception:
            pass

    return fallback


# ── Translation (deep-translator → Google Translate, free, no key) ─────────────

def translate_to_english(text: str, src_lang: str = "auto") -> str:
    """
    Translate text to English.
    Returns original text if translation fails or src is already English.
    """
    if not text or not text.strip():
        return text
    if src_lang in ("en", "english"):
        return text

    try:
        from deep_translator import GoogleTranslator
        source = src_lang if src_lang != "auto" else "auto"
        translated = GoogleTranslator(source=source, target="en").translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"[Translator] to_english failed ({src_lang}): {e}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate English text to target language.
    Returns original text if translation fails or target is English.
    """
    if not text or not text.strip():
        return text
    if target_lang in ("en", "english"):
        return text

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
        return translated if translated else text
    except Exception as e:
        print(f"[Translator] from_english failed ({target_lang}): {e}")
        return text


def translate_text(text: str, src: str, tgt: str) -> str:
    """Generic translation from src to tgt language codes."""
    if src == tgt:
        return text
    if tgt == "en":
        return translate_to_english(text, src)
    if src == "en":
        return translate_from_english(text, tgt)
    # Indirect: src → en → tgt
    en_text = translate_to_english(text, src)
    return translate_from_english(en_text, tgt)


# ── Whisper Speech-to-Text ─────────────────────────────────────────────────────

_whisper_model = None
_whisper_available = None


def _check_whisper() -> bool:
    global _whisper_available
    if _whisper_available is None:
        try:
            import whisper  # noqa: F401
            _whisper_available = True
        except ImportError:
            _whisper_available = False
    return _whisper_available


def load_whisper_model(model_size: str = "base"):
    """Load Whisper model (cached after first load)."""
    global _whisper_model
    if _whisper_model is None:
        if not _check_whisper():
            raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")
        import whisper
        print(f"[Whisper] Loading '{model_size}' model...")
        _whisper_model = whisper.load_model(model_size)
        print("[Whisper] Model loaded.")
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, user_lang: str = "en") -> dict:
    """
    Transcribe audio bytes using Whisper.

    Args:
        audio_bytes: Raw audio bytes (WAV from st.audio_input)
        user_lang:   Expected language code (hint to Whisper)

    Returns:
        {
          "text": str,           # Transcribed text (in original language)
          "language": str,       # Detected language code (e.g. "hi")
          "english_text": str,   # English translation (via Whisper translate task)
          "success": bool,
          "error": str
        }
    """
    if not _check_whisper():
        return {"text": "", "language": user_lang, "english_text": "", "success": False,
                "error": "Whisper not installed. Run: pip install openai-whisper"}

    tmp_path = None
    try:
        model = load_whisper_model("base")
        import whisper

        # Write to temp file (Whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        # Transcribe in original language
        transcribe_result = model.transcribe(tmp_path, task="transcribe")
        original_text = transcribe_result["text"].strip()
        detected_lang = transcribe_result.get("language", user_lang)

        # If non-English, also get Whisper's English translation
        if detected_lang != "en":
            translate_result = model.transcribe(tmp_path, task="translate")
            english_text = translate_result["text"].strip()
        else:
            english_text = original_text

        return {
            "text": original_text,
            "language": detected_lang,
            "english_text": english_text,
            "success": True,
            "error": ""
        }

    except Exception as e:
        return {
            "text": "",
            "language": user_lang,
            "english_text": "",
            "success": False,
            "error": str(e)
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Utility ────────────────────────────────────────────────────────────────────

def get_language_name(code: str) -> str:
    """Get display name from ISO code."""
    return CODE_TO_LANG.get(code, code.upper())


def is_english(text: str) -> bool:
    """Quick check if text appears to be English (ASCII-dominant)."""
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) < 0.1


if __name__ == "__main__":
    # Quick self-test
    test_cases = [
        ("mai udaas hu", "hi"),
        ("bohut bura chal raha hai", "hi"),
        ("मैं बहुत थका हुआ हूं", "hi"),
        ("nahi so pa raha hoon bilkul bhi", "hi"),
        ("Me siento muy triste", "es"),
        ("I'm feeling really low lately", "en"),
    ]
    print("Translator Self-Test\n" + "=" * 50)
    for text, src in test_cases:
        detected = detect_language(text)
        en = translate_to_english(text, src)
        back = translate_from_english(en, src) if src != "en" else en
        print(f"Input  : {text}")
        print(f"Detect : {detected}  |  EN: {en}")
        print(f"Back   : {back}")
        print()
