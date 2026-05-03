"""
Multilingual Support Module — IndicTrans2 (exclusive for Indian languages)

Primary engine:  IndicTrans2 (ai4bharat) for ALL 22 Indian languages
                 Handles native script AND romanized input (e.g. "mai udaas hu")
                 Google Translate is NOT used for any Indic language.

Fallback engine: Google Translate (deep-translator) for non-Indic languages
                 (Spanish, French, Arabic, etc.)

Models (downloaded from HuggingFace on first use, ~200 MB each distilled):
  ai4bharat/indictrans2-indic-en-dist-200M   (Indic → English)
  ai4bharat/indictrans2-en-indic-dist-200M   (English → Indic)
"""

import os
import tempfile
from typing import Optional

# torch is imported lazily inside functions that need it (avoids MPS mutex
# deadlock on macOS when the module is imported at Streamlit startup).

# Stub out transformers.onnx — removed in transformers>=4.44 but still
# imported by IndicTrans2's configuration_indictrans.py (only needed for
# ONNX export, never for inference).
import sys as _sys
import types as _types

# Stub out transformers.onnx AND transformers.onnx.utils — removed in
# transformers>=4.44 but still imported by IndicTrans2's
# configuration_indictrans.py. We make it a proper "package" by setting
# __path__ so sub-module access like `transformers.onnx.utils` works.
if "transformers.onnx" not in _sys.modules:
    _onnx_stub = _types.ModuleType("transformers.onnx")
    _onnx_stub.__path__ = []          # marks it as a package
    _onnx_stub.__package__ = "transformers.onnx"

    class _OnnxConfig:
        default_fixed_batch = 2
        default_fixed_sequence = 8
    class _OnnxSeq2SeqConfigWithPast(_OnnxConfig): pass

    _onnx_stub.OnnxConfig = _OnnxConfig
    _onnx_stub.OnnxSeq2SeqConfigWithPast = _OnnxSeq2SeqConfigWithPast

    _utils_stub = _types.ModuleType("transformers.onnx.utils")
    def _compute_effective_axis_dimension(dimension, fixed_dimension, num_token_to_add=0):
        if dimension == -1:
            return fixed_dimension
        return dimension
    _utils_stub.compute_effective_axis_dimension = _compute_effective_axis_dimension
    _onnx_stub.utils = _utils_stub

    _sys.modules["transformers.onnx"] = _onnx_stub
    _sys.modules["transformers.onnx.utils"] = _utils_stub

# ── Flores-200 language codes (IndicTrans2 format) ────────────────────────────

# ISO 639-1 → Flores-200 (native/primary script)
ISO_TO_FLORES = {
    "hi":    "hin_Deva",   # Hindi        — Devanagari
    "ur":    "urd_Arab",   # Urdu         — Arabic script
    "bn":    "ben_Beng",   # Bengali      — Bengali script
    "mr":    "mar_Deva",   # Marathi      — Devanagari
    "ta":    "tam_Taml",   # Tamil        — Tamil script
    "te":    "tel_Telu",   # Telugu       — Telugu script
    "gu":    "guj_Gujr",   # Gujarati     — Gujarati script
    "kn":    "kan_Knda",   # Kannada      — Kannada script
    "ml":    "mal_Mlym",   # Malayalam    — Malayalam script
    "pa":    "pan_Guru",   # Punjabi      — Gurmukhi
    "or":    "ory_Orya",   # Odia         — Odia script
    "as":    "asm_Beng",   # Assamese     — Bengali script
    "ne":    "npi_Deva",   # Nepali       — Devanagari
    "si":    "sin_Sinh",   # Sinhala
    "en":    "eng_Latn",   # English
}

# Romanized (Latin-script) Flores codes — hin_Latn is NOT in IndicTrans2's
# vocabulary, so romanized Hindi is treated as hin_Deva (the model handles
# mixed-script input reasonably well, and Google Translate covers the rest).
ROMANIZED_FLORES: dict = {}

# Set of ISO codes handled by IndicTrans2
INDIC_LANGS = set(ISO_TO_FLORES.keys()) - {"en"}

# ── Language display options for the UI ───────────────────────────────────────

LANGUAGE_OPTIONS = {
    "English":              "en",
    "Hindi (हिन्दी)":       "hi",
    "Urdu (اردو)":          "ur",
    "Bengali (বাংলা)":      "bn",
    "Marathi (मराठी)":      "mr",
    "Tamil (தமிழ்)":        "ta",
    "Telugu (తెలుగు)":      "te",
    "Gujarati (ગુજરાતી)":   "gu",
    "Kannada (ಕನ್ನಡ)":     "kn",
    "Malayalam (മലയാളം)":   "ml",
    "Punjabi (ਪੰਜਾਬੀ)":    "pa",
    "Odia (ଓଡ଼ିଆ)":        "or",
    # Non-Indic (handled by Google Translate)
    "Arabic (العربية)":     "ar",
    "Spanish (Español)":    "es",
    "French (Français)":    "fr",
    "Portuguese":           "pt",
    "German (Deutsch)":     "de",
    "Japanese (日本語)":     "ja",
    "Korean (한국어)":       "ko",
    "Chinese (中文)":        "zh-CN",
}

CODE_TO_LANG = {v: k for k, v in LANGUAGE_OPTIONS.items()}


# ── Unicode script detection (offline, instant) ───────────────────────────────

_SCRIPT_RANGES = [
    ("ऀ", "ॿ", "hi"),   # Devanagari → Hindi / Marathi
    ("؀", "ۿ", "ur"),   # Arabic     → Urdu / Arabic
    ("ঀ", "৿", "bn"),   # Bengali
    ("଀", "୿", "or"),   # Odia
    ("஀", "௿", "ta"),   # Tamil
    ("ఀ", "౿", "te"),   # Telugu
    ("઀", "૿", "gu"),   # Gujarati
    ("ಀ", "೿", "kn"),   # Kannada
    ("ഀ", "ൿ", "ml"),   # Malayalam
    ("਀", "੿", "pa"),   # Gurmukhi  → Punjabi
    ("一", "鿿", "zh-CN"),# CJK       → Chinese
    ("぀", "ヿ", "ja"),   # Kana      → Japanese
    ("가", "힯", "ko"),   # Hangul    → Korean
]


def detect_script(text: str) -> Optional[str]:
    """Return ISO code based on Unicode script, or None for Latin/ASCII."""
    for char in text:
        for start, end, lang in _SCRIPT_RANGES:
            if start <= char <= end:
                return lang
    return None


def detect_language(text: str, fallback: str = "en") -> str:
    """
    Detect language:
    1. Unicode script detection (reliable for Devanagari, Arabic, etc.)
    2. langdetect for longer Latin-script text
    3. Fallback
    """
    if not text or not text.strip():
        return fallback

    script = detect_script(text)
    if script:
        return script

    # Latin-script: only trust langdetect on 5+ word inputs
    if len(text.split()) >= 5:
        try:
            from langdetect import detect
            _FP = {"af", "so", "sw", "yo", "ig", "ha", "tl", "id", "ms"}
            detected = detect(text)
            if detected and detected not in _FP:
                return detected
        except Exception:
            pass

    return fallback


def get_flores_code(iso_code: str, text: str = "") -> str:
    """
    Return the Flores-200 code for IndicTrans2.
    For Hindi: hin_Latn if text is romanized, hin_Deva if Devanagari.
    """
    if iso_code in ROMANIZED_FLORES:
        if detect_script(text) is None:
            # No native script detected → romanized
            return ROMANIZED_FLORES[iso_code]
    return ISO_TO_FLORES.get(iso_code, "eng_Latn")


def is_indic(lang_code: str) -> bool:
    return lang_code in INDIC_LANGS


# ── IndicTrans2 lazy model loader ─────────────────────────────────────────────

_indic_en_model  = None
_indic_en_tok    = None
_en_indic_model  = None
_en_indic_tok    = None
_indic_processor = None

# Set to an error string if IndicTrans2 models are inaccessible (gated, no auth)
# so we don't keep retrying every turn.
_indictrans2_error: Optional[str] = None

INDIC_EN_MODEL  = "ai4bharat/indictrans2-indic-en-dist-200M"
EN_INDIC_MODEL  = "ai4bharat/indictrans2-en-indic-dist-200M"
DEVICE = "cpu"   # MPS or CUDA if available, but cpu is safest cross-platform


def _hf_token() -> Optional[str]:
    """Read HuggingFace token from environment (HF_TOKEN or HUGGINGFACE_TOKEN)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _is_gated_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(k in msg for k in ("401", "gated", "restricted", "authentication", "log in"))


def _patch_model_for_transformers5(model):
    """
    Apply two patches required for transformers>=5.x compatibility:

    1. Reinitialize IndicTransSinusoidalPositionalEmbedding.weights buffers.
       transformers 5.x initialize_weights() corrupts non-persistent buffers
       on remote-code models (they become uninitialized memory → NaN in encoder).

    2. Add __getitem__ to EncoderDecoderCache so that the IndicTrans2 decoder
       can access per-layer cache via past_key_values[idx] (legacy tuple API).
    """
    import torch as _t

    # Fix 1: reinit sinusoidal positional embedding buffers
    for m in model.modules():
        if "SinusoidalPositionalEmbedding" in type(m).__name__:
            m.make_weights(m.weights.size(0), m.embedding_dim, m.padding_idx)

    # Fix 2: EncoderDecoderCache legacy subscript support
    try:
        from transformers.cache_utils import EncoderDecoderCache
        if not hasattr(EncoderDecoderCache, "__getitem__"):
            def _edc_getitem(self, idx):
                sa, ca = self.self_attention_cache, self.cross_attention_cache
                def _kv(cache, i):
                    return (cache.layers[i].keys, cache.layers[i].values) \
                        if i < len(cache.layers) and cache.layers[i].is_initialized \
                        else (None, None)
                sk, sv = _kv(sa, idx)
                ck, cv = _kv(ca, idx)
                return None if (sk is None and ck is None) else (sk, sv, ck, cv)
            EncoderDecoderCache.__getitem__ = _edc_getitem
    except Exception:
        pass


def _load_indic_processor():
    global _indic_processor
    if _indic_processor is None:
        from IndicTransToolkit.processor import IndicProcessor
        _indic_processor = IndicProcessor(inference=True)
    return _indic_processor


def _load_indic_en():
    """Load Indic → English model (lazy, cached)."""
    global _indic_en_model, _indic_en_tok, _indictrans2_error
    if _indictrans2_error:
        raise RuntimeError(_indictrans2_error)
    if _indic_en_model is None:
        print(f"[IndicTrans2] Loading {INDIC_EN_MODEL} …")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = _hf_token()
        _indic_en_tok   = AutoTokenizer.from_pretrained(
            INDIC_EN_MODEL, trust_remote_code=True, token=tok)
        _indic_en_model = AutoModelForSeq2SeqLM.from_pretrained(
            INDIC_EN_MODEL, trust_remote_code=True, token=tok
        ).to(DEVICE).eval()
        _patch_model_for_transformers5(_indic_en_model)
        print("[IndicTrans2] Indic→EN model ready.")
    return _indic_en_model, _indic_en_tok


def _load_en_indic():
    """Load English → Indic model (lazy, cached)."""
    global _en_indic_model, _en_indic_tok, _indictrans2_error
    if _indictrans2_error:
        raise RuntimeError(_indictrans2_error)
    if _en_indic_model is None:
        print(f"[IndicTrans2] Loading {EN_INDIC_MODEL} …")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = _hf_token()
        _en_indic_tok   = AutoTokenizer.from_pretrained(
            EN_INDIC_MODEL, trust_remote_code=True, token=tok)
        _en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
            EN_INDIC_MODEL, trust_remote_code=True, token=tok
        ).to(DEVICE).eval()
        _patch_model_for_transformers5(_en_indic_model)
        print("[IndicTrans2] EN→Indic model ready.")
    return _en_indic_model, _en_indic_tok


def _run_indictrans2(sentences: list, src_flores: str, tgt_flores: str,
                     model, tokenizer) -> list:
    """Run IndicTrans2 translation on a list of sentences."""
    import torch  # lazy import — avoids MPS mutex hang at startup on macOS

    ip = _load_indic_processor()
    batch = ip.preprocess_batch(sentences, src_lang=src_flores, tgt_lang=tgt_flores)

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=4,
            num_return_sequences=1,
        )

    decoded = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return ip.postprocess_batch(decoded, lang=tgt_flores)


# ── Public translation API ────────────────────────────────────────────────────

def _google_translate(text: str, src: str, tgt: str) -> Optional[str]:
    """Call Google Translate; returns result string or None on failure."""
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source=src, target=tgt).translate(text)
        if result and result.strip():
            return result.strip()
    except Exception as e:
        print(f"[GoogleTranslate] {src}→{tgt} failed: {e}")
    return None


def translate_to_english(text: str, src_lang: str = "auto") -> str:
    """
    Translate *text* to English.

    Routing for Indic languages:
    1. IndicTrans2 (best quality, requires HuggingFace auth for first download)
    2. Google Translate fallback (if IndicTrans2 unavailable / gated)

    Non-Indic: Google Translate only.
    """
    global _indictrans2_error
    if not text or not text.strip():
        return text
    if src_lang in ("en", "eng_Latn"):
        return text

    # ── Indic: try IndicTrans2 first ─────────────────────────────────────────
    if is_indic(src_lang) and not _indictrans2_error:
        try:
            flores_src = get_flores_code(src_lang, text)
            model, tok = _load_indic_en()
            results = _run_indictrans2([text], flores_src, "eng_Latn", model, tok)
            if results and results[0].strip():
                return results[0].strip()
        except Exception as e:
            if _is_gated_error(e):
                _indictrans2_error = (
                    "IndicTrans2 models are gated on HuggingFace. "
                    "Accept terms at https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M "
                    "and set HF_TOKEN env var, then restart."
                )
            print(f"[IndicTrans2] to_en failed: {e}")

    # ── Fallback: Google Translate ────────────────────────────────────────────
    result = _google_translate(text, src_lang if src_lang != "auto" else "auto", "en")
    if result:
        return result

    return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate English *text* into *target_lang*.

    Routing for Indic languages:
    1. IndicTrans2 (best quality, requires HuggingFace auth for first download)
    2. Google Translate fallback (if IndicTrans2 unavailable / gated)

    Non-Indic: Google Translate only.
    """
    global _indictrans2_error
    if not text or not text.strip():
        return text
    if target_lang in ("en", "eng_Latn"):
        return text

    # ── Indic: try IndicTrans2 first ─────────────────────────────────────────
    if is_indic(target_lang) and not _indictrans2_error:
        try:
            flores_tgt = ISO_TO_FLORES.get(target_lang, "hin_Deva")
            model, tok = _load_en_indic()
            results = _run_indictrans2([text], "eng_Latn", flores_tgt, model, tok)
            if results and results[0].strip():
                return results[0].strip()
        except Exception as e:
            if _is_gated_error(e):
                _indictrans2_error = (
                    "IndicTrans2 models are gated on HuggingFace. "
                    "Accept terms at https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M "
                    "and set HF_TOKEN env var, then restart."
                )
            print(f"[IndicTrans2] from_en failed: {e}")

    # ── Fallback: Google Translate ────────────────────────────────────────────
    result = _google_translate(text, "en", target_lang)
    if result:
        return result

    return text


def are_indic_models_loaded() -> bool:
    """Return True only when BOTH IndicTrans2 models are in memory."""
    return _indic_en_model is not None and _en_indic_model is not None


def get_indictrans2_error() -> Optional[str]:
    """Return the IndicTrans2 error message if models are inaccessible, else None."""
    return _indictrans2_error


def load_all_indic_models():
    """
    Load processor + both IndicTrans2 models synchronously.
    Raises RuntimeError if models are gated and no HF_TOKEN is set.
    """
    _load_indic_processor()
    _load_indic_en()
    _load_en_indic()


# ── Whisper Speech-to-Text ────────────────────────────────────────────────────

_whisper_model = None


def load_whisper_model(model_size: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            print(f"[Whisper] Loading '{model_size}' model…")
            _whisper_model = whisper.load_model(model_size)
            print("[Whisper] Ready.")
        except ImportError:
            raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, user_lang: str = "en") -> dict:
    """
    Transcribe audio using Whisper (runs locally).
    Returns original-language transcription + English translation.
    """
    try:
        import whisper
    except ImportError:
        return {"text": "", "language": user_lang, "english_text": "",
                "success": False, "error": "openai-whisper not installed"}

    tmp_path = None
    try:
        model = load_whisper_model("base")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        # Transcribe in detected language
        result   = model.transcribe(tmp_path, task="transcribe")
        orig_txt = result["text"].strip()
        det_lang = result.get("language", user_lang)

        # English translation via Whisper's own translate task (very accurate)
        if det_lang != "en":
            en_result = model.transcribe(tmp_path, task="translate")
            eng_txt   = en_result["text"].strip()
        else:
            eng_txt = orig_txt

        return {"text": orig_txt, "language": det_lang,
                "english_text": eng_txt, "success": True, "error": ""}

    except Exception as e:
        return {"text": "", "language": user_lang, "english_text": "",
                "success": False, "error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_language_name(code: str) -> str:
    return CODE_TO_LANG.get(code, code.upper())


def is_english(text: str) -> bool:
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) < 0.1


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("IndicTrans2 Translator Self-Test")
    print("=" * 60)

    # Script detection (no model needed)
    script_tests = [
        "मैं उदास हूं",
        "mai udaas hun",
        "نہیں سو پا رہا ہوں",
        "আমি খুব ক্লান্ত",
        "I feel really sad",
    ]
    print("\n[Script Detection — offline, instant]")
    for t in script_tests:
        s = detect_script(t)
        d = detect_language(t)
        print(f"  [{s or 'Latin':6}|{d:5}] {t}")

    # Translation test (loads models — ~800 MB download on first run)
    if "--translate" in sys.argv:
        print("\n[Translation — loading IndicTrans2 models]")
        tests = [
            ("mai udaas hun",          "hi", "→ EN (romanized Hindi)"),
            ("मैं उदास हूं",            "hi", "→ EN (Devanagari Hindi)"),
            ("bohut bura chal raha hai","hi", "→ EN (romanized, longer)"),
            ("مجھے نیند نہیں آتی",       "ur", "→ EN (Urdu)"),
            ("আমি খুব ক্লান্ত",          "bn", "→ EN (Bengali)"),
        ]
        for text, src, label in tests:
            result = translate_to_english(text, src)
            print(f"  {label}")
            print(f"    Input:  {text}")
            print(f"    Output: {result}")

        print("\n[English → Hindi]")
        en_text = "I understand, that sounds really difficult. How long has this been going on?"
        hi = translate_from_english(en_text, "hi")
        print(f"  EN: {en_text}")
        print(f"  HI: {hi}")
