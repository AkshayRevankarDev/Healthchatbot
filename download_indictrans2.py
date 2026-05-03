"""
Download IndicTrans2 models to local HuggingFace cache.

Run once from terminal BEFORE starting the app:
    /opt/anaconda3/bin/python download_indictrans2.py hf_YOUR_TOKEN_HERE

Models are ~900 MB each; ~1.8 GB total.
Uses snapshot_download so files are patched BEFORE the model is instantiated.
"""

import os
import sys
import glob
import types
import traceback

# ── Stub transformers.onnx before any other import ──────────────────────────
_onnx_stub = types.ModuleType("transformers.onnx")
_onnx_stub.__path__ = []
_onnx_stub.__package__ = "transformers.onnx"

class _OnnxConfig:
    default_fixed_batch = 2
    default_fixed_sequence = 8

class _OnnxSeq2SeqConfigWithPast(_OnnxConfig):
    pass

def _compute_effective_axis_dimension(dimension, fixed_dimension, num_token_to_add=0):
    return fixed_dimension if dimension == -1 else dimension

_utils_stub = types.ModuleType("transformers.onnx.utils")
_utils_stub.compute_effective_axis_dimension = _compute_effective_axis_dimension
_onnx_stub.OnnxConfig = _OnnxConfig
_onnx_stub.OnnxSeq2SeqConfigWithPast = _OnnxSeq2SeqConfigWithPast
_onnx_stub.utils = _utils_stub
sys.modules["transformers.onnx"] = _onnx_stub
sys.modules["transformers.onnx.utils"] = _utils_stub

# ── Get HF token ─────────────────────────────────────────────────────────────
HF_TOKEN = ""
if len(sys.argv) > 1 and sys.argv[1].startswith("hf_"):
    HF_TOKEN = sys.argv[1].strip()
if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if not HF_TOKEN:
    for _p in ["~/.huggingface/token", "~/.cache/huggingface/token"]:
        _p = os.path.expanduser(_p)
        if os.path.exists(_p):
            with open(_p) as _f:
                HF_TOKEN = _f.read().strip()
            break
if not HF_TOKEN:
    try:
        HF_TOKEN = input("Enter your HuggingFace token (hf_...): ").strip()
    except (EOFError, KeyboardInterrupt):
        pass

if not HF_TOKEN:
    print("\nERROR: No HuggingFace token found.")
    print("Usage:  /opt/anaconda3/bin/python download_indictrans2.py hf_YOUR_TOKEN")
    print("Get a token: https://huggingface.co/settings/tokens  (Read access)")
    sys.exit(1)

print(f"\nUsing token: {HF_TOKEN[:8]}{'*' * max(0, len(HF_TOKEN) - 8)}")

# ── Patch helpers ─────────────────────────────────────────────────────────────
def _patch_file(path: str, patches: list) -> bool:
    """Apply (old, new) string replacements. Returns True if changed."""
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    original = src
    for old, new in patches:
        src = src.replace(old, new)
    if src != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        return True
    return False


CONFIG_PATCH = [(
    "from transformers.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast\n"
    "from transformers.onnx.utils import compute_effective_axis_dimension",
    "try:\n"
    "    from transformers.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast\n"
    "    from transformers.onnx.utils import compute_effective_axis_dimension\n"
    "except (ImportError, ModuleNotFoundError):\n"
    "    class OnnxConfig:\n"
    "        default_fixed_batch = 2\n"
    "        default_fixed_sequence = 8\n"
    "    class OnnxSeq2SeqConfigWithPast(OnnxConfig): pass\n"
    "    def compute_effective_axis_dimension(dimension, fixed_dimension, num_token_to_add=0):\n"
    "        return fixed_dimension if dimension == -1 else dimension",
)]

# Two possible positions where _special_tokens_map init might be needed
TOKENIZER_PATCH = [
    # Pattern A — after self.src_vocab_fp line (original file layout)
    (
        "        self.src_vocab_fp = src_vocab_fp\n"
        "        self.tgt_vocab_fp = tgt_vocab_fp",
        "        # Pre-initialize _special_tokens_map (transformers>=4.44 checks it\n"
        "        # in __setattr__ before super().__init__ is called).\n"
        "        if not hasattr(self, '_special_tokens_map'):\n"
        "            self._special_tokens_map = {}\n"
        "        self.src_vocab_fp = src_vocab_fp\n"
        "        self.tgt_vocab_fp = tgt_vocab_fp",
    ),
]

MODELING_PATCH = [
    (
        "    def tie_weights(self):\n"
        "        if self.config.share_decoder_input_output_embed:\n"
        "            self._tie_or_clone_weights(self.model.decoder.embed_tokens, self.lm_head)",
        "    def tie_weights(self, **kwargs):\n"
        "        if self.config.share_decoder_input_output_embed:\n"
        "            # _tie_or_clone_weights removed in transformers>=4.44; tie directly.\n"
        "            self.lm_head.weight = self.model.decoder.embed_tokens.weight",
    ),
    # Belt-and-suspenders: already has **kwargs but body still uses old method
    (
        "    def tie_weights(self, **kwargs):\n"
        "        if self.config.share_decoder_input_output_embed:\n"
        "            self._tie_or_clone_weights(self.model.decoder.embed_tokens, self.lm_head)",
        "    def tie_weights(self, **kwargs):\n"
        "        if self.config.share_decoder_input_output_embed:\n"
        "            # _tie_or_clone_weights removed in transformers>=4.44; tie directly.\n"
        "            self.lm_head.weight = self.model.decoder.embed_tokens.weight",
    ),
    # EncoderDecoderCache — new cache format in transformers>=4.44 is not subscriptable
    (
        "        # past_key_values_length\n"
        "        past_key_values_length = (\n"
        "            past_key_values[0][0].shape[2] if past_key_values is not None else 0\n"
        "        )",
        "        # past_key_values_length — handle both legacy tuple format and new\n"
        "        # EncoderDecoderCache format introduced in transformers>=4.44.\n"
        "        if past_key_values is None:\n"
        "            past_key_values_length = 0\n"
        "        elif hasattr(past_key_values, 'self_attention_cache'):\n"
        "            kc = past_key_values.self_attention_cache.key_cache\n"
        "            past_key_values_length = kc[0].shape[2] if kc else 0\n"
        "        else:\n"
        "            past_key_values_length = past_key_values[0][0].shape[2]",
    ),
]


def patch_snapshot(snapshot_dir: str) -> None:
    for fname, patches in [
        ("configuration_indictrans.py", CONFIG_PATCH),
        ("tokenization_indictrans.py",  TOKENIZER_PATCH),
        ("modeling_indictrans.py",       MODELING_PATCH),
    ]:
        fpath = os.path.join(snapshot_dir, fname)
        if os.path.exists(fpath):
            changed = _patch_file(fpath, patches)
            label = "✏️  Patched" if changed else "✅ Already OK"
            print(f"    {label}: {fname}")


def patch_all_snapshots(cache_slug: str) -> None:
    base = os.path.expanduser(
        f"~/.cache/huggingface/modules/transformers_modules/ai4bharat/{cache_slug}"
    )
    snapshots = glob.glob(os.path.join(base, "*"))
    if not snapshots:
        print(f"    ⚠️  No snapshots found under {base}")
        return
    for snap in snapshots:
        if os.path.isdir(snap):
            patch_snapshot(snap)


# ── Models to download ───────────────────────────────────────────────────────
MODELS = [
    ("ai4bharat/indictrans2-indic-en-dist-200M",
     "indictrans2_hyphen_indic_hyphen_en_hyphen_dist_hyphen_200M"),
    ("ai4bharat/indictrans2-en-indic-dist-200M",
     "indictrans2_hyphen_en_hyphen_indic_hyphen_dist_hyphen_200M"),
]

from huggingface_hub import snapshot_download

for model_id, cache_slug in MODELS:
    print(f"\n{'='*60}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    # ── Step 1: Download all files (no instantiation) ────────────────────────
    print("  → Downloading files via snapshot_download...")
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            token=HF_TOKEN,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"  ✅ Files downloaded to: {local_dir}")
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 2: Patch cached source files BEFORE loading ────────────────────
    print("  → Patching cached source files...")
    patch_all_snapshots(cache_slug)

    # ── Step 3: Load tokenizer (this also copies source files to transformers_modules) ──
    print("  → Loading tokenizer (verify)...")
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, token=HF_TOKEN,
        )
        print("  ✅ Tokenizer OK")
    except Exception as e:
        print(f"  ❌ Tokenizer load failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Step 4: Patch AGAIN — modeling_indictrans.py only appears in
    #    transformers_modules AFTER the first trust_remote_code load above. ──
    print("  → Re-patching (modeling file now present)...")
    patch_all_snapshots(cache_slug)

    print("  → Loading model weights (verify)...")
    try:
        from transformers import AutoModelForSeq2SeqLM
        m = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, token=HF_TOKEN,
        )
        n_params = sum(p.numel() for p in m.parameters()) // 1_000_000
        print(f"  ✅ Model OK — {n_params}M params")
        del m  # free memory before next model
    except Exception as e:
        print(f"  ❌ Model load failed: {e}")
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*60)
print("✅ ALL MODELS DOWNLOADED, PATCHED, AND VERIFIED!")
print("Start the app:  streamlit run app.py")
print("="*60)
