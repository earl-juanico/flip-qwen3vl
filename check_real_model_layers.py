#!/usr/bin/env python3
"""
check_real_model_layers.py

Diagnostic tool that verifies the FLIP monkeypatch (patch_qwen.py) has been
correctly applied to the Qwen3-VL model before starting the full vLLM server.

── WHAT IT DOES ─────────────────────────────────────────────────────────────

  1. Loads patch_qwen.py in-process, which monkeypatches the vLLM Qwen3-VL
     model class to insert the FLIP hidden-state perturbation hook.

  2. Constructs a minimal VllmConfig (no distributed init, no compilation)
     and attempts to instantiate Qwen3VLForConditionalGeneration.

  3. Walks every decoder layer and prints:
       layer <i>: wrapped=<True|False>, target_layers=<set|None>
     A layer with wrapped=True confirms the FLIP forward hook is active.

  4. If model construction fails (e.g. tensor-parallel group not initialised
     when running outside the full vLLM engine), the script falls back to
     static inspection of config.json and reports:
       • text_config.num_hidden_layers   — expected decoder depth
       • vision_config.depth             — vision encoder depth
       • FLIP_LAYER_INDICES parsed targets — which layers would be wrapped

  Run this script after editing patch_qwen.py to confirm the patch targets
  the intended layers without having to start the full server.

── USAGE ────────────────────────────────────────────────────────────────────

  python3 check_real_model_layers.py

  Optionally set environment variables before running to simulate the same
  configuration that serve_with_patch.py will use at serving time:

    export FLIP_VARTHETA_FILE=config10/vartheta.txt  # which config to read
    export FLIP_VARTHETA=-2.0                        # override scalar directly
    export FLIP_LAYER_INDICES=0-7                    # restrict to first 8 layers
    export FLIP_APPLY_ON_FINAL=1                     # apply on final hidden state

── MODEL PATH ───────────────────────────────────────────────────────────────

  MODEL_DIR is hard-coded to <repo_root>/model/Qwen3-VL-4B-Instruct.
  To inspect a different variant (e.g. 8B), edit MODEL_DIR at the top of
  this script or symlink the desired weights directory to that path.
"""
import os
import sys
import json
import pathlib
import traceback
from types import SimpleNamespace

# Make repo root importable so your local patch_qwen is used
_REPO_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_DIR))

# Import the patch to apply monkeypatches
import patch_qwen  # noqa: F401

MODEL_DIR = str(_REPO_DIR / "model" / "Qwen3-VL-4B-Instruct")
config_path = os.path.join(MODEL_DIR, "config.json")
quant_path_candidates = [
    os.path.join(MODEL_DIR, "quant_config.json"),
    os.path.join(MODEL_DIR, "quant.json"),
]

def dict_to_obj(d):
    if isinstance(d, dict):
        ns = SimpleNamespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_obj(v))
        return ns
    if isinstance(d, list):
        return [dict_to_obj(x) for x in d]
    return d

# Build vllm_config object expected by the model constructor
vllm_config = SimpleNamespace()
vllm_config.model_config = SimpleNamespace()
hf_cfg_obj = SimpleNamespace()

if os.path.isfile(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        hf_cfg_obj = dict_to_obj(cfg)
        print(f"Loaded config.json from {config_path}")
    except Exception as e:
        print("Warning: failed to parse config.json:", e)
        hf_cfg_obj = SimpleNamespace()
else:
    print("Warning: config.json not found at", config_path)

vllm_config.model_config.hf_config = hf_cfg_obj

from vllm.config.multimodal import MultiModalConfig  # Import multimodal config class to instantiate a valid multimodal_config

# Provide a real MultiModalConfig instance so required multimodal attributes exist
try:
    vllm_config.model_config.multimodal_config = MultiModalConfig()
except Exception:
    # fallback to a simple namespace if import/instantiation fails
    vllm_config.model_config.multimodal_config = SimpleNamespace()

# Prefer data TP mode for encoder during local inspection to avoid tensor-parallel init
try:
    vllm_config.model_config.multimodal_config.mm_encoder_tp_mode = "data"
except Exception:
    setattr(vllm_config.model_config.multimodal_config, "mm_encoder_tp_mode", "data")

# Load quant config if present, else provide conservative defaults
quant_cfg_obj = None
for qp in quant_path_candidates:
    if os.path.isfile(qp):
        try:
            with open(qp, "r", encoding="utf-8") as f:
                qcfg = json.load(f)
            quant_cfg_obj = dict_to_obj(qcfg)
            print(f"Loaded quant config from {qp}")
            break
        except Exception:
            pass

if quant_cfg_obj is None:
    print("No quant config file found; using quant_config=None")

# For local inspection, use non-quantized path expected by vLLM layers.
vllm_config.quant_config = None

# Optionally provide other commonly-accessed attributes (safe defaults)
vllm_config.device = getattr(vllm_config, "device", None)
vllm_config.tokenizer = getattr(vllm_config, "tokenizer", None)

print("Using MODEL_DIR =", MODEL_DIR)
print("Env FLIP_VARTHETA_FILE =", os.environ.get("FLIP_VARTHETA_FILE"))
print("Env FLIP_VARTHETA =", os.environ.get("FLIP_VARTHETA"))
print("Env FLIP_LAYER_INDICES =", os.environ.get("FLIP_LAYER_INDICES"))
print("Env FLIP_APPLY_ON_FINAL =", os.environ.get("FLIP_APPLY_ON_FINAL"))

from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.config.vllm import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode

import inspect

print("Qwen3VLForConditionalGeneration:", Qwen3VLForConditionalGeneration)
try:
    sig = inspect.signature(Qwen3VLForConditionalGeneration.__init__)
    print("Constructor signature:", sig)
except Exception:
    print("Could not get constructor signature for Qwen3VLForConditionalGeneration.__init__")
try:
    src = inspect.getsource(Qwen3VLForConditionalGeneration.__init__)
    print("Constructor source (truncated):\n", src[:2000])
except Exception:
    print("Could not get constructor source; maybe it's C-implemented or not available")

def ensure_attr_on(obj, name, default=None):
    if not hasattr(obj, name):
        setattr(obj, name, default if default is not None else SimpleNamespace())
        return True
    return False

# Create a real VllmConfig and set it as current so custom ops/functions
# that call get_current_vllm_config() during module init can access it.
vllm_current = VllmConfig()
# Populate minimal fields expected by code paths during init
vllm_current.model_config = SimpleNamespace()
vllm_current.model_config.hf_config = vllm_config.model_config.hf_config
vllm_current.model_config.multimodal_config = vllm_config.model_config.multimodal_config
vllm_current.quant_config = None

# Reduce compilation and parallel features to safe defaults for local inspection
try:
    vllm_current.compilation_config = CompilationConfig()
    vllm_current.compilation_config.mode = CompilationMode.NONE
    # Ensure custom_ops is explicitly set to avoid default assertion checks
    try:
        vllm_current.compilation_config.custom_ops = ["none"]
    except Exception:
        pass
except Exception:
    pass
try:
    # ensure parallel sizes are 1 to avoid distributed initialization
    vllm_current.parallel_config.tensor_parallel_size = 1
    vllm_current.parallel_config.pipeline_parallel_size = 1
    vllm_current.parallel_config.data_parallel_size = 1
except Exception:
    pass

m = None
with set_current_vllm_config(vllm_current):
    try:
        m = Qwen3VLForConditionalGeneration(vllm_config=vllm_config)
    except Exception as e:
        print("Model construction failed:", type(e).__name__, e)
        # In standalone scripts (outside vLLM engine runtime), TP group is often
        # not initialized. Fall back to static config validation instead of hard fail.
        if isinstance(e, AssertionError) and "tensor model parallel group is not initialized" in str(e):
            print("Detected standalone runtime without initialized TP group; using static validation fallback.")
            m = None
        else:
            traceback.print_exc()
            sys.exit(1)

if m is not None:
    print("Model instantiated successfully.")

if m is not None:
    # Helpful debug prints for user: show model repr, type, and top-level attrs
    try:
        print("Model repr:", m)
        print("Model type:", type(m))
        top_attrs = [a for a in dir(m) if not a.startswith("__")]
        print("Top-level attributes:", top_attrs)
        if hasattr(m, "language_model"):
            lm = m.language_model
            print("language_model type:", type(lm))
            if hasattr(lm, "model"):
                mod = lm.model
                print("language_model.model type:", type(mod))
                if hasattr(mod, "layers"):
                    try:
                        print("num layers:", len(mod.layers))
                    except Exception:
                        print("Could not determine length of mod.layers")
    except Exception:
        print("Failed to print model debug info")

def resolve_layers(model):
    candidates = [
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.layers", lambda m: m.model.layers),
        ("layers", lambda m: m.layers),
    ]
    for name, getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                return layers, name
        except Exception:
            continue
    return None, None

if m is not None:
    layers, path = resolve_layers(m)
    if layers is None:
        print("Could not resolve decoder layers on the instantiated model.")
    else:
        print(f"Resolved decoder layers via path: {path} (num_layers={len(layers)})")
        for i, layer in enumerate(layers):
            wrapped = getattr(layer, "_flip_forward_wrapped", False)
            target = getattr(layer, "_flip_target_layers", None)
            print(f"layer {i}: wrapped={wrapped}, target_layers={target}")
else:
    # Static fallback: use config.json to report expected decoder depth and
    # patch target layers from FLIP_LAYER_INDICES.
    text_layers = getattr(hf_cfg_obj, "text_config", SimpleNamespace())
    num_layers = getattr(text_layers, "num_hidden_layers", None)
    vision_cfg = getattr(hf_cfg_obj, "vision_config", SimpleNamespace())
    vision_depth = getattr(vision_cfg, "depth", None)
    print("Static model config summary:")
    print("- text_config.num_hidden_layers =", num_layers)
    print("- vision_config.depth =", vision_depth)
    if isinstance(num_layers, int):
        spec = os.environ.get("FLIP_LAYER_INDICES")
        try:
            targets = patch_qwen._parse_layer_spec(spec, num_layers)
            print("- FLIP_LAYER_INDICES parsed targets =", sorted(targets))
        except Exception:
            print("- Could not parse FLIP_LAYER_INDICES with patch helper")