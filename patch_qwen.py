# patch_qwen.py — FLIP + Permutation patch for Qwen3-VL (vLLM)
#
# OVERVIEW
# --------
# This module monkey-patches Qwen3VLForConditionalGeneration at import time
# to apply Final Layer Inference-time Probe (FLIP) interventions on the
# model's hidden states.  Two complementary interventions are supported:
#
#   1. Layer-wise FLIP  — apply a floor (and optional permutation) to the
#      output hidden states of selected decoder layers during the forward pass.
#
#   2. Final FLIP       — apply the same floor/permutation to the hidden states
#      passed into compute_logits, just before the vocabulary projection.
#
# Each intervention can independently use either:
#   • Plain flooring   : h' = max(h, vartheta)
#   • Permute + floor  : randomly shuffle a fraction of hidden-state dimensions
#                        among themselves, then floor the result.
#
# STATISTICS (Section 5.1)
# ------------------------
# When FLIP_LOG_STATS=1, the module accumulates token-level pre-floor statistics
# at every active intervention site, keyed by (site, vartheta) so each ϑ value
# in a sweep gets its own bucket.  Two quantities are tracked per token i at
# layer ℓ, computed over the d hidden dimensions before any floor is applied:
#
#   A_{ℓ,i}(ϑ) = (1/d) Σ_k 1{z_ik^(ℓ) < ϑ}   — fraction of dims clamped
#   M_{ℓ,i}(ϑ) = (1/d) Σ_k (ϑ − z_ik^(ϑ))₊   — mean clamp magnitude
#
# The aggregate reported is E_{samples,t}[A_{ℓ,t}] ± std and likewise for M,
# where t ranges over logit-readout token positions T:
#   • Final site   : hidden_states passed to compute_logits are already T
#                    (vLLM pre-selects readout positions), so accumulation is exact.
#   • Layer sites  : all token positions in each forward call are included;
#                    single-token decode steps are exact, prefill is approximate.
#
# Stats are independent of whether the floor intervention is active at a given
# site: FLIP_LOG_STATS=1 measures A and M even when FLIP_APPLY_ON_FINAL=0.
#
# USAGE
# -----
# Import this module before vLLM loads the model (serve_with_patch.py does
# this automatically).  All behaviour is controlled by environment variables.
#
# ENVIRONMENT VARIABLES
# ---------------------
#
# FLIP_VARTHETA          (float | "none")
#     The flooring vartheta t.  Hidden-state values below t are raised to t.
#     Set to "none" or leave unset to disable flooring entirely.
#     Example: FLIP_VARTHETA=0.0
#
# FLIP_VARTHETA_FILE     (path)
#     Path to a plain-text file containing a single float (or "none").
#     When set, the vartheta is re-read from this file on every forward pass
#     if the file's mtime has changed, allowing runtime adjustment without
#     restarting the server.  Takes priority over FLIP_VARTHETA.
#     Example: FLIP_VARTHETA_FILE=config/vartheta.txt
#
# FLIP_LAYER_INDICES      (spec | "all" | unset)
#     Which decoder layers receive the layer-wise FLIP intervention.
#     Accepted forms:
#       all          — every decoder layer
#       28           — layer 28 only
#       28,29,31     — layers 28, 29, and 31
#       24:32        — layers 24–31 (exclusive upper bound)
#       28:          — layers 28 through the last layer
#       :8           — layers 0–7
#     Leave unset (or empty) to skip layer-wise FLIP entirely.
#     Example: FLIP_LAYER_INDICES=28:
#
# FLIP_APPLY_ON_FINAL     (0 | 1, default 0)
#     When 1, also apply FLIP to the hidden states just before the vocabulary
#     projection (inside compute_logits).
#     Example: FLIP_APPLY_ON_FINAL=1
#
# FLIP_PERMUTE_FRACTION   (float 0.0–1.0 | unset)
#     Fraction of hidden-state dimensions to permute before flooring.
#     When set, both the layer-wise and final interventions use permute+floor
#     instead of plain flooring.
#     k = max(1, int(fraction * hidden_dim)) dimensions are selected at random,
#     shuffled among themselves, then the floor is applied.
#     Leave unset to use plain flooring only.
#     Example: FLIP_PERMUTE_FRACTION=0.2
#
# FLIP_PERMUTE_SEED       (int, default 1234)
#     Integer seed for the permutation RNG.  The same seed produces the same
#     permutation on every call (deterministic per run for a given hidden dim).
#     Example: FLIP_PERMUTE_SEED=42
#
# FLIP_LOG_LAYER_FLOOR_ONCE  (0 | 1, default 0)
#     When 1, emit a one-time INFO log the first time FLIP is applied on each
#     wrapped decoder layer.  Useful for confirming which layers are active.
#
# FLIP_LOG_STATS          (0 | 1, default 0)
#     When 1, accumulate per-token A_{ℓ,i}(ϑ) and M_{ℓ,i}(ϑ) at every active
#     site and periodically log mean ± std to stderr.  Stats are keyed by
#     (site, vartheta) so each ϑ value in a sweep produces a separate bucket.
#     Stats are always on pre-floor hidden states (measuring what would be
#     clamped) and are independent of FLIP_APPLY_ON_FINAL.
#     A final summary is emitted at process exit via atexit.
#
# FLIP_LOG_STATS_INTERVAL (int, default 100)
#     Log a running mean ± std summary every N forward calls per (site, ϑ)
#     bucket.  The summary covers all tokens accumulated in that bucket.
#     Set to 1 to log after every forward call.
#
# FLIP_STATS_CSV          (path | unset)
#     When set, write accumulated stats to this CSV file on every periodic
#     summary and at process exit.  The CSV has one row per (ϑ, site) bucket
#     with columns: vartheta, site, n_tokens, n_calls, A_mean, A_std,
#     M_mean, M_std.  The file is overwritten on each write so it stays
#     current mid-sweep.  Intended as a temporary scratchpad; sweep_leftright.py
#     copies it into --output-dir after the sweep completes.
#     Example: FLIP_STATS_CSV=/tmp/flip_am_stats.csv
#
# NOTE — CUDA graphs and layer-wise stats
# ----------------------------------------
# vLLM uses CUDA graphs for decode steps by default.  After the graph is
# captured, only CUDA kernels replay; the Python wrapped_forward code
# (including the stats block in _apply) does NOT execute.  As a result,
# layer-wise A/M stats accumulate only during prefill, giving far fewer
# tokens than the Final site (which is called outside the graph boundary
# on every token).  To collect layer-wise stats over all tokens — including
# decode steps — restart serve_with_patch.py with:
#
#   FLIP_ENFORCE_EAGER=1
#
# This passes --enforce-eager to vLLM, disabling CUDA graph capture so that
# Python executes on every forward call.  Decode throughput will be lower.
#
# QWEN3_DISABLE_VIDEO    (0 | 1, default 0)
#     When 1, skip video embedding processing entirely
#     (adds --limit-mm-per-prompt video=0 equivalent at the model level).
#
# EXAMPLE INVOCATIONS
# -------------------
#
# Plain flooring on layers 28 and 29 only:
#   FLIP_VARTHETA=0.0 FLIP_LAYER_INDICES=28,29 python serve_with_patch.py
#
# Permute 20% of dims + floor, applied at all layers and before logits:
#   FLIP_VARTHETA=0.0 FLIP_LAYER_INDICES=all \
#   FLIP_PERMUTE_FRACTION=0.2 FLIP_APPLY_ON_FINAL=1 \
#   python serve_with_patch.py
#
# Permute 20% of dims + floor on layer 8, with stats collection for a sweep:
#   FLIP_VARTHETA=0.0 FLIP_LAYER_INDICES=8 \
#   FLIP_PERMUTE_FRACTION=0.2 \
#   FLIP_LOG_STATS=1 FLIP_STATS_CSV=/tmp/flip_am_stats.csv \
#   python serve_with_patch.py
#
# Runtime vartheta adjustment via file (no restart needed):
#   FLIP_VARTHETA_FILE=config/vartheta.txt \
#   FLIP_LAYER_INDICES=28: FLIP_PERMUTE_FRACTION=0.1 \
#   python serve_with_patch.py
#   # then: echo "0.5" > config/vartheta.txt

import logging
import os
import inspect
import functools
from collections import deque
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import types
import threading
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

log = logging.getLogger("patch_qwen")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)
log.info(
    "patch_qwen imported; FLIP_VARTHETA=%s FLIP_LAYER_INDICES=%s FLIP_APPLY_ON_FINAL=%s "
    "FLIP_PERMUTE_FRACTION=%s FLIP_PERMUTE_SEED=%s",
    os.environ.get("FLIP_VARTHETA"),
    os.environ.get("FLIP_LAYER_INDICES"),
    os.environ.get("FLIP_APPLY_ON_FINAL"),
    os.environ.get("FLIP_PERMUTE_FRACTION"),
    os.environ.get("FLIP_PERMUTE_SEED"),
)


def _parse_vartheta_from_env(env_name: str = "FLIP_VARTHETA"):
    val = os.environ.get(env_name)
    if val is None:
        return None
    txt = val.strip()
    if txt.lower() == "none":
        return None
    try:
        return float(txt)
    except Exception:
        log.warning("Unable to parse %s=%r as float; falling back to None", env_name, val)
        return None


def _parse_bool_env(env_name: str, default: bool = False):
    val = os.environ.get(env_name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(env_name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.environ.get(env_name)
    if val is None:
        return default
    try:
        return int(val.strip())
    except Exception:
        log.warning("Unable to parse %s=%r as int; falling back to %s", env_name, val, default)
        return default


FLIP_VARTHETA = _parse_vartheta_from_env()
DISABLE_QWEN3_VIDEO = _parse_bool_env("QWEN3_DISABLE_VIDEO", default=False)
FLIP_PERMUTE_SEED = _parse_int_env("FLIP_PERMUTE_SEED", default=1234)
FLIP_PERMUTE_FRACTION = _parse_vartheta_from_env("FLIP_PERMUTE_FRACTION")

# New: which decoder layers should receive flooring during forward pass.
# Examples:
#   FLIP_LAYER_INDICES=all
#   FLIP_LAYER_INDICES=28,29,31
#   FLIP_LAYER_INDICES=24:32
#   FLIP_LAYER_INDICES=28:
#   FLIP_LAYER_INDICES=:8
FLIP_LAYER_INDICES_ENV = os.environ.get("FLIP_LAYER_INDICES")
FLIP_APPLY_ON_FINAL = _parse_bool_env("FLIP_APPLY_ON_FINAL", default=False)
FLIP_LOG_LAYER_FLOOR_ONCE = _parse_bool_env("FLIP_LOG_LAYER_FLOOR_ONCE", default=False)
FLIP_LOG_STATS = _parse_bool_env("FLIP_LOG_STATS", default=False)
FLIP_LOG_STATS_INTERVAL = _parse_int_env("FLIP_LOG_STATS_INTERVAL", default=100)
FLIP_STATS_CSV = os.environ.get("FLIP_STATS_CSV")

# Optional: file to override vartheta at runtime
FLIP_VARTHETA_FILE = os.environ.get("FLIP_VARTHETA_FILE")
_flip_file_cached_value = None
_flip_file_cached_mtime = None
_flip_lock = threading.Lock()

def _get_flip_vartheta():
    """
    Return current FLIP vartheta.

    Priority:
      1. If FLIP_VARTHETA_FILE is set and readable, use its latest value.
      2. Otherwise, fall back to FLIP_VARTHETA from the environment.
    """
    global _flip_file_cached_value, _flip_file_cached_mtime

    if not FLIP_VARTHETA_FILE:
        return FLIP_VARTHETA

    # Protect stat/read/update with a lock to avoid races across threads
    with _flip_lock:
        try:
            st = os.stat(FLIP_VARTHETA_FILE)
        except FileNotFoundError:
            return FLIP_VARTHETA
        except Exception as e:
            log.warning(
                "Error stat() on %s: %s; using env FLIP_VARTHETA=%s",
                FLIP_VARTHETA_FILE,
                e,
                FLIP_VARTHETA,
            )
            return FLIP_VARTHETA

        if _flip_file_cached_mtime is None or st.st_mtime != _flip_file_cached_mtime:
            # update mtime and cached value under the same lock
            try:
                with open(FLIP_VARTHETA_FILE, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                if raw == "" or raw.lower() == "none":
                    _flip_file_cached_value = None
                else:
                    _flip_file_cached_value = float(raw)
                _flip_file_cached_mtime = st.st_mtime
                log.info(
                    "FLIP vartheta updated from file %s: %s",
                    FLIP_VARTHETA_FILE,
                    _flip_file_cached_value,
                )
            except Exception as e:
                log.warning(
                    "Error reading/parsing %s: %s; keeping previous value %s "
                    "or env FLIP_VARTHETA=%s",
                    FLIP_VARTHETA_FILE,
                    e,
                    _flip_file_cached_value,
                    FLIP_VARTHETA,
                )

    if _flip_file_cached_value is not None:
        return _flip_file_cached_value
    return FLIP_VARTHETA


def _floor_hidden_states(x: torch.Tensor, t: Optional[float]) -> torch.Tensor:
    """
    Flooring operator:
        h' = max(h, t)
    """
    if t is None:
        return x
    return x.clamp(min=t)


def _compute_am_stats_tokens(h: torch.Tensor, t: float):
    """
    Compute per-token A_{ℓ,i}(ϑ) and M_{ℓ,i}(ϑ) from Section 5.1.

        A_{ℓ,i}(ϑ) = (1/d) Σ_k 1{z_ik < ϑ}   — fraction of dims clamped, token i
        M_{ℓ,i}(ϑ) = (1/d) Σ_k (ϑ − z_ik)₊   — mean clamp magnitude, token i

    h: [..., d] — last dim is hidden dimension; any leading dims are token positions.
    Returns (A_vals, M_vals): Python lists, one float per token position.
    """
    h_f = h.detach().float()
    if h_f.dim() == 1:
        h_f = h_f.unsqueeze(0)
    A_vals = (h_f < t).float().mean(dim=-1).tolist()
    M_vals = (t - h_f).clamp(min=0).mean(dim=-1).tolist()
    return A_vals, M_vals


# Keyed by (site_label, vartheta_str) so each ϑ value gets its own bucket.
# Example keys: ("Final", "-2.0"), ("Layer 8", "-0.5")
_stats_accum: dict = {}
_stats_accum_lock = threading.Lock()


def _accumulate_stats(label: str, t: float, A_vals: list, M_vals: list) -> int:
    """Extend per-token accumulators for (site, vartheta); return call count."""
    key = (label, str(t))
    with _stats_accum_lock:
        if key not in _stats_accum:
            _stats_accum[key] = {"A": [], "M": [], "n_calls": 0}
        _stats_accum[key]["A"].extend(A_vals)
        _stats_accum[key]["M"].extend(M_vals)
        _stats_accum[key]["n_calls"] += 1
        return _stats_accum[key]["n_calls"]


def _log_stats_summary(label: Optional[str] = None) -> None:
    """
    Log mean ± std of accumulated A_{ℓ,i} and M_{ℓ,i} values.

    Reports E_{samples,t}[A_{ℓ,t}(ϑ)] and E_{samples,t}[M_{ℓ,t}(ϑ)] with
    standard deviation, matching the paper's appendix table format.
    Pass label=None to log all (site, vartheta) buckets.
    """
    with _stats_accum_lock:
        if label is not None:
            snapshot = [((lbl, thr), dict(v))
                        for (lbl, thr), v in _stats_accum.items() if lbl == label]
        else:
            snapshot = [((lbl, thr), dict(v)) for (lbl, thr), v in _stats_accum.items()]
    for (lbl, thr), data in snapshot:
        A_list = data["A"]
        M_list = data["M"]
        n_calls = data["n_calls"]
        if not A_list:
            continue
        n = len(A_list)
        A_t = torch.tensor(A_list)
        M_t = torch.tensor(M_list)
        log.info(
            "[FLIP stats] %s | vartheta=%s n_tokens=%d n_calls=%d | "
            "A(ϑ)=%.4f±%.4f  M(ϑ)=%.6f±%.6f",
            lbl, thr, n, n_calls,
            A_t.mean().item(), A_t.std().item() if n > 1 else 0.0,
            M_t.mean().item(), M_t.std().item() if n > 1 else 0.0,
        )


def _write_stats_csv() -> None:
    """
    Write accumulated A and M stats to FLIP_STATS_CSV, one row per (ϑ, site).

    CSV columns: vartheta, site, n_tokens, n_calls, A_mean, A_std, M_mean, M_std

    Safe to call repeatedly; overwrites the file each time so the CSV is always
    up-to-date even if the server is still running during a long sweep.
    """
    if not FLIP_STATS_CSV:
        return
    with _stats_accum_lock:
        snapshot = [((lbl, thr), dict(v)) for (lbl, thr), v in _stats_accum.items()]
    if not snapshot:
        return

    def _thr_sort_key(thr_str: str) -> float:
        try:
            return float(thr_str)
        except (ValueError, TypeError):
            return float("inf")

    sorted_rows = sorted(snapshot, key=lambda x: (_thr_sort_key(x[0][1]), x[0][0]))

    import csv as _csv
    try:
        os.makedirs(os.path.dirname(os.path.abspath(FLIP_STATS_CSV)) or ".", exist_ok=True)
        with open(FLIP_STATS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(["vartheta", "site", "n_tokens", "n_calls",
                             "A_mean", "A_std", "M_mean", "M_std"])
            for (lbl, thr), data in sorted_rows:
                A_list = data["A"]
                M_list = data["M"]
                if not A_list:
                    continue
                n = len(A_list)
                A_t = torch.tensor(A_list)
                M_t = torch.tensor(M_list)
                writer.writerow([
                    thr, lbl, n, data["n_calls"],
                    f"{A_t.mean().item():.6f}",
                    f"{A_t.std().item():.6f}" if n > 1 else "0.000000",
                    f"{M_t.mean().item():.8f}",
                    f"{M_t.std().item():.8f}" if n > 1 else "0.00000000",
                ])
        log.info("[FLIP stats] CSV written → %s", FLIP_STATS_CSV)
    except Exception as e:
        log.warning("[FLIP stats] Failed to write CSV to %s: %s", FLIP_STATS_CSV, e)


def _permute_and_floor_hidden_states(
    x: torch.Tensor,
    t: float,
    lam: float,
    seed: int,
) -> torch.Tensor:
    """
    Permute a random fraction `lam` of hidden-state dimensions among themselves,
    then apply the FLIP flooring vartheta t.

        1. Select k = ceil(lam * D) dimension indices at random.
        2. Shuffle those k values among the k positions.
        3. Apply h' = max(h_permuted, t).

    The permutation is deterministic given seed and tensor shape. We derive it
    via argsort of integer hash values rather than torch.Generator, which is a
    C++ builtin that torch.compile/dynamo cannot trace.
    """
    last_dim = x.shape[-1]
    k = max(1, int(lam * last_dim))

    # Hash-based deterministic permutation — fully traceable by torch.compile.
    # Multiplier 1664525 is the Numerical Recipes LCG constant; fits in int32
    # so Triton does not reject it as an out-of-range scalar.
    arange_d = torch.arange(last_dim, device=x.device, dtype=torch.int64)
    h_d = (arange_d * 1664525 + seed) & 0x7FFFFFFF
    selected = torch.argsort(h_d)[:k]

    arange_k = torch.arange(k, device=x.device, dtype=torch.int64)
    h_k = (arange_k * 1664525 + (seed + 1)) & 0x7FFFFFFF
    perm_of_selected = torch.argsort(h_k)

    x2 = x.clone()
    x2[..., selected] = x[..., selected[perm_of_selected]]

    t_tensor = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    return torch.maximum(x2, t_tensor)


def _parse_layer_spec(spec: Optional[str], num_layers: int) -> Set[int]:
    """
    Parse FLIP_LAYER_INDICES into a set of layer indices.

    Supported forms:
      all
      28
      28,29,31
      24:32     -> [24, 25, ..., 31]
      28:       -> [28, ..., num_layers-1]
      :8        -> [0, ..., 7]
    """
    if spec is None or spec.strip() == "":
        return set()

    txt = spec.strip().lower()
    if txt == "all":
        return set(range(num_layers))

    out = set()
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            a, b = p.split(":", 1)
            start = int(a) if a != "" else 0
            end = int(b) if b != "" else num_layers
            start = max(0, start)
            end = min(num_layers, end)
            if start < end:
                out.update(range(start, end))
        else:
            idx = int(p)
            if idx < 0:
                idx = num_layers + idx
            if 0 <= idx < num_layers:
                out.add(idx)

    return out


def _get_decoder_layers(model) -> Optional[list]:
    """
    Try common Qwen3-VL/vLLM paths for decoder layers.
    If direct paths fail, fall back to a bounded attribute scan and prefer
    the list whose length matches text_config.num_hidden_layers.
    """
    expected_num_layers = None
    try:
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "num_hidden_layers"):
            expected_num_layers = int(text_cfg.num_hidden_layers)
        elif cfg is not None and hasattr(cfg, "num_hidden_layers"):
            expected_num_layers = int(cfg.num_hidden_layers)
    except Exception:
        expected_num_layers = None

    candidates = [
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
        ("language_model.model.model.layers", lambda m: m.language_model.model.model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.layers", lambda m: m.model.layers),
        ("layers", lambda m: m.layers),
    ]
    for name, getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                if expected_num_layers is not None and len(layers) != expected_num_layers:
                    log.warning(
                        "Resolved layers via %s but len=%d != expected decoder layers=%d",
                        name,
                        len(layers),
                        expected_num_layers,
                    )
                log.info("Resolved decoder layers via path: %s (num_layers=%d)", name, len(layers))
                return layers
        except Exception:
            pass

    # Fallback scan: breadth-first over object attributes (bounded), looking
    # for `.layers`-like containers of modules.
    best = None
    best_name = None
    visited = set()
    q = deque([("self", model, 0)])
    max_depth = 4
    max_nodes = 300
    nodes = 0

    while q and nodes < max_nodes:
        path, obj, depth = q.popleft()
        nodes += 1
        oid = id(obj)
        if oid in visited:
            continue
        visited.add(oid)

        if depth > max_depth:
            continue

        # Probe obvious list containers under this object.
        for attr_name in ("layers", "blocks", "h"):
            try:
                val = getattr(obj, attr_name, None)
                if val is None:
                    continue
                n = len(val)
                if n <= 0:
                    continue
                # Prefer exact decoder-depth match when available.
                if expected_num_layers is not None and n == expected_num_layers:
                    log.info(
                        "Resolved decoder layers via fallback path: %s.%s (num_layers=%d)",
                        path,
                        attr_name,
                        n,
                    )
                    return val
                if best is None or n > len(best):
                    best = val
                    best_name = f"{path}.{attr_name}"
            except Exception:
                pass

        # Continue BFS through public attributes.
        try:
            obj_dict = vars(obj)
        except Exception:
            obj_dict = {}
        for k, v in obj_dict.items():
            if k.startswith("_"):
                continue
            if v is None:
                continue
            # Skip simple scalars
            if isinstance(v, (str, int, float, bool, bytes)):
                continue
            q.append((f"{path}.{k}", v, depth + 1))

    if best is not None:
        log.warning(
            "Using fallback layer container %s (num_layers=%d); expected=%s",
            best_name,
            len(best),
            expected_num_layers,
        )
        return best

    log.warning("Could not resolve decoder layers on Qwen3VLForConditionalGeneration instance")
    return None


def _wrap_layer_forward(layer, layer_idx: int):
    """
    Wrap a single decoder layer so we can floor its output hidden states
    during the forward pass.
    """
    if getattr(layer, "_flip_forward_wrapped", False):
        return

    original_forward = layer.forward

    def _log_floor_applied_once(vartheta: float) -> None:
        if not FLIP_LOG_LAYER_FLOOR_ONCE:
            return
        if getattr(layer, "_flip_floor_log_emitted", False):
            return
        layer._flip_floor_log_emitted = True
        log.info(
            "FLIP floor applied on decoder layer %d (vartheta=%s)",
            layer_idx,
            vartheta,
        )

    def wrapped_forward(_self, *args, **kwargs):
        # `original_forward` is already bound to `layer`, so forward only
        # the real layer inputs (exclude `_self` from MethodType binding).
        out = original_forward(*args, **kwargs)

        target_layers = getattr(layer, "_flip_target_layers", None)
        if target_layers is None or layer_idx not in target_layers:
            return out

        # Avoid lock/file I/O in TorchDynamo compiled regions.
        try:
            is_compiling = torch.compiler.is_compiling()
        except Exception:
            try:
                is_compiling = torch._dynamo.is_compiling()
            except Exception:
                is_compiling = False

        if is_compiling:
            t = _flip_file_cached_value if _flip_file_cached_value is not None else FLIP_VARTHETA
        else:
            t = _get_flip_vartheta()
        if t is None:
            return out

        # Decoder blocks usually return either:
        #   hidden_states (Tensor)
        #   (hidden_states, ...)
        # or an object with attributes like `last_hidden_state` or `hidden_states`.
        should_log_floor = not is_compiling

        lam = FLIP_PERMUTE_FRACTION
        seed = FLIP_PERMUTE_SEED if FLIP_PERMUTE_SEED is not None else 1234

        def _apply(h: torch.Tensor) -> torch.Tensor:
            if FLIP_LOG_STATS and should_log_floor:
                A_vals, M_vals = _compute_am_stats_tokens(h, t)
                n_calls = _accumulate_stats(f"Layer {layer_idx}", t, A_vals, M_vals)
                if n_calls == 1 or n_calls % FLIP_LOG_STATS_INTERVAL == 0:
                    _log_stats_summary(f"Layer {layer_idx}")
                    _write_stats_csv()
            if lam is not None:
                return _permute_and_floor_hidden_states(h, t, float(lam), seed)
            return _floor_hidden_states(h, t)

        if isinstance(out, torch.Tensor):
            if should_log_floor:
                _log_floor_applied_once(t)
            return _apply(out)
        elif isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            floored = _apply(out[0])
            if should_log_floor:
                _log_floor_applied_once(t)
            return (floored,) + out[1:]
        else:
            for attr in ("last_hidden_state", "hidden_states"):
                if hasattr(out, attr):
                    try:
                        val = getattr(out, attr)
                        if isinstance(val, torch.Tensor):
                            floored = _apply(val)
                            try:
                                setattr(out, attr, floored)
                                if should_log_floor:
                                    _log_floor_applied_once(t)
                                return out
                            except Exception:
                                return out
                    except Exception:
                        pass
            return out

    # Bind the wrapper as a method on the instance to preserve method semantics
    layer.forward = types.MethodType(wrapped_forward, layer)
    layer._flip_forward_wrapped = True
    layer._flip_original_forward = original_forward
    log.info("Wrapped decoder layer %d for FLIP flooring", layer_idx)


def _install_layerwise_flooring(self):
    """
    Resolve decoder layers and wrap selected layers exactly once per model instance.
    """
    if getattr(self, "_flip_layers_installed", False):
        return

    layers = _get_decoder_layers(self)
    if not layers:
        log.warning("FLIP: no decoder layers resolved; layerwise flooring not installed")
        return

    target_layers = _parse_layer_spec(FLIP_LAYER_INDICES_ENV, len(layers))
    self._flip_target_layers = target_layers

    for i, layer in enumerate(layers):
        # Only wrap decoder layers that are explicitly targeted. Wrapping
        # every layer (even when not targeted) can change execution
        # characteristics; limit wrapping to the requested indices so
        # behaviour is unchanged when `FLIP_LAYER_INDICES` is empty
        # or when the vartheta is None.
        if i in target_layers:
            _wrap_layer_forward(layer, i)
        # Always record the target set on the layer for the wrapper to
        # consult if present.
        layer._flip_target_layers = target_layers

    self._flip_layers_installed = True
    log.info("Installed FLIP layerwise flooring on layers=%s", sorted(target_layers))


# Conv3d / F.conv3d fallbacks to avoid GET on depth=1 (image) inputs and disable cuDNN otherwise
_original_conv3d_forward = nn.Conv3d.forward
_original_f_conv3d = F.conv3d
_conv3d_fallback_warned = False
_f_conv3d_warned = False


def _to_hw(arg):
    if isinstance(arg, tuple):
        return arg[-2:] if len(arg) == 3 else arg
    return (arg, arg)


def _conv2d_from_depth1(input5d, weight5d, bias, stride, padding, dilation, groups, log_once_fn):
    b, c, _, h, w = input5d.shape
    input2d = input5d.view(b, c, h, w)
    weight2d = weight5d.sum(dim=2)
    stride_hw = _to_hw(stride)
    pad_hw = _to_hw(padding)
    dil_hw = _to_hw(dilation)
    out2d = F.conv2d(input2d, weight2d, bias, stride_hw, pad_hw, dil_hw, groups)
    return out2d.unsqueeze(2)


def _f_conv3d_with_image_fallback(input, weight, bias=None,
                                  stride=1, padding=0, dilation=1, groups=1):
    global _f_conv3d_warned
    if (
        isinstance(input, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and input.ndim == 5
        and input.shape[2] == 1
        and (isinstance(stride, int) or stride[0] == 1)
        and (isinstance(dilation, int) or dilation[0] == 1)
    ):
        if not _f_conv3d_warned:
            log.info("Intercepting F.conv3d depth=1; routing to conv2d (depth-summed) to avoid GET")
            _f_conv3d_warned = True
        with cudnn.flags(enabled=False):
            return _conv2d_from_depth1(input, weight, bias, stride, padding, dilation, groups, log.info)
    with cudnn.flags(enabled=False):
        return _original_f_conv3d(input, weight, bias, stride, padding, dilation, groups)


def _conv3d_with_image_fallback(self, input, *args, **kwargs):
    global _conv3d_fallback_warned
    if (
        isinstance(input, torch.Tensor)
        and input.ndim == 5
        and input.shape[2] == 1
        and self.stride[0] == 1
        and self.dilation[0] == 1
    ):
        if not _conv3d_fallback_warned:
            log.info("Conv3d depth=1 rerouted to conv2d (depth-summed) to avoid GET")
            _conv3d_fallback_warned = True
        weight2d = self.weight.sum(dim=2)
        b, c, _, h, w = input.shape
        x2d = input.view(b, c, h, w)
        stride_hw = (self.stride[1], self.stride[2])
        pad_hw = (self.padding[1], self.padding[2])
        dil_hw = (self.dilation[1], self.dilation[2])
        with cudnn.flags(enabled=False):
            out2d = F.conv2d(x2d, weight2d, self.bias, stride_hw, pad_hw, dil_hw, self.groups)
        return out2d.unsqueeze(2)
    with cudnn.flags(enabled=False):
        return _original_conv3d_forward(self, input, *args, **kwargs)


F.conv3d = _f_conv3d_with_image_fallback
nn.Conv3d.forward = _conv3d_with_image_fallback

# Save original implementations so we can delegate
_original_init = Qwen3VLForConditionalGeneration.__init__
_original_compute_logits = Qwen3VLForConditionalGeneration.compute_logits
_original_process_video_input = Qwen3VLForConditionalGeneration._process_video_input


@functools.wraps(_original_init)
def _flip_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    try:
        _install_layerwise_flooring(self)
    except Exception as e:
        log.exception("Failed to install FLIP layerwise flooring: %s", e)


# Keep an explicit signature so vLLM init-arg inference sees keyword-only args
# like `vllm_config` when inspecting patched model classes.
try:
    _flip_init.__signature__ = inspect.signature(_original_init)
except Exception:
    pass


def _flip_compute_logits(self, hidden_states, *args, **kwargs):
    """
    Optional final pre-logit FLIP.  Enabled with FLIP_APPLY_ON_FINAL=1.

    If FLIP_PERMUTE_FRACTION is set, applies permutation+flooring:
        1. Shuffle a random fraction of hidden-state dimensions.
        2. Apply h' = max(h_permuted, vartheta).
    Otherwise applies plain flooring:
        h' = max(h, vartheta)

    Stats note: hidden_states here contains only logit-readout positions (vLLM
    extracts them before calling compute_logits), so A and M accumulation at
    this site is exact over T = {logit-readout token positions}.
    """
    t = _get_flip_vartheta()
    # Stats are independent of FLIP_APPLY_ON_FINAL: measure A and M at the Final
    # site whenever FLIP_LOG_STATS=1 and a vartheta is set, regardless of whether
    # the floor intervention is active here.
    if FLIP_LOG_STATS and t is not None:
        A_vals, M_vals = _compute_am_stats_tokens(hidden_states, t)
        n_calls = _accumulate_stats("Final", t, A_vals, M_vals)
        if n_calls == 1 or n_calls % FLIP_LOG_STATS_INTERVAL == 0:
            _log_stats_summary("Final")
            _write_stats_csv()
    if FLIP_APPLY_ON_FINAL and t is not None:
        lam = FLIP_PERMUTE_FRACTION
        if lam is not None:
            seed = FLIP_PERMUTE_SEED if FLIP_PERMUTE_SEED is not None else 1234
            hidden_states = _permute_and_floor_hidden_states(
                hidden_states, t, float(lam), seed
            )
        else:
            hidden_states = _floor_hidden_states(hidden_states, t)
    return _original_compute_logits(self, hidden_states, *args, **kwargs)


_video_warning_emitted = False


def _maybe_disable_video_input(self, multimodal_input, *args, **kwargs):
    global _video_warning_emitted
    if DISABLE_QWEN3_VIDEO:
        if not _video_warning_emitted:
            log.info(
                "Skipping Qwen3 video embeddings because QWEN3_DISABLE_VIDEO=%s",
                os.environ.get("QWEN3_DISABLE_VIDEO"),
            )
            _video_warning_emitted = True
        return None
    return _original_process_video_input(self, multimodal_input, *args, **kwargs)


Qwen3VLForConditionalGeneration.__init__ = _flip_init
Qwen3VLForConditionalGeneration.compute_logits = _flip_compute_logits
Qwen3VLForConditionalGeneration._process_video_input = _maybe_disable_video_input

log.info(
    "Patched Qwen3VLForConditionalGeneration: vartheta=%s (file=%s), "
    "layer_indices=%s, final_flooring=%s, permute_fraction=%s, permute_seed=%s, "
    "video inputs %s",
    _get_flip_vartheta(),
    FLIP_VARTHETA_FILE,
    FLIP_LAYER_INDICES_ENV,
    FLIP_APPLY_ON_FINAL,
    FLIP_PERMUTE_FRACTION,
    FLIP_PERMUTE_SEED,
    "disabled" if DISABLE_QWEN3_VIDEO else "enabled",
)

if FLIP_LOG_STATS:
    import atexit
    atexit.register(_log_stats_summary)
    atexit.register(_write_stats_csv)
    log.info(
        "FLIP stats accumulation enabled; summary every %d calls + CSV+log at exit "
        "(FLIP_STATS_CSV=%s)",
        FLIP_LOG_STATS_INTERVAL,
        FLIP_STATS_CSV or "<not set — CSV output disabled>",
    )