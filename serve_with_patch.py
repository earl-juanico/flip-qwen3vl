#!/usr/bin/env python3
"""
serve_with_patch.py

Starts a vLLM OpenAI-compatible API server with the FLIP monkeypatch applied
(patch_qwen.py loaded in-process before vLLM initialises its model).  Also
launches a lightweight Python HTTP server alongside vLLM so that image URLs
of the form http://127.0.0.1:<http_port>/data/coco/val2017/<file>.jpg resolve
correctly during inference.

── QUICK START ──────────────────────────────────────────────────────────────

  python3 serve_with_patch.py

All settings are controlled through environment variables — no CLI flags are
needed for a typical run.

── REQUIRED ─────────────────────────────────────────────────────────────────

  MODEL_PATH   Path to the model weights directory (must contain config.json).
               There is no globally installed default; the directory
               <repo_root>/model/Qwen3-VL-4B-Instruct is checked first, but
               you almost certainly need to set this explicitly.

               Examples:
                 # 4B model at a custom location
                 export MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct

                 # Switch to the 8B variant
                 export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct

               The script exits immediately with a clear error if the
               directory does not exist.

── OPTIONAL ─────────────────────────────────────────────────────────────────

  FLIP_VARTHETA_FILE
               Path to the plain-text file that the FLIP patch reads at
               inference time to determine the vartheta scalar.  The file
               contains a single line: either a float (e.g. -2.0) or the
               word 'none' to disable the perturbation.

               Default: <repo_root>/config/vartheta.txt

               probe_and_sweep.py writes this file automatically for each
               sweep iteration, so you only need to set it manually when
               running the server standalone or with a non-default config
               directory.

               Example:
                 export FLIP_VARTHETA_FILE=/path/to/config10/vartheta.txt

  MEDIA_PATH   Root directory served by the lightweight HTTP image server.
               Image URLs sent to the vLLM endpoint are resolved relative to
               this directory.  Must be an ancestor of the COCO image files.

               Default: <repo_root>   (images are then reachable as
               http://127.0.0.1:<http_port>/data/coco/val2017/<file>.jpg)

               Override only when your dataset lives outside the repo tree:
                 export MEDIA_PATH=/mnt/datasets

               The HTTP server port is derived from VLLM_PORT by replacing
               the leading '8' with '9' (e.g. vLLM on 8010 → HTTP on 9010).

  VLLM_PORT / PORT
               Port for the vLLM API server.  Default: 8008.
                 export VLLM_PORT=8020

  VLLM_API_KEY          API key checked by vLLM.  Default: foobar.
  MAX_MODEL_LEN         Maximum sequence length.  Default: 4096.
  VLLM_GPU_MEMORY_UTILIZATION
                        Fraction of GPU memory to use.  Default: 0.9.
  QWEN3_DISABLE_VIDEO   Set to 1 to tell vLLM to ignore video inputs.
  HFTOKEN_FILE          Path to a file containing your HuggingFace token.
                        Default: <repo_root>/HFTOKEN.txt.
  FLIP_ENFORCE_EAGER    Set to 1 to pass --enforce-eager to vLLM, disabling
                        CUDA graph capture.  Required when FLIP_LAYER_INDICES
                        is set with a dynamic vartheta file (FLIP_VARTHETA_FILE)
                        because vLLM bakes the floor vartheta into the CUDA graph
                        at capture time — if the vartheta later changes across a
                        sweep, the captured graph keeps the old value and the
                        intervention is silently wrong for all decode steps.
                        FLIP_LOG_STATS layer-level stats also depend on this flag
                        (Python stats code does not run during CUDA graph replay).
                        Without FLIP_LAYER_INDICES set, CUDA graphs are safe.
  FLIP_STATS_CSV        Path for the temporary per-sweep stats CSV written by
                        the server (FLIP_LOG_STATS=1).  sweep_leftright.py copies
                        this into --output-dir after the sweep completes.
                        Example: FLIP_STATS_CSV=/tmp/flip_am_stats.csv
"""
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import logging
import subprocess
import atexit
from importlib.metadata import entry_points

# Basic logging setup
log = logging.getLogger("serve_with_patch")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    log.addHandler(handler)
log.setLevel(logging.INFO)

# Default FLIP vartheta config path relative to the repo root (can be overridden by env)
os.environ.setdefault(
    "FLIP_VARTHETA_FILE",
    str(Path(__file__).resolve().parent / "config" / "vartheta.txt"),
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Only load the patch, do NOT load a custom architecture
patch_path = os.path.join(os.path.dirname(__file__), "patch_qwen.py")
load("patch_qwen", patch_path)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def main():
    _default_model = Path(__file__).resolve().parent / "model" / "Qwen3-VL-4B-Instruct"
    model_dir = os.environ.get("MODEL_PATH", str(_default_model))

    if not os.path.isdir(model_dir):
        log.error(
            "Model directory not found: %s\n"
            "  Set MODEL_PATH to the directory containing the model weights, e.g.:\n"
            "    export MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct",
            model_dir,
        )
        sys.exit(1)

    # Load HuggingFace token from file if not already provided in env.
    # File path can be overridden by HFTOKEN_FILE; default is 'HFTOKEN.txt' next to this script.
    if not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        hftoken_file = os.environ.get("HFTOKEN_FILE", os.path.join(os.path.dirname(__file__), "HFTOKEN.txt"))
        try:
            if os.path.isfile(hftoken_file):
                with open(hftoken_file, "r", encoding="utf-8") as f:
                    token = f.read().strip()
                if token:
                    os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                    log.info("Loaded HUGGINGFACE_HUB_TOKEN from %s", hftoken_file)
                else:
                    log.info("HFTOKEN file %s is empty; not setting HUGGINGFACE_HUB_TOKEN", hftoken_file)
            else:
                log.debug("HFTOKEN file %s not found; skipping", hftoken_file)
        except Exception as e:
            log.warning("Failed to read HFTOKEN file %s: %s", hftoken_file, e)

    # Pass MODEL_PATH through directly; let vLLM interpret directory vs file
    model_path = model_dir
    api_key = os.environ.get("VLLM_API_KEY", "foobar")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "4096")

    raw_gpu_env = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
    gpu_mem_util = raw_gpu_env if raw_gpu_env is not None else "0.9"

    if raw_gpu_env is None:
        log.info(
            "VLLM_GPU_MEMORY_UTILIZATION not set in env; using default %s",
            gpu_mem_util,
        )
    else:
        log.info(
            "VLLM_GPU_MEMORY_UTILIZATION set from env: %s",
            gpu_mem_util,
        )

    # Port can be configured via env `VLLM_PORT` or `PORT`; default 8008
    port = os.environ.get("VLLM_PORT", os.environ.get("PORT", "8008"))

    # MEDIA_PATH: directory served by the lightweight HTTP server for image URLs.
    # Defaults to the repo root (same convention as probe_and_sweep.py).
    media_path = os.environ.get(
        "MEDIA_PATH",
        str(Path(__file__).resolve().parent),
    )

    # derive http server port by replacing leading '8' with '9' when present
    def _derive_http_port(p: str) -> str:
        if p and len(p) >= 1 and p[0] == "8":
            return "9" + p[1:]
        try:
            return str(int(p) + 1000)
        except Exception:
            return "9000"

    http_port = _derive_http_port(str(port))

    # Start a simple HTTP server to serve MEDIA_PATH so image_url requests work locally.
    http_proc = None
    try:
        if os.path.isdir(media_path):
            http_proc = subprocess.Popen(
                [sys.executable, "-m", "http.server", str(http_port), "--bind", "0.0.0.0"],
                cwd=media_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            log.info("Started local http.server serving %s on port %s (pid=%s)", media_path, http_port, getattr(http_proc, "pid", None))
        else:
            log.warning("MEDIA_PATH %s does not exist or is not a directory; skipping http server", media_path)
    except Exception as e:
        log.warning("Failed to start http.server for %s on port %s: %s", media_path, http_port, e)

    # Ensure we terminate the http server on exit
    def _cleanup():
        try:
            if http_proc is not None:
                http_proc.terminate()
                http_proc.wait(timeout=5)
                log.info("Stopped http.server (pid=%s)", getattr(http_proc, "pid", None))
        except Exception:
            pass

    atexit.register(_cleanup)

    # Build argv to emulate the vllm serve CLI
    argv = [
        "vllm",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "1",
        "--allowed-local-media-path",
        media_path,
        "--api-key",
        api_key,
        "--tensor-parallel-size",
        "1",#"2",
    ]

    # --enforce-eager disables vLLM's CUDA graph capture so that Python layer
    # wrappers execute on every token.  This is required whenever
    # FLIP_LAYER_INDICES is set with a dynamic vartheta (FLIP_VARTHETA_FILE),
    # because vLLM bakes the floor vartheta into the CUDA graph at capture
    # time — if the vartheta later changes (e.g. across a vartheta sweep),
    # the captured graph keeps the old value and the intervention is silently
    # wrong for all decode steps.  FLIP_LOG_STATS layer-level stats also depend
    # on this flag for the same reason (Python stats code does not run during
    # CUDA graph replay).  Without FLIP_LAYER_INDICES set, CUDA graphs are safe.
    flip_enforce_eager = _parse_bool_env("FLIP_ENFORCE_EAGER", default=False)
    if flip_enforce_eager:
        log.info("FLIP_ENFORCE_EAGER=1: adding --enforce-eager (disables CUDA graphs; "
                 "slower decode but required for layer-wise stats collection)")
        argv.append("--enforce-eager")

    # If QWEN3_DISABLE_VIDEO is true, let vLLM ignore video inputs
    if _parse_bool_env("QWEN3_DISABLE_VIDEO", default=False):
        log.info("QWEN3_DISABLE_VIDEO=true; adding --limit-mm-per-prompt video=0")
        argv.extend(
            [
                "--limit-mm-per-prompt",
                "video=0",
            ]
        )

    sys.argv[:] = argv

    # Prefer invoking the installed `vllm` console-script entrypoint in-process
    # so that the previously-loaded `patch_qwen` remains active.
    try:
        for ep in entry_points(group="console_scripts"):
            if getattr(ep, "name", None) == "vllm":
                func = ep.load()
                log.info("Launching vllm via console-script entrypoint in-process")
                try:
                    func()
                except SystemExit:
                    pass
                break
        else:
            # If console script not found, try running module directly
            log.info("vllm console-script not found; trying run_module fallback")
            try:
                runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
            except Exception:
                # final fallback: invoke installed CLI subprocess
                log.info("run_module fallback failed; invoking vllm as subprocess")
                subprocess.run(["vllm"] + sys.argv[1:], check=True)
    except Exception as e:
        log.exception("Failed to start vllm: %s", e)


if __name__ == "__main__":
    main()
