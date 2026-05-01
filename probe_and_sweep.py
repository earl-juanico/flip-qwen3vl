#!/usr/bin/env python3
"""
probe_and_sweep.py

Sweeps across a list of FLIP vartheta values, running the full synchronized set of
query jobs (detect, reason, indout) across all bootstrap batch sizes (10, 20, 40, 60, 80,
and optionally 100 for the full dataset) for each value. All runs for a given vartheta
value launch in parallel; the script waits for all to finish before advancing to the next
value.

The sample name (e.g. s00, s01) is configurable via --sample and is substituted into
both the input questions filename and the output JSONL filename. When --pct 100 is used,
--sample defaults to s00 and does not need to be specified.

When pct=100 (full dataset, no bootstrap), the output directory omits the 'bootstrap_100'
infix: answers/answers_{task}_clustered instead of answers/answers_bootstrap_100_{task}_clustered.

Usage examples:
  python3 probe_and_sweep.py                                  # default vartheta list, s00
  python3 probe_and_sweep.py --sample s01                     # default vartheta list, custom sample
  python3 probe_and_sweep.py --sample s01 -- 0.2 -0.5 -1.0 none
  python3 probe_and_sweep.py --pct 20 40 -- 0.0 0.2 -0.5
  python3 probe_and_sweep.py --pct 20 40 --sample s01 -- 0.0 0.2 -0.5
  python3 probe_and_sweep.py --pct 100                        # full dataset, s00 (default)
  python3 probe_and_sweep.py --pct 100 -- 0.0 -0.5 none      # full dataset, specific vartheta values
  python3 probe_and_sweep.py --pct 100 --dry-run -- 0.0 -0.5 # dry-run for full dataset
  python3 probe_and_sweep.py --dry-run --sample s01           # preview default vartheta list, custom sample
  python3 probe_and_sweep.py --dry-run --sample s01 -- 0.2 -0.5
  python3 probe_and_sweep.py --port 20:8021                   # override port for pct=20
  python3 probe_and_sweep.py --port 20:8021 --port 60:8061    # override multiple ports
  python3 probe_and_sweep.py --port 100:8100                  # override port for pct=100
  python3 probe_and_sweep.py --pct 20 40 --port 20:8021 --dry-run -- 0.0 -0.5
  python3 probe_and_sweep.py --config-file 20:/path/to/config20_s1/vartheta.txt
  python3 probe_and_sweep.py --port 20:8120 --config-file 20:/path/to/config20_s1/vartheta.txt
  python3 probe_and_sweep.py --model-path /path/to/model      # override model path
  python3 probe_and_sweep.py --model-path /path/to/model --sample s01 -- 0.0 0.2 -0.5
  python3 probe_and_sweep.py --output-suffix _v2              # write to *_clustered_v2/ dirs
  python3 probe_and_sweep.py --pct 100 --output-suffix _v2 -- 0.0 -0.5 none
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — derived from the location of this script (repo root)
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "model" / "Qwen3-VL-4B-Instruct")
IMAGE_DIR  = str(BASE_DIR / "data/coco/val2017")
MEDIA_PATH = str(BASE_DIR)          # HTTP server root; port = 9{pct}
MAX_TOKENS = 512

BATCH_CONFIGS = [
    {"pct": 10,  "port": 8010,  "config_file": BASE_DIR / "config10"  / "vartheta.txt"},
    {"pct": 20,  "port": 8020,  "config_file": BASE_DIR / "config20"  / "vartheta.txt"},
    {"pct": 40,  "port": 8040,  "config_file": BASE_DIR / "config40"  / "vartheta.txt"},
    {"pct": 60,  "port": 8060,  "config_file": BASE_DIR / "config60"  / "vartheta.txt"},
    {"pct": 80,  "port": 8080,  "config_file": BASE_DIR / "config80"  / "vartheta.txt"},
    {"pct": 100, "port": 8100,  "config_file": BASE_DIR / "config100" / "vartheta.txt"},
]

TASKS = [
    {"name": "detect", "script": "infer/query_detect.py"},
    {"name": "reason", "script": "infer/query_reason.py"},
    {"name": "indout", "script": "infer/query_indout.py"},
]

# Default sweep list used when no vartheta values are given on the command line
DEFAULT_VARTHETA_VALUES = [
    "none", 
    "0.0", "0.1", "0.2", "0.4", "0.5", "1.0",
    "-0.2", "-0.3", "-0.5", "-1.0", "-1.5", "-2.0", "-2.5", "-4.0", "-5.0", "-50.0",
]


def _vartheta_tag(vartheta_str: str) -> str:
    """Return the filename-safe tag for an VARTHETA value (mirrors the existing convention).

    Numeric strings are normalised through float so that '-1' becomes '-1.0'
    to match the output filenames already produced by earlier runs.
    """
    s = vartheta_str.strip()
    if s.lower() == "none":
        return "none"
    try:
        return str(float(s))
    except ValueError:
        return s


def _write_vartheta_config(config_file: Path, vartheta_str: str) -> None:
    tmp = str(config_file) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(vartheta_str.strip() + "\n")
    os.replace(tmp, str(config_file))
    print(f"  FLIP_VARTHETA_FILE <- {vartheta_str!r}  ({config_file})")


def _build_cmd(pct: int, port: int, config_file: Path, task: dict, vartheta_str: str, tag: str, sample: str, model_path: str = MODEL_PATH, output_suffix: str = "") -> list:
    questions = str(
        BASE_DIR / f"questions/question_pct{pct}_{sample}.jsonl"
    )
    if pct == 100:
        out_dir = BASE_DIR / "answers" / f"answers_{task['name']}_clustered{output_suffix}"
    else:
        out_dir = BASE_DIR / "answers" / f"answers_bootstrap_{pct}_{task['name']}_clustered{output_suffix}"
    out_file = str(out_dir / f"Qwen3-VL-4B-Instruct_{sample}_vartheta_{tag}.jsonl")
    script   = str(BASE_DIR / task["script"])

    return [
        sys.executable, script,
        "--model-path", model_path,
        "--questions",  questions,
        "--max-tokens", str(MAX_TOKENS),
        "--image-dir",  IMAGE_DIR,
        "--media-path", MEDIA_PATH,
        "--api-port",   str(port),
        "--config-file",   str(config_file),
        "--flip-vartheta", vartheta_str,
        "--output",     out_file,
    ], out_file, out_dir


def run_sweep(vartheta_values: list, batch_configs: list, sample: str = "s00", dry_run: bool = False, model_path: str = MODEL_PATH, output_suffix: str = "") -> None:
    total = len(vartheta_values)
    for idx, vartheta_str in enumerate(vartheta_values, start=1):
        tag = _vartheta_tag(vartheta_str)
        print(f"\n{'='*64}")
        print(f"[{idx}/{total}] VARTHETA = {vartheta_str!r}  →  filename tag: {tag!r}")
        print(f"{'='*64}")

        # Write the new VARTHETA value to every config file first
        for cfg in batch_configs:
            config_file = cfg["config_file"]
            if dry_run:
                print(f"  DRY-RUN: would write {vartheta_str!r} -> {config_file}")
            else:
                _write_vartheta_config(config_file, vartheta_str)

        # Build all job specs first, then check for conflicting output files before launching
        job_specs: list[tuple[str, int, list, str, Path]] = []
        for cfg in batch_configs:
            pct, port = cfg["pct"], cfg["port"]
            for task in TASKS:
                cmd, out_file, out_dir = _build_cmd(pct, port, cfg["config_file"], task, vartheta_str, tag, sample, model_path, output_suffix)
                label = f"bs{pct}-{task['name']}"
                job_specs.append((label, port, cmd, out_file, out_dir))

        # Abort if the specific output file for this vartheta already exists.
        # The directory may legitimately be non-empty (files from other vartheta
        # values written by earlier iterations of this sweep).
        if not dry_run:
            existing = [
                (label, out_file) for label, _, _, out_file, _ in job_specs
                if Path(out_file).exists()
            ]
            if existing:
                lines = [f"ERROR: output files already exist for VARTHETA={vartheta_str!r}:"]
                for label, f in existing:
                    lines.append(f"  [{label}] {f}")
                    lines.append(f"           Suggestion: delete or rename this file, or rerun with --output-suffix to write to a different directory.")
                raise SystemExit("\n".join(lines))

        # Launch all jobs in parallel
        procs: list[tuple[str, int, subprocess.Popen | None]] = []
        for label, port, cmd, out_file, out_dir in job_specs:
            print(f"  → {label} (port {port}): {out_file}")
            if dry_run:
                print(f"     cmd: {' '.join(cmd)}")
                procs.append((label, port, None))
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                p = subprocess.Popen(cmd, cwd=str(BASE_DIR))
                procs.append((label, port, p))

        if dry_run:
            print("  [dry-run: no processes started]")
            continue

        # Wait for all parallel jobs to finish, printing status every 30 s
        print(f"\n  Waiting for {len(procs)} jobs  [VARTHETA={vartheta_str!r}] …")
        POLL_INTERVAL = 30
        finished: dict[str, int] = {}
        t_start = time.monotonic()
        while True:
            all_done = True
            still_running = []
            for label, port, p in procs:
                if label in finished:
                    continue
                rc = p.poll()
                if rc is None:
                    all_done = False
                    still_running.append(f"{label}(:{port})")
                else:
                    finished[label] = rc
                    status = "done" if rc == 0 else f"FAILED(rc={rc})"
                    elapsed = time.monotonic() - t_start
                    print(f"    [{elapsed:6.0f}s] VARTHETA={vartheta_str!r}  {label} port:{port}  → {status}",
                          flush=True)
            if all_done:
                break
            elapsed = time.monotonic() - t_start
            print(f"    [{elapsed:6.0f}s] VARTHETA={vartheta_str!r}  still running ({len(still_running)}): "
                  f"{', '.join(still_running)}", flush=True)
            time.sleep(POLL_INTERVAL)

        any_failed = any(rc != 0 for rc in finished.values())

        if any_failed:
            print(f"  WARNING: one or more jobs failed for VARTHETA={vartheta_str!r}")
        else:
            print(f"  All jobs for VARTHETA={vartheta_str!r} completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep vartheta values across bootstrap batch sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "vartheta_values",
        nargs="*",
        metavar="VARTHETA",
        help=f"VARTHETA values to sweep. Defaults to: {DEFAULT_VARTHETA_VALUES}",
    )
    parser.add_argument(
        "--pct",
        type=int,
        nargs="+",
        choices=[10, 20, 40, 60, 80, 100],
        metavar="PCT",
        help="Restrict sweep to specific batch-size groups (default: all). "
             "Use 100 for the full dataset (no bootstrap); --sample defaults to s00.",
    )
    parser.add_argument(
        "--sample",
        default="s00",
        metavar="SAMPLE",
        help="Sample name used in questions filename and output filename (default: s00).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and config writes without executing them.",
    )
    parser.add_argument(
        "--port",
        action="append",
        metavar="PCT:PORT",
        dest="port_overrides",
        default=[],
        help="Override the port for a batch size, e.g. --port 20:8021. Repeatable.",
    )
    parser.add_argument(
        "--config-file",
        action="append",
        metavar="PCT:PATH",
        dest="config_file_overrides",
        default=[],
        help="Override the vartheta.txt path for a batch size, "
             "e.g. --config-file 20:/path/to/config20_s1/vartheta.txt. Repeatable.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        metavar="PATH",
        help=f"Model path passed to each query script (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        metavar="SUFFIX",
        help="Suffix appended to every output answer directory, e.g. '_v2' writes to "
             "answers/answers_bootstrap_{pct}_{task}_clustered_v2/ instead of the default. "
             "Useful when the default directories already contain a complete sweep and "
             "you want to run a second sweep without overwriting existing files.",
    )
    args = parser.parse_args()

    # Parse PCT:PORT overrides
    port_map: dict[int, int] = {}
    for override in args.port_overrides:
        try:
            pct_s, port_s = override.split(":")
            port_map[int(pct_s)] = int(port_s)
        except ValueError:
            parser.error(f"--port must be in PCT:PORT format, got: {override!r}")

    # Parse PCT:PATH config-file overrides
    config_file_map: dict[int, Path] = {}
    for override in args.config_file_overrides:
        try:
            pct_s, path_s = override.split(":", 1)
            config_file_map[int(pct_s)] = Path(path_s)
        except ValueError:
            parser.error(f"--config-file must be in PCT:PATH format, got: {override!r}")

    vartheta_values   = args.vartheta_values if args.vartheta_values else DEFAULT_VARTHETA_VALUES
    batch_configs = (
        [c for c in BATCH_CONFIGS if c["pct"] in args.pct]
        if args.pct
        else BATCH_CONFIGS
    )
    # Apply port and vartheta-file overrides (copy dicts so BATCH_CONFIGS is not mutated)
    batch_configs = [
        {
            **c,
            **( {"port": port_map[c["pct"]]} if c["pct"] in port_map else {}),
            **( {"config_file": config_file_map[c["pct"]]} if c["pct"] in config_file_map else {}),
        }
        for c in batch_configs
    ]

    batch_summary = [f"{c['pct']}(:{c['port']}, {c['config_file']})" for c in batch_configs]
    print(f"Batch sizes    : {batch_summary}")
    print(f"Tasks          : {[t['name'] for t in TASKS]}")
    print(f"Sample         : {args.sample}")
    print(f"Output suffix  : {args.output_suffix!r}")
    print(f"VARTHETA sweep : {vartheta_values}")
    print(f"Total runs     : {len(vartheta_values)} × {len(batch_configs)} × {len(TASKS)} = "
          f"{len(vartheta_values) * len(batch_configs) * len(TASKS)}")

    run_sweep(vartheta_values, batch_configs, sample=args.sample, dry_run=args.dry_run, model_path=args.model_path, output_suffix=args.output_suffix)

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
