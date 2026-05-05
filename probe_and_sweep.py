#!/usr/bin/env python3
"""
probe_and_sweep.py

Sweeps across a list of FLIP vartheta values, running the full synchronized set of
query jobs (detect, reason, indout, leftright) for each value.

Bootstrap tasks (detect, reason, indout) run across all selected bootstrap batch sizes
(10, 20, 40, 60, 80, and optionally 100 for the full dataset), all in parallel per
vartheta value.

The leftright task (spatial left/right evaluation) runs once per vartheta value on
its own dedicated server port, independent of the bootstrap batch sizes.  It uses a
separate questions file and is evaluated with lr_accuracy.py after the full sweep.
AM stats (A_{ℓ,i} and M_{ℓ,i} per vartheta, from Section 5.1) accumulated by the
leftright server are copied from the temp file into the output directory when the
sweep finishes.

The sample name (e.g. s00, s01) is configurable via --sample and is substituted into
both the input questions filename and the output JSONL filename for bootstrap tasks.
When --pct 100 is used, --sample defaults to s00 and does not need to be specified.

When pct=100 (full dataset, no bootstrap), the output directory omits the 'bootstrap_100'
infix: answers/answers_{task}_clustered instead of answers/answers_bootstrap_100_{task}_clustered.

Usage examples:

  ── Basic sweep (bootstrap + leftright, default vartheta list) ──────────────

  # All batch sizes (pct 10/20/40/60/80/100), sample s00, leftright on port 8001
  python3 probe_and_sweep.py

  # Same, with a custom sample for bootstrap tasks
  python3 probe_and_sweep.py --sample s01

  # Sweep a specific subset of vartheta values (use -- before negative numbers)
  python3 probe_and_sweep.py -- none 0.0 -0.5 -1.0

  # Custom sample + specific vartheta values
  python3 probe_and_sweep.py --sample s01 -- 0.2 -0.5 -1.0 none

  ── Restrict bootstrap to selected pct values ───────────────────────────────

  # Bootstrap pct 20 and 40 only (leftright still runs once per vartheta)
  python3 probe_and_sweep.py --pct 20 40 -- 0.0 0.2 -0.5

  # Full dataset (pct=100) bootstrap, sample s00 (default), leftright enabled
  python3 probe_and_sweep.py --pct 100

  # Full dataset, specific vartheta values
  python3 probe_and_sweep.py --pct 100 -- 0.0 -0.5 none

  ── Bootstrap-only (skip leftright) ─────────────────────────────────────────

  python3 probe_and_sweep.py --no-leftright
  python3 probe_and_sweep.py --no-leftright --pct 20 40 -- 0.0 -0.5
  python3 probe_and_sweep.py --no-leftright --pct 100 --sample s01 -- 0.0 -0.5

  ── Leftright dataset overrides ──────────────────────────────────────────────

  # Leftright always shares the first batch config's server — no separate port needed.
  # Override questions or GT only when using non-default files.
  python3 probe_and_sweep.py \
      --leftright-questions ./questions/question_leftright_clustered.jsonl \
      --leftright-gt ./gt/coco_gt_val2017_spatial_lr_clustered.jsonl

  ── AM stats (Section 5.1) ───────────────────────────────────────────────────

  # Server must be launched with FLIP_LOG_STATS=1 and matching FLIP_STATS_CSV.
  # The temp CSV is copied to answers/answers_leftright_clustered/am_stats.csv
  # after the sweep completes (or on server exit).
  #
  #   FLIP_LOG_STATS=1 FLIP_STATS_CSV=/tmp/flip_am_stats.csv \
  #   FLIP_VARTHETA_FILE=./config/vartheta.txt \
  #   python3 serve_with_patch.py
  #
  python3 probe_and_sweep.py --flip-stats-csv /tmp/flip_am_stats.csv

  # Custom temp path (must match FLIP_STATS_CSV on the server)
  python3 probe_and_sweep.py --flip-stats-csv /tmp/my_run_stats.csv

  ── Output organisation ──────────────────────────────────────────────────────

  # Append suffix to all output dirs (avoids overwriting a previous sweep)
  python3 probe_and_sweep.py --output-suffix _v2

  # Full dataset + leftright + suffix, sweep subset
  python3 probe_and_sweep.py --pct 100 --output-suffix _v2 -- 0.0 -0.5 none

  # Write all output subdirectories under a custom root instead of ./answers/
  python3 probe_and_sweep.py --output-dir /path/to/results

  # Custom root combined with suffix and a vartheta subset
  python3 probe_and_sweep.py --output-dir /path/to/results --output-suffix _v2 -- 0.0 -0.5 none

  ── Port and config-file overrides ──────────────────────────────────────────

  # --port and --config-file apply to bootstrap tasks AND leftright, since
  # leftright shares the first batch config's server.  When running a single
  # pct, that one override covers both.

  # Single server for everything: pct=100 only, leftright shares port 8100
  python3 probe_and_sweep.py --pct 100

  # Single server on a non-default port
  python3 probe_and_sweep.py --pct 100 --port 100:8033

  # pct=20 only — leftright shares port 8020 automatically
  python3 probe_and_sweep.py --pct 20

  # pct=20 with a non-default port — leftright follows
  python3 probe_and_sweep.py --pct 20 --port 20:8021

  # Multi-pct: leftright shares the first (pct=20) server
  python3 probe_and_sweep.py --pct 20 40 --port 20:8021 --port 40:8041

  # Override the vartheta config file for a batch size (leftright follows if it is the first)
  python3 probe_and_sweep.py --pct 20 --config-file 20:/path/to/config20_s1/vartheta.txt

  # Combine port and config-file overrides for a single-server run
  python3 probe_and_sweep.py \
      --pct 100 \
      --port 100:8033 \
      --config-file 100:/path/to/config_run2/vartheta.txt

  ── Model path override ──────────────────────────────────────────────────────

  python3 probe_and_sweep.py --model-path /path/to/Qwen3-VL-4B-Instruct
  python3 probe_and_sweep.py \
      --model-path /path/to/Qwen3-VL-4B-Instruct \
      --sample s01 -- 0.0 0.2 -0.5

  ── Dry run (preview without executing) ─────────────────────────────────────

  # Preview all commands and config writes for the default sweep
  python3 probe_and_sweep.py --dry-run

  # Dry run for a subset of pct and vartheta values
  python3 probe_and_sweep.py --dry-run --pct 20 40 -- 0.0 -0.5

  # Dry run including leftright (leftright shares first batch config's server)
  python3 probe_and_sweep.py --dry-run --pct 100 -- 0.0 -0.5

  # Dry run for full dataset
  python3 probe_and_sweep.py --dry-run --pct 100 -- 0.0 -0.5
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

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

# Left/right spatial evaluation — runs once per vartheta on the same server as
# the first batch config (shares its port and vartheta config file).  Uses its
# own fixed questions/GT dataset regardless of pct.
LR_ACCURACY_SCRIPT = BASE_DIR / "lr_accuracy.py"
LEFTRIGHT_TASK = {
    "name": "leftright",
    "script": "infer/query_leftright.py",
    "max_tokens": 16,
}
DEFAULT_LEFTRIGHT_QUESTIONS = str(BASE_DIR / "questions" / "question_leftright_clustered.jsonl")
DEFAULT_LEFTRIGHT_GT        = str(BASE_DIR / "gt" / "coco_gt_val2017_spatial_lr_clustered.jsonl")
DEFAULT_FLIP_STATS_CSV = "/tmp/flip_am_stats.csv"

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


def _build_cmd(pct: int, port: int, config_file: Path, task: dict, vartheta_str: str, tag: str, sample: str, model_path: str = MODEL_PATH, output_suffix: str = "", output_dir: Path = BASE_DIR / "answers") -> list:
    questions = str(
        BASE_DIR / f"questions/question_pct{pct}_{sample}.jsonl"
    )
    if pct == 100:
        out_dir = output_dir / f"answers_{task['name']}_clustered{output_suffix}"
    else:
        out_dir = output_dir / f"answers_bootstrap_{pct}_{task['name']}_clustered{output_suffix}"
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


def _build_leftright_cmd(
    port: int,
    config_file: str,
    vartheta_str: str,
    tag: str,
    model_path: str,
    questions: str,
    output_suffix: str = "",
    output_dir: Path = BASE_DIR / "answers",
) -> Tuple[List[str], Path, Path]:
    model_name = os.path.basename(model_path.rstrip("/"))
    out_dir = output_dir / f"answers_leftright_clustered{output_suffix}"
    out_file = out_dir / f"{model_name}_vartheta_{tag}.jsonl"
    script = str(BASE_DIR / LEFTRIGHT_TASK["script"])
    cmd = [
        sys.executable, script,
        "--model-path", model_path,
        "--questions",  questions,
        "--max-tokens", str(LEFTRIGHT_TASK["max_tokens"]),
        "--image-dir",  IMAGE_DIR,
        "--media-path", MEDIA_PATH,
        "--api-port",   str(port),
        "--config-file", config_file,
        "--flip-vartheta", vartheta_str,
        "--output",     str(out_file),
    ]
    return cmd, out_file, out_dir


def _evaluate_leftright_output(out_file: Path, gt: str, dry_run: bool) -> Optional[float]:
    """Run lr_accuracy.py on out_file and return the accuracy float, or None on error."""
    cmd = [sys.executable, str(LR_ACCURACY_SCRIPT), str(out_file), "--gt", gt]
    if dry_run:
        print(f"  [dry-run] eval: {' '.join(cmd)}")
        return None
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
        output = result.stdout.strip()
        m = re.search(r"Accuracy:\s*([\d.]+)", output)
        if m:
            return float(m.group(1))
        print(f"  [WARN] could not parse accuracy from: {output!r}")
        return None
    except Exception as e:
        print(f"  [ERROR] lr_accuracy.py failed for {out_file}: {e}")
        return None


def run_eval_leftright(
    leftright_results: List[Tuple[str, Path, bool]],
    gt: str,
    csv_output: Path,
    dry_run: bool,
) -> None:
    """Evaluate leftright output files with lr_accuracy.py and write the summary CSV."""
    print(f"\n{'='*64}")
    print("LEFTRIGHT EVALUATION (lr_accuracy.py)")
    print(f"{'='*64}")

    rows: List[Tuple[str, Optional[float]]] = []
    for vartheta_str, out_file, success in leftright_results:
        if not success:
            print(f"  Skipping vartheta={vartheta_str!r} (query failed)")
            rows.append((vartheta_str, None))
            continue
        if not dry_run and not out_file.exists():
            print(f"  Skipping vartheta={vartheta_str!r}: output file missing: {out_file}")
            rows.append((vartheta_str, None))
            continue
        print(f"  Evaluating vartheta={vartheta_str!r}: {out_file.name}")
        acc = _evaluate_leftright_output(out_file, gt, dry_run)
        rows.append((vartheta_str, acc))
        if acc is not None:
            print(f"    → lr_accuracy = {acc:.4f}")

    if not dry_run:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            import csv as _csv
            writer = _csv.writer(f)
            writer.writerow(["vartheta", "lr_accuracy"])
            for vartheta_str, acc in rows:
                writer.writerow([vartheta_str, "" if acc is None else f"{acc:.4f}"])
        print(f"\nLeftright accuracy CSV written → {csv_output}")
    else:
        print(f"\n[dry-run] would write leftright accuracy CSV to: {csv_output}")
        print("  vartheta, lr_accuracy")
        for vartheta_str, _ in rows:
            print(f"  {vartheta_str}, <evaluated>")


def run_sweep(vartheta_values: list, batch_configs: list, sample: str = "s00", dry_run: bool = False, model_path: str = MODEL_PATH, output_suffix: str = "", leftright_cfg: Optional[dict] = None, output_dir: Path = BASE_DIR / "answers") -> List[Tuple[str, Path, bool]]:
    """Run all query jobs for each vartheta value.

    Bootstrap tasks (detect/reason/indout) run once per (batch_config × task).
    The leftright task (if leftright_cfg is not None) runs once per vartheta,
    independent of batch_configs.

    Returns list of (vartheta_str, out_file, success) for leftright jobs only.
    """
    leftright_results: List[Tuple[str, Path, bool]] = []
    total = len(vartheta_values)

    for idx, vartheta_str in enumerate(vartheta_values, start=1):
        tag = _vartheta_tag(vartheta_str)
        print(f"\n{'='*64}")
        print(f"[{idx}/{total}] VARTHETA = {vartheta_str!r}  →  filename tag: {tag!r}")
        print(f"{'='*64}")

        # Write the new VARTHETA value to every config file first.
        # Leftright shares batch_configs[0]'s config file, so it is covered here.
        for cfg in batch_configs:
            config_file = cfg["config_file"]
            if dry_run:
                print(f"  DRY-RUN: would write {vartheta_str!r} -> {config_file}")
            else:
                _write_vartheta_config(config_file, vartheta_str)

        # Build all job specs first, then check for conflicting output files before launching
        job_specs: List[Tuple[str, int, list, str, Path]] = []
        for cfg in batch_configs:
            pct, port = cfg["pct"], cfg["port"]
            for task in TASKS:
                cmd, out_file, out_dir = _build_cmd(pct, port, cfg["config_file"], task, vartheta_str, tag, sample, model_path, output_suffix, output_dir)
                label = f"bs{pct}-{task['name']}"
                job_specs.append((label, port, cmd, out_file, out_dir))

        # Append the leftright job (runs once, on its own port)
        lr_out_file: Optional[Path] = None
        if leftright_cfg is not None:
            lr_cmd, lr_out_file, lr_out_dir = _build_leftright_cmd(
                port=leftright_cfg["port"],
                config_file=leftright_cfg["config_file"],
                vartheta_str=vartheta_str,
                tag=tag,
                model_path=model_path,
                questions=leftright_cfg["questions"],
                output_suffix=output_suffix,
                output_dir=output_dir,
            )
            job_specs.append(("leftright", leftright_cfg["port"], lr_cmd, str(lr_out_file), lr_out_dir))

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
        procs: List[Tuple[str, int, Optional[subprocess.Popen]]] = []
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
            if leftright_cfg is not None and lr_out_file is not None:
                leftright_results.append((vartheta_str, lr_out_file, True))
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

        if leftright_cfg is not None and lr_out_file is not None:
            lr_rc = finished.get("leftright", 1)
            leftright_results.append((vartheta_str, lr_out_file, lr_rc == 0))

    return leftright_results


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
             "Also applied to the leftright output directory and CSV filenames. "
             "Useful when the default directories already contain a complete sweep and "
             "you want to run a second sweep without overwriting existing files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "answers"),
        metavar="DIR",
        help=f"Root directory for all output answer subdirectories "
             f"(default: {BASE_DIR / 'answers'}).",
    )
    # ── Leftright task options ──────────────────────────────────────────────
    parser.add_argument(
        "--no-leftright",
        action="store_true",
        help="Skip the leftright spatial evaluation task (bootstrap tasks only).",
    )
    parser.add_argument(
        "--leftright-questions",
        default=DEFAULT_LEFTRIGHT_QUESTIONS,
        metavar="JSONL",
        help=f"Questions JSONL for the leftright task (default: {DEFAULT_LEFTRIGHT_QUESTIONS}).",
    )
    parser.add_argument(
        "--leftright-gt",
        default=DEFAULT_LEFTRIGHT_GT,
        metavar="JSONL",
        help=f"Ground-truth JSONL for lr_accuracy.py (default: {DEFAULT_LEFTRIGHT_GT}).",
    )
    parser.add_argument(
        "--flip-stats-csv",
        default=DEFAULT_FLIP_STATS_CSV,
        metavar="FILE",
        help="Temporary path where the leftright vLLM server writes AM stats "
             "(must match FLIP_STATS_CSV set when launching the server). "
             "Copied to answers/answers_leftright_clustered{suffix}/am_stats{suffix}.csv "
             f"after the sweep. (default: {DEFAULT_FLIP_STATS_CSV})",
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

    # Build leftright config dict; None when --no-leftright is given.
    # Leftright shares the first batch config's server (port + vartheta file),
    # so --port and --config-file overrides for that pct also cover leftright.
    leftright_cfg: Optional[dict] = None
    if not args.no_leftright:
        lr_base = batch_configs[0]
        leftright_cfg = {
            "port":        lr_base["port"],
            "config_file": str(lr_base["config_file"]),
            "questions":   args.leftright_questions,
        }

    output_dir = Path(args.output_dir)
    lr_out_dir = output_dir / f"answers_leftright_clustered{args.output_suffix}"
    lr_csv_output = lr_out_dir / f"lr_sweep{args.output_suffix}.csv"
    flip_stats_temp = Path(args.flip_stats_csv)
    flip_stats_out  = lr_out_dir / f"am_stats{args.output_suffix}.csv"

    bootstrap_runs = len(vartheta_values) * len(batch_configs) * len(TASKS)
    leftright_runs = len(vartheta_values) if leftright_cfg is not None else 0
    total_runs = bootstrap_runs + leftright_runs

    batch_summary = [f"{c['pct']}(:{c['port']}, {c['config_file']})" for c in batch_configs]
    print(f"Batch sizes    : {batch_summary}")
    print(f"Bootstrap tasks: {[t['name'] for t in TASKS]}")
    if leftright_cfg:
        lr_pct = batch_configs[0]["pct"]
        print(f"Leftright task : enabled — sharing pct={lr_pct} server "
              f"(port {leftright_cfg['port']}, config {leftright_cfg['config_file']})")
    else:
        print("Leftright task : disabled (--no-leftright)")
    print(f"Sample         : {args.sample}")
    print(f"Output suffix  : {args.output_suffix!r}")
    print(f"VARTHETA sweep : {vartheta_values}")
    print(f"Total runs     : {bootstrap_runs} bootstrap  +  {leftright_runs} leftright  =  {total_runs}")
    if leftright_cfg:
        print(f"Leftright GT   : {args.leftright_gt}")
        print(f"LR accuracy CSV: {lr_csv_output}")
        print(f"AM stats temp  : {flip_stats_temp}  (server writes here; set FLIP_STATS_CSV to this)")
        print(f"AM stats out   : {flip_stats_out}  (copied here after sweep)")

    leftright_results = run_sweep(
        vartheta_values, batch_configs,
        sample=args.sample,
        dry_run=args.dry_run,
        model_path=args.model_path,
        output_suffix=args.output_suffix,
        leftright_cfg=leftright_cfg,
        output_dir=output_dir,
    )

    # Post-sweep: evaluate leftright outputs and write lr_sweep CSV
    if leftright_cfg is not None and leftright_results:
        run_eval_leftright(leftright_results, args.leftright_gt, lr_csv_output, args.dry_run)

    print("\nSweep complete.")

    # Copy AM stats CSV from server temp file into the leftright output directory
    if leftright_cfg is not None and not args.dry_run:
        if flip_stats_temp.exists():
            lr_out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(flip_stats_temp, flip_stats_out)
            print(f"AM stats CSV   : {flip_stats_out}  ✓  (copied from {flip_stats_temp})")
        else:
            print(f"AM stats CSV   : {flip_stats_temp}  (not found — was the leftright server "
                  f"launched with FLIP_STATS_CSV={flip_stats_temp} ?)")
            print(f"  → The CSV is also written at server exit (stop the tmux session).")
    elif leftright_cfg is not None and args.dry_run:
        print(f"[dry-run] would copy AM stats: {flip_stats_temp} → {flip_stats_out}")


if __name__ == "__main__":
    main()
