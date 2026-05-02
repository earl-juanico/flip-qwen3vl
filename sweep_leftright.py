#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_leftright.py

Sweeps across a list of FLIP vartheta values, running
infer/query_leftright.py once per value, then evaluating each output file with
lr_accuracy.py and writing the results to a CSV.

Output files land in <repo>/answers/answers_leftright_control/ (created if absent).
Each output JSONL is named:

    {model_name}{output_suffix}_vartheta_{tag}.jsonl

where {model_name} is the basename of --model-path and {tag} is the
normalised vartheta string (e.g. "-0.5", "none").

After all query runs finish, lr_accuracy.py is called for each output file
and the results are written to a CSV whose first column is the vartheta value
and second column is the corresponding lr accuracy.

USAGE
-----
    python sweep_leftright.py [OPTIONS] [-- VARTHETA ...]

    Vartheta values come after the optional "--" separator so that negative
    numbers (e.g. -0.5) are not mis-parsed as flags.

REQUIRED (have usable defaults, but override for your setup)
    --questions JSONL   Questions file produced by generate_questions_leftright.py
                        (default: ./questions/question_leftright_clustered.jsonl )
    --gt JSONL          Ground-truth JSONL for lr_accuracy.py
                        (default: ./gt/coco_gt_val2017_spatial_lr_clustered.jsonl)

KEY OPTIONS
    --output-suffix SUFFIX
        Appended after the model name in the output filename, e.g. "_run2"
        → Qwen3-VL-4B-Instruct_run2_vartheta_-0.5.jsonl
        Also used to select which JSONL files to evaluate in the accuracy sweep.
        (default: "")
    --output-dir DIR    Directory where output JSONL files are written (created if
                        absent). The CSV default also lands here.
                        (default: ./answers/)
    --api-port PORT     vLLM chat-completions API port (default: 8001)
    --model-path PATH   Model path; its basename becomes the filename prefix
                        (default: <script_dir>/model/Qwen3-VL-4B-Instruct)
    --image-dir DIR     Full path to the COCO val2017 image directory.
                        The HTTP server root (media-path) is derived automatically
                        as three levels up (i.e. strip /data/coco/val2017).
                        (default: ./data/coco/val2017)
    --flip-file PATH     FLIP vartheta file path, passed to infer/query_leftright.py
                        as --config-file before each query.
                        Defaults to $FLIP_VARTHETA_FILE env var, or
                        <script_dir>/config/vartheta.txt.
    --csv-output FILE   Path for the summary CSV
                        (default: ./answers/answers_leftright_control/lr_sweep{suffix}.csv)
    --dry-run           Print commands and paths without executing anything.
    --max-tokens N      Max output tokens per query (default: 16)

EXAMPLES
--------
  # Sweep the default vartheta list with defaults for everything else:
  python sweep_leftright.py

  python sweep_leftright.py --output-dir /path/to/output
    --api-port 8002 --image-dir /path/to/images --model-path /path/to/model

  # Sweep specific vartheta values, label the run with a suffix:
  python sweep_leftright.py --output-suffix _run2 -- 0.0 -0.5 -1.0 none


  # Custom questions / GT / port, sweep subset of values:
  python sweep_leftright.py \\
      --questions ./questions/question_leftright_clustered.jsonl \\
      --gt ./gt/coco_gt_val2017_spatial_lr_clustered.jsonl \\
      --api-port 8001 \\
      --output-suffix _ctrl \\
      -- none 0.0 -0.2 -0.5 -1.0

  # Dry run to preview commands without executing:
  python sweep_leftright.py --dry-run --output-suffix _test -- 0.0 -0.5

  # Point at an alternate model directory:
  python sweep_leftright.py \\
      --model-path /path/to/Qwen3-VL-4B-Instruct \\
      --output-suffix _altmodel \\
      -- none -0.5

  # Override the FLIP config file (must match what the vLLM server reads):
  python sweep_leftright.py \\
      --flip-file ./config/vartheta.txt \\
      --output-suffix _cfgA3B1 \\
      -- none 0.0 -0.5

  # Write outputs to a custom directory (created if it does not exist):
  python sweep_leftright.py \\
      --output-dir ./answers/answers_experiment1 \\
      --output-suffix _exp1 \\
      -- none 0.0 -0.5 -1.0
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
QUERY_SCRIPT = BASE_DIR / "infer" / "query_leftright.py"
LR_ACCURACY_SCRIPT = BASE_DIR / "lr_accuracy.py"
OUTPUT_DIR = BASE_DIR / "answers" / "answers_leftright_control"

DEFAULT_MODEL_PATH = str(BASE_DIR / "model" / "Qwen3-VL-4B-Instruct")
DEFAULT_QUESTIONS = str(BASE_DIR / "questions" / "question_leftright_clustered.jsonl")
DEFAULT_GT = str(BASE_DIR / "gt" / "coco_gt_val2017_spatial_lr_clustered.jsonl")
DEFAULT_IMAGE_DIR = str(BASE_DIR / "data" / "coco" / "val2017")
DEFAULT_FLIP_FILE = os.environ.get(
    "FLIP_VARTHETA_FILE",
    str(BASE_DIR / "config" / "vartheta.txt"),
)

DEFAULT_VARTHETA_VALUES = [
    "none",
    "0.0", "0.1", "0.2", "0.4", "0.5", "1.0",
    "-0.2", "-0.3", "-0.5", "-1.0", "-1.5", "-2.0", "-2.5", "-4.0", "-5.0", "-50.0",
]

# permutation additive shift
# DEFAULT_VARTHETA_VALUES = [
#     "none",
#     "0.0", "0.2", "0.5", "0.8", "1.0", "1.5",
#     "2.0", "2.5", "3.0", "4.0", "5.0",
#     "-0.2", "-0.3", "-0.5", "-1.0", "-1.5", 
#     "-2.0", "-2.5", "-4.0", "-5.0", "-50.0",
# ]


POLL_INTERVAL = 30  # seconds between progress checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vartheta_tag(vartheta_str: str) -> str:
    """Return the filename-safe tag for a vartheta value.

    Numeric strings are normalised through float so '-1' becomes '-1.0'.
    'none' is kept as 'none'.
    """
    s = vartheta_str.strip()
    if s.lower() == "none":
        return "none"
    try:
        return str(float(s))
    except ValueError:
        return s


def _flip_vartheta_arg(vartheta_str: str) -> str:
    """Convert vartheta string to the value passed as --flip-vartheta."""
    s = vartheta_str.strip()
    if s.lower() == "none":
        return "None"
    return s


def _build_query_cmd(
    vartheta_str: str,
    tag: str,
    model_name: str,
    output_suffix: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[List[str], Path]:
    out_file = output_dir / f"{model_name}{output_suffix}_vartheta_{tag}.jsonl"
    # Derive the HTTP server root by stripping the three trailing components
    # of image_dir (which is always structured as {root}/data/coco/val2017).
    media_path = str(Path(args.image_dir).parent.parent.parent)
    cmd = [
        sys.executable, str(QUERY_SCRIPT),
        "--questions",      args.questions,
        "--image-dir",      args.image_dir,
        "--api-port",       str(args.api_port),
        "--model-path",     args.model_path,
        "--max-tokens",     str(args.max_tokens),
        "--config-file",    args.flip_file,
        "--flip-vartheta",  _flip_vartheta_arg(vartheta_str),
        "--media-path",     media_path,
        "--output",         str(out_file),
    ]
    return cmd, out_file


def _run_query(label: str, cmd: list[str], out_file: Path, dry_run: bool) -> bool:
    """Launch query_leftright.py and wait for it to finish. Returns True on success."""
    if dry_run:
        print(f"  [dry-run] {label}: {' '.join(cmd)}")
        return True

    print(f"  Launching {label} → {out_file}")
    t_start = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=str(BASE_DIR))

    while True:
        rc = proc.poll()
        if rc is not None:
            elapsed = time.monotonic() - t_start
            status = "done" if rc == 0 else f"FAILED(rc={rc})"
            print(f"    [{elapsed:6.0f}s] {label} → {status}")
            return rc == 0
        elapsed = time.monotonic() - t_start
        print(f"    [{elapsed:6.0f}s] {label} still running …", flush=True)
        time.sleep(POLL_INTERVAL)


def _evaluate_output(out_file: Path, gt: str, dry_run: bool) -> Optional[float]:
    """Run lr_accuracy.py on out_file and return the accuracy float, or None on error."""
    cmd = [sys.executable, str(LR_ACCURACY_SCRIPT), str(out_file), "--gt", gt]
    if dry_run:
        print(f"  [dry-run] eval: {' '.join(cmd)}")
        return None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        output = result.stdout.strip()
        # Expected format: "Accuracy: 0.6543 (a/b)"
        m = re.search(r"Accuracy:\s*([\d.]+)", output)
        if m:
            return float(m.group(1))
        print(f"  [WARN] could not parse accuracy from: {output!r}")
        return None
    except Exception as e:
        print(f"  [ERROR] lr_accuracy.py failed for {out_file}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------

def run_sweep(vartheta_values: List[str], args: argparse.Namespace, output_dir: Path) -> List[Tuple[str, Path, bool]]:
    """Run query_leftright.py for each vartheta value. Returns list of (vartheta, out_file, success)."""
    model_name = os.path.basename(args.model_path.rstrip("/"))
    results = []
    total = len(vartheta_values)

    for idx, vartheta_str in enumerate(vartheta_values, start=1):
        tag = _vartheta_tag(vartheta_str)
        print(f"\n{'='*64}")
        print(f"[{idx}/{total}] VARTHETA = {vartheta_str!r}  →  tag: {tag!r}")
        print(f"{'='*64}")

        cmd, out_file = _build_query_cmd(vartheta_str, tag, model_name, args.output_suffix, args, output_dir)

        if not args.dry_run and out_file.exists():
            print(f"  ERROR: output file already exists: {out_file}")
            print(f"         Delete it or use --output-suffix to write to a new set of files.")
            raise SystemExit(1)

        success = _run_query(f"vartheta={vartheta_str!r}", cmd, out_file, args.dry_run)
        results.append((vartheta_str, out_file, success))

    return results


def run_eval(
    results: List[Tuple[str, Path, bool]],
    gt: str,
    output_suffix: str,
    csv_output: Path,
    dry_run: bool,
) -> None:
    """Evaluate each output file and write the summary CSV."""
    print(f"\n{'='*64}")
    print("EVALUATION SWEEP (lr_accuracy.py)")
    print(f"{'='*64}")

    rows: List[Tuple[str, Optional[float]]] = []
    for vartheta_str, out_file, success in results:
        tag = _vartheta_tag(vartheta_str)
        if not success:
            print(f"  Skipping vartheta={vartheta_str!r} (query failed)")
            rows.append((vartheta_str, None))
            continue
        if not dry_run and not out_file.exists():
            print(f"  Skipping vartheta={vartheta_str!r}: output file missing: {out_file}")
            rows.append((vartheta_str, None))
            continue
        print(f"  Evaluating vartheta={vartheta_str!r}: {out_file.name}")
        acc = _evaluate_output(out_file, gt, dry_run)
        rows.append((vartheta_str, acc))
        if acc is not None:
            print(f"    → lr_accuracy = {acc:.4f}")

    # Write CSV
    if not dry_run:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["vartheta", "lr_accuracy"])
            for vartheta_str, acc in rows:
                writer.writerow([vartheta_str, "" if acc is None else f"{acc:.4f}"])
        print(f"\nResults written to: {csv_output}")
    else:
        print(f"\n[dry-run] would write CSV to: {csv_output}")
        print("  vartheta, lr_accuracy")
        for vartheta_str, _ in rows:
            print(f"  {vartheta_str}, <evaluated>")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep vartheta values for left/right spatial evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "vartheta_values",
        nargs="*",
        metavar="VARTHETA",
        help=f"Vartheta values to sweep (default: {DEFAULT_VARTHETA_VALUES}). "
             "Use -- before negative numbers, e.g.: -- none 0.0 -0.5 -1.0",
    )
    parser.add_argument(
        "--questions", "-q",
        default=DEFAULT_QUESTIONS,
        metavar="JSONL",
        help=f"Input questions JSONL (default: {DEFAULT_QUESTIONS})",
    )
    parser.add_argument(
        "--gt",
        default=DEFAULT_GT,
        metavar="JSONL",
        help=f"Ground-truth JSONL for lr_accuracy.py (default: {DEFAULT_GT})",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        metavar="SUFFIX",
        help='Suffix in output filename after the model name, e.g. "_run2". '
             'Also used to match JSONL files during the accuracy sweep. (default: "")',
    )
    parser.add_argument(
        "--api-port", "-p",
        type=int,
        default=8001,
        metavar="PORT",
        help="vLLM chat-completions API port (default: 8001)",
    )
    parser.add_argument(
        "--model-path", "-m",
        default=DEFAULT_MODEL_PATH,
        metavar="PATH",
        help=f"Model path; its basename is used in output filenames (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--image-dir", "-i",
        default=DEFAULT_IMAGE_DIR,
        metavar="DIR",
        help=f"Host-side image directory (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--flip-file",
        default=DEFAULT_FLIP_FILE,
        metavar="PATH",
        help=f"FLIP vartheta config file written before each query "
             f"(default: $FLIP_VARTHETA_FILE or {DEFAULT_FLIP_FILE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        metavar="N",
        help="Max output tokens per query (default: 16)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        metavar="DIR",
        help=f"Directory for output JSONL files (created if absent, default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        metavar="FILE",
        help="Path for the summary CSV "
             f"(default: {OUTPUT_DIR}/lr_sweep{{suffix}}.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and paths without executing anything.",
    )
    args = parser.parse_args()

    vartheta_values = args.vartheta_values if args.vartheta_values else DEFAULT_VARTHETA_VALUES

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    csv_output = (
        Path(args.csv_output)
        if args.csv_output
        else output_dir / f"lr_sweep{args.output_suffix}.csv"
    )

    model_name = os.path.basename(args.model_path.rstrip("/"))

    print(f"Model name     : {model_name}")
    print(f"Output dir     : {output_dir}")
    print(f"Output suffix  : {args.output_suffix!r}")
    print(f"Questions      : {args.questions}")
    print(f"Ground truth   : {args.gt}")
    print(f"API port       : {args.api_port}")
    print(f"Image dir      : {args.image_dir}")
    print(f"Media path     : {str(Path(args.image_dir).parent.parent.parent)!r}")
    print(f"FLIP file      : {args.flip_file}")
    print(f"CSV output     : {csv_output}")
    print(f"VARTHETA sweep : {vartheta_values}")
    print(f"Total runs     : {len(vartheta_values)}")

    results = run_sweep(vartheta_values, args, output_dir)

    run_eval(results, args.gt, args.output_suffix, csv_output, args.dry_run)

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
