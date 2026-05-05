#!/usr/bin/env python3
"""
evaluate_sweep.py

Automates the four-step post-sweep evaluation for each (pct, vartheta) combination
produced by probe_and_sweep.py.  For every vartheta value across the requested pct
groups:

  Step 1 — evaluate_FitAP_vlm.py  : score detection + reasoning for this vartheta.
  Step 2 — rename prc/ directory  : adds the bootstrap_pct infix to the prc dir name
                                    so step-3 CSV paths are independent of --split.
                                    Skipped when --no-rename is given or pct=100.
  Step 3 — evaluate_FitAP_vlm.py  : pooled-GLM comparison vs. the baseline vartheta.
                                    Skipped for the baseline value itself.
  Step 4 — fliprate.py             : indoor/outdoor flip-rate vs. baseline.
                                    Skipped for the baseline value itself.

When --no-rename is used, steps 3 and 4 look for CSVs / JSOLs under the raw prc
directory name produced by step 1 (prc-{model}{split}_{vartheta}/), which is fully
deterministic and requires no renaming.

Usage examples:
  python3 evaluate_sweep.py                                               # all pcts, default vartheta list
  python3 evaluate_sweep.py -- -50.0 none                                # specific values, all pcts
  python3 evaluate_sweep.py --pct 80 -- -50.0 none                       # pct=80 only
  python3 evaluate_sweep.py --pct 80 --sample s04                        # custom sample
  python3 evaluate_sweep.py --pct 100                                    # full dataset (no bootstrap infix)
  python3 evaluate_sweep.py --baseline none                              # baseline for comparison (default)
  python3 evaluate_sweep.py --no-rename                                  # skip prc directory rename (step 2)
  python3 evaluate_sweep.py --dry-run                                    # preview all commands without running
  python3 evaluate_sweep.py --pct 80 --dry-run -- -50.0 none
  python3 evaluate_sweep.py --pct 80 --output-csv results_pct80.csv      # → reports/results_pct80.csv
  python3 evaluate_sweep.py --pct 80 --sample s04 --output-csv results_pct80_s04.csv
  python3 evaluate_sweep.py --pct 100 --output-csv results_full.csv      # full dataset + CSV
  python3 evaluate_sweep.py --no-rename --output-csv results.csv         # skip rename, write CSV
  python3 evaluate_sweep.py --pct 80 --output-suffix _v2                  # data in *_clustered_v2/ dirs
  python3 evaluate_sweep.py --answers-root /data/sweep_v2/answers          # answers in a custom directory
  python3 evaluate_sweep.py --pct 80 --answers-root ../other_run/answers   # relative path also accepted
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# By the time evaluate_sweep.py runs, probe_and_sweep.py has already written
# all model outputs to answers/answers_bootstrap_{pct}_{task}_clustered/ directories
# (or answers/answers_{task}_clustered/ for pct=100).  The vLLM servers are no longer
# needed, so 'port' is omitted.  The 'pct' values are still required here
# because they determine which of those pre-computed output directories to read
# from (step 1) and how to name the prc/ subdirectory after step-2 rename.
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent
MODEL_DEFAULT = "Qwen3-VL-4B-Instruct"

# Environment variables derived from BASE_DIR that evaluate_FitAP_vlm.py needs.
# COCO_ROOT must end in a directory named 'data' so that evaluate_FitAP_vlm.py
# sets DATA_ROOT = COCO_ROOT and resolves annotations at DATA_ROOT/coco/annotations/.
# EVAL_DIR defaults to the script's own directory (BASE_DIR) so it is not overridden here.
_FITAP_ENV = {
    "COCO_ROOT": str(BASE_DIR / "data"),
}

BATCH_CONFIGS = [
    {"pct": 10},
    {"pct": 20},
    {"pct": 40},
    {"pct": 60},
    {"pct": 80},
    {"pct": 100},
]

DEFAULT_VARTHETA_VALUES = [
    "none", "0.0", "0.1", "0.2", "0.4", "0.5", "1.0",
    "-0.2", "-0.3", "-0.5", "-1.0", "-1.5", "-2.0", "-2.5", "-4.0", "-5.0", "-50.0",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _vartheta_tag(vartheta_str: str) -> str:
    s = vartheta_str.strip()
    if s.lower() == "none":
        return "none"
    try:
        return str(float(s))
    except ValueError:
        return s


def _vartheta_arg(sample: str, tag: str) -> str:
    """Value passed as --flip-vartheta to evaluate_FitAP_vlm.py, e.g. 's00_vartheta_-50.0'."""
    return f"{sample}_vartheta_{tag}"


def _questions_file(pct: int, sample: str) -> Path:
    return BASE_DIR / "questions" / f"question_pct{pct}_{sample}.jsonl"


def _answers_dir(pct: int, task: str, suffix: str = "", answers_root: Path = None) -> Path:
    root = answers_root if answers_root is not None else BASE_DIR / "answers"
    if pct == 100:
        return root / f"answers_{task}_clustered{suffix}"
    return root / f"answers_bootstrap_{pct}_{task}_clustered{suffix}"


def _prc_dir_raw(model: str, split: str, vartheta_arg: str) -> Path:
    """Directory produced by evaluate_FitAP_vlm.py step 1.

    evaluate_FitAP_vlm.py sets:
        PRC_DIR = prc/prc-{model}{split}_{vartheta_arg}
    where split already carries its leading underscore (e.g. _novel_clustered).
    """
    return BASE_DIR / "prc" / f"prc-{model}{split}_{vartheta_arg}"


def _prc_dir_named(model: str, pct: int, vartheta_arg: str) -> Path:
    """Target prc directory after the step-2 rename.

    Replaces the split infix with the bootstrap_pct infix:
        pct != 100: prc/prc-{model}_bootstrap_{pct}_{vartheta_arg}
        pct == 100: prc/prc-{model}_{vartheta_arg}  (no bootstrap infix)
    """
    if pct == 100:
        return BASE_DIR / "prc" / f"prc-{model}_{vartheta_arg}"
    return BASE_DIR / "prc" / f"prc-{model}_bootstrap_{pct}_{vartheta_arg}"


def _metrics_csv(prc_dir: Path, model: str, vartheta_arg: str) -> Path:
    """Path to the per-sample metrics CSV written by evaluate_FitAP_vlm.py step 1."""
    return prc_dir / f"metrics_reason_det_{model}_{vartheta_arg}.csv"


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run(cmd: list[str], env_extra: dict | None = None, dry_run: bool = False) -> int:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    prefix = (" ".join(f"{k}={v}" for k, v in env_extra.items()) + " ") if env_extra else ""
    print(f"    $ {prefix}{' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.run(cmd, env=env, cwd=str(BASE_DIR)).returncode


def _run_capture(cmd: list[str], env_extra: dict | None = None, dry_run: bool = False) -> tuple[int, str]:
    """Like _run but captures stdout+stderr and returns (returncode, combined_output)."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    prefix = (" ".join(f"{k}={v}" for k, v in env_extra.items()) + " ") if env_extra else ""
    print(f"    $ {prefix}{' '.join(cmd)}")
    if dry_run:
        return 0, ""
    result = subprocess.run(
        cmd, env=env, cwd=str(BASE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    return result.returncode, result.stdout or ""


# ---------------------------------------------------------------------------
# Output parsers for step 3 (evaluate_FitAP_vlm.py) and step 4 (fliprate.py)
# ---------------------------------------------------------------------------

_NUM = r"[-+]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"


def _parse_step3_output(output: str) -> dict:
    row: dict = {}

    # irr_indirect + 95_ci_irr:
    #   "  IRR_indirect (exp(a×b)):   1.2345  95% CI=[0.5678, 2.3456]"
    m = re.search(
        r'IRR_indirect\s*\(exp\(a[×x]b\)\)[^:\n]*:\s*(' + _NUM + r')\s+95%\s*CI=\[(' + _NUM + r'),\s*(' + _NUM + r')\]',
        output,
    )
    if m:
        row['irr_indirect'] = m.group(1)
        row['95_ci_irr'] = f"[{m.group(2)}, {m.group(3)}]"

    # irr_direct:
    #   "[info] IRR for is_experiment after controlling ΔIoU (strict responses): 1.234"
    m = re.search(
        r'\[info\]\s*IRR for is_experiment after controlling[^:\n]*:\s*(' + _NUM + r')',
        output,
    )
    if m:
        row['irr_direct'] = m.group(1)

    # dr_50:
    #   "[info] Coef (experiment vs control) on ΔIoU: 0.1234"
    m = re.search(
        r'\[info\]\s*Coef\s*\(experiment vs control\) on[^:\n]*IoU[^:\n]*:\s*(' + _NUM + r')',
        output,
    )
    if m:
        row['dr_50'] = m.group(1)

    # 95_ci_dr_50: last two columns of the is_experiment row in the GLM summary table.
    # Statsmodels format:  "is_experiment  coef  std_err  z  P>|z|  ci_lo  ci_hi"
    _N = r'(' + _NUM + r')'
    m = re.search(
        r'^is_experiment\s+' + _N + r'\s+' + _N + r'\s+' + _N + r'\s+' + _N + r'\s+' + _N + r'\s+' + _N,
        output, re.MULTILINE,
    )
    if m:
        row['95_ci_dr_50'] = f"[{m.group(5)}, {m.group(6)}]"

    return row


def _parse_mediation_output(output: str) -> dict:
    row: dict = {}

    # Leg 1 (a): coef and SE
    #   "  Leg 1 (a)  — is_experiment → Δrecall:          coef=0.123456  SE=0.123456"
    m = re.search(
        r'Leg\s+1\s*\(a\)[^\n]*coef\s*=\s*(' + _NUM + r')\s+SE\s*=\s*(' + _NUM + r')',
        output,
    )
    if m:
        row['a'] = m.group(1)
        row['SE(a)'] = m.group(2)

    # Leg 2 (b): coef and SE
    #   "  Leg 2 (b)  — Δrecall → log(count_error_tol):  coef=0.123456  SE=0.123456"
    m = re.search(
        r'Leg\s+2\s*\(b\)[^\n]*coef\s*=\s*(' + _NUM + r')\s+SE\s*=\s*(' + _NUM + r')',
        output,
    )
    if m:
        row['b'] = m.group(1)
        row['SE(b)'] = m.group(2)

    # Indirect effect (a×b): standalone value and SE
    #   "  Indirect effect (a×b) on log(T):              0.123456  SE=0.123456"
    m = re.search(
        r'Indirect\s+effect\s*\(a[×x]b\)[^\n]*:\s+(' + _NUM + r')\s+SE\s*=\s*(' + _NUM + r')',
        output,
    )
    if m:
        row['axb'] = m.group(1)
        row['SE(axb)'] = m.group(2)

    # p-value from "  z=1.234  p=0.0123  95% CI=[...]"
    m = re.search(r'z\s*=\s*' + _NUM + r'\s+p\s*=\s*(' + _NUM + r')', output)
    if m:
        row['p-value'] = m.group(1)

    # irr_indirect and 95_pct_ci_irr
    #   "  IRR_indirect (exp(a×b)):                       1.2345  95% CI=[0.5678, 2.3456]"
    m = re.search(
        r'IRR_indirect\s*\(exp\(a[×x]b\)\)[^:\n]*:\s*(' + _NUM + r')\s+95%\s*CI=\[(' + _NUM + r'),\s*(' + _NUM + r')\]',
        output,
    )
    if m:
        row['irr_indirect'] = m.group(1)
        row['95_pct_ci_irr'] = f"[{m.group(2)}, {m.group(3)}]"

    return row


def _parse_step4_output(output: str) -> dict:
    row: dict = {}
    # fliprate.py prints the rate as a bare float on its own line
    m = re.search(r'^(' + _NUM + r')\s*$', output, re.MULTILINE)
    if m:
        row['switchrate'] = m.group(1)
    return row


# ---------------------------------------------------------------------------
# Per-(pct, vartheta) evaluation
# ---------------------------------------------------------------------------

def _evaluate_one(
    *,
    pct: int,
    vartheta_str: str,
    baseline_str: str,
    sample: str,
    model: str,
    split: str,
    width: int,
    height: int,
    rename: bool,
    output_suffix: str,
    answers_root: Path,
    dry_run: bool,
) -> dict:
    tag                  = _vartheta_tag(vartheta_str)
    baseline_tag         = _vartheta_tag(baseline_str)
    vartheta_arg         = _vartheta_arg(sample, tag)
    baseline_vartheta_arg = _vartheta_arg(sample, baseline_tag)
    is_baseline          = (tag == baseline_tag)

    detect_dir = _answers_dir(pct, "detect", output_suffix, answers_root)
    reason_dir = _answers_dir(pct, "reason", output_suffix, answers_root)
    indout_dir = _answers_dir(pct, "indout", output_suffix, answers_root)

    label = f"pct={pct}  vartheta={vartheta_str!r}"
    print(f"\n  ── {label} ──")

    # ── Ensure prc/ parent directory exists ──────────────────────────────────
    if not dry_run:
        (BASE_DIR / "prc").mkdir(parents=True, exist_ok=True)

    # ── Step 1: evaluate detection + reasoning ───────────────────────────────
    print("  [1] evaluate_FitAP_vlm.py")
    rc = _run([
        sys.executable, str(BASE_DIR / "eval" / "evaluate_FitAP_vlm.py"),
        "--model",          model,
        "--width",          str(width),
        "--height",         str(height),
        "--answers-in",     str(detect_dir),
        "--reasons-in",     str(reason_dir),
        "--questions-file", str(_questions_file(pct, sample)),
        "--no-draw", "--no-plots",
        "--split",          split,
        "--flip-vartheta",  vartheta_arg,
    ], env_extra=_FITAP_ENV, dry_run=dry_run)
    if rc != 0:
        print(f"  [WARN] step 1 returned rc={rc}")

    # ── Step 2: rename prc directory ─────────────────────────────────────────
    prc_raw   = _prc_dir_raw(model, split, vartheta_arg)
    prc_final = _prc_dir_named(model, pct, vartheta_arg) if rename else prc_raw

    if rename and prc_raw != prc_final:
        print(f"  [2] rename prc dir")
        print(f"      {prc_raw}")
        print(f"      → {prc_final}")
        if not dry_run:
            if not prc_raw.exists():
                print(f"  [WARN] source prc dir not found; skipping rename: {prc_raw}")
                prc_final = prc_raw          # fall back to raw so step 3 can still try
            elif prc_final.exists():
                print(f"  [WARN] target prc dir already exists; skipping rename: {prc_final}")
            else:
                prc_raw.rename(prc_final)
    else:
        print("  [2] rename skipped")

    row: dict = {"vartheta": tag}

    # ── Steps 3 & 4 only for non-baseline values ──────────────────────────────
    if is_baseline:
        print("  [3–4] skipped (this is the baseline)")
        return row

    # ── Step 3: pooled-GLM comparison against baseline ───────────────────────
    print("  [3] evaluate_FitAP_vlm.py --compare-csvs")
    baseline_prc = _prc_dir_named(model, pct, baseline_vartheta_arg) if rename else _prc_dir_raw(model, split, baseline_vartheta_arg)
    baseline_csv = _metrics_csv(baseline_prc, model, baseline_vartheta_arg)
    current_csv  = _metrics_csv(prc_final,    model, vartheta_arg)

    if not dry_run:
        for p in (baseline_csv, current_csv):
            if not p.exists():
                print(f"  [WARN] CSV not found (step 3 may fail): {p}")

    rc, step3_out = _run_capture([
        sys.executable, str(BASE_DIR / "eval" / "evaluate_FitAP_vlm.py"),
        "--model",          model,
        "--width",          str(width),
        "--height",         str(height),
        "--questions-file", str(_questions_file(pct, sample)),
        "--compare-csvs",   f"{baseline_csv}, {current_csv}",
        "--split",          split,
    ], env_extra={**_FITAP_ENV, "LAMBDA_FMT_FAIL": "10"}, dry_run=dry_run)
    if rc != 0:
        print(f"  [WARN] step 3 returned rc={rc}")
    row.update(_parse_step3_output(step3_out))
    row.update(_parse_mediation_output(step3_out))

    # ── Step 4: indoor/outdoor flip rate ─────────────────────────────────────
    print("  [4] fliprate.py")
    baseline_indout = indout_dir / f"{model}_{baseline_vartheta_arg}.jsonl"
    current_indout  = indout_dir / f"{model}_{vartheta_arg}.jsonl"

    if not dry_run:
        for p in (baseline_indout, current_indout):
            if not p.exists():
                print(f"  [WARN] indout JSONL not found (step 4 may fail): {p}")

    rc, step4_out = _run_capture([
        sys.executable, str(BASE_DIR / "fliprate.py"),
        str(baseline_indout),
        str(current_indout),
    ], dry_run=dry_run)
    if rc != 0:
        print(f"  [WARN] step 4 returned rc={rc}")
    row.update(_parse_step4_output(step4_out))

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vartheta sweep results produced by probe_and_sweep.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "vartheta_values",
        nargs="*",
        metavar="VARTHETA",
        help=f"Vartheta values to evaluate (default: {DEFAULT_VARTHETA_VALUES})",
    )
    parser.add_argument(
        "--pct",
        type=int,
        nargs="+",
        choices=[10, 20, 40, 60, 80, 100],
        metavar="PCT",
        help="Batch-size groups to evaluate (default: all). Use 100 for full dataset.",
    )
    parser.add_argument(
        "--sample", default="s00", metavar="SAMPLE",
        help="Sample name matching the probe_and_sweep.py run (default: s00). "
             "When --pct 100, s00 is the default and need not be specified.",
    )
    parser.add_argument(
        "--model", default=MODEL_DEFAULT, metavar="MODEL",
        help=f"Model name (default: {MODEL_DEFAULT}).",
    )
    parser.add_argument(
        "--split", default="_novel_clustered", metavar="SPLIT",
        help="Dataset split suffix passed to evaluate_FitAP_vlm.py "
             "(default: _novel_clustered).",
    )
    parser.add_argument("--width",  type=int, default=1000,
                        help="Image width override (default: 1000).")
    parser.add_argument("--height", type=int, default=1000,
                        help="Image height override (default: 1000).")
    parser.add_argument(
        "--baseline", default="none", metavar="VARTHETA",
        help="Baseline vartheta value used as comparison reference (default: none). "
             "If present in the vartheta list, it is always evaluated first.",
    )
    parser.add_argument(
        "--no-rename", action="store_true",
        help="Skip step 2 (prc directory rename). Steps 3 and 4 will use the raw "
             "prc directory names produced by step 1.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all commands without executing them.",
    )
    parser.add_argument(
        "--output-csv", metavar="PATH",
        help="Write per-vartheta metrics CSV (columns: vartheta, irr_indirect, "
             "95_ci_irr, irr_direct, dr_50, 95_ci_dr_50, switchrate).  "
             "A bare filename (no directory) is placed in reports/ under the repo root.",
    )
    parser.add_argument(
        "--output-suffix", default="", metavar="SUFFIX",
        help="Suffix appended to every answer directory name, e.g. '_v2' to read "
             "from answers/answers_bootstrap_{pct}_{task}_clustered_v2/ instead of the "
             "default.  Must match the --output-suffix used in the corresponding "
             "probe_and_sweep.py run (default: empty, i.e. use the standard name).",
    )
    parser.add_argument(
        "--answers-root", default=None, metavar="DIR",
        help="Root directory containing the answers_* subdirectories "
             "(default: <repo>/answers/).  Use this to point at a directory other "
             "than answers/, e.g. --answers-root /data/sweep_v2/answers.",
    )
    args = parser.parse_args()

    answers_root = Path(args.answers_root).resolve() if args.answers_root else BASE_DIR / "answers"
    vartheta_values = args.vartheta_values if args.vartheta_values else DEFAULT_VARTHETA_VALUES
    batch_configs = (
        [c for c in BATCH_CONFIGS if c["pct"] in args.pct]
        if args.pct else BATCH_CONFIGS
    )
    baseline_tag = _vartheta_tag(args.baseline)

    # Ensure the baseline value (if present) is evaluated before all others,
    # so its CSV exists when step 3 runs for subsequent values.
    ordered: list[str] = []
    rest:    list[str] = []
    for v in vartheta_values:
        (ordered if _vartheta_tag(v) == baseline_tag else rest).append(v)
    if not ordered:
        print(f"[WARN] baseline vartheta {args.baseline!r} not in the value list; "
              f"step 3 will rely on a pre-existing baseline CSV.")
    ordered += rest

    print(f"Subsample fraction  : {[c['pct'] for c in batch_configs]}")
    print(f"Sample       : {args.sample}")
    print(f"Model        : {args.model}")
    print(f"Split        : {args.split}")
    print(f"Baseline     : {args.baseline!r}  (tag: {baseline_tag!r})")
    print(f"Rename prc    : {not args.no_rename}")
    print(f"Output suffix : {args.output_suffix!r}")
    print(f"Answers root  : {answers_root}")
    print(f"Vartheta list : {ordered}")
    print(f"Dry run       : {args.dry_run}")
    print(f"Output CSV    : {args.output_csv or '(none)'}")

    # Clean up any existing prc/ directory from a previous run before writing
    # new step-1 outputs.  If the directory is non-empty, prompt the user to
    # rename it first so results from the previous run are not lost.
    prc_dir = BASE_DIR / "prc"
    if prc_dir.exists():
        contents = list(prc_dir.iterdir())
        if contents:
            pct_tag = "_".join(str(c["pct"]) for c in batch_configs)
            suffix_tag = args.output_suffix or ""
            suggested = f"prc_{args.sample}_pct{pct_tag}{suffix_tag}"
            print(f"\n[WARN] {prc_dir}/ is non-empty ({len(contents)} entr{'y' if len(contents)==1 else 'ies'}).")
            print(f"       It likely contains results from a previous run.  Rename it to")
            print(f"       preserve those results, e.g.:")
            print(f"         mv {prc_dir} {prc_dir.parent / suggested}")
            if not args.dry_run:
                try:
                    input("\n  Rename it now if needed, then press Enter to delete and continue (Ctrl+C to abort): ")
                except KeyboardInterrupt:
                    print("\nAborted.")
                    sys.exit(1)
        if args.dry_run:
            print(f"\n[dry-run] Would delete {prc_dir}/")
        else:
            shutil.rmtree(prc_dir)
            print(f"\n  Deleted {prc_dir}/")

    CSV_COLUMNS = ["vartheta", "irr_indirect", "95_ci_irr", "irr_direct", "dr_50", "95_ci_dr_50", "switchrate"]
    MEDIATION_COLUMNS = ["vartheta", "a", "SE(a)", "b", "SE(b)", "axb", "SE(axb)", "p-value", "irr_indirect", "95_pct_ci_irr"]
    rows: list[dict] = []

    for cfg in batch_configs:
        pct = cfg["pct"]
        print(f"\n{'='*64}")
        print(f"PCT = {pct}")
        print(f"{'='*64}")
        for vartheta_str in ordered:
            row = _evaluate_one(
                pct=pct,
                vartheta_str=vartheta_str,
                baseline_str=args.baseline,
                sample=args.sample,
                model=args.model,
                split=args.split,
                width=args.width,
                height=args.height,
                rename=not args.no_rename,
                output_suffix=args.output_suffix,
                answers_root=answers_root,
                dry_run=args.dry_run,
            )
            rows.append(row)

    if args.output_csv:
        csv_path = Path(args.output_csv)
        if not csv_path.parent.name or csv_path.parent == Path("."):
            csv_path = BASE_DIR / "reports" / csv_path.name
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {csv_path}")
        med_csv_path = csv_path.with_name(csv_path.stem + "_delta-mediation" + csv_path.suffix)
        with open(med_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MEDIATION_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {med_csv_path}")

    print("\nEvaluation sweep complete.")


if __name__ == "__main__":
    main()
