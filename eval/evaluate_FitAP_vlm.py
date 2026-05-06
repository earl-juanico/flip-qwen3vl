"""
evaluate_FitAP_vlm.py — FitAP evaluation and pooled GLM comparison for VLM bounding-box predictions on COCO.

The script has two operating modes:

  1. DETECTION PIPELINE  — parse a model's answer JSONL, compute per-sample
     IoU-weighted detection quality (det_quality = tp_iou / gt_count, i.e.
     IoU-weighted recall), FitAP (mean AP across IoU thresholds 0.50–0.95),
     and optionally tolerant count error (count_error_tol) from a paired
     reasoning JSONL.  Results are written to a per-condition CSV under prc/.

  2. POOLED GLM COMPARISON  — read two or more pre-built CSVs (one per FLIP
     vartheta variant) and run cluster-robust GLMs to estimate the treatment effect
     of the experimental condition on detection quality and count accuracy.
     This mode is triggered by --compare-csvs and does NOT require running
     the detection pipeline.


── ENVIRONMENT ──────────────────────────────────────────────────────────────

  export EVAL_DIR=/path/to/evaluate_FitAP_vlm.py   # default: script directory
  export COCO_ROOT=/path/to/coco/data               # default: /data/vlm/playground/


── DETECTION PIPELINE (mode 1) ──────────────────────────────────────────────

  python3 evaluate_FitAP_vlm.py \
      --model  <MODEL_NAME>          \  # e.g. Qwen3-VL-8B-Instruct
      --flip-vartheta <FLIP_VARTHETA_TAG> \  # e.g. s00_vartheta_none, s00_vartheta_-2.0
      --answers-in  <ANSWERS_FOLDER>     \  # folder of per-condition answer JSONL files
      [--reasons-in <REASONS_FOLDER>]\  # optional: folder of reasoning JSONL files
      [--width  <W>] [--height <H>]  \  # override VLM image space (e.g. 1000 1000)
      [--split  <SUFFIX>]               # dataset split suffix (default: _novel)

  --answers-in   Folder containing JSONL files named <MODEL>_<FLIP_VARTHETA>.jsonl.
                 Each line is a prediction record with 'question_id' and
                 either a 'bounding_boxes' key or bbox coordinates embedded
                 in 'text' (fenced JSON, bracketed lists, or LLaVA format).

  --reasons-in   Folder of reasoning JSONL files (same naming convention).
                 Each 'text' field should be a direct response to the prompt
                 "How many <object> are there? Answer briefly with integer."
                 Strict integer parsing (parse_strict_int) determines whether
                 a response is a format success (strict_ok=1) or failure
                 (strict_ok=0, penalised with LAMBDA_FMT_FAIL=10).
                 Enables count_error_tol and the consolidated E_combined metric.

  --width/--height  Override the image coordinate space assumed when
                    normalising raw VLM bbox coordinates (e.g. 1000×1000
                    for Qwen3-VL).  Per-image sizes from COCO metadata are
                    used when this flag is omitted.

  --split        Suffix appended to question/GT filenames, e.g. '_novel'
                 selects flip_qset_val_novel.jsonl (default: _novel).


── POOLED GLM COMPARISON (mode 2) ───────────────────────────────────────────

  python3 evaluate_FitAP_vlm.py \
      --model       <MODEL_NAME>   \
      --compare-csvs "<CSV1>,<CSV2>[,<CSV3>...]" \
      [--det-only-glm]             \  # skip count-error models; detection GLM only
      [--show-mismatches]             # print format-fail rows for debugging

  --compare-csvs   Comma-separated paths to per-condition CSVs (produced by
                   mode 1 with --reasons-in).  Each CSV represents one FLIP
                   vartheta variant; the first CSV is treated as the control condition.
                   run_pooled_glms() fits the following models:

                   • Binomial GLM  — P(format_fail=1) ~ is_experiment
                   • ZIP (T-model) — count_error_tol ~ is_experiment + det_quality
                     on strict-parse rows only (F=0), cluster-robust by image.
                   • Gaussian GLM  — det_quality_delta ~ is_experiment
                     pooled and stratified by size_bin (small / large).
                   • Poisson GLM  — count_error_tol ~ is_experiment + det_quality_delta
                   • Mediation    — product-of-coefficients (delta method) test of the
                     indirect effect of θ on tolerant count error T via the detection
                     pathway.  Formally:
                       a  = coef(is_experiment) from the Gaussian detection GLM
                       b  = coef(det_quality_delta) from the Poisson mediation GLM
                       indirect effect on log(T) = a × b
                       SE(a×b) = sqrt(b²·se_a² + a²·se_b²)   [Sobel / delta method]
                     Reports z-statistic, two-sided p-value, 95% CI, and the
                     exponentiated IRR_indirect = exp(a×b).  An IRR_indirect < 1
                     that is statistically significant supports the claim that θ
                     indirectly reduces counting error through improved detection,
                     i.e. grounding-proxy compatibility of the θ→T pathway via detection recall.
                     Requires scipy (scipy.stats.norm); no additional CLI flag needed.

                   The consolidated metric E_combined per condition is:
                     E_i = P_hat(F=1)*LAMBDA_FMT_FAIL + (1-P_hat(F=1))*T_hat_i
                   averaged per condition; ΔE_combined = E_exp − E_ctrl.

  --det-only-glm   Run only the Gaussian detection-quality GLM.  Use when
                   --reasons-in was not available and count_error columns are absent.

  --show-mismatches  Print pooled rows where strict_ok=0 or where the parsed
                     integer count differs from the detected box count, to
                     help diagnose format-failure imputations.


── KEY METRICS ──────────────────────────────────────────────────────────────

  det_quality          IoU-weighted recall per sample:
                         tp_iou / gt_count  where  tp_iou = tp × mean_IoU
  det_quality_delta    Deviation of det_quality from the control-condition mean.
  count_error_tol      Tolerant count error: |predicted_count − gt_count|
                       clipped to [0, gt_count] (partial credit).
                       Set to LAMBDA_FMT_FAIL (=10) when strict_ok=0.
  strict_ok            1 if the reasoning 'text' parsed as a bare integer; 0 otherwise.
  size_bin             Stratification by GT object count per image:
                         'small' if gt_count ≤ 3, 'large' if gt_count > 3.
                       Note: this is not bounding-box area — it is the number
                       of ground-truth instances of the queried class in the image.
  E_combined           Consolidated expected error per condition (see above).


── EXAMPLE ──────────────────────────────────────────────────────────────────

  # Compare flip_none (control) vs flip_-2.0 and flip_-2.5 (experiments)
  python3 evaluate_FitAP_vlm.py \
      --model Qwen3-VL-8B-Instruct \
      --compare-csvs \
        "prc/prc-Qwen3-VL-8B-Instruct_novel_flip_none/metrics_reason_det_Qwen3-VL-8B-Instruct_flip_none.csv,\
         prc/prc-Qwen3-VL-8B-Instruct_novel_flip_-2.0/metrics_reason_det_Qwen3-VL-8B-Instruct_flip_-2.0.csv,\
         prc/prc-Qwen3-VL-8B-Instruct_novel_flip_-2.5/metrics_reason_det_Qwen3-VL-8B-Instruct_flip_-2.5.csv"


── PERFORMANCE OPTIMIZATIONS ────────────────────────────────────────────────

  The following CPU-side optimisations were applied to the detection
  pipeline.  All numerical results are identical to the original code.

  1. IoU matrix pre-computation (Change 1)
     The IoU matrix L, class-match matrix N, and confidence-weighted
     matrix C are now computed ONCE per sample before the iou_th loop.
     Previously they were recomputed redundantly for each of the 10 IoU
     thresholds (0.50-0.95), giving 10x wasted work.  Only the
     assignment matrix Q (which depends on iou_th) is rebuilt per
     threshold, using numpy np.where instead of Python nested for-loops.
     Expected speedup: ~10x on the inner detection loop.

  2. Optional image annotation (Change 2)   --no-draw
     draw_bboxes() reads a full JPEG from disk, annotates it with
     OpenCV, and writes it back for every sample at IoU=0.50.  This is
     pure CPU + sequential disk I/O.  Pass --no-draw to skip it when
     only the metrics CSV is needed.  Visualisation files are still
     produced by default.

  3. Optional PR-curve PNG output (Change 3)  --no-plots
     calculate_ap() renders a matplotlib figure for every (class,
     iou_threshold) pair (up to N_classes x 10 PNG files).  Pass
     --no-plots to skip savefig() when the PNG files are not needed.
     AP values are computed and printed regardless.

  4. O(n) cumsum for AP precision-recall (Change 4)
     The inner AP loop previously called sum(sorted_list[:k+1]) for
     each k, giving O(n^2) cost.  This is replaced with numpy cumsum,
     reducing it to O(n).
"""
import os
import re
import json
import ast
import argparse
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import sys

# retained constant (used as conservative penalty for consolidated metric)
LAMBDA_FMT_FAIL = float(os.environ.get("LAMBDA_FMT_FAIL", 0.0))
_INT_STRICT_RE = re.compile(r"^[+-]?\d+$")

def parse_strict_int(s: str):
    """Return int if the entire string is a strict integer token; else None."""
    if s is None:
        return None
    t = str(s).strip()
    if _INT_STRICT_RE.match(t):
        try:
            return int(t)
        except Exception:
            return None
    return None

# EVAL_DIR is the repo root (one level above eval/).  Configurable via env var.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.environ.get('EVAL_DIR', os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)  # so eval_coco / calculate_ap_multiple are importable

from eval_coco import get_predictions_from_llava, draw_bboxes
from calculate_ap_multiple import PR, calculate_ap, calc_iou

# CLI args: allow overriding image width/height used for normalization
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--width', type=int, default=None, help='Override image width (pixels) for bbox normalization')
parser.add_argument('--height', type=int, default=None, help='Override image height (pixels) for bbox normalization')
parser.add_argument('--flip-vartheta', type=str, default="none", help='FLIP vartheta tag to use')
parser.add_argument('--model', type=str, default="Qwen2.5-VL-7B-Instruct", help='Model name to use')
parser.add_argument('--answers-in', type=str, default=None, help='Folder containing answers JSONL files')
parser.add_argument('--reasons-in', type=str, default=None, help='Folder containing reasoning JSONL files')
# NEW: allow pooled CSV comparison via CLI
parser.add_argument('--compare-csvs', type=str, default=None, help='Comma-separated CSV paths to compare (pooled GLMs)')
# RESTORED: control to run only detection GLM (skip count-error models)
parser.add_argument('--det-only-glm', action='store_true', help='Run only detection quality delta GLM, skip count-error models')
# RESTORED: print pooled mismatch rows (format failures / strict mismatches)
parser.add_argument('--show-mismatches', action='store_true', help='Print mismatches between GT and reasoning counts')
parser.add_argument('--split', type=str, default='_novel', help='Dataset split suffix for questions/GT files (default: _novel)')
parser.add_argument('--questions-file', type=str, default=None,
                    help='Path to questions JSONL (overrides the default questions file derived from --split). '
                         'Use this to evaluate on a bootstrap subsample, e.g. question_pct20_s00.jsonl')
# Change 2: skip per-sample image annotation (cv2 read+draw+write) when not needed
parser.add_argument('--no-draw', action='store_true', help='Skip draw_bboxes image annotation output (faster metric-only runs)')
# Change 3: skip matplotlib PR-curve PNG generation when not needed
parser.add_argument('--no-plots', action='store_true', help='Skip saving PR-curve PNG files (AP values are still computed and printed)')
args, _ = parser.parse_known_args()

# When --compare-csvs is the only goal, skip the detection pipeline entirely.
_compare_csvs_only = bool(getattr(args, 'compare_csvs', None))

OVERRIDE_W = args.width
OVERRIDE_H = args.height
if OVERRIDE_W is not None or OVERRIDE_H is not None:
    print(f"[info] Using override image size: width={OVERRIDE_W} height={OVERRIDE_H}")

ablation = ''
split = args.split

DIR = EVAL_DIR
questions_file = (
    os.path.expanduser(args.questions_file)
    if getattr(args, 'questions_file', None)
    else os.path.join(DIR, 'questions', f'question_pct100_s00.jsonl')
)

LOADER="qwen"
MODEL = args.model
FLIP_VARTHETA = args.flip_vartheta
answers_folder = args.answers_in
reasons_folder = args.reasons_in
if answers_folder is None:
    answers_folder = f"answers_{LOADER}_dissector"
answers_file = os.path.join(DIR, answers_folder, f"{MODEL}_{FLIP_VARTHETA}.jsonl")

if reasons_folder is not None:
    reasons_file = os.path.join(DIR, reasons_folder, f"{MODEL}_{FLIP_VARTHETA}.jsonl")
    print(f"[info] Using reasoning file: {reasons_file}")

MODEL_NAME = MODEL
prompt_mapping = {
    '': 'f',
    '_ablation00': '0',
    '_ablation01': '1',
    '_ablation02': '2',
    '_ablation03': '3',
    '_ablation04': '4',
    '_ablation05': '5',
    '_ablation06': '6',
    '_ablation07': '7',
    '_ablation08': '8',
    '_ablation09': '9',
}
ROOT_DIR = DIR
TRUTHS = f"coco_gt_val2017{split}.jsonl"

# COCO root configurable
COCO_ROOT = os.environ.get('COCO_ROOT', '/data/vlm/playground/')
if os.path.basename(COCO_ROOT.rstrip('/')) == 'data':
    DATA_ROOT = COCO_ROOT
else:
    DATA_ROOT = os.path.join(COCO_ROOT, 'data')

coco_data_json = os.path.join(DATA_ROOT, 'coco', 'annotations', 'instances_val2017.json')
image_dir = os.path.join(DATA_ROOT, 'coco', 'val2017')
PRC_DIR = os.path.join(ROOT_DIR, 'prc', f'prc-{MODEL_NAME}{split}{ablation}_{FLIP_VARTHETA}')
os.makedirs(PRC_DIR, exist_ok=True)

with open(coco_data_json, 'r') as f:
    coco_data = json.load(f)
objects = coco_data.get('annotations', [])
images = coco_data.get('images', [])
cats = coco_data.get('categories', [])

questions = []
with open(questions_file, 'r') as f:
    for line in f:
        q = json.loads(line)
        questions.append(q)

# Index questions by question_id for O(1) join with GT and answers.
q_by_qid = {str(q.get('question_id')): q for q in questions}

coco_gt_file = os.path.join(ROOT_DIR, 'gt', TRUTHS)

# Load the full GT file into a dict keyed by question_id so that bootstrap
# subsamples (which are reordered subsets) can be joined correctly instead of
# relying on coincidental line-number alignment.
gt_by_qid = {}
with open(coco_gt_file, 'r') as f:
    for line in f:
        gt = json.loads(line)
        gt_by_qid[str(gt.get('question_id'))] = gt

# Parallel lists are rebuilt after answer loading via question_id join (below).
# Kept as empty placeholders so code that checks their existence still works.
image_name = []
index = []
grounds = []
cls = []

qid_to_image = {q.get('question_id'): q.get('image') for q in questions}

def normalize_bbox(bbox, w, h):
    """Normalize a bbox to normalized XYXY ([x_min,y_min,x_max,y_max]) coordinates.

    Behavior:
    - If input values are already in [0,1], try to preserve the input format
      semantics: detect normalized XYWH and convert it to normalized XYXY.
      If values already look like normalized XYXY, return them unchanged.
    - If input values are absolute (outside [0,1]), detect whether they are
      absolute XYWH (x,y,w,h) or absolute XYXY (x_min,y_min,x_max,y_max) and
      convert to normalized XYXY.
    """
    try:
        vals = [float(v) for v in bbox]
    except Exception:
        raise ValueError(f"Invalid bbox values: {bbox}")

    if len(vals) != 4:
        raise ValueError(f"Invalid bbox format: {bbox}. Expected 4 values.")

    # Case A: already normalized in [0,1]
    if all(0.0 <= v <= 1.0 for v in vals):
        x0, y0, x2, y2 = vals
        # Heuristic: if third/fourth elements are widths/heights (XYWH), then
        # x0 + x2 <= 1 and y0 + y2 <= 1 typically hold. Convert to XYXY.
        if (x0 + x2 <= 1.0) and (y0 + y2 <= 1.0) and not (x2 > x0 and y2 > y0):
            x_min = x0
            y_min = y0
            x_max = min(1.0, x0 + x2)
            y_max = min(1.0, y0 + y2)
            return [x_min, y_min, x_max, y_max]
        # Otherwise assume already XYXY normalized; return as-is
        return vals

    # Case B: absolute coordinates (not in [0,1])
    # Detect absolute XYWH if x + w <= image width and y + h <= image height
    x0, y0, x2, y2 = vals
    try:
        fw = float(w)
        fh = float(h)
    except Exception:
        fw = float(w)
        fh = float(h)

    if (x0 + x2 <= fw) and (y0 + y2 <= fh):
        # Treat as XYWH absolute
        x_min = x0 / fw
        y_min = y0 / fh
        x_max = (x0 + x2) / fw
        y_max = (y0 + y2) / fh
        return [x_min, y_min, x_max, y_max]
    else:
        # Treat as XYXY absolute
        x_min = x0 / fw
        y_min = y0 / fh
        x_max = x2 / fw
        y_max = y2 / fh
        return [x_min, y_min, x_max, y_max]

def parse_json_blocks(text, query_label=None):
    blocks = re.findall(r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```', text, flags=re.IGNORECASE)
    results = []
    for b in blocks:
        try:
            obj = json.loads(b)
        except Exception:
            try:
                obj = ast.literal_eval(b)
            except Exception:
                continue
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    for k in ('bbox_2d','bbox','bbox2d','bbox_2D'):
                        if k in item and isinstance(item[k], (list,tuple)) and len(item[k])==4:
                            if query_label is not None:
                                item_label = item.get('label', '')
                                if isinstance(item_label, list):
                                    item_label = item_label[0] if item_label else ''
                                if str(item_label).lower() != query_label.lower():
                                    continue
                            results.append(list(map(float,item[k])))
                elif isinstance(item, (list,tuple)) and len(item)==4:
                    results.append(list(map(float,item)))
        elif isinstance(obj, dict):
            for k in ('bbox_2d','bbox','bbox2d','bbox_2D'):
                if k in obj and isinstance(obj[k], (list,tuple)) and len(obj[k])==4:
                    if query_label is not None:
                        item_label = obj.get('label', '')
                        if isinstance(item_label, list):
                            item_label = item_label[0] if item_label else ''
                        if str(item_label).lower() != query_label.lower():
                            continue
                    results.append(list(map(float,obj[k])))
    return results

def parse_any_bracketed_lists(text):
    cand = re.findall(r'\[([^\[\]]+)\]', text)
    results = []
    for s in cand:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) == 4:
            try:
                nums = [float(x) for x in parts]
                results.append(nums)
            except Exception:
                continue
    return results

def is_all_zero(bboxes):
    if not bboxes:
        return True
    for b in bboxes:
        try:
            if any(float(v) != 0.0 for v in b):
                return False
        except Exception:
            return False
    return True

'''
This part processes the model's answers
by parsing and mapping them to 
normalized bounding boxes.
'''

predictions = []
pred_qids = []  # parallel to predictions; used for question_id-based join below
if not _compare_csvs_only:
    with open(answers_file, 'r') as f:
        for line_no, line in enumerate(f, 1):
            pred = json.loads(line)
            text = pred.get('text', '') or ''
            provided = pred.get('bounding_boxes', None)
            bboxes = []
            source = None

            if provided and isinstance(provided, list) and provided:
                bboxes = provided
                source = 'bounding_boxes'
            else:
                # Look up the GT class for this specific question_id so that
                # label-filtered bbox parsing works correctly for subsamples
                # (previously used cls[line_no-1] which assumed line-order alignment).
                _qid_str = str(pred.get('question_id'))
                query_label = gt_by_qid.get(_qid_str, {}).get('class', None)
                extr = parse_json_blocks(text, query_label=query_label)
                if extr:
                    bboxes = extr
                    source = 'fenced_json'
                else:
                    extr = parse_any_bracketed_lists(text)
                    if extr:
                        bboxes = extr
                        source = 'bracket_regex'
                    else:
                        try:
                            extr = get_predictions_from_llava(text)
                        except Exception:
                            extr = []
                        if extr and not is_all_zero(extr):
                            bboxes = extr
                            source = 'get_predictions_from_llava'
                        else:
                            bboxes = []
                            source = 'none'

            snippet = text.replace('\n',' ')[:200]
            print(f"[debug] line={line_no} qid={pred.get('question_id')} bbox_source={source} raw_bboxes={bboxes} text_snippet='{snippet}'")

            qid = pred.get('question_id')
            img_name = qid_to_image.get(qid, '')
            img_path = os.path.join(image_dir, img_name) if img_name else ''
            w, h = None, None

            if img_path and os.path.exists(img_path) and (OVERRIDE_W is None or OVERRIDE_H is None):
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        w_img, h_img = img.size
                        w, h = w_img, h_img
                except Exception as e:
                    print(f"[warn] Error opening image {img_path}: {e}")
                    w, h = None, None

            if (w is None or h is None) and img_name:
                meta = next((im for im in images if im.get('file_name') == img_name or im.get('file_name','').endswith(img_name)), None)
                if meta:
                    if w is None:
                        w = meta.get('width')
                    if h is None:
                        h = meta.get('height')

            if OVERRIDE_W is not None:
                w = OVERRIDE_W
            if OVERRIDE_H is not None:
                h = OVERRIDE_H

            if w is None or h is None:
                w, h = 1000, 1000

            norm_bboxes = []
            for bbox in bboxes:
                try:
                    norm_bbox = normalize_bbox(bbox, w, h)
                except Exception as e:
                    print(f"[warn] Failed to normalize bbox {bbox} for image {img_path}: {e}")
                    continue
                norm_bboxes.append(norm_bbox)

            print(f"[debug] qid={qid} normalized_bboxes={norm_bboxes} image_size=({w},{h})")

            predictions.append([
                (bbox, round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 4))
                for bbox in norm_bboxes
            ])
            pred_qids.append(str(pred.get('question_id')))

# ── question_id-based join ────────────────────────────────────────────────────
# Rebuild the parallel arrays (image_name, index, cls, grounds, predictions) in
# answers-file order, joined by question_id.  This replaces the previous implicit
# assumption that questions, GT, and answers share the same line ordering — an
# assumption that breaks for bootstrap subsamples (which are reordered subsets).
# For the full dataset (where all files are already aligned) the output is identical.
if not _compare_csvs_only:
    _joined = []
    for qid_str, pred_bboxes in zip(pred_qids, predictions):
        q  = q_by_qid.get(qid_str)
        gt = gt_by_qid.get(qid_str)
        if q is None or gt is None:
            print(f"[warn] question_id={qid_str} missing from questions or GT file; skipping")
            continue
        _joined.append((
            q.get('image', ''),
            q.get('question_id', ''),
            gt.get('class', ''),
            gt.get('answers', []),
            pred_bboxes,
        ))
    image_name, index, cls, grounds, predictions = map(list, zip(*_joined)) if _joined else ([], [], [], [], [])
    print(f"[info] Joined {len(image_name)} samples via question_id (answers={len(pred_qids)}, questions={len(q_by_qid)}, gt={len(gt_by_qid)})")
# ─────────────────────────────────────────────────────────────────────────────

'''
This part calculates the FitAP using the
Q-matrix method to optimally match
predicted and ground truth boxes.

Visualize the PR curves in /prc/ directory.
'''

detections = defaultdict(list)
truths = defaultdict(list)
for k, (im, idx, c, g, p) in enumerate(zip(image_name, index, cls, grounds, predictions)):
    detections[(im, str(idx))].append([(c, f'{bbox[0]}', round(bbox[1], 2)) for bbox in p])
    truths[(im, str(idx))].append([(c, f'{bbox}') for bbox in g])

iou_th_values = np.round(np.arange(0.5, 1.0, 0.05), 2) if not _compare_csvs_only else np.array([])
classes = [cat['name'] for cat in cats]
total_grounds_per_class = Counter({c: 0 for c in classes})
for ((image, idx), dets), ((image2, idx2), grnds) in zip(detections.items(), truths.items()):
    dt_bboxes = dets[0]
    gt_bboxes = grnds[0]
    gt_counts = Counter([g[0] for g in gt_bboxes])
    for c in gt_counts:
        total_grounds_per_class[c] += gt_counts[c]

classes = [c for c in classes if total_grounds_per_class[c] > 0]

det_metrics_iou50 = {}  # key: str(qid) -> dict with TP/FP/gt_count/det_count/precision/recall/image/class
pr = defaultdict(list)

# Change 1: Pre-compute IoU (L), class-match (N), and confidence*IoU (C) matrices once
# per sample before the iou_th loop.  These are threshold-independent; only Q depends on
# iou_th and is rebuilt cheaply each iteration using numpy vectorised ops (np.where).
_sample_cache = {}  # (image, idx) -> (dt_bboxes, gt_bboxes, L, N, C)
if not _compare_csvs_only:
    for ((image, idx), dets), ((_, __), grnds) in zip(detections.items(), truths.items()):
        dt_bboxes = dets[0]
        gt_bboxes = grnds[0]
        ng, nd = len(gt_bboxes), len(dt_bboxes)
        L = np.zeros((ng, nd))
        N = np.zeros((ng, nd))
        C = np.zeros((ng, nd))
        for i, g in enumerate(gt_bboxes):
            for j, d in enumerate(dt_bboxes):
                try:
                    L[i, j] = calc_iou(ast.literal_eval(g[1]), ast.literal_eval(d[1]))
                except Exception:
                    L[i, j] = 0.0
                N[i, j] = 1.0 if g[0] == d[0] else -1.0
                C[i, j] = d[2] * L[i, j]
        _sample_cache[(image, idx)] = (dt_bboxes, gt_bboxes, L, N, C)

for iou_th in iou_th_values:
    for ((image, idx), dets), ((image2, idx2), grnds) in zip(detections.items(), truths.items()):
        # Change 1: reuse pre-computed L, N, C; build Q with numpy vectorised ops
        dt_bboxes, gt_bboxes, L, N, C = _sample_cache[(image, idx)]
        if L.size > 0:
            Q = np.where(L >= iou_th,
                         np.where(N == 1, L, -L),
                         np.where(L != 0, -1.0 - L, 0.0))
        else:
            Q = np.zeros_like(L)
        if Q.size > 0:
            Q_row_min = np.max(-Q, axis=1, keepdims=True)
            Q = np.where(Q < -1, np.where(Q == -Q_row_min, -1, -2), Q)
            Q_row_max = np.max(np.where(Q == 0, -3, Q), axis=1, keepdims=True)
            temp = np.where(Q == Q_row_max, Q, -4)
            Q = np.where(Q == temp, Q, -4)
            Q_col_max = np.max(np.where(Q == -4, -5, Q), axis=0)
            temp = np.where(Q == Q_col_max, Q, 0)
            Q = np.where(Q == temp, Q, 0)
        Q_row_sum = np.sum(Q, axis=1)
        nonzero_indices = [np.nonzero(row)[0] for row in Q]
        for i, g in enumerate(gt_bboxes):
            for j in nonzero_indices[i]:
                if Q_row_sum[i] > 0:
                    pr[(g[0], round(iou_th, 2))].append((C[i, j], 1))
                elif Q_row_sum[i] < 0:
                    pr[(g[0], round(iou_th, 2))].append((C[i, j], 0))
        dts = [dt_box[1] for dt_box in dt_bboxes]
        gts = [gt_box[1] for gt_box in gt_bboxes]
        iou = np.amax(L, axis=0) if L.size>0 else np.array([])
        cls_name = [dt_box[0] for dt_box in dt_bboxes]
        TP = np.count_nonzero(Q_row_sum > 0)
        FP = np.count_nonzero(Q_row_sum < 0)
        if round(iou_th,2) == 0.5:
            image_path = os.path.join(image_dir, image)
            image_out_path = os.path.join(PRC_DIR, image)
            # Change 2: skip per-sample image annotation when --no-draw is set
            if not args.no_draw:
                draw_bboxes(image_path, cls_name, gts, dts, TP, FP, iou, image_out_path)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / len(gt_bboxes) if len(gt_bboxes) > 0 else 0
        ntruths = len(gt_bboxes)
        non_zero_elements = L[Q != 0] if L.size>0 else np.array([])
        q_average = np.mean(non_zero_elements) if non_zero_elements.size > 0 else 0
        # --- NEW: collect per-sample detection metrics at IoU=0.50 ---
        if round(iou_th, 2) == 0.5:
            qid_str = str(idx)
            det_count = len(dt_bboxes)
            gt_count = len(gt_bboxes)
            fn = max(0, gt_count - int(TP))
            det_metrics_iou50[qid_str] = {
                "image": image,
                "qid": qid_str,
                "class": gt_bboxes[0][0] if len(gt_bboxes) > 0 else None,
                "gt_count": int(gt_count),
                "det_count": int(det_count),
                "tp": int(TP),
                "fp": int(FP),
                "fn": int(fn),
                "precision": float(precision),
                "recall": float(recall),
                "q_avg_iou": float(q_average),
            }

# save det_metrics_iou50 for later use (GLM with reasoning)
if not _compare_csvs_only:
    try:
        det_metrics_out = os.path.join(PRC_DIR, f'det_metrics_iou50_{MODEL}_{FLIP_VARTHETA}.json')
        with open(det_metrics_out, 'w') as dmf:
            json.dump(det_metrics_iou50, dmf, indent=2)
        print(f"[info] Saved per-sample IoU=0.50 detection metrics to: {det_metrics_out}")
    except Exception as e:
        print(f"[warn] Failed to save det_metrics_iou50: {e}")

# NEW: det-only CSV export (no reasons needed)
if not _compare_csvs_only and reasons_folder is None and args.det_only_glm:
    try:
        import pandas as pd
        det_rows = []
        for qid, dm in det_metrics_iou50.items():
            det_rows.append({
                "model": MODEL,
                "flip_vartheta": FLIP_VARTHETA,
                "image": dm.get("image"),
                "qid": qid,
                "class": dm.get("class"),
                "gt_count": int(dm.get("gt_count", 0)),
                "det_count": int(dm.get("det_count", 0)),
                "tp": int(dm.get("tp", 0)),
                "fp": int(dm.get("fp", 0)),
                "fn": int(dm.get("fn", 0)),
                "precision": float(dm.get("precision", 0.0)),
                "recall": float(dm.get("recall", 0.0)),
                "q_avg_iou": float(dm.get("q_avg_iou", 0.0)),
            })
        det_df = pd.DataFrame(det_rows)
        out_csv = os.path.join(PRC_DIR, f"metrics_det_only_{MODEL}_{FLIP_VARTHETA}.csv")
        det_df.to_csv(out_csv, index=False)
        print(f"[info] Saved det-only per-sample metrics to: {out_csv}")
    except Exception as e:
        print(f"[warn] Failed to save det-only CSV: {e}")

'''
Calculate the FitAP from the PR data
for a given IoU threshold and class.
The overall FitAP is the mean
across all classes and IoU thresholds,
similar to the definition of mean AP.
'''

AP = []
if not _compare_csvs_only:
    for iou_th in iou_th_values:
        iou_th = round(iou_th, 2)
        aP = []
        for c in classes:
            key = (c, iou_th)
            sorted_list = sorted(pr.get(key, []), key=lambda x: x[0], reverse=True)
            # Change 4: O(n) cumsum replaces the O(n^2) repeated sum(sorted_list[:k+1])
            G = total_grounds_per_class[key[0]]
            PrecisionRecall = defaultdict(list)
            if sorted_list:
                labels = np.array([p[1] for p in sorted_list], dtype=float)
                cum_tp = np.cumsum(labels)
                for k in range(len(sorted_list)):
                    precision = cum_tp[k] / (k + 1)
                    recall = cum_tp[k] / G if G > 0 else 0.0
                    PrecisionRecall[key].append((precision, recall))
            points = PrecisionRecall.get(key, [])
            ap, Curve = calculate_ap(points)
            aP.append(ap)
            # Change 3: skip figure annotation and PNG save when --no-plots is set
            if not args.no_plots:
                plt.text(0.95, 0.95, f'$\Theta$ = {iou_th:.2f}\n AP = {ap:.2f}',
                         horizontalalignment='right', verticalalignment='top',
                         transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
                Curve.savefig(os.path.join(PRC_DIR, f'AP_{c}__iou_{iou_th:.2f}.png'), format='png')
            plt.close()
        print(f'AP@{iou_th}: {np.mean(aP) if aP else 0}')
        AP.append(aP)
    print('*' * 10)
    print(f'AP: {np.mean(AP) if AP else 0}')
    print('*' * 10)


#================================================
# Determine correlation between object detection
# and counting reasoning (if reasoning file provided)
# To avoid mismatches in dimension, only consider
# those question IDs present in both predictions
# and reasoning files.
#================================================
if not _compare_csvs_only and reasons_folder is not None:
    # read reasoning counts
    reasoning_counts = {}
    with open(reasons_file, 'r') as f:
        for line_no, line in enumerate(f, 1):
            resp = json.loads(line)
            qid = resp.get('question_id')
            text = resp.get('text', '') or ''

            strict_int = parse_strict_int(text)
            if strict_int is not None:
                strict_ok = 1
                strict_val = strict_int
                first_int = strict_int
            else:
                strict_ok = 0
                strict_val = None
                m_first = re.search(r'\b([+-]?\d+)\b', text)
                first_int = int(m_first.group(1)) if m_first else None

            reasoning_counts[str(qid)] = {
                "strict_ok": int(strict_ok),
                "reason_count_strict": strict_val,
                "reason_count_first": first_int,
            }

    # Join detection metrics (IoU=0.50) with reasoning counts
    rows = []
    common_qids = set(reasoning_counts.keys()) & set(det_metrics_iou50.keys())

    for qid in sorted(common_qids):
        r = reasoning_counts.get(qid, {})
        dm = det_metrics_iou50[qid]
        gt_count = dm["gt_count"]

        strict_ok = int(r.get("strict_ok", 0))
        rc_strict = r.get("reason_count_strict", None)
        rc_first = r.get("reason_count_first", None)

        _MAX_COUNT = 100  # guard against VLM hallucinating absurdly large integers
        if rc_strict is not None:
            rc_strict = min(int(rc_strict), _MAX_COUNT)
        if rc_first is not None:
            rc_first = min(int(rc_first), _MAX_COUNT)

        strict_format_failure = int(strict_ok == 0)

        rc_for_diag = rc_strict if strict_ok == 1 and rc_strict is not None else (rc_first if rc_first is not None else 0)

        abs_err = int(abs(int(rc_for_diag) - int(gt_count))) if strict_ok == 1 and rc_for_diag is not None else None

        valid_mismatch = 0
        if strict_ok == 1 and rc_strict is not None:
            valid_mismatch = int(abs(int(rc_strict) - int(gt_count)) > 1)

        if strict_ok == 1 and rc_strict is not None:
            ce_mag = int(max(abs(int(rc_strict) - int(gt_count)) - 1, 0))
            tolerant_ok = int(abs(int(rc_strict) - int(gt_count)) <= 1)
        else:
            ce_mag = None
            tolerant_ok = 0

        row = {
            "model": MODEL,
            "flip_vartheta": FLIP_VARTHETA,
            "image": dm["image"],
            "qid": qid,
            "class": dm["class"],
            "gt_count": int(gt_count),
            "det_count": int(dm["det_count"]),
            "tp": int(dm["tp"]),
            "fp": int(dm["fp"]),
            "fn": int(dm["fn"]),
            "precision": float(dm["precision"]),
            "recall": float(dm["recall"]),
            "q_avg_iou": float(dm["q_average"] if "q_average" in dm else dm.get("q_avg_iou", 0)),
            "reason_count": int(rc_for_diag),
            "exact_match": int((strict_ok == 1) and (rc_strict is not None) and (int(rc_strict) == int(gt_count))),
            "count_error": int(abs_err) if abs_err is not None else None,
            "count_error_tol": int(ce_mag) if ce_mag is not None else None,
            "count_error_norm": (float(ce_mag) / (1.0 + int(gt_count))) if (gt_count is not None and ce_mag is not None) else None,
            "strict_ok": int(strict_ok),
            "strict_format_failure": int(strict_format_failure),
            "valid_mismatch": int(valid_mismatch),
            "tolerant_ok": int(tolerant_ok),
        }
        rows.append(row)

    if not rows:
        print("[info] No valid joined (detection, reasoning, GT) rows found.")
    else:
        import pandas as pd
        dfm = pd.DataFrame(rows)

        if "q_avg_iou" in dfm.columns:
            dfm["tp_iou"] = dfm["tp"].astype(float) * dfm["q_avg_iou"].astype(float)
        else:
            dfm["tp_iou"] = dfm["tp"].astype(float)

        if "count_error" not in dfm.columns:
            dfm["count_error"] = (dfm["reason_count"].astype(float) - dfm["gt_count"].astype(float)).abs().astype(float).fillna(np.nan)

        dfm["format_fail"] = dfm.get("strict_format_failure", dfm.get("strict_ok", 0)).astype(int)

        if "count_error_tol" in dfm.columns:
            dfm["count_error_tol"] = pd.to_numeric(dfm["count_error_tol"], errors="coerce")
        else:
            dfm["count_error_tol"] = np.nan
            mask_strict = dfm["strict_ok"].astype(int) == 1
            if "count_error" in dfm.columns:
                derived = (pd.to_numeric(dfm.loc[mask_strict, "count_error"], errors="coerce") - 1).clip(lower=0)
                dfm.loc[mask_strict, "count_error_tol"] = derived.fillna(np.nan)

        out_csv = os.path.join(PRC_DIR, f"metrics_reason_det_{MODEL}_{FLIP_VARTHETA}.csv")
        dfm.to_csv(out_csv, index=False)
        print(f"[info] Saved per-sample metrics to: {out_csv}")
        print(f"[info] Joined rows: {len(dfm)}")

        em_rate = dfm["exact_match"].mean()
        fmt_rate = dfm["format_fail"].mean()
        mean_tol_valid = dfm.loc[dfm["format_fail"] == 0, "count_error_tol"].dropna().mean()
        tol_err_rate_valid = (dfm.loc[dfm["format_fail"] == 0, "count_error_tol"].dropna() > 0).mean() if dfm.loc[dfm["format_fail"] == 0].shape[0] > 0 else float("nan")
        print(f"[info] Exact-match reasoning accuracy vs GT: {em_rate:.3f}")
        print(f"[info] Format-fail rate P(F=1): {fmt_rate:.3f} | Mean T among F=0: {mean_tol_valid:.3f} | P(T>0 | F=0): {tol_err_rate_valid:.3f}")
        print(f"[info] Mean TP: {dfm['tp'].mean():.3f} | Mean Recall: {dfm['recall'].mean():.3f} | Mean Precision: {dfm['precision'].mean():.3f}")

        # GLM analyses under Option A:
        try:
            import statsmodels.formula.api as smf
            import statsmodels.api as sm
            import numpy as _np

            # GLM 1: Format-failure (Binomial / logit) ~ tp_iou
            try:
                try:
                    m_fmt = smf.glm(formula="format_fail ~ tp_iou", data=dfm, family=sm.families.Binomial()).fit(
                        cov_type='cluster', cov_kwds={'groups': dfm["image"]})
                    print("\n=== Binomial GLM: format_fail ~ tp_iou (cluster-robust by image) ===")
                except Exception as e_cl:
                    print(f"[warn] Cluster-robust SE failed for Binomial GLM (format_fail ~ tp_iou): {e_cl}")
                    m_fmt = smf.glm(formula="format_fail ~ tp_iou", data=dfm, family=sm.families.Binomial()).fit()
                    print("\n=== Binomial GLM: format_fail ~ tp_iou ===")
                print(m_fmt.summary())
            except Exception as e_fmt:
                print(f"[warn] Format-failure GLM failed: {e_fmt}")

            # GLM 2: Tolerant magnitude (T) on subset F==0 (strict responses only)
            dfm_valid = dfm[dfm["format_fail"] == 0].copy()
            if dfm_valid.shape[0] == 0:
                print("[info] No strict responses (F==0) to model tolerant magnitude T.")
            else:
                dfm_valid["count_error_tol"] = pd.to_numeric(dfm_valid["count_error_tol"], errors="coerce").fillna(0).astype(int)
                try:
                    m_err_tp = smf.glm(formula="count_error_tol ~ tp_iou", data=dfm_valid, family=sm.families.Poisson()).fit(
                        cov_type='cluster', cov_kwds={'groups': dfm_valid["image"]})
                    print("\n=== Poisson GLM: count_error_tol ~ tp_iou (cluster-robust by image) ===")
                except Exception as e_cl:
                    print(f"[warn] Cluster-robust SE failed for Poisson GLM (count_error_tol ~ tp_iou): {e_cl}")
                    m_err_tp = smf.glm(formula="count_error_tol ~ tp_iou", data=dfm_valid, family=sm.families.Poisson()).fit()
                    print("\n=== Poisson GLM: count_error_tol ~ tp_iou ===")
                print(m_err_tp.summary())
                irr_tp = float(_np.exp(m_err_tp.params.get("tp_iou", np.nan)))
                print(f"[info] IRR per +1 IoU-weighted TP (T tolerant magnitude): {irr_tp:.3f}")

                try:
                    m_err_recall = smf.glm(formula="count_error_tol ~ recall", data=dfm_valid, family=sm.families.Poisson()).fit(
                        cov_type='cluster', cov_kwds={'groups': dfm_valid["image"]})
                    print("\n=== Poisson GLM: count_error_tol ~ recall (cluster-robust by image) ===")
                except Exception as e_cl:
                    print(f"[warn] Cluster-robust SE failed for Poisson GLM (count_error_tol ~ recall): {e_cl}")
                    m_err_recall = smf.glm(formula="count_error_tol ~ recall", data=dfm_valid, family=sm.families.Poisson()).fit()
                    print("\n=== Poisson GLM: count_error_tol ~ recall ===")
                print(m_err_recall.summary())
                irr_recall = float(_np.exp(m_err_recall.params.get("recall", np.nan)))
                print(f"[info] IRR per +1.0 recall (T tolerant magnitude): {irr_recall:.3f}")

            # Poisson GLM: reason_count ~ det_count (keep as before)
            try:
                m_pois = smf.glm(formula="reason_count ~ det_count", data=dfm, family=sm.families.Poisson()).fit(
                    cov_type='cluster', cov_kwds={'groups': dfm["image"]})
                print("\n=== Poisson GLM: reason_count ~ det_count (cluster-robust by image) ===")
            except Exception as e_cl:
                print(f"[warn] Cluster-robust SE failed for Poisson GLM (reason_count ~ det_count): {e_cl}")
                m_pois = smf.glm(formula="reason_count ~ det_count", data=dfm, family=sm.families.Poisson()).fit()
                print("\n=== Poisson GLM: reason_count ~ det_count ===")
            print(m_pois.summary())
            irr = float(_np.exp(m_pois.params.get("det_count", np.nan)))
            print(f"[info] IRR per +1 detection (reason count): {irr:.3f}")

        except Exception as e:
            print(f"[warn] GLM analysis failed: {e}")

        # Plot: Reasoning count vs detection count + fitted Poisson curve (log y)
        out_png = os.path.splitext(reasons_file)[0] + '.png'
        xs_arr = dfm["det_count"].to_numpy()
        ys_arr = dfm["reason_count"].to_numpy()

        plt.figure(figsize=(6,6))
        plt.scatter(xs_arr, np.maximum(ys_arr, 1e-6), alpha=0.7, edgecolor='k')
        plt.yscale('log')
        plt.xlabel('Detection count')
        plt.ylabel('Reasoning count (log scale)')
        plt.title('Reasoning vs Detection Counts (log y vs x)')

        try:
            import statsmodels.api as sm
            X = sm.add_constant(xs_arr.astype(float), has_constant='add')
            y = np.maximum(0, np.round(ys_arr)).astype(int)
            res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            b0, b1 = float(res.params[0]), float(res.params[1])
            grid = np.linspace(xs_arr.min(), xs_arr.max(), 100)
            yfit = res.predict(sm.add_constant(grid, has_constant='add'))
            yfit = np.maximum(yfit, 1e-6)
            plt.plot(grid, yfit, linewidth=2, label=f'E[y]=exp({b0:.3f}+{b1:.3f}*x)')
            plt.legend(loc='center right', bbox_to_anchor=(0.98, 0.5))
        except Exception as e:
            print(f"[warn] Plot Poisson fit failed: {e}")

        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[info] Saved scatter plot to: {out_png}")

def run_pooled_glms(csv_paths, det_only_glm: bool = False, show_mismatches: bool = False):
    
    import os
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    dfs = []
    for p in csv_paths:
        p = p.strip()
        if not p:
            continue
        if not os.path.exists(p):
            print(f"[warn] CSV not found: {p}")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] Failed to read {p}: {e}")
            continue
        df['_source_file'] = os.path.basename(p)
        dfs.append(df)

    if not dfs:
        print("[warn] No CSVs loaded for pooled comparison.")
        return

    d = pd.concat(dfs, ignore_index=True)

    CONTROL_TAGS = {"vartheta_none", "none", "control", "baseline"}
    if 'flip_vartheta' in d.columns:
        r = d['flip_vartheta'].astype(str).str.strip().str.lower()
        r_clean = r.str.replace(r'\.\d+$', '', regex=True)
        # strip leading session prefix like "s00_" so "s00_vartheta_none" -> "vartheta_none"
        r_clean = r_clean.str.replace(r'^s\d+_', '', regex=True)
        d['is_experiment'] = (~r_clean.isin(CONTROL_TAGS)).astype(int)
    elif 'is_experiment' in d.columns:
        d['is_experiment'] = pd.to_numeric(d['is_experiment'], errors='coerce').fillna(0).astype(int)
    else:
        d['is_experiment'] = 0
        print("[warn] Neither 'flip_vartheta' nor 'is_experiment' found; defaulting is_experiment=0 for all rows.")

    if "gt_count" not in d.columns or "tp" not in d.columns:
        print("[warn] Required columns missing in pooled CSVs ('gt_count' and 'tp').")
        return

    # FAIL-CLOSED: require strict_ok to exist for Option A
    if "strict_ok" not in d.columns:
        print("[warn] 'strict_ok' column missing in pooled CSVs; Option A requires strict parse indicator. Aborting pooled count-error analyses.")
        return
    # normalize strict_ok
    d["strict_ok"] = pd.to_numeric(d["strict_ok"], errors="coerce").fillna(0).astype(int)
    d["format_fail"] = (d["strict_ok"].astype(int) == 0).astype(int)

    # ensure count_error exists or derive it
    if "count_error" not in d.columns:
        if "reason_count" in d.columns:
            rc = pd.to_numeric(d["reason_count"], errors="coerce")
            gt = pd.to_numeric(d["gt_count"], errors="coerce").fillna(0).astype(int)
            d["count_error"] = (rc - gt).abs()
        else:
            if not det_only_glm:
                print("[warn] 'count_error' not found and cannot be derived; aborting.")
                return

    # Preserve existing count_error_tol if present; otherwise derive for strict rows only
    if "count_error_tol" in d.columns:
        d["count_error_tol"] = pd.to_numeric(d["count_error_tol"], errors="coerce")
    else:
        d["count_error_tol"] = np.nan
        mask_strict = d["strict_ok"].astype(int) == 1
        if mask_strict.any() and "count_error" in d.columns:
            derived = (pd.to_numeric(d.loc[mask_strict, "count_error"], errors="coerce") - 1).clip(lower=0)
            d.loc[mask_strict, "count_error_tol"] = derived.fillna(np.nan)

    # ensure numeric
    d["count_error_tol"] = pd.to_numeric(d["count_error_tol"], errors="coerce")

    d["count_error_norm"] = (d["count_error_tol"].astype(float) / (pd.to_numeric(d["gt_count"], errors="coerce").fillna(0).astype(float) + 1.0))
    if "q_avg_iou" in d.columns:
        d['tp_iou'] = d['tp'].astype(float) * d['q_avg_iou'].astype(float)
    else:
        d['tp_iou'] = d['tp'].astype(float)
    d['gt_count_f'] = d['gt_count'].astype(float).replace(0, np.nan)
    d['tp_iou_norm'] = (d['tp_iou'] / d['gt_count_f']).fillna(0.0)
    # Use recall (TP/gt_count at IoU>=0.5) as detection quality: the direct
    # grounding proxy for counting error (coverage of GT objects found).
    if "recall" in d.columns:
        d['det_quality'] = pd.to_numeric(d['recall'], errors='coerce').fillna(0.0)
    else:
        d['det_quality'] = d['tp_iou_norm'].astype(float)
        print("[warn] 'recall' column not found in pooled CSVs; falling back to tp_iou_norm for det_quality.")
    d.loc[d['tp'].astype(int) == 0, 'det_quality'] = 0.0

    control_mean = d.loc[d['is_experiment'] == 0, 'det_quality'].mean()
    if np.isnan(control_mean):
        control_mean = 0.0
    d['det_quality_delta'] = d['det_quality'] - float(control_mean)

    d['size_bin'] = np.where(d['gt_count'].astype(int) <= 3, 'small', 'large')
    groups = d["image"] if "image" in d.columns else None

    print("\n================ POOL SUMMARY ================")
    print("rows:", len(d))
    print("is_experiment value counts:\n", d["is_experiment"].value_counts(dropna=False))
    print("mean tolerant count_error (T) by condition (among F==0):\n", d.loc[d['format_fail'] == 0].groupby("is_experiment")["count_error_tol"].mean())
    det_quality_label = "recall" if "recall" in d.columns else "tp_iou_norm"
    print(f"mean detection quality ({det_quality_label}) by condition:\n", d.groupby("is_experiment")["det_quality"].mean())
    print("control mean det_quality (used for ΔIoU):", float(control_mean))
    print("mean ΔIoU by condition:\n", d.groupby("is_experiment")["det_quality_delta"].mean())
    print("rows by size_bin:\n", d['size_bin'].value_counts())

    # ---- Consolidated metric combining per-row P(F=1) + per-row t_hat ----
    pen = float(LAMBDA_FMT_FAIL)  # conservative penalty for F==1 when creating consolidated metric

    # Fit binomial model for format failures (predict P(F=1))
    try:
        m_fmt_pool = smf.glm(formula="format_fail ~ is_experiment + tp_iou_norm", data=d, family=sm.families.Binomial()).fit()
        d['p_fmt_hat'] = m_fmt_pool.predict(d)
        p0 = float(d.loc[d['is_experiment'] == 0, 'p_fmt_hat'].mean())
        p1 = float(d.loc[d['is_experiment'] == 1, 'p_fmt_hat'].mean()) if (d['is_experiment']==1).any() else float('nan')
        print("[info] F-model fitted (pooled): mean predicted P(F=1) control/exp:", p0, p1)
    except Exception as e_pf:
        print(f"[warn] P(F) model failed: {e_pf}")
        d['p_fmt_hat'] = d['format_fail'].astype(float)
        p0 = float(d.loc[d['is_experiment'] == 0, 'p_fmt_hat'].mean())
        p1 = float(d.loc[d['is_experiment'] == 1, 'p_fmt_hat'].mean()) if (d['is_experiment']==1).any() else float('nan')

    # Fit T-model on strict responses (train on F==0), then PREDICT t_hat per-row across all rows
    d_valid = d[d['format_fail'] == 0].copy()
    # require finite count_error_tol in strict subset before fitting
    if d_valid.shape[0] > 0:
        d_valid = d_valid.dropna(subset=["count_error_tol"])
    t_model_used = None
    t_pred_full = None
    if d_valid.shape[0] == 0:
        print("[info] No strict responses with finite count_error_tol available to fit T-model.")
        d['t_hat'] = np.nan
    else:
        try:
            from statsmodels.discrete.count_model import ZeroInflatedPoisson
            try:
                zip_mod = ZeroInflatedPoisson.from_formula("count_error_tol ~ is_experiment + det_quality", data=d_valid, inflation="logit")
                zip_res = zip_mod.fit(disp=False)
                t_model_used = 'ZIP'
                # attempt to predict on full data; fallback to predicting on d_valid only
                try:
                    t_pred_full = zip_res.predict(d)
                except Exception:
                    t_pred_full = zip_res.predict(d_valid)
            except Exception:
                # fallback to Poisson GLM trained on strict rows
                m_err_pool = smf.glm(formula="count_error_tol ~ is_experiment + det_quality", data=d_valid, family=sm.families.Poisson()).fit()
                t_model_used = 'Poisson'
                try:
                    t_pred_full = m_err_pool.predict(d)
                except Exception:
                    t_pred_full = m_err_pool.predict(d_valid)
            # assign t_hat per-row; if prediction only on d_valid, map back and fill others with mean
            if t_pred_full is not None:
                if len(t_pred_full) == len(d):
                    d['t_hat'] = np.maximum(0.0, pd.to_numeric(t_pred_full, errors="coerce").astype(float))
                else:
                    # predictions correspond to d_valid
                    d['t_hat'] = np.nan
                    d.loc[d_valid.index, 't_hat'] = np.maximum(0.0, pd.to_numeric(t_pred_full, errors="coerce").astype(float))
                    mean_t_valid = float(d.loc[d_valid.index, 't_hat'].mean())
                    d['t_hat'].fillna(mean_t_valid, inplace=True)
            else:
                d['t_hat'] = np.nan
        except Exception as e_t:
            print(f"[warn] T-model training/prediction failed: {e_t}")
            # fallback: use empirical mean T among strict rows to fill t_hat
            mean_t_valid = float(d_valid['count_error_tol'].dropna().mean()) if d_valid.shape[0] > 0 else float('nan')
            d['t_hat'] = mean_t_valid
            t_model_used = 'fallback-mean'
        # compute group-level means for reporting
        try:
            t0_hat = float(d.loc[d['is_experiment'] == 0, 't_hat'].mean()) if (d['is_experiment']==0).any() else float('nan')
            t1_hat = float(d.loc[d['is_experiment'] == 1, 't_hat'].mean()) if (d['is_experiment']==1).any() else float('nan')
            print(f"[info] T-model ({t_model_used}) predictions: mean predicted T control/exp: {t0_hat}, {t1_hat}")
        except Exception:
            pass

    # Per-row consolidated expected tolerant magnitude
    # E_i = p_fmt_hat_i * pen + (1 - p_fmt_hat_i) * t_hat_i
    # ensure p_fmt_hat exists
    if 'p_fmt_hat' not in d.columns:
        d['p_fmt_hat'] = d['format_fail'].astype(float)
    if 't_hat' not in d.columns:
        d['t_hat'] = np.nan
    # fill any remaining NaN t_hat with 0 (or with mean of strict if available)
    if d['t_hat'].isna().any():
        fallback_t = float(d.loc[d['format_fail'] == 0, 't_hat'].dropna().mean()) if (d['format_fail'] == 0).any() else 0.0
        d['t_hat'].fillna(fallback_t, inplace=True)

    d['e_combined_hat'] = d['p_fmt_hat'].astype(float) * pen + (1.0 - d['p_fmt_hat'].astype(float)) * d['t_hat'].astype(float)

    # condition-level summaries
    E0 = float(d.loc[d['is_experiment'] == 0, 'e_combined_hat'].mean()) if (d['is_experiment'] == 0).any() else float('nan')
    E1 = float(d.loc[d['is_experiment'] == 1, 'e_combined_hat'].mean()) if (d['is_experiment'] == 1).any() else float('nan')
    delta_E_combined = E1 - E0 if (E1 == E1 and E0 == E0) else float('nan')
    print("[info] Consolidated expected tolerant magnitude (per-row averaging using PENALTY=LAMBDA_FMT_FAIL=%s):" % int(pen))
    print(f"  E_combined control: {E0:.3f} | E_combined experiment: {E1:.3f} | ΔE_combined (exp - ctrl): {delta_E_combined:.3f}")
    print("  (Per-row: E_i = p_hat_i*pen + (1-p_hat_i)*t_hat_i; averaged by condition)")

    # Detection GLM (ΔIoU)
    try:
        if groups is not None:
            try:
                m_det = smf.glm(formula="det_quality_delta ~ is_experiment", data=d, family=sm.families.Gaussian()).fit(
                    cov_type='cluster', cov_kwds={'groups': groups})
                print("\n=== GLM (Gaussian): ΔIoU ~ is_experiment (cluster-robust) ===")
            except Exception as e_cl:
                print(f"[warn] Cluster-robust SE failed for detection GLM: {e_cl}")
                m_det = smf.glm(formula="det_quality_delta ~ is_experiment", data=d, family=sm.families.Gaussian()).fit()
                print("\n=== GLM (Gaussian): ΔIoU ~ is_experiment (standard) ===")
        else:
            m_det = smf.glm(formula="det_quality_delta ~ is_experiment", data=d, family=sm.families.Gaussian()).fit()
            print("\n=== GLM (Gaussian): ΔIoU ~ is_experiment ===")
        print(m_det.summary())
        coef = m_det.params.get("is_experiment", np.nan)
        print("[info] Coef (experiment vs control) on ΔIoU:", float(coef))
    except Exception as e:
        print(f"[warn] Detection-quality GLM failed: {e}")

    if det_only_glm:
        return

    # If print mismatches requested
    if show_mismatches:
        # Prefer computing mismatches from the raw "text" in the --reasons-in JSONL
        reason_texts = {}
        if 'reasons_file' in globals() and reasons_file and os.path.exists(reasons_file):
            try:
                with open(reasons_file, 'r') as rf:
                    for ln in rf:
                        try:
                            j = json.loads(ln)
                            qid = str(j.get('question_id'))
                            txt = j.get('text', '') or ''
                            reason_texts[qid] = txt
                        except Exception:
                            continue
                print(f"[info] Loaded reason texts from: {reasons_file} (rows={len(reason_texts)})")
            except Exception as e:
                print(f"[warn] Failed to load reasons_file {reasons_file}: {e}")
                reason_texts = None
        else:
            reason_texts = None

        # helper to parse strict int and first-int from free text
        def _parse_from_text(txt):
            if txt is None:
                return (0, None, None)
            s = parse_strict_int(txt)
            if s is not None:
                return (1, int(s), int(s))
            m = re.search(r'\b([+-]?\d+)\b', txt)
            return (0, None, int(m.group(1)) if m else None)

        # Attach file-derived columns when available; otherwise fall back to existing columns
        if reason_texts is not None:
            d['_reason_text_file'] = d['qid'].astype(str).map(reason_texts).fillna('')
            parsed = d['_reason_text_file'].apply(lambda t: _parse_from_text(t))
            d['strict_ok_file'] = parsed.apply(lambda x: int(x[0]))
            d['reason_count_strict_file'] = parsed.apply(lambda x: x[1])
            d['reason_count_first_file'] = parsed.apply(lambda x: x[2])
        else:
            # best-effort: use existing columns if present
            d['strict_ok_file'] = pd.to_numeric(d.get('strict_ok', pd.Series([0]*len(d), index=d.index)), errors='coerce').fillna(0).astype(int)
            d['reason_count_strict_file'] = pd.to_numeric(d.get('reason_count', pd.Series([np.nan]*len(d), index=d.index)), errors='coerce')

        # define mismatch masks (based on file-derived strict parse where available)
        fmt_fail_mask_file = (d['strict_ok_file'].astype(int) == 0)
        int_mismatch_mask = (d['strict_ok_file'].astype(int) == 1) & (
            pd.to_numeric(d['reason_count_strict_file'], errors='coerce') != pd.to_numeric(d['gt_count'], errors='coerce')
        )

        n_total = len(d)
        n_fmt_fail = int(fmt_fail_mask_file.sum())
        n_int_mismatch = int(int_mismatch_mask.sum())
        n_any = int(((fmt_fail_mask_file) | (int_mismatch_mask)).sum())

        print(
            f"[info] Mismatches summary (based on reasons-in 'text'): "
            f"rows={n_total} | format_failures={n_fmt_fail} | int_mismatches={n_int_mismatch} | total_mismatch_rows={n_any}"
        )

        # print examples (limit to 20 each)
        limit = 20
        if n_int_mismatch > 0:
            print("[info] Examples: integer-parse mismatches (strict integer parsed but != gt_count)")
            for _, r in d[int_mismatch_mask].head(limit).iterrows():
                txt = (r.get('_reason_text_file') or "")[:200].replace("\n", " ")
                print(
                    f"[mismatch-int] qid={r.get('qid')} gt={r.get('gt_count')} parsed={r.get('reason_count_strict_file')} "
                    f"src={r.get('_source_file')} text_snippet='{txt}'"
                )

        if n_fmt_fail > 0:
            print("[info] Examples: format failures (non-integer 'text')")
            for _, r in d[fmt_fail_mask_file].head(limit).iterrows():
                txt = (r.get('_reason_text_file') or "")[:200].replace("\n", " ")
                # show any first-int found for diagnostics
                first_int = r.get('reason_count_first_file') if 'reason_count_first_file' in r.index else None
                print(
                    f"[format-fail] qid={r.get('qid')} gt={r.get('gt_count')} first_int={first_int} "
                    f"src={r.get('_source_file')} text_snippet='{txt}'"
                )

        # final tally
        print(
            f"[info] Done printing mismatches-from-text. totals: format_failures={n_fmt_fail}, int_mismatches={n_int_mismatch}, combined={n_any}"
        )

    # Stratified detection GLMs
    for size in ['small', 'large']:
        try:
            sub = d[d['size_bin'] == size]
            if len(sub) < 10:
                print(f"[info] Skipping stratified GLM for {size} (n={len(sub)})")
                continue
            sub_groups = sub["image"] if "image" in sub.columns else None
            if sub_groups is not None:
                try:
                    m_det_s = smf.glm(formula="det_quality_delta ~ is_experiment", data=sub, family=sm.families.Gaussian()).fit(
                        cov_type='cluster', cov_kwds={'groups': sub_groups})
                    print(f"\n=== GLM (Gaussian) ΔIoU ~ is_experiment for size_bin={size} (cluster-robust, n={len(sub)}) ===")
                except Exception as e_cl:
                    print(f"[warn] Cluster-robust SE failed for stratified detection GLM ({size}): {e_cl}")
                    m_det_s = smf.glm(formula="det_quality_delta ~ is_experiment", data=sub, family=sm.families.Gaussian()).fit()
                    print(f"\n=== GLM (Gaussian) ΔIoU ~ is_experiment for size_bin={size} (n={len(sub)}) ===")
            else:
                m_det_s = smf.glm(formula="det_quality_delta ~ is_experiment", data=sub, family=sm.families.Gaussian()).fit()
                print(f"\n=== GLM (Gaussian) ΔIoU ~ is_experiment for size_bin={size} (n={len(sub)}) ===")
            print(m_det_s.summary())
        except Exception as e:
            print(f"[warn] Stratified detection GLM failed for {size}: {e}")

    # Count-error modeling under Option A: only on strict responses (format_fail==0)
    d_valid = d[d['format_fail'] == 0].copy()
    if d_valid.shape[0] > 0:
        d_valid = d_valid.dropna(subset=["count_error_tol"])
    if d_valid.shape[0] == 0:
        print("[info] No strict responses across pooled CSVs to model tolerant magnitude T; skipping count-error GLMs.")
    else:
        # Zero-inflated Poisson (or fallback) for count_error_tol magnitude on strict responses
        zip_res = None
        m_err = None
        zip_groups = d_valid["image"].values if "image" in d_valid.columns else None
        try:
            zip_ok = False
            try:
                from statsmodels.discrete.count_model import ZeroInflatedPoisson
                try:
                    zip_mod = ZeroInflatedPoisson.from_formula("count_error_tol ~ is_experiment + det_quality", data=d_valid, inflation="logit")
                    try:
                        zip_res = zip_mod.fit(disp=False, cov_type='cluster', cov_kwds={'groups': zip_groups}) if zip_groups is not None else zip_mod.fit(disp=False)
                    except Exception as e_cl:
                        print(f"[warn] Cluster-robust SE failed for ZIP (formula): {e_cl}")
                        zip_res = zip_mod.fit(disp=False)
                    zip_ok = True
                except Exception:
                    endog = d_valid['count_error_tol'].astype(int)
                    exog = sm.add_constant(d_valid[['is_experiment', 'det_quality']].astype(float), has_constant='add')
                    exog_infl = sm.add_constant(d_valid[['is_experiment']].astype(float), has_constant='add')
                    zip_mod = ZeroInflatedPoisson(endog, exog, exog_infl=exog_infl, inflation='logit')
                    try:
                        zip_res = zip_mod.fit(disp=False, cov_type='cluster', cov_kwds={'groups': zip_groups}) if zip_groups is not None else zip_mod.fit(disp=False)
                    except Exception as e_cl:
                        print(f"[warn] Cluster-robust SE failed for ZIP (manual): {e_cl}")
                        zip_res = zip_mod.fit(disp=False)
                    zip_ok = True
            except Exception as e_zip:
                print(f"[warn] ZeroInflatedPoisson not available or failed to initialize: {e_zip}")
                zip_ok = False

            if zip_ok and zip_res is not None:
                label = "cluster-robust by image" if zip_groups is not None else "standard"
                print(f"\n=== ZeroInflatedPoisson: count_error_tol ~ is_experiment + det_quality ({det_quality_label}) ({label}, strict responses only) ===")
                print(zip_res.summary())
                try:
                    params = zip_res.params
                    if 'det_quality' in params.index:
                        print(f"[info] IRR per +1 det_quality ({det_quality_label}) (T tolerant magnitude): {float(np.exp(params['det_quality'])):.3f}")
                    elif 'tp_iou_norm' in params.index:
                        print(f"[info] IRR per +1 tp_iou_norm (T tolerant magnitude): {float(np.exp(params['tp_iou_norm'])):.3f}")
                    coef_exp = params.get('is_experiment', np.nan)
                    print(f"[info] IRR for is_experiment (T tolerant magnitude): {float(np.exp(coef_exp)):.3f}")
                except Exception:
                    pass
            else:
                if groups is not None:
                    try:
                        m_err = smf.glm(formula="count_error_tol ~ is_experiment + det_quality", data=d_valid, family=sm.families.Poisson()).fit(
                            cov_type='cluster', cov_kwds={'groups': d_valid["image"]})
                        print(f"\n=== Poisson GLM (fallback): count_error_tol ~ is_experiment + det_quality ({det_quality_label}) (cluster-robust) ===")
                    except Exception as e_cl:
                        print(f"[warn] Cluster-robust SE failed for Poisson fallback GLM: {e_cl}")
                        m_err = smf.glm(formula="count_error_tol ~ is_experiment + det_quality", data=d_valid, family=sm.families.Poisson()).fit()
                        print("\n=== Poisson GLM (fallback, strict responses only) ===")
                else:
                    m_err = smf.glm(formula="count_error_tol ~ is_experiment + det_quality", data=d_valid, family=sm.families.Poisson()).fit()
                    print("\n=== Poisson GLM (fallback, strict responses only) ===")
                print(m_err.summary())
                params = m_err.params
                if 'tp_iou_norm' in params.index:
                    print(f"[info] IRR per +1 tp_iou_norm (T tolerant magnitude, fallback): {float(np.exp(params['tp_iou_norm'])):.3f}")
                coef_exp = params.get('is_experiment', np.nan)
                print(f"[info] IRR for is_experiment (T tolerant magnitude, fallback): {float(np.exp(coef_exp)):.3f}")

        except Exception as e:
            print(f"[warn] Zero-inflated/Poisson analysis failed: {e}")

        # Mediation-style: count_error_tol ~ is_experiment + det_quality_delta on strict responses
        try:
            if groups is not None:
                try:
                    m_med = smf.glm(formula="count_error_tol ~ is_experiment + det_quality_delta", data=d_valid, family=sm.families.Poisson()).fit(
                        cov_type='cluster', cov_kwds={'groups': d_valid["image"]})
                    print("\n=== Poisson GLM: count_error_tol ~ is_experiment + det_quality_delta (cluster-robust, strict responses) ===")
                except Exception as e_cl:
                    print(f"[warn] Cluster-robust SE failed for mediation GLM: {e_cl}")
                    m_med = smf.glm(formula="count_error_tol ~ is_experiment + det_quality_delta", data=d_valid, family=sm.families.Poisson()).fit()
                    print("\n=== Poisson GLM mediation (standard, strict responses) ===")
            else:
                m_med = smf.glm(formula="count_error_tol ~ is_experiment + det_quality_delta", data=d_valid, family=sm.families.Poisson()).fit()
                print("\n=== Poisson GLM mediation (no clustering, strict responses) ===")
            print(m_med.summary())
            irr_det = float(np.exp(m_med.params.get("det_quality_delta", np.nan)))
            irr_exp_after = float(np.exp(m_med.params.get("is_experiment", np.nan)))
            print(f"[info] IRR per +1 ΔIoU (T tolerant magnitude): {irr_det:.3f}")
            print(f"[info] IRR for is_experiment after controlling ΔIoU (strict responses): {irr_exp_after:.3f}")

            # ── Formal mediation: product-of-coefficients (delta method) ──────
            # Leg 1 (a path): is_experiment → det_quality_delta  (from m_det)
            # Leg 2 (b path): det_quality_delta → count_error_tol (from m_med, log scale)
            # Indirect effect on log(T) = a × b; exponentiated = IRR_indirect
            try:
                a     = float(m_det.params.get("is_experiment", np.nan))
                se_a  = float(m_det.bse.get("is_experiment", np.nan))
                b     = float(m_med.params.get("det_quality_delta", np.nan))
                se_b  = float(m_med.bse.get("det_quality_delta", np.nan))

                # cluster-robust SE can be NaN when within-cluster variance is zero;
                # fall back to standard (non-robust) SE so mediation can still proceed
                if np.isnan(se_a):
                    try:
                        m_det_std = smf.glm(formula="det_quality_delta ~ is_experiment", data=d, family=sm.families.Gaussian()).fit()
                        se_a = float(m_det_std.bse.get("is_experiment", np.nan))
                        print("[info] Mediation: using standard SE for a-path (cluster-robust was NaN)")
                    except Exception:
                        pass
                if np.isnan(se_b):
                    try:
                        m_med_std = smf.glm(formula="count_error_tol ~ is_experiment + det_quality_delta", data=d_valid, family=sm.families.Poisson()).fit()
                        se_b = float(m_med_std.bse.get("det_quality_delta", np.nan))
                        print("[info] Mediation: using standard SE for b-path (cluster-robust was NaN)")
                    except Exception:
                        pass

                if not any(np.isnan(v) for v in [a, se_a, b, se_b]):
                    from scipy.stats import norm as _norm
                    indirect     = a * b
                    se_indirect  = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
                    z_indirect   = indirect / se_indirect if se_indirect > 0 else np.nan
                    p_indirect   = float(2 * (1 - _norm.cdf(abs(z_indirect)))) if not np.isnan(z_indirect) else np.nan
                    ci_lo        = indirect - 1.96 * se_indirect
                    ci_hi        = indirect + 1.96 * se_indirect
                    irr_indirect = float(np.exp(indirect))
                    irr_ci_lo    = float(np.exp(ci_lo))
                    irr_ci_hi    = float(np.exp(ci_hi))
                    print("\n=== Mediation analysis: indirect effect of θ on T via detection (delta method) ===")
                    print(f"  Leg 1 (a)  — is_experiment → Δrecall:          coef={a:.6f}  SE={se_a:.6f}")
                    print(f"  Leg 2 (b)  — Δrecall → log(count_error_tol):  coef={b:.6f}  SE={se_b:.6f}")
                    print(f"  Indirect effect (a×b) on log(T):              {indirect:.6f}  SE={se_indirect:.6f}")
                    print(f"  z={z_indirect:.3f}  p={p_indirect:.4f}  95% CI=[{ci_lo:.6f}, {ci_hi:.6f}]")
                    print(f"  IRR_indirect (exp(a×b)):                       {irr_indirect:.4f}  95% CI=[{irr_ci_lo:.4f}, {irr_ci_hi:.4f}]")
                    sig_label = "significant" if p_indirect < 0.05 else "not significant"
                    print(f"  Interpretation: indirect effect is {sig_label} at α=0.05.")
                    print(f"  (A value <1 means better detection induced by θ indirectly reduces counting error.)")
                else:
                    print("[warn] Mediation delta method skipped: missing coefficients or SEs.")
            except Exception as e_med:
                print(f"[warn] Mediation delta method failed: {e_med}")
            # ─────────────────────────────────────────────────────────────────

        except Exception as e:
            print(f"[warn] Mediation-style GLM failed: {e}")

# Trigger pooled comparison if requested
if getattr(args, "compare_csvs", None):
    csvs = [s.strip() for s in args.compare_csvs.split(",") if s.strip()]
    run_pooled_glms(csvs, det_only_glm=getattr(args, "det_only_glm", False), show_mismatches=getattr(args, "show_mismatches", False))
    sys.exit(0)