#!/usr/bin/env python3
"""
lr_accuracy.py

Computes accuracy of a model's yes/no responses to left/right spatial questions
against ground-truth answers. Predictions come from infer/query_leftright.py output;
ground truth comes from the corresponding GT JSONL.

USAGE
-----
    python lr_accuracy.py PREDICTIONS_JSONL --gt GT_JSONL [OPTIONS]

REQUIRED
    PREDICTIONS_JSONL
        Model output JSONL produced by infer/query_leftright.py.
        Each line: {"question_id": int, "prompt": ..., "text": <model response>, "flip_vartheta": ...}
    --gt GT_JSONL
        Ground truth JSONL (e.g. coco_gt_val2017_spatial_lr_clustered.jsonl).
        Each line: {"question_id": int, "answers": ["yes"|"no"], ...}

OPTIONS
    --mismatches    Print each wrong prediction with its objects and image ID.
    --breakdown     Print accuracy split by novel-pair vs non-novel-pair subsets
                    (uses the 'pair_has_novel' field in the GT).

EXAMPLES
    # Basic accuracy
    python lr_accuracy.py results_lr.jsonl \\
        --gt ./gt/coco_gt_val2017_spatial_lr_clustered.jsonl

    # Show wrong predictions
    python lr_accuracy.py results_lr.jsonl \\
        --gt ./gt/coco_gt_val2017_spatial_lr_clustered.jsonl \\
        --mismatches

    # Novel vs non-novel breakdown
    python lr_accuracy.py results_lr.jsonl \\
        --gt ./gt/coco_gt_val2017_spatial_lr_clustered.jsonl \\
        --breakdown

GENERATE PREDICTIONS
    python infer/query_leftright.py \\
        --questions ./questions/question_leftright_clustered.jsonl \\
        --image-dir ./data/coco/val2017 \\
        --model-path ./model/Qwen3-VL-4B-Instruct \\
        --config-file ./config/vartheta.txt \\
        --flip-vartheta none \\
        --output results_lr.jsonl \\
        --api-port 8001
"""
import json
import argparse
import re


def _normalize_text(t):
    if t is None:
        return None
    s = str(t).strip().lower()
    m = re.search(r"[a-z]+", s)
    return m.group(0) if m else s


def load_gt(gt_path):
    """Return dict: question_id -> list[str] (normalized answers)."""
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            # Strip leading non-JSON characters (e.g. BOM or stray prefix bytes)
            line = line.lstrip("\ufeff\x00\x78\x60").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Find the first '{' and retry in case of a corrupt line prefix
                brace = line.find("{")
                if brace == -1:
                    print(f"Warning: skipping malformed line {lineno} in {gt_path}")
                    continue
                obj = json.loads(line[brace:])
            qid = obj.get("question_id")
            answers = [_normalize_text(a) for a in obj.get("answers", [])]
            gt[qid] = {"answers": answers, "meta": obj}
    return gt


def load_predictions(pred_path):
    """Return list of (question_id, normalized_text) from model output JSONL."""
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            text = _normalize_text(obj.get("text"))
            preds.append((qid, text))
    return preds


def accuracy(pred_path, gt_path):
    gt = load_gt(gt_path)
    preds = load_predictions(pred_path)
    correct = 0
    total = 0
    for qid, pred in preds:
        if qid not in gt:
            continue
        total += 1
        if pred in gt[qid]["answers"]:
            correct += 1
    return correct / total if total > 0 else 0.0, correct, total


def print_mismatches(pred_path, gt_path):
    gt = load_gt(gt_path)
    preds = load_predictions(pred_path)
    mismatches = 0
    for qid, pred in preds:
        if qid not in gt:
            continue
        if pred not in gt[qid]["answers"]:
            mismatches += 1
            meta = gt[qid]["meta"]
            print(
                f"[WRONG] qid={qid} pred={pred!r} gt={gt[qid]['answers']} "
                f"obj1={meta.get('object1')!r} obj2={meta.get('object2')!r} "
                f"image={meta.get('image_id')}"
            )
    return mismatches


def breakdown_by_novel(pred_path, gt_path):
    gt = load_gt(gt_path)
    preds = load_predictions(pred_path)
    buckets = {
        "all": [0, 0],
        "novel_pair": [0, 0],
        "non_novel_pair": [0, 0],
    }
    for qid, pred in preds:
        if qid not in gt:
            continue
        meta = gt[qid]["meta"]
        is_correct = pred in gt[qid]["answers"]
        for bucket in ["all"]:
            buckets[bucket][1] += 1
            if is_correct:
                buckets[bucket][0] += 1
        if meta.get("pair_has_novel"):
            buckets["novel_pair"][1] += 1
            if is_correct:
                buckets["novel_pair"][0] += 1
        else:
            buckets["non_novel_pair"][1] += 1
            if is_correct:
                buckets["non_novel_pair"][0] += 1
    return buckets


def main():
    p = argparse.ArgumentParser(
        description="Compute accuracy of left/right spatial predictions against ground truth."
    )
    p.add_argument("predictions", help="Model output JSONL (from infer/query_leftright.py)")
    p.add_argument("--gt", required=True, help="Ground truth JSONL (e.g. coco_gt_val2017_spatial_lr_clustered.jsonl)")
    p.add_argument("--mismatches", action="store_true", help="Print wrong predictions with metadata")
    p.add_argument("--breakdown", action="store_true", help="Print accuracy breakdown by novel/non-novel pairs")
    args = p.parse_args()

    if args.mismatches:
        n = print_mismatches(args.predictions, args.gt)
        print(f"Total wrong: {n}")
    elif args.breakdown:
        buckets = breakdown_by_novel(args.predictions, args.gt)
        for label, (correct, total) in buckets.items():
            acc = correct / total if total > 0 else 0.0
            print(f"{label}: {acc:.4f} ({correct}/{total})")
    else:
        acc, correct, total = accuracy(args.predictions, args.gt)
        print(f"Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
