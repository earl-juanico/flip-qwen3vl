"""
Generate a clustered question file for evaluate_FitAP_vlm.py.

This script produces
multiple queries per image — one per category (novel AND base) present in
that image — enabling cluster-robust inference grouped by image.

Strategy (Option 1):
  For images that have at least 1 novel category, include queries for ALL
  COCO categories present in that image (novel + base). This guarantees
  multiple rows per image even when only 1 novel category is present, as
  long as any base category co-occurs. The GT file marks each row with
  is_novel so the evaluation GLM can stratify or filter by category type.

Outputs:
  questions/question_clustered.jsonl
  gt/coco_gt_val2017_novel_clustered.jsonl

Usage:
  python3 generate_questions_clustered.py [options]

Options:
  --min-total-cats INT  Minimum total categories (novel+base) per image to
                        include, ensuring genuine multi-query clusters (default: 2)
  --min-novel INT       Minimum novel categories required per image (default: 1)
  --max-images INT      Cap on number of images sampled (default: no cap)
  --seed INT            Random seed for image sampling (default: 42)
  --start-qid INT       Starting question_id (default: 10000, avoids overlap
                        with existing file whose IDs go up to ~2528)
  --suffix STR          Output file suffix (default: _clustered)
"""

import json
import random
import argparse
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Novel categories — same 17 used in questions_pct100_s00.jsonl
# ---------------------------------------------------------------------------
NOVEL_CATS = {
    "airplane", "bus", "cake", "cat", "couch", "cow", "cup", "dog",
    "elephant", "keyboard", "knife", "scissors", "sink", "skateboard",
    "snowboard", "tie", "umbrella",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_ANN = os.path.join(SCRIPT_DIR, "data", "coco", "annotations", "instances_val2017.json")


def bbox_to_normalized_xyxy(x, y, w, h, W, H):
    """Convert COCO absolute XYWH bbox to normalized XYXY string."""
    x1 = x / W
    y1 = y / H
    x2 = (x + w) / W
    y2 = (y + h) / H
    return f"[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-total-cats", type=int, default=2,
                        help="Minimum total categories (novel+base) per image (default: 2)")
    parser.add_argument("--min-novel", type=int, default=1,
                        help="Minimum novel categories required per image (default: 1)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum images to sample (default: all qualifying)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--start-qid", type=int, default=10000,
                        help="Starting question_id (default: 10000)")
    parser.add_argument("--suffix", type=str, default="_clustered",
                        help="Output filename suffix (default: _clustered)")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load COCO annotations
    # -----------------------------------------------------------------------
    print(f"[info] Loading COCO annotations from {COCO_ANN} ...")
    with open(COCO_ANN, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    novel_cat_ids = {cid for cid, name in cat_id_to_name.items() if name in NOVEL_CATS}
    img_id_to_meta = {i["id"]: i for i in coco["images"]}

    print(f"[info] Novel category IDs ({len(novel_cat_ids)}): {sorted(novel_cat_ids)}")
    print(f"[info] Total COCO categories: {len(cat_id_to_name)}")

    # -----------------------------------------------------------------------
    # Build: image_id -> {cat_id -> [bbox_str, ...]}  (ALL categories)
    # -----------------------------------------------------------------------
    img_to_all_cats = defaultdict(lambda: defaultdict(list))

    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        meta = img_id_to_meta.get(img_id)
        if meta is None:
            continue
        W, H = meta["width"], meta["height"]
        x, y, w, h = ann["bbox"]
        bbox_str = bbox_to_normalized_xyxy(x, y, w, h, W, H)
        img_to_all_cats[img_id][cat_id].append(bbox_str)

    # -----------------------------------------------------------------------
    # Filter: images with >= min_novel novel cats AND >= min_total_cats total
    # -----------------------------------------------------------------------
    qualifying = []
    for img_id, cats in img_to_all_cats.items():
        n_novel = sum(1 for cid in cats if cid in novel_cat_ids)
        n_total = len(cats)
        if n_novel >= args.min_novel and n_total >= args.min_total_cats:
            qualifying.append(img_id)

    print(f"[info] Images with >= {args.min_novel} novel cat(s) and "
          f">= {args.min_total_cats} total cats: {len(qualifying)}")

    rng = random.Random(args.seed)
    rng.shuffle(qualifying)

    if args.max_images is not None:
        qualifying = qualifying[:args.max_images]
        print(f"[info] Capped to {len(qualifying)} images (--max-images {args.max_images})")

    # -----------------------------------------------------------------------
    # Generate question and GT rows (all categories per image)
    # -----------------------------------------------------------------------
    questions = []
    ground_truths = []
    qid = args.start_qid

    for img_id in qualifying:
        meta = img_id_to_meta[img_id]
        file_name = meta["file_name"]
        cats_in_image = img_to_all_cats[img_id]

        # Sort by category ID for deterministic ordering
        for cat_id in sorted(cats_in_image.keys()):
            cat_name = cat_id_to_name[cat_id]
            bboxes = cats_in_image[cat_id]
            is_novel = cat_id in novel_cat_ids

            text = (
                f"Give the normalized bounding box coordinates in the format "
                f"[x1, y1, x2, y2] of all instances of {cat_name} in the image."
            )

            questions.append({
                "question_id": qid,
                "image": file_name,
                "text": text,
            })
            ground_truths.append({
                "question_id": qid,
                "answers": bboxes,
                "class": cat_name,
                "is_novel": is_novel,
            })
            qid += 1

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print(f"[info] Total question-answer pairs generated: {len(questions)}")

    cats_per_image = defaultdict(int)
    novel_per_image = defaultdict(int)
    for q, g in zip(questions, ground_truths):
        cats_per_image[q["image"]] += 1
        if g["is_novel"]:
            novel_per_image[q["image"]] += 1

    total_counts = list(cats_per_image.values())
    novel_counts = list(novel_per_image.values())
    singletons = sum(1 for v in total_counts if v == 1)

    print(f"[info] Queries per image   — min: {min(total_counts)}, "
          f"max: {max(total_counts)}, mean: {sum(total_counts)/len(total_counts):.2f}")
    print(f"[info] Novel queries/image — min: {min(novel_counts)}, "
          f"max: {max(novel_counts)}, mean: {sum(novel_counts)/len(novel_counts):.2f}")
    print(f"[info] Singleton clusters (1 query/image): {singletons} "
          f"({100*singletons/len(total_counts):.1f}%)")
    n_novel_rows = sum(1 for g in ground_truths if g["is_novel"])
    print(f"[info] Novel rows: {n_novel_rows} / {len(ground_truths)} total "
          f"({100*n_novel_rows/len(ground_truths):.1f}%)")

    # -----------------------------------------------------------------------
    # Write output files
    # -----------------------------------------------------------------------
    os.makedirs(os.path.join(SCRIPT_DIR, "questions"), exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "gt"), exist_ok=True)

    q_out = os.path.join(SCRIPT_DIR, "questions", f"question{args.suffix}.jsonl")
    gt_out = os.path.join(SCRIPT_DIR, "gt", f"coco_gt_val2017_novel{args.suffix}.jsonl")

    with open(q_out, "w") as f:
        for item in questions:
            f.write(json.dumps(item) + "\n")
    print(f"[info] Written: {q_out}")

    with open(gt_out, "w") as f:
        for item in ground_truths:
            f.write(json.dumps(item) + "\n")
    print(f"[info] Written: {gt_out}")


if __name__ == "__main__":
    main()
