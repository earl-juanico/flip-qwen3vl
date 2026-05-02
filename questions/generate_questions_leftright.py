"""
Generate a clustered spatial-relation control set from MS COCO val2017.

This script builds a left/right negative-control dataset for VLM evaluation:
  "Is the <object1> to the left of the <object2>? Answer only yes or no."

Unlike the main clustered localization set, this script only uses object-category
pairs that are UNAMBIGUOUS within an image:
  - exactly one instance of object1
  - exactly one instance of object2
  - sufficient horizontal separation
  - optional area / overlap filters

The output remains clustered by image, so downstream analysis can still use
cluster-robust inference grouped by image.

Design choices:
  * Qualifying images are filtered similarly to the current clustered generator:
      - at least `--min-novel` novel categories present
      - at least `--min-total-cats` total categories present
  * Candidate control pairs are then formed only from singleton categories.
  * By default, at most one pair is sampled per image to avoid overweighting
    busy scenes.
  * Labels are balanced globally by alternating true-order and swapped-order
    prompts, yielding near-exact 50/50 yes/no balance.

Outputs:
  questions/question_leftright<suffix>.jsonl
  gt/coco_gt_val2017_spatial_lr<suffix>.jsonl

Usage:
  python3 generate_questions_leftright.py [options]

Example:
  python3 generate_questions_leftright.py \
      --min-total-cats 2 \
      --min-novel 1 \
      --max-pairs-per-image 1 \
      --min-area-frac 0.01 \
      --min-x-sep-frac 0.15 \
      --max-iou 0.3 \
      --seed 42 \
      --suffix _clustered
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from itertools import combinations

# ---------------------------------------------------------------------------
# Novel categories — same 17 used in the current clustered generator
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
COCO_ANN = os.path.join(
    SCRIPT_DIR, "data", "coco", "annotations", "instances_val2017.json"
)


def bbox_to_normalized_xyxy(x, y, w, h, W, H):
    """Convert COCO absolute XYWH bbox to normalized XYXY string."""
    x1 = x / W
    y1 = y / H
    x2 = (x + w) / W
    y2 = (y + h) / H
    return f"[{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]"


def xywh_to_xyxy(box):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou_xywh(box_a, box_b):
    """IoU for COCO XYWH boxes."""
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return 0.0 if union <= 0 else inter / union


def box_center_x(box):
    """Center x-coordinate for COCO XYWH box."""
    x, _, w, _ = box
    return x + 0.5 * w


def box_area_frac(box, W, H):
    """Area fraction of the full image for COCO XYWH box."""
    _, _, w, h = box
    return (w * h) / float(W * H)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Image qualification (same spirit as current clustered generator)
    parser.add_argument(
        "--min-total-cats",
        type=int,
        default=2,
        help="Minimum total categories (novel+base) per image (default: 2)",
    )
    parser.add_argument(
        "--min-novel",
        type=int,
        default=1,
        help="Minimum novel categories required per image (default: 1)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images to sample after qualification (default: all)",
    )

    # Spatial-pair construction
    parser.add_argument(
        "--max-pairs-per-image",
        type=int,
        default=1,
        help="Maximum number of eligible object pairs to keep per image (default: 1)",
    )
    parser.add_argument(
        "--require-novel-in-pair",
        action="store_true",
        help="Require at least one queried object in the pair to be novel",
    )
    parser.add_argument(
        "--min-area-frac",
        type=float,
        default=0.01,
        help="Minimum area fraction for each queried singleton object (default: 0.01)",
    )
    parser.add_argument(
        "--min-x-sep-frac",
        type=float,
        default=0.15,
        help="Minimum horizontal center separation as a fraction of image width (default: 0.15)",
    )
    parser.add_argument(
        "--max-iou",
        type=float,
        default=0.30,
        help="Maximum IoU allowed between the two queried singleton boxes (default: 0.30)",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--start-qid",
        type=int,
        default=50000,
        help="Starting question_id (default: 50000)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_clustered",
        help="Output filename suffix (default: _clustered)",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

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
    # Build: image_id -> {cat_id -> [instance_dict, ...]}
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
        bbox_xywh = [x, y, w, h]

        inst = {
            "bbox_xywh": bbox_xywh,
            "bbox_norm_xyxy": bbox_to_normalized_xyxy(x, y, w, h, W, H),
            "center_x": box_center_x(bbox_xywh),
            "area_frac": box_area_frac(bbox_xywh, W, H),
            "iscrowd": ann.get("iscrowd", 0),
            "area": ann.get("area", w * h),
        }
        img_to_all_cats[img_id][cat_id].append(inst)

    # -----------------------------------------------------------------------
    # Filter qualifying images (same high-level image-centric logic as current set)
    # -----------------------------------------------------------------------
    qualifying = []
    for img_id, cats in img_to_all_cats.items():
        n_novel = sum(1 for cid in cats if cid in novel_cat_ids)
        n_total = len(cats)
        if n_novel >= args.min_novel and n_total >= args.min_total_cats:
            qualifying.append(img_id)

    print(
        f"[info] Images with >= {args.min_novel} novel cat(s) and "
        f">= {args.min_total_cats} total cats: {len(qualifying)}"
    )

    rng.shuffle(qualifying)
    if args.max_images is not None:
        qualifying = qualifying[:args.max_images]
        print(f"[info] Capped to {len(qualifying)} images (--max-images {args.max_images})")

    # -----------------------------------------------------------------------
    # Build eligible singleton spatial pairs
    # -----------------------------------------------------------------------
    base_candidates = []

    for img_id in qualifying:
        meta = img_id_to_meta[img_id]
        W, H = meta["width"], meta["height"]
        file_name = meta["file_name"]
        cats_in_image = img_to_all_cats[img_id]

        # Keep only categories with exactly one instance
        singleton_cats = {}
        for cat_id, instances in cats_in_image.items():
            if len(instances) != 1:
                continue
            inst = instances[0]
            if inst["iscrowd"]:
                continue
            if inst["area_frac"] < args.min_area_frac:
                continue
            singleton_cats[cat_id] = inst

        # Need at least two singleton categories to form a pair
        if len(singleton_cats) < 2:
            continue

        img_candidates = []

        for cat_id_a, cat_id_b in combinations(sorted(singleton_cats.keys()), 2):
            inst_a = singleton_cats[cat_id_a]
            inst_b = singleton_cats[cat_id_b]

            # Optional restriction: at least one object in the pair is novel
            if args.require_novel_in_pair:
                if (cat_id_a not in novel_cat_ids) and (cat_id_b not in novel_cat_ids):
                    continue

            # Exclude near-overlapping / near-tied pairs
            iou = iou_xywh(inst_a["bbox_xywh"], inst_b["bbox_xywh"])
            if iou > args.max_iou:
                continue

            x_sep_frac = abs(inst_a["center_x"] - inst_b["center_x"]) / float(W)
            if x_sep_frac < args.min_x_sep_frac:
                continue

            # Canonical left/right order from GT centers
            if inst_a["center_x"] < inst_b["center_x"]:
                left_id, right_id = cat_id_a, cat_id_b
                left_inst, right_inst = inst_a, inst_b
            else:
                left_id, right_id = cat_id_b, cat_id_a
                left_inst, right_inst = inst_b, inst_a

            img_candidates.append(
                {
                    "img_id": img_id,
                    "image": file_name,
                    "width": W,
                    "height": H,
                    "left_cat_id": left_id,
                    "right_cat_id": right_id,
                    "left_cat_name": cat_id_to_name[left_id],
                    "right_cat_name": cat_id_to_name[right_id],
                    "left_inst": left_inst,
                    "right_inst": right_inst,
                    "x_sep_frac": x_sep_frac,
                    "iou": iou,
                    "left_is_novel": left_id in novel_cat_ids,
                    "right_is_novel": right_id in novel_cat_ids,
                }
            )

        if not img_candidates:
            continue

        rng.shuffle(img_candidates)
        keep = img_candidates[: args.max_pairs_per_image]
        base_candidates.extend(keep)

    print(f"[info] Images with >=1 eligible singleton spatial pair: "
          f"{len(set(c['img_id'] for c in base_candidates))}")
    print(f"[info] Total eligible sampled base candidates: {len(base_candidates)}")

    if not base_candidates:
        raise RuntimeError(
            "No eligible left/right singleton pairs found. "
            "Try lowering --min-area-frac or --min-x-sep-frac, "
            "raising --max-iou, or disabling --require-novel-in-pair."
        )

    # -----------------------------------------------------------------------
    # Balance yes/no labels by alternating true-order vs swapped-order prompts
    # -----------------------------------------------------------------------
    rng.shuffle(base_candidates)

    questions = []
    ground_truths = []
    qid = args.start_qid

    yes_count = 0
    no_count = 0

    for idx, cand in enumerate(base_candidates):
        left_name = cand["left_cat_name"]
        right_name = cand["right_cat_name"]

        # Alternate labels for near-exact global 50/50 balance
        ask_true_order = (idx % 2 == 0)

        if ask_true_order:
            ask_obj1 = left_name
            ask_obj2 = right_name
            answer = "yes"
            obj1_inst = cand["left_inst"]
            obj2_inst = cand["right_inst"]
            obj1_is_novel = cand["left_is_novel"]
            obj2_is_novel = cand["right_is_novel"]
        else:
            ask_obj1 = right_name
            ask_obj2 = left_name
            answer = "no"
            obj1_inst = cand["right_inst"]
            obj2_inst = cand["left_inst"]
            obj1_is_novel = cand["right_is_novel"]
            obj2_is_novel = cand["left_is_novel"]

        text = f"Is the {ask_obj1} to the left of the {ask_obj2}? Answer only yes or no."

        questions.append(
            {
                "question_id": qid,
                "image": cand["image"],
                "text": text,
            }
        )

        ground_truths.append(
            {
                "question_id": qid,
                "answers": [answer],
                "task": "spatial_left_right",
                "object1": ask_obj1,
                "object2": ask_obj2,
                "canonical_left_object": left_name,
                "canonical_right_object": right_name,
                "bbox1": obj1_inst["bbox_norm_xyxy"],
                "bbox2": obj2_inst["bbox_norm_xyxy"],
                "object1_is_novel": obj1_is_novel,
                "object2_is_novel": obj2_is_novel,
                "pair_has_novel": cand["left_is_novel"] or cand["right_is_novel"],
                "x_sep_frac": round(cand["x_sep_frac"], 4),
                "pair_iou": round(cand["iou"], 4),
                "image_id": cand["img_id"],
            }
        )

        if answer == "yes":
            yes_count += 1
        else:
            no_count += 1

        qid += 1

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print(f"[info] Total question-answer pairs generated: {len(questions)}")

    pairs_per_image = defaultdict(int)
    novel_pairs_per_image = defaultdict(int)
    for q, g in zip(questions, ground_truths):
        pairs_per_image[q["image"]] += 1
        if g["pair_has_novel"]:
            novel_pairs_per_image[q["image"]] += 1

    cluster_sizes = list(pairs_per_image.values())
    novel_cluster_sizes = list(novel_pairs_per_image.values())
    singletons = sum(1 for v in cluster_sizes if v == 1)

    print(
        f"[info] Pairs per image        — min: {min(cluster_sizes)}, "
        f"max: {max(cluster_sizes)}, mean: {sum(cluster_sizes)/len(cluster_sizes):.2f}"
    )
    print(
        f"[info] Novel pairs / image   — min: {min(novel_cluster_sizes)}, "
        f"max: {max(novel_cluster_sizes)}, mean: {sum(novel_cluster_sizes)/len(novel_cluster_sizes):.2f}"
    )
    print(
        f"[info] Singleton clusters    — {singletons} / {len(cluster_sizes)} "
        f"({100*singletons/len(cluster_sizes):.1f}%)"
    )
    print(
        f"[info] Label balance         — yes: {yes_count}, no: {no_count} "
        f"({100*yes_count/len(ground_truths):.1f}% yes)"
    )
    n_novel_pairs = sum(1 for g in ground_truths if g["pair_has_novel"])
    print(
        f"[info] Pairs with novel obj  — {n_novel_pairs} / {len(ground_truths)} "
        f"({100*n_novel_pairs/len(ground_truths):.1f}%)"
    )

    # -----------------------------------------------------------------------
    # Write output files
    # -----------------------------------------------------------------------
    os.makedirs(os.path.join(SCRIPT_DIR, "questions"), exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "gt"), exist_ok=True)

    q_out = os.path.join(
        SCRIPT_DIR,
        "questions",
        f"question_leftright{args.suffix}.jsonl",
    )
    gt_out = os.path.join(
        SCRIPT_DIR,
        "gt",
        f"coco_gt_val2017_spatial_lr{args.suffix}.jsonl",
    )

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