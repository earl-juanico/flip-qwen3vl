# Dataset Card

## Dataset used: MS COCO val2017

| Field | Value |
|---|---|
| **Name** | Microsoft COCO (Common Objects in Context) — 2017 validation split |
| **Version / split** | val2017 (5,000 images) |
| **License** | CC BY 4.0 |
| **Homepage** | <https://cocodataset.org/> |
| **Citation** | Lin et al., ECCV 2014 (`arxiv.org/abs/1405.0312`) |

### Contents used

| File | Purpose in this project |
|---|---|
| `val2017/*.jpg` | Source images served to the VLM via a local HTTP server |
| `annotations/instances_val2017.json` | Ground-truth bounding boxes and category labels for FitAP scoring |
| `annotations/captions_val2017.json` | Not directly used; present for completeness |

### Subset construction and question file generation

The full 5,000-image validation split is used.  Bootstrap sub-experiments draw
stratified random samples at fractions pct ∈ {10, 20, 40, 60, 80, 100} of the
full set; the pct = 100 condition is equivalent to the full val2017 split.
Sampling seeds are fixed per bootstrap seed index (s00–s04) to ensure
reproducibility.  Ground-truth JSONL files derived from
`instances_val2017.json` are stored under `gt/`.

Two scripts in `questions/` generate the JSONL question and ground-truth files
from `instances_val2017.json`:

| Script | Purpose | Primary output |
|---|---|---|
| `generate_questions_clustered.py` | Clustered detection set — one query per category per qualifying image; used in the core detection/counting sweep | `question_clustered.jsonl`, `coco_gt_val2017_novel_clustered.jsonl` |
| `generate_questions_leftright.py` | Spatial left/right negative-control set — singleton-object pairs with balanced yes/no labels; used to verify that FLIP does not spuriously disrupt spatial reasoning | `question_leftright_clustered.jsonl`, `coco_gt_val2017_spatial_lr_clustered.jsonl` |

Pre-generated files are committed and do not need to be recreated for standard
replication.  See `README.md § Question file generation` for full CLI usage.

### Image-cluster subsampling

A secondary subsampling strategy partitions images by visual cluster (based on
scene features) to test result stability across geographically or semantically
distinct subsets.  Cluster assignments are precomputed and stored in
`questions/` alongside the standard bootstrap JSONL files.

### Categories

Evaluation targets 80 COCO object categories.  Detection queries use natural-
language prompts that request all visible objects; category matching during
scoring uses the standard COCO label set with minor normalisation (e.g.,
`"hair drier"` ↔ `"hair dryer"`).

### Data access

The dataset is **not distributed with this repository**.  To obtain it:

```bash
# Images (~1 GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/coco/

# Annotations (~240 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data/coco/
```

Set `COCO_ROOT=data/` (or to the parent of `coco/`) before running any
evaluation script.  See `README.md` for full instructions.

### Known limitations and caveats

- **Evaluation only:** COCO val2017 is used exclusively for inference and
  evaluation; no images or annotations are used for training or fine-tuning.
- **Open-vocabulary detection:** The VLM is prompted in an open-vocabulary
  fashion; category recall depends on the model's ability to name objects in
  free-form text, not on a fixed label vocabulary.
- **Crowd annotations:** Instances marked `iscrowd=1` are retained in
  ground-truth files but are handled identically to non-crowd instances for
  simplicity; no special crowd suppression is applied.
- **Licence:** Images are licenced CC BY 4.0 from their original contributors.
  Users must comply with both the CC BY 4.0 terms and the COCO Terms of Use
  (<https://cocodataset.org/#termsofuse>).
