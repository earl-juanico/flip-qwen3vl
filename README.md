# flip-qwen3vl

Causal probing of Qwen3-VL using **FLIP** (Final Layer Inference-time Probe) interventions.
The study sweeps a hidden-state flooring threshold *ϑ* (vartheta) across decoder layers and
measures its effect on object detection quality and counting accuracy on COCO val2017.

## Overview

The FLIP intervention clamps hidden-state values below *ϑ* to *ϑ* during inference (a floor
operation applied in-place to selected decoder layers and/or the pre-logit projection).
By sweeping *ϑ* across a range of values and comparing against a no-intervention baseline
(`vartheta=none`), the pipeline estimates the causal dose-response relationship between the
intervention strength and model behaviour, mediated through detection quality (recall).

The optional **permutation** variant (`FLIP_PERMUTE_FRACTION`) shuffles a fraction of
hidden-state dimensions before flooring, enabling a feature-coherence experiment.

## Hardware requirements

| | Minimum |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB (or equivalent ≥ 40 GB VRAM) |
| CUDA | 12.8 |
| Driver | 570.195.03 or newer |
| RAM | 64 GB system RAM recommended |

The default server configuration runs a single A100. For the 4B model, `VLLM_GPU_MEMORY_UTILIZATION=0.9`
leaves ~4 GB headroom for OS/CUDA overhead.

## Environment setup

The reference environment is Python 3.12.3, CUDA 12.8.

```bash
# 1. Create a fresh conda environment
conda create -n flip-qwen3vl python=3.12.3 -y
conda activate flip-qwen3vl

# 2. Install PyTorch with the cu128 build
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install the remaining dependencies
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

> **Note:** `flashinfer` ships pre-compiled wheels for specific CUDA + Python combinations.
> If the wheel for cu128 / Python 3.12 is not yet on PyPI, install from the FlashInfer
> nightly index:
> ```bash
> pip install flashinfer==0.6.3 \
>     --index-url https://flashinfer.ai/whl/cu128/torch2.9/
> ```


All scripts use `sys.executable` internally, so activating the environment before running
any script is sufficient — no additional installation is needed.

### Verify the environment

```bash
python - <<'EOF'
import torch, vllm, transformers, statsmodels
print("torch  :", torch.__version__)
print("vllm   :", vllm.__version__)
print("cuda ok:", torch.cuda.is_available())
EOF
```

Expected output:
```
torch  : 2.9.1+cu128
vllm   : 0.16.0
cuda ok: True
```

## Repository layout

```
flip-qwen3vl/
├── serve_with_patch.py       # Step 1 — launch vLLM + FLIP patch
├── patch_qwen.py             # FLIP monkeypatch for Qwen3VLForConditionalGeneration
├── probe_and_sweep.py        # Step 2 — sweep vartheta, run infer jobs in parallel
├── evaluate_sweep.py         # Step 3 — post-sweep GLM evaluation pipeline
├── fliprate.py               # Indoor/outdoor scene-label flip-rate helper
├── check_real_model_layers.py  # Diagnostic: verify patch applied to correct layers
│
├── infer/
│   ├── query_detect.py       # Detection queries (bounding boxes)
│   ├── query_reason.py       # Counting/reasoning queries
│   └── query_indout.py       # Indoor/outdoor classification queries
│
├── eval/
│   ├── evaluate_FitAP_vlm.py # FitAP scoring + pooled GLM + mediation analysis
│   ├── eval_coco.py          # COCO bbox parsing helpers
│   └── calculate_ap_multiple.py  # mAP / PR-curve calculation
│
├── config/                   # vartheta.txt files read at runtime by patch_qwen
│
├── questions/
│   ├── generate_questions_clustered.py   # Generate clustered detection question set
│   ├── generate_questions_leftright.py   # Generate spatial left/right control set
│   └── question_pct*.jsonl / question_*_clustered.jsonl  # Pre-generated JSONL files
│
├── gt/                       # Ground-truth JSONL files (COCO val2017 subsets)
├── data/                     # COCO val2017 images + annotations (not tracked in git)
├── model/                    # Model weights directory (not tracked in git)
├── answers/                  # Inference outputs written by probe_and_sweep.py
├── prc/                      # Per-sample metrics CSVs + PR-curve PNGs (intermediate)
└── reports/                  # Final CSV reports written by evaluate_sweep.py
```

## Data and model

### COCO val2017

The dataset is not included in this repository.  Download it from the official
COCO website and place it under `data/`:

```bash
mkdir -p data/coco/annotations

# Validation images (~1 GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/coco/
rm val2017.zip

# Annotations (~240 MB; includes instances_val2017.json and captions_val2017.json)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data/coco/
rm annotations_trainval2017.zip
```

After extraction the layout should be:

```
data/
└── coco/
    ├── annotations/
    │   ├── instances_val2017.json
    │   └── captions_val2017.json
    └── val2017/
        └── *.jpg          (~5 000 images)
```

If you already have COCO on disk, you can symlink instead:

```bash
ln -s /path/to/existing/coco data/coco
```

Set `COCO_ROOT` to the `data/` directory if it lives outside the repo:

```bash
export COCO_ROOT=/path/to/data
```

### Model weights

After cloning this repository, pull the model weights directly from HuggingFace with `git clone`:

```bash
cd model
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
cd ..
```

This places the weights at `model/Qwen3-VL-4B-Instruct/`, which is the default path the
server expects.  Git LFS must be installed for the large weight files to download correctly:

```bash
# Install git-lfs if not already present
git lfs install
cd model
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
cd ..
```

To use a different variant (e.g. 8B) or a weights directory that already exists elsewhere,
set `MODEL_PATH` before running the server:

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

Optionally store a HuggingFace access token in `HFTOKEN.txt` in the repo root; the server
will pick it up automatically.

### Question file generation

Pre-generated JSONL files for the bootstrap core sweep (`question_pct*_s*.jsonl`) are
committed under `questions/` and do not need to be regenerated for standard replication.
Two scripts are provided to recreate them or produce alternative variants:

#### `questions/generate_questions_clustered.py`

Generates the clustered detection question set used in the primary FLIP detection/counting
sweep.  For each COCO val2017 image that contains at least one *novel* category and at
least two total categories, it emits one detection query per category present — enabling
cluster-robust inference grouped by image.  Outputs:

- `questions/question_clustered.jsonl`
- `gt/coco_gt_val2017_novel_clustered.jsonl`

```bash
python questions/generate_questions_clustered.py \
    --min-total-cats 2 \
    --min-novel 1 \
    --seed 42
```

Key options: `--max-images`, `--start-qid`, `--suffix`.

#### `questions/generate_questions_leftright.py`

Generates the spatial left/right negative-control dataset.  From the same qualifying
images it selects *singleton* object pairs (exactly one instance of each category) with
sufficient horizontal separation and low overlap, then constructs
*"Is the \<obj1\> to the left of the \<obj2\>? Answer only yes or no."* prompts.
Labels are balanced to a near-exact 50/50 yes/no split by alternating the prompt order.
Outputs:

- `questions/question_spatial_lr_clustered.jsonl`
- `gt/coco_gt_val2017_spatial_lr_clustered.jsonl`

```bash
python questions/generate_questions_leftright.py \
    --min-total-cats 2 \
    --min-novel 1 \
    --max-pairs-per-image 1 \
    --min-area-frac 0.01 \
    --min-x-sep-frac 0.15 \
    --max-iou 0.3 \
    --seed 42 \
    --suffix _clustered
```

Key options: `--require-novel-in-pair`, `--max-pairs-per-image`, `--min-x-sep-frac`,
`--max-iou`, `--start-qid`.

Both scripts resolve `data/coco/annotations/instances_val2017.json` relative to the
**repo root** (not their own directory), so run them from the repo root or set
`COCO_ROOT` appropriately.

## Workflow

### Step 1 — Deploy the patched server

```bash
conda activate flip-qwen3vl   # or rmr2
python serve_with_patch.py
```

`serve_with_patch.py` loads `patch_qwen.py` in-process **before** vLLM initialises the model,
ensuring the FLIP hook is active for every forward pass.  It also starts a lightweight HTTP
server on port `9008` (derived from the vLLM port `8008`) to serve local COCO images.

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model/Qwen3-VL-4B-Instruct` | Path to model weights directory |
| `VLLM_PORT` | `8008` | vLLM API port |
| `FLIP_VARTHETA_FILE` | `config/vartheta.txt` | Live-reload vartheta config file |
| `FLIP_LAYER_INDICES` | *(unset — no layer-wise FLIP)* | Decoder layers to intercept (e.g. `28:`, `all`) |
| `FLIP_APPLY_ON_FINAL` | `0` | Apply FLIP before vocabulary projection |
| `FLIP_PERMUTE_FRACTION` | *(unset)* | Fraction of dims to permute (negative control) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction |
| `MAX_MODEL_LEN` | `4096` | Maximum sequence length |
| `QWEN3_DISABLE_VIDEO` | `0` | Skip video input handling |

To verify the patch is applied before starting a full sweep:

```bash
python check_real_model_layers.py
```

### Step 2 — Probe and sweep

In a **separate terminal** (with the server from step 1 still running):

```bash
python probe_and_sweep.py
```

This sweeps across the default list of *ϑ* values:

```
none  0.0  0.1  0.2  0.4  0.5  1.0  -0.2  -0.3  -0.5  -1.0  -1.5  -2.0  -2.5  -4.0  -5.0  -50.0
```

For each *ϑ* value, the sweep runs detection (`query_detect.py`), reasoning (`query_reason.py`),
and indoor/outdoor (`query_indout.py`) queries **in parallel** across all bootstrap batch sizes
(pct = 10, 20, 40, 60, 80, 100).  Results are written to `answers/`.

Common options:

```bash
# Custom vartheta values
python probe_and_sweep.py -- -2.0 -0.5 none

# Specific bootstrap fractions only
python probe_and_sweep.py --pct 80 100 -- -2.0 none

# Preview all commands without executing
python probe_and_sweep.py --dry-run

# Run with the permutation negative control (set on the server side)
export FLIP_PERMUTE_FRACTION=0.2
python serve_with_patch.py &
python probe_and_sweep.py --output-suffix _permute -- -2.0 none
```

### Step 3 — Evaluate

```bash
python evaluate_sweep.py --output-csv results.csv
```

For each *(pct, vartheta)* pair this runs:

1. **`evaluate_FitAP_vlm.py` (detection pipeline)** — scores bounding boxes with FitAP,
   computes per-sample IoU-weighted recall and tolerant count error, writes a metrics CSV
   under `prc/`.
2. **prc directory rename** — adds the bootstrap-pct infix to make paths deterministic.
3. **`evaluate_FitAP_vlm.py --compare-csvs` (GLM comparison)** — pools baseline vs.
   experimental metrics and fits cluster-robust GLMs (Gaussian ΔIoU, Poisson count error,
   mediation delta-method).
4. **`fliprate.py`** — computes the indoor/outdoor scene-label flip rate vs. baseline.

Two CSV files are written to `reports/`:

| File | Columns |
|---|---|
| `results.csv` | `vartheta`, `irr_indirect`, `95_ci_irr`, `irr_direct`, `dr_50`, `95_ci_dr_50`, `switchrate` |
| `results_delta-mediation.csv` | `vartheta`, `a`, `SE(a)`, `b`, `SE(b)`, `axb`, `SE(axb)`, `p-value`, `irr_indirect`, `95_pct_ci_irr` |

Common options:

```bash
# Specific pct and vartheta subset
python evaluate_sweep.py --pct 80 --output-csv results_pct80.csv -- -2.0 -0.5 none

# Full dataset (no bootstrap infix)
python evaluate_sweep.py --pct 100 --output-csv results_full.csv

# Skip prc directory rename (use raw names from step 1)
python evaluate_sweep.py --no-rename --output-csv results.csv

# Permutation-run results (reads from *_permute/ answer dirs)
python evaluate_sweep.py --output-suffix _permute --output-csv results_permute.csv
```

### Step 4 — (Optional) Archive prc outputs

Before starting a new sweep, rename or copy `prc/` to preserve intermediate results:

```bash
mv prc/ prc_s00_pct80/
```

`evaluate_sweep.py` will prompt you to do this if `prc/` is non-empty when it starts.

### Step 5 — Inspect reports

Open `reports/` to find:

| Analysis | Source columns | Interpretation |
|---|---|---|
| **Dose-response** | `dr_50`, `95_ci_dr_50` | Effect of *ϑ* on ΔIoU-weighted recall vs. baseline |
| **Mediation** | `a`, `b`, `axb`, `p-value`, `irr_indirect` | Indirect effect of *ϑ* on counting error through the detection pathway |
| **Negative control** | `switchrate` | Indoor↔outdoor scene-label flip rate; should not change systematically with *ϑ* under FLIP |
| **With/without permutation** | compare `results.csv` vs. `results_permute.csv` | Separates flooring from dimension shuffling effects |

A significant `irr_indirect < 1` (p < 0.05) supports the claim that *ϑ* indirectly reduces
counting error through improved detection.

## Environment variables summary

| Variable | Used by | Description |
|---|---|---|
| `MODEL_PATH` | `serve_with_patch.py` | Model weights directory |
| `VLLM_PORT` | `serve_with_patch.py`, `probe_and_sweep.py` | vLLM API port |
| `MEDIA_PATH` | `serve_with_patch.py` | Root served by the local HTTP image server |
| `FLIP_VARTHETA_FILE` | `patch_qwen.py` | Live-reload vartheta config (polled on mtime) |
| `FLIP_VARTHETA` | `patch_qwen.py` | Override vartheta scalar directly |
| `FLIP_LAYER_INDICES` | `patch_qwen.py` | Layer specification (e.g. `28:`, `all`, `28,29`) |
| `FLIP_APPLY_ON_FINAL` | `patch_qwen.py` | Also apply before vocabulary projection |
| `FLIP_PERMUTE_FRACTION` | `patch_qwen.py` | Permutation fraction for negative-control runs |
| `FLIP_PERMUTE_SEED` | `patch_qwen.py` | RNG seed for permutation (default: 1234) |
| `COCO_ROOT` | `evaluate_FitAP_vlm.py` | Path to the `data/` directory |
| `LAMBDA_FMT_FAIL` | `evaluate_FitAP_vlm.py` | Penalty for format failures (default: 0) |

## Troubleshooting

**"tensor model parallel group is not initialized"**
This is expected when running `check_real_model_layers.py` standalone outside the vLLM engine.
The script falls back to static config validation, which is sufficient to confirm the patch
targets the intended layers.

**Server not responding on port 8008**
Check that `serve_with_patch.py` has finished loading the model (look for `Application startup
complete` in its output).  The model load typically takes 2–4 minutes on an A100.

**`prc/` directory is non-empty warning**
`evaluate_sweep.py` detected leftover results from a previous run.  Rename or delete `prc/`
before proceeding, or use `--no-rename` to avoid the rename step entirely.

**flashinfer wheel not found**
Use the FlashInfer nightly index for your specific CUDA and PyTorch combination:
```bash
pip install flashinfer \
    --index-url https://flashinfer.ai/whl/cu128/torch2.9/
```
