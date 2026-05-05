# Reproducibility Guide

This document provides the information needed to reproduce all reported
experiments from scratch.  For a full narrative description of the pipeline,
see `README.md`.

## Hardware and software environment

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| GPU count | 1 per job |
| Driver | NVIDIA-SMI 570.195.03 |
| CUDA | 12.8 |
| OS | Linux 6.8.0 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu128 |
| vLLM | 0.16.0 |
| Transformers | 4.57.3 |

All Python dependencies are pinned in `requirements.txt`.  See `README.md` for
the full conda / pip installation commands.

## Randomness and seeds

| Source of randomness | Control |
|---|---|
| Bootstrap sample order | JSONL files in `questions/` (committed; fixed per pct/seed index) |
| Permutation negative control | `FLIP_PERMUTE_SEED=1234` (default) |
| vLLM sampling | `temperature=0, top_p=1.0` (greedy decoding) throughout |
| GLM / mediation | Deterministic closed-form estimators (no MCMC or random restarts) |

No other sources of non-determinism are present in the reported pipeline.
vLLM's CUDA kernel scheduling may introduce floating-point non-determinism at
the bit level, but this does not materially affect summary statistics.

## Step-by-step reproduction

### 0. Prerequisites

```bash
# Clone repository
git clone <repo-url> flip-qwen3vl
cd flip-qwen3vl

# Install environment (see README.md for full commands)
conda create -n flip-qwen3vl python=3.12.3 -y
conda activate flip-qwen3vl
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Download COCO val2017
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
mkdir -p data/coco
unzip val2017.zip          -d data/coco/
unzip annotations_trainval2017.zip -d data/coco/

# Pull model weights (example: 4B)
cd model && git lfs install && git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct && cd ..
```

### 1. Primary FLIP detection/counting sweep

```bash
# Terminal A — patched vLLM server
conda activate flip-qwen3vl
export MODEL_PATH=model/Qwen3-VL-4B-Instruct
python serve_with_patch.py

# Terminal B — sweep (wait for "Application startup complete" in Terminal A)
# Runs detect / reason / indout (bootstrap) + leftright (spatial control) in parallel per ϑ.
# The leftright accuracy summary is written to answers/answers_leftright_clustered/lr_sweep.csv.
python probe_and_sweep.py

# Evaluate
python evaluate_sweep.py --output-csv reports/results_4b.csv
```

To collect per-token A/M stats (Section 5.1) during the leftright queries, launch the server with:

```bash
FLIP_LOG_STATS=1 FLIP_STATS_CSV=/tmp/flip_am_stats.csv python serve_with_patch.py
```

Then run the sweep as above; the stats CSV is copied to `answers/answers_leftright_clustered/am_stats.csv` automatically.

> **Layer-wise stats note**: layer-level stats accumulate only during prefill when CUDA graphs
> are active (the default).  To collect stats over all tokens (including decode steps), set
> `FLIP_ENFORCE_EAGER=1` when launching the server; this disables CUDA graph capture
> and is also required when sweeping `FLIP_VARTHETA_FILE` dynamically with `FLIP_LAYER_INDICES` set.

Repeat with `MODEL_PATH=model/Qwen3-VL-8B-Instruct` and
`MODEL_PATH=model/Kimi-VL-A3B-Instruct` for the 8B and Kimi runs.

Archive intermediate results before each new model:

```bash
mv prc/ prc_4b_primary/
mv answers/ answers_4b_primary/
```

### 2. Permutation negative control

```bash
# Terminal A
export MODEL_PATH=model/Qwen3-VL-4B-Instruct
export FLIP_PERMUTE_FRACTION=0.2
export FLIP_PERMUTE_SEED=1234
python serve_with_patch.py

# Terminal B — leftright spatial control also runs alongside bootstrap tasks
python probe_and_sweep.py --output-suffix _permute

# Evaluate
python evaluate_sweep.py --output-suffix _permute \
    --output-csv reports/results_4b_permute.csv
```

### 3. Cross-layer checks

Re-run the primary sweep with non-default `FLIP_LAYER_INDICES` values:

```bash
# Example: FLIP on layers 0–13 only (first half of 28-layer model)
# FLIP_ENFORCE_EAGER=1 is required because FLIP_LAYER_INDICES + FLIP_VARTHETA_FILE
# would otherwise bake the initial vartheta into CUDA graphs (stale across sweep steps).
export FLIP_LAYER_INDICES=0:14
FLIP_ENFORCE_EAGER=1 python serve_with_patch.py &
python probe_and_sweep.py --output-suffix _layer0_13
python evaluate_sweep.py --output-suffix _layer0_13 \
    --output-csv reports/results_4b_layer0_13.csv
```

### 4. Same-site operator comparisons

Apply alternative interventions at the same layer sites by modifying
`patch_qwen.py` to use a ceiling or absolute-value clamp operator, then
re-running steps 1–3 with the modified server.

### 5. Feature-coherence controls

Run structured-shuffle (block permutation) vs. unstructured random permutation
at matched magnitudes using `FLIP_PERMUTE_FRACTION` with different seed values:

```bash
for SEED in 42 100 999 1234 5678; do
    export FLIP_PERMUTE_SEED=$SEED
    python probe_and_sweep.py --output-suffix _perm_seed${SEED}
done
```

### 6. Image-cluster subsampling

Question JSONL files for cluster-stratified subsets are under `questions/`.
Run the primary sweep with `--pct` flags corresponding to cluster sizes:

```bash
python probe_and_sweep.py --pct 10 20 40
python evaluate_sweep.py --pct 10 --output-csv reports/results_4b_pct10.csv
```

## Verifying the environment

```bash
python - <<'EOF'
import torch, vllm, transformers, statsmodels
print("torch      :", torch.__version__)
print("vllm       :", vllm.__version__)
print("transformers:", transformers.__version__)
print("statsmodels:", statsmodels.__version__)
print("cuda ok    :", torch.cuda.is_available())
EOF
```

Expected output:
```
torch      : 2.9.1+cu128
vllm       : 0.16.0
transformers: 4.57.3
statsmodels: 0.14.6
cuda ok    : True
```

Verify the FLIP patch targets the correct layers before running a full sweep:

```bash
python check_real_model_layers.py
```

## Expected outputs

| File | Description |
|---|---|
| `reports/results_*.csv` | GLM/mediation summary statistics per *(pct, vartheta)* |
| `reports/results_*_delta-mediation.csv` | Delta-method mediation decomposition (`a`, `b`, `axb`, `SE`, `p-value`, `irr_indirect`) |
| `prc_*/` | Per-sample IoU and count-error metrics (intermediate) |
| `answers_*/` | Raw VLM inference outputs (JSON-per-line) |
| `answers/answers_leftright_clustered/lr_sweep.csv` | Per-ϑ yes/no spatial accuracy from `probe_and_sweep.py` |
| `answers/answers_leftright_clustered/am_stats.csv` | Per-ϑ A_{ℓ,i}/M_{ℓ,i} stats (only when `FLIP_LOG_STATS=1`) |

A successful primary sweep for the 4B model produces `results_4b.csv` with 17
*ϑ* rows (including `none`) for each bootstrap fraction, containing non-null
`irr_indirect`, `irr_direct`, `dr_50`, and `switchrate` columns.  A corresponding
`results_4b_delta-mediation.csv` is always written alongside it.

## Checklist

- [ ] `requirements.txt` dependencies installed with pinned versions
- [ ] COCO val2017 images and annotations present under `data/coco/`
- [ ] Model weights present under `model/`
- [ ] `python check_real_model_layers.py` completes without errors
- [ ] `FLIP_PERMUTE_SEED=1234` for all permutation runs
- [ ] `FLIP_ENFORCE_EAGER=1` set when using `FLIP_LAYER_INDICES` with a dynamic `FLIP_VARTHETA_FILE`
- [ ] `prc/` and `answers/` archived between model runs to prevent cross-contamination
- [ ] `reports/` inspected for null values before reporting results
- [ ] `answers/answers_leftright_clustered/lr_sweep.csv` present after each sweep (confirm leftright ran)
- [ ] `reports/results_*_delta-mediation.csv` present alongside each `results_*.csv`
