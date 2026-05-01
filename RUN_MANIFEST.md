# Run Manifest

All COCO-based core sweeps were executed as **single-GPU inference jobs** on one
**NVIDIA A100-SXM4-40GB** GPU (NVIDIA-SMI / driver 570.195.03, CUDA 12.8) with
`VLLM_GPU_MEMORY_UTILIZATION=0.9`.  Post-hoc parsing, scoring, GLM/mediation
analysis, subsample aggregation, and figure generation are CPU-side analyses and
required no GPU time.

## Per-experiment wall-clock / GPU-hour estimates

Times are reported in GPU-hours (= wall-clock hours × number of GPUs; all runs
used exactly one GPU, so the two quantities are equal).  Estimates are rounded to
the nearest 0.5 h for the 4B model and to the nearest whole hour for 8B / Kimi.

| Experiment | Qwen3-VL-4B | Qwen3-VL-8B | Kimi-VL-A3B |
|---|---:|---:|---:|
| Primary FLIP detection/counting sweep | 2.5 h | 24 h | 86 h |
| Permutation negative control | 0.5 h | 4.5 h | 17 h |
| Cross-layer checks | 1.0 h | 9 h | 34 h |
| Same-site operator comparisons | 0.5 h | 5 h | 17 h |
| Feature-coherence controls | 0.5 h | 5 h | 17 h |
| Image-cluster subsampling runs | 0.5 h | 5 h | 17 h |
| **Total (reported experiments)** | **~5–6 h** | **~52–53 h** | **~189 h** |

Preliminary or failed runs outside the reported experiments required
approximately **8 GPU-hours** (not attributed to any single model or experiment).

### Notes

- The **primary detection/counting sweep** covers 17 *ϑ* values
  (`none 0.0 0.1 0.2 0.4 0.5 1.0 -0.2 -0.3 -0.5 -1.0 -1.5 -2.0 -2.5 -4.0 -5.0 -50.0`)
  × 6 bootstrap fractions (pct = 10, 20, 40, 60, 80, 100) per model.
- The **permutation negative control** sweeps the same *ϑ* grid with
  `FLIP_PERMUTE_FRACTION=0.2` (seed 1234) to disentangle flooring from
  dimension-shuffling effects.
- **Cross-layer checks** probe non-default `FLIP_LAYER_INDICES` configurations
  (e.g. first half, second half, single mid-layer) to localise layer sensitivity.
- **Same-site operator comparisons** apply alternative interventions (e.g. ceiling,
  absolute-value clamp) at the same layer sites to test operator specificity.
- **Feature-coherence controls** measure structured-shuffle versus unstructured
  permutation at matched magnitudes.
- **Image-cluster subsampling** re-runs the primary sweep on geographically or
  semantically stratified COCO subsets to check result stability.
- **Cross-architecture replications** are implicit in the three model rows above;
  the same protocol was applied to each architecture without modification.

## Hardware specification

| Item | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| Driver | 570.195.03 |
| CUDA | 12.8 |
| GPU count per job | 1 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.9 |
| System RAM | 64 GB |
| Python | 3.12.3 |
| vLLM | 0.16.0 |
| PyTorch | 2.9.1+cu128 |
