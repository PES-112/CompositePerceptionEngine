# Walkable-segmentation speed pilot, MacBook Air M4 (16 GB), MPS — 2026-10-17

One epoch on a seeded 500-train / 100-val subset of `data/sanpo_walkable_seg` (symlinks; built while the
full download was still running alongside, so timings are slightly pessimistic). Same `train()` arguments
as `training/scripts/train_yolo26_sem_walkable.py`, plus `--device mps --workers 4 --cache disk`, AMP on.
s/iter is the median after 5 warm-up iterations; peak memory is `torch.mps.driver_allocated_memory()`.

| config | batch | s/iter | s/img | peak MPS GB | loss first→last | NaN | val s/img | min/epoch* | 40 ep, h* |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| yolo26n-sem @640 | 8 | 0.62 | 0.077 | 3.6 | 10.7 → 5.0 | no | 0.20 | 39 | 26 |
| yolo26s-sem @640 | 8 | 1.16 | 0.146 | 5.9 | 12.6 → 4.9 | no | 0.29 | 68 | 45 |
| yolo26s-sem @768 | 8 | 3.09 | 0.386 | 7.9 (swapping) | 12.7 → 5.0 | no | 0.36 | 157 | 105 |
| yolo26s-sem @768 | 4 | 1.46 | 0.365 | 4.7 | 13.2 → 5.3 | no | 0.36 | 150 | 100 |

\* Projected for the full split: 21,394 train images (11,691 human + 9,703 propagated) and 3,246 val images
per epoch (training + one val pass). Patience 10 can stop earlier.

**Choice (declared before training): yolo26s-sem, imgsz 640, batch 8.** Rule: the largest config whose
projected run is practical on this laptop, s over n, then larger imgsz. s@768 (~100 h worst case, memory-bound
at batch 8, no per-image gain at batch 4) is not; s@640 (~45 h) is. Cost: thinner curb/stairs edges than the
docstring's 1024 lab-server plan; reported via curb/stairs boundary F.
