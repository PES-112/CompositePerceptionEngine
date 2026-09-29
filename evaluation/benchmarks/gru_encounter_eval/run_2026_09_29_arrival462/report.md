# Arrival-time GRUs and ensemble — all 462 curated sessions (2026-09-29)

This run repeats `run_2026_09_29_arrival/` (387 sessions) on the **full curated set**. The setup,
label, models and protocol are the same; see that report. This one covers the numbers and what
changed.

- **Data.** 462 sessions, of which 364 contain tracked objects and 109 contain real encounters
  (21–22 per fold). There are 21,265 test objects (encounter rate 0.168) and 4,897 frames with ≥ 2
  objects.
- **Dataset.** `dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_f5_s0/`.
- **Runs.**
  - `models/GRU/{GRU,GRU_attention}/runs/arrival462_2026_09_29_fold*_seed*/`
  - Size-matched control: `models/GRU/GRU/runs/arrival462h96_2026_09_29_*`
- **Reproduce.**
  - `python -m models.GRU.summarize --run-name arrival462_2026_09_29`
  - `python -m models.Ensemble.ensemble --run-name arrival462_2026_09_29` — the full ensemble
    report is in `evaluation/benchmarks/ensemble_eval/arrival462_2026_09_29/`.

## Results — all frames

Values are mean ± std over 3 seeds.

| Method | AUROC | Avg. precision | encounter_top1 | soonest_top1 | arrival_order | flicker_rate ↓ |
|---|---:|---:|---:|---:|---:|---:|
| gru | 0.887 ± 0.001 | 0.741 ± 0.001 | 0.481 ± 0.001 | 0.629 ± 0.003 | 0.676 ± 0.004 | **0.590** ± 0.010 |
| gru_attention | 0.893 ± 0.001 | 0.737 ± 0.002 | 0.464 ± 0.003 | 0.602 ± 0.002 | 0.666 ± 0.005 | 0.655 ± 0.012 |
| ensemble_stack | **0.894** ± 0.001 | **0.752** ± 0.002 | 0.478 ± 0.002 | 0.630 ± 0.004 | **0.688** ± 0.003 | 0.636 ± 0.008 |
| ensemble_rank | 0.879 ± 0.000 | 0.687 ± 0.001 | 0.473 ± 0.001 | 0.620 ± 0.006 | 0.674 ± 0.002 | 0.651 ± 0.006 |
| K0 | 0.571 | 0.202 | 0.394 | 0.469 | 0.585 | 0.660 |
| TTC | 0.656 | 0.504 | 0.422 | 0.528 | 0.612 | 0.676 |
| nearest | 0.872 | 0.727 | 0.469 | 0.644 | 0.657 | 0.730 |
| nearest_in_cone | 0.871 | 0.739 | **0.481** | **0.663** | 0.681 | 0.727 |

## Paired differences (95% session-bootstrap CI; ✓ = CI excludes 0)

| Model | vs | AUROC | arrival_order |
|---|---|---|---|
| gru | K0 | **+0.316** [+0.264, +0.353] ✓ | **+0.091** [+0.054, +0.122] ✓ |
| gru | TTC | **+0.231** [+0.190, +0.262] ✓ | **+0.064** [+0.031, +0.090] ✓ |
| gru | nearest_in_cone | **+0.016** [+0.002, +0.029] ✓ | −0.005 [−0.028, +0.019] |
| gru_attention | gru | +0.006 [−0.001, +0.014] | −0.010 [−0.020, +0.000] |
| gru_attention | gru, hidden 96 (size-matched) | +0.007 [−0.000, +0.016] | −0.008 [−0.019, +0.001] |
| gru, hidden 96 | gru, hidden 64 | −0.002 [−0.004, +0.001] | −0.002 [−0.007, +0.003] |
| ensemble_stack | nearest_in_cone | **+0.023** [+0.013, +0.034] ✓ | +0.007 [−0.008, +0.021] |
| ensemble_stack | gru | **+0.007** [+0.001, +0.014] ✓ | +0.012 [−0.002, +0.028] |
| ensemble_stack | gru_attention | +0.001 [−0.003, +0.006] | **+0.022** [+0.009, +0.037] ✓ |
| ensemble_stack | ensemble_rank | **+0.015** [+0.009, +0.021] ✓ | **+0.015** [+0.003, +0.025] ✓ |

## What changed from 387 to 462 sessions

| Claim | 387 | 462 |
|---|---|---|
| GRU vs. nearest-in-cone, AUROC | +0.007, tie | **+0.016, small real gain** |
| Attention vs. size-matched GRU, AUROC | +0.010, significant | **+0.007, tie** (CI touches 0) |
| Everything else | — | Same verdicts; CIs slightly tighter |

## Conclusions (final for this label and feature set)

1. **K0 is not an arrival ranking.** It has AUROC 0.57 for "will this object come within 3 m in
   front within 2 s", and every learned or distance-based ranker beats it by about 0.3.
2. **A learned GRU slightly beats the nearest-object rule at predicting *whether* objects come
   close** (+0.016 AUROC), and its top pick is much steadier (flicker 0.59 vs. 0.73).
3. **No model beats the nearest-in-cone rule at ranking *who arrives first*.** That holds for the
   GRU, the attention GRU and both ensembles. Arrival ordering appears limited by the perception
   features, not the model.
4. **Scene context (cross-object attention) adds nothing measurable** on full data. SANPO walks
   rarely contain the multi-agent interactions it would need.
5. **The best overall predictor is the stacked ensemble** (AUROC 0.894, +0.023 over the rule). It
   is worth using only if the "will it come close" probability is consumed downstream, for example
   by SLM-1 or Physics Verification.

The limitations are the same as in `run_2026_09_29_arrival/report.md`. The labels are perception
output, not ground truth, and the 3 m / 2 s definition is a design choice.
