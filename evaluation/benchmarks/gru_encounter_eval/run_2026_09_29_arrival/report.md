# Arrival-time GRUs vs. K0, TTC and nearest-object rules — 387 sessions (2026-09-29)

**Question.** Can a GRU rank tracked objects by which will come close soonest, better than the
kinetic score (K0), TTC, or a plain nearest-object rule? Does letting objects see each other
(attention) help?

**Answer.**
- **K0 and TTC: yes.** The GRU clearly beats both, at predicting which objects will come close and
  at ordering them by arrival.
- **Nearest-object rules: no.** It only ties the nearest-object-in-cone rule.
- **Attention:** it adds a small but real gain in predicting *whether* an object comes close, which
  survives a size-matched control. It adds nothing to predicting *when*.

## Setup

- **Data.** 387 SANPO-Real sessions: the full curated list is 462, and the rest were still
  downloading. 302 sessions contain tracked objects, and 97 of those contain real encounters.
  - The data is Stage-1 output from the v3 detector at 10 Hz (`data/processed/sanpo_real_387/`).
  - Untracked `obs_*` depth blobs are dropped (`--tracked-only`). They are mostly ground; see
    `run_2026_09_28/report.md`.
- **Label.** The object comes within **3 m** inside the ±30° cone within 2 s. The arrival time is
  the frame at which it first does.
- **Models.** `gru` and `gru_attention` with `--time-bins 4`, which gives probabilities for
  arriving in 0–0.4 s, 0.5–0.9 s, 1.0–1.4 s, 1.5–2.0 s, or not at all. How-soon score = minus the
  expected arrival time; will-it-arrive score = 1 − P(not at all).
- **Protocol.**
  - 5-fold session-level cross-validation. Sessions are dealt into folds by encounter count, so
    each fold has 19–20 encounter sessions.
  - Seeds 0, 1 and 2.
  - The test predictions of all folds are pooled, so every session is scored exactly once.
  - Differences get 95% CIs from 1,000 paired session resamples.
- **Dataset.** `dataset/GRU/sanpo_real_387_tracked_d3_h8_current_m32_f5_s0/`.
- **Runs.**
  - `models/GRU/{GRU,GRU_attention}/runs/arrival387_2026_09_29_fold*_seed*/`
  - Size-matched control: `models/GRU/GRU/runs/arrival387h96_2026_09_29_fold*_seed*/`
- **Reproduce.** `python -m models.GRU.summarize --run-name arrival387_2026_09_29`

## Results — all frames

18,182 objects (encounter rate 0.175); 4,195 frames with ≥ 2 objects, 1,708 of them with an
encounter. Values are mean ± std over 3 seeds.

| Method | AUROC | Avg. precision | encounter_top1 | soonest_top1 | arrival_order | flicker_rate ↓ |
|---|---:|---:|---:|---:|---:|---:|
| gru | 0.877 ± 0.002 | 0.732 ± 0.004 | 0.470 ± 0.001 | 0.608 ± 0.008 | 0.669 ± 0.002 | **0.593** ± 0.009 |
| gru_attention | **0.886** ± 0.004 | 0.729 ± 0.005 | 0.463 ± 0.006 | 0.597 ± 0.016 | 0.656 ± 0.003 | 0.664 ± 0.007 |
| K0 | 0.569 | 0.210 | 0.395 | 0.460 | 0.586 | 0.666 |
| TTC | 0.651 | 0.507 | 0.420 | 0.515 | 0.610 | 0.683 |
| nearest | 0.871 | 0.733 | 0.465 | 0.633 | 0.656 | 0.733 |
| nearest_in_cone | 0.870 | **0.746** | **0.477** | **0.652** | **0.680** | 0.731 |

What the columns mean:
- **arrival_order:** for pairs of objects in a frame where one arrives strictly sooner, how often
  it is ranked higher. 0.5 = random.
- **soonest_top1:** in frames where something visible arrives, whether the #1 pick is one of the
  first to arrive.

## Paired differences (model minus other, 95% session-bootstrap CI)

A difference counts only if its CI excludes 0.

| Model | vs | AUROC | arrival_order |
|---|---|---|---|
| gru | K0 | **+0.308** [+0.245, +0.348] | **+0.083** [+0.042, +0.117] |
| gru | TTC | **+0.226** [+0.179, +0.257] | **+0.059** [+0.020, +0.091] |
| gru | nearest_in_cone | +0.007 [−0.007, +0.021] | −0.011 [−0.035, +0.012] |
| gru_attention | nearest_in_cone | **+0.017** [+0.003, +0.033] | **−0.025** [−0.046, −0.001] |
| gru_attention | gru | **+0.010** [+0.001, +0.020] | −0.013 [−0.028, +0.004] |
| gru_attention | gru, hidden 96 (≈67k params, size-matched) | **+0.010** [+0.002, +0.021] | −0.015 [−0.030, +0.002] |
| gru, hidden 96 | gru, hidden 64 | −0.001 [−0.003, +0.002] | +0.002 [−0.005, +0.008] |

## Interaction slice

This is frames with ≥ 2 objects within 5 m, or a crosswalk: 879 frames, 3,780 objects, encounter
rate 0.52.

| Method | AUROC | arrival_order | soonest_top1 |
|---|---:|---:|---:|
| gru | 0.817 | 0.717 | 0.773 |
| gru_attention | 0.805 | 0.699 | 0.758 |
| nearest_in_cone | 0.820 | 0.733 | 0.794 |
| K0 | 0.564 | 0.600 | 0.626 |

Attention does **not** help more where objects interact. Its small overall AUROC gain disappears
on this slice.

## Forecast vs. result

The forecast was made before the run.

| Claim | Forecast | Result |
|---|---|---|
| GRU vs. K0, AUROC | 0.88–0.93 vs. 0.55 | 0.877 vs. 0.569 ✓ (low end) |
| GRU vs. K0, arrival order | 0.70–0.78 vs. 0.58 | 0.669 vs. 0.586 — a smaller win than forecast, but significant |
| GRU vs. nearest-in-cone | +0.01–0.05 AUROC, +0.02–0.08 arrival | +0.007 / −0.011 — a tie, not a win ✗ |
| Attention vs. GRU | tie | +0.010 AUROC (significant, not capacity), arrival tie ≈ |

The 139-session preliminary run (`arrival139_2026_09_28`) pointed the same way. The only change is
that GRU-vs-K0 arrival ordering became significant with more data. Those runs are not committed; they
can be reproduced from `dataset/GRU/ablation_30pct_tracked_d3_h8_current_m32_f5_s0`.

## Reading it

1. **K0 is the wrong tool for "what reaches me first".**
   - It is barely above chance at predicting who comes within 3 m (AUROC 0.57). Every learned or
     distance-based ranker beats it by about 0.3.
   - Reasons: it has no bearing term, and speed² × mass rank fast, heavy, distant objects first.
   - That fits its design (consequence, not arrival). But for announcing what a blind pedestrian
     will reach first, K0 should not be the only ranking.
2. **"Nearest object in the forward cone" is a hard baseline.** With the perception stack's current
   depth and velocity noise, 0.8 s of track history adds no measurable ranking skill over it. TTC
   is the physics estimate of arrival, and it does worse than both.
3. **The GRU's practical advantage is stability.** Its top pick changes identity 59% of frames,
   against 73% for nearest-in-cone and 67% for K0, at equal accuracy. For a narrator, fewer flips
   means fewer contradictory announcements.
4. **Scene context (attention) is not worth it here yet.**
   - It gives +0.01 AUROC, real and not explained by capacity.
   - It does not improve arrival ranking, it is worse on flicker, and it does not help on the
     interaction slice.
   - SANPO walks rarely contain the multi-agent interactions it is meant to exploit.

## Limitations

- **Incomplete data:** 387 of 462 curated sessions. The remaining 75 were still downloading.
- **The labels are perception output, not ground truth.** They come from the same stack as the
  inputs, including depth noise and ByteTrack ID switches.
- **The 3 m / 2 s encounter definition is a design choice.** It departs from the K0 ablation's
  1.5 m definition to get enough positives: 45 → 263 encounter tracks on 139 sessions.
- **Several things are untested:** the future-only label (the default counts objects already
  inside the zone as arriving at 0 s), other bin counts, and longer histories.
