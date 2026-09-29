# GRU vs. GRU + attention encounter predictors — first run (2026-09-28)

**Headline.** Neither GRU architecture beats the plain "nearest object" rule, and attention does
not help. The run's main finding is about the label: **99.7% of encounters in the K0 ablation's
label are untracked depth blobs**, which are mostly the ground ahead of the walker. That affects the
ablation's `encounter_top1` numbers as well as this experiment.

## Setup

- **Data.** The same 30% SANPO-Real sample as the K0 ablation (139 sessions, seed 20260819,
  10 Hz), regenerated on 2026-09-28 with `tools/stream_sanpo_perception.py` and the v3 detector.
  The regenerated CSVs reproduce the ablation: 19,452 scored frames vs. 19,402, and K0
  encounter_top1 0.297 [0.254, 0.344] vs. 0.307 [0.263, 0.355].
- **Datasets** (`dataset/GRU/`, split seed 0):
  - `ablation_30pct_h8_current_m32_s0` — all objects, exactly the ablation's label.
  - `ablation_30pct_tracked_h8_current_m32_s0` — `obs_*` depth blobs removed.
- **Models.** `gru` (per-object GRU, ~30k parameters) and `gru_attention` (plus cross-object
  attention, ~64k parameters). Seeds 0, 1 and 2, default hyperparameters. Each run took under
  1 minute on an RTX 5060 laptop GPU.
- **Reproduce.**
  - `python -m models.GRU.summarize --run-name all_2026_09_28` (and `tracked_2026_09_28`)
  - `python evaluation/encounter_blob_check.py`

## Finding 1 — the encounter label is dominated by depth blobs

`detect_unlabeled_obstacles` (`src/perception_stack/pipeline.py`) emits one `unlabeled_obstacle`
per fixed column (20% of the width) of the lower 60% of every frame. Each reports the closest
depth pixel YOLO didn't claim, within 0.3–8 m, with no ground removal.

The data fits "ground ahead of the walker":
- 89% of blob boxes end at the image bottom, and all start at 40% of the image height.
- Their median distance is 2.5 m.

Two further consequences:
- **K0 is always 0 for them.** They are re-IDed every frame (`obs_<frame>_<cx>`), so they have no
  history and velocity 0.
- **Most encounter frames are unwinnable.** A blob that is close at T+3 counts as an encounter for
  frame T, but it doesn't exist at T. Only 36% of all-objects test encounter frames (438 / 1,215)
  have the encounter object visible, which caps every method's `encounter_top1`.

`evaluation/encounter_blob_check.py`, all 139 sessions, 95% session-bootstrap CIs:

| encounter_top1 | K0 | no-velocity | TTC |
|---|---|---|---|
| All objects (ablation definition) | 0.297 [0.254, 0.344] | 0.339 [0.290, 0.389] | 0.033 [0.013, 0.057] |
| Tracked objects only | 0.305 [0.177, 0.427] | 0.306 [0.203, 0.409] | 0.310 [0.177, 0.440] |

With the blobs removed, the three formulas **tie**. Two ablation results come from which formula
picks the ground blob, not from real encounters:
- TTC's much lower encounter recall;
- no-velocity's edge over K0.

Other ablation metrics (flicker, rank stability) were not re-checked here.

## Finding 2 — GRU results

Values are mean ± std over 3 seeds, on the 21 held-out test sessions (17 for tracked-only).
Baselines are scored on the same frames.

### All objects (ablation label)

| Method | AUROC | Avg. precision | encounter_top1 | flicker_rate |
|---|---:|---:|---:|---:|
| gru | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.302 ± 0.000 | 0.966 ± 0.002 |
| gru_attention | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.302 ± 0.000 | 0.960 ± 0.001 |
| K0 | 0.490 | 0.118 | 0.274 | 0.978 |
| nearest | 1.000 | 1.000 | 0.302 | 0.999 |
| nearest_in_cone | 1.000 | 1.000 | 0.302 | 0.999 |

Here the label is "the ground blob is within 1.5 m now", and current distance alone separates it
perfectly. Both GRUs learn exactly the nearest-object rule and sit at the reachable ceiling. This
variant cannot test prediction or scene context.

### Tracked objects only

| Method | AUROC | Avg. precision | encounter_top1 | flicker_rate |
|---|---:|---:|---:|---:|
| gru | 0.905 ± 0.054 | 0.388 ± 0.135 | 0.353 ± 0.064 | 0.581 ± 0.011 |
| gru_attention | 0.879 ± 0.022 | 0.233 ± 0.049 | 0.328 ± 0.031 | 0.570 ± 0.026 |
| K0 | 0.481 | 0.031 | 0.289 | 0.625 |
| nearest | 0.937 | 0.513 | 0.366 | 0.701 |
| nearest_in_cone | 0.940 | 0.515 | 0.366 | 0.678 |

Interaction slice (≥ 2 objects within 5 m, or a crosswalk; 41 encounter frames):
- gru 0.443 ± 0.072
- gru_attention 0.401 ± 0.041
- K0 0.351
- nearest 0.451

### Reading it

- **K0 does not predict close encounters.** Its per-object AUROC is 0.48, chance level. It ranks
  fast objects that pass at a distance above objects that actually come close. That is consistent
  with its design goal (consequence, not arrival), but it means `encounter_top1` is the wrong yardstick
  for K0.
- **The GRUs learn something real but do not beat "nearest".** They reach AUROC ≈ 0.9 against 0.94
  for nearest. Their top pick is steadier (flicker 0.57–0.58 vs. 0.68–0.70) at a slightly lower
  encounter_top1.
- **Attention adds nothing measurable.** It is slightly worse on average precision, within seed
  noise.
- **The tracked-only data is too thin to settle anything.**
  - Training has ~300 positive object-frames.
  - Validation has only 7, so early stopping is close to random: attention seeds stopped at epochs
    1–3.
  - The test set has 45 encounter frames from a few sessions. One seed's session-bootstrap CI for
    encounter_top1 is [0.000, 0.442].

## Next steps

1. **Relax the tracked-object label**, for example within 3 m inside the cone within 2 s, to get
   enough positives. It then no longer matches the ablation's definition.
2. **Make validation meaningful.** Use a larger validation fraction or cross-validation by
   session, so early stopping isn't driven by 7 positives.
3. **Add ground removal to `detect_unlabeled_obstacles`** (or drop the blobs from evaluation
   labels), then re-run the ablation's `encounter_top1`.
