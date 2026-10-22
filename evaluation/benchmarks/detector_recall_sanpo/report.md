# Pole recall on SANPO: inference size and pole confidence (2026-10-22)

`evaluation/detector_recall_sweep.py`. Objects and matching are those of methodology §4.21
(`evaluation/detector_recall_sanpo.py` on v2-segmentation): a connected component of a SANPO class,
≥ 0.3% of the image, median depth ≤ 10 m; found if a v3 box of a matching class covers ≥ 50% of its
bounding box. Raw detector boxes (no tracker, no size check). 95% Wilson intervals.

**Knobs (no retraining).** A: input size 640 (Stage 1 today) / 960 / 1280. B: a lower confidence for
pole and bollard boxes only (0.30 / 0.20 / 0.15 / 0.10); other classes stay at 0.30.

**Selection** on 172 human-annotated head-camera frames of 43 official-train sessions (the detector
never saw SANPO). Rule fixed before looking: highest pole recall with pedestrian recall ≥ baseline −
0.05 and pole-box support ≥ 0.5. Chosen: **960 px, pole conf 0.10** (`selection_train.json`).

| train (172 frames) | poles | pedestrians | vehicles |
|---|---|---|---|
| 640 / 0.30 (today) | 0/50 | 0.59 | 0.18 |
| best pole setting (960 / 0.10) | 3/50 | 0.62 | 0.21 |
| 1280 / any | 1/50 | **0.71** [0.63, 0.79] | 0.18 |

**Test, scored once** (80 frames, 10 sessions; the §4.21 frames; `test.json`):

| setting | poles | pedestrians | vehicles | pole boxes/frame | pole-box support |
|---|---|---|---|---|---|
| before: 640, conf 0.30 | 1/22 = 0.05 [0.01, 0.22] | 28/44 = 0.64 [0.49, 0.76] | 1/8 | 0.09 | 0.86 |
| A: 960 | 1/22 = 0.05 [0.01, 0.22] | 29/44 = 0.66 [0.51, 0.78] | 1/8 | 0.09 | 1.00 |
| B: pole conf 0.10 | 1/22 = 0.05 [0.01, 0.22] | 28/44 | 1/8 | 0.11 | 0.89 |
| A+B | 2/22 = 0.09 [0.03, 0.28] | 29/44 = 0.66 [0.51, 0.78] | 1/8 | 0.16 | 1.00 |

Pole-box support: share of predicted pole/bollard boxes with ≥ 20% thin-vertical SANPO pixels (pole,
sign, light, bike rack, obstacle, bus stop, rails); a precision proxy.

**Reading.**
- **A and B don't fix poles.** 1/22 → 2/22 on test; the intervals overlap fully.
- **The detector puts no box on most SANPO poles at all.** At any of the three sizes and confidence
  ≥ 0.05, no box covers ≥ 50% of 40/50 train and 20/22 test poles. They aren't small: the median pole
  is 175 px (train) and 123 px (test) wide in the 2208 px frame. So the cause is not resolution or
  threshold but what v3 learned a "pole" looks like (Roboflow poles vs. SANPO's street furniture seen
  from head height). Fine-tuning on SANPO (boxes from its panoptic masks) is the fix; future work.
- **Side finding:** 1280 px raises pedestrian recall on train from 0.59 to 0.71 (CIs barely overlap).
  Not tested on test (it was not the declared rule's choice), and it costs about 4× detector time on
  the phone.
- **Pedestrians here 28/44, §4.21 reports 34/44** on what should be the same frames and rule. Not
  resolved: possibly a different Ultralytics version (8.4.38 here) or frame list. The pole count
  (1/22) matches.

**Videos** (`tools/render_pole_comparison.py --pick 3`; written to `data/demo/poles/`, not committed):
the three test frames with the most SANPO poles, 8 s around each, before | after side by side, SANPO
pole mask in cyan. Picked for having poles; illustration, not measurement.
