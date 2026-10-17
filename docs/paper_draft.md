# Paper draft: hazard-arrival forecasting and alert policy for blind pedestrians, on SANPO

Working draft, started 2026-10-17. Each result is either **final** (with its source file) or marked
**[PENDING: what, owner]**. Methods detail lives in `methodology.md`; this file is the paper's text
and the checklist of what is still missing. Framing follows `reports/CPE v2 next steps.md`: an
open-loop benchmark of alert policies, aimed at a CVPR/ICCV accessibility workshop or a WACV
applications paper; the ASSETS version needs the listening study.

**Claim policy.** The replay is open-loop: the recorded walker never heard an alert. We claim
predictive accuracy against named baselines, precision and recall at frozen alert budgets,
opportunity-normalised lead time, announcement stability, a depth-realism bound and measured
on-device latency. We do **not** claim "safer", "reduces collisions", "usable by blind people" or
that any alert rate is acceptable.

---

## Status of each result

| § | Result | State | Source / owner |
|---|---|---|---|
| 6.1 | Baseline ladder on the corridor label | **[PENDING: re-run on the corrected labels]** — the current table (`methodology.md` §4.14 on `v2-kalman-lidar`) predates the camera-intrinsics fix | teammate, after merging `v2-threat-engine` |
| 6.2 | System replay at alert budgets | final for the declared tracker; **[PENDING: re-run with the fitted Kalman noise model]** | `threat_engine_replay/corridor_2026_10_17/`; teammate |
| 6.3 | Scene context, oracle (SANPO masks) | final | `scene_context/oracle_gt_2026_10_17/`, `gt_train_*`, `gt_val_test_*` |
| 6.4 | Edge-ahead alerts, oracle | final (rule not tuned) | `edge_alerts/gt_2026_10_17/`, `gt_sustained_2026_10_17/` |
| 6.5 | Segmentation model (YOLO26s-sem) | **[PENDING: training on the M4]** | `v2-seg-training` |
| 6.6 | Scene context with the model's masks, all 71 test sessions | **[PENDING: needs 6.5]** | `v2-seg-training` |
| 6.7 | Depth-realism (LiDAR) bound | **[PENDING: re-run on corrected labels]** | teammate |
| 6.8 | On-device latency, latency-aware replay | **[PENDING: TestFlight run]** | teammate |
| 6.9 | Announcement order (VLM referee, protocol v2) | **[PENDING: GPU-served VLMs]** | teammate |
| — | Audio clip library, clip durations vs. the announcer's 1.2 s | **[PENDING]** | `v2-narration` |
| — | BLV listening study (ASSETS version only) | **[PENDING: ethics]** | teammate (`listening_study_plan.md`) |

---

## Abstract (draft)

Phone-based hazard alerts for blind and low-vision pedestrians must decide not only *what* is
around the walker but *which* object to announce, and *when* to stay silent. Most systems rank
objects by proximity or a hand-built risk score and are evaluated by detection accuracy, which
says little about the alerts a user would hear. We frame alerting as **hazard-arrival
forecasting**: will this object enter the walker's path within the next 4 s? We build arrival
labels for SANPO-Real from measured depth and camera motion, on SANPO's official split. On this
benchmark, we compare a small GRU against a ladder of physics and learned baselines, and evaluate
complete alert engines by replaying every test session at frozen alert budgets. A deterministic
time-to-collision envelope bounds every learned decision. At five false alerts per minute, the
learned engine warns about [PENDING: final] of arrivals against [PENDING] for the strongest
geometric baseline, with [PENDING] fewer announcement switches. Walkable-surface segmentation adds
scene context; with SANPO's own masks as an oracle, one of three declared context rules points the
wrong way, and we report it. All models run on an iPhone 17 Pro Max in [PENDING] ms (p95). We
release labels, code and the replay harness.

---

## 1. Introduction

- **Problem.** A blind pedestrian using a camera-based aid hears at most one short phrase every
  1–2 s. Detection is a solved-enough sub-problem; the open one is the **alert policy**: which of
  the many detected objects deserves the next phrase, and when silence is better.
- **Gap.**
  - Assistive systems evaluate detectors (mAP) or run small user studies; neither isolates the
    alert policy.
  - Hand-built risk scores (our own v1 used `severity · v² / d`) are not tested against what
    actually happens next.
  - Systems that do measure outcomes need people walking (Pundlik et al. 2021); none offers a
    reproducible open-loop benchmark.
- **Idea.** Label every tracked object in SANPO-Real by whether it entered the walker's path within
  4 s, using SANPO's depth and camera poses. This turns alert policy into a measurable forecasting
  problem and lets whole engines be replayed.
- **Contributions.**
  1. An **arrival benchmark on SANPO-Real** (official split, 364 sessions with tracked objects):
     corridor arrival labels, opportunity windows and a replay harness, released.
  2. A **baseline ladder** for arrival forecasting: K0, TTC, constant-velocity, Kalman (fitted
     noise), nearest-in-corridor, logistic regression, gradient-boosted trees, and GRUs with 1, 4
     and 8 history steps. We report ties plainly.
  3. **Alert-policy evaluation at frozen budgets**: warned share, false alerts per minute and
     announcement switches, with operating points chosen on validation and
     opportunity-normalised lead time (Kaplan–Meier).
  4. **Scene context from walkable-surface segmentation**, with an oracle bound from SANPO's masks.
     It includes a negative result: the "near a boundary" rule points the wrong way.
  5. **On-device measurement** on an iPhone 17 Pro Max, and a depth-realism bound.

## 2. Related work

Full list: `related_work.md`. Must cite and distinguish:
- SANPO (Waghmare et al., WACV 2025): the data.
- Pundlik et al., JAMA Ophthalmology 2021: the closed-loop trial. Its silent-mode arm is the model
  for our open-loop framing.
- BlinkBud (arXiv 2512.01366): shares Kalman tracking and head-motion compensation.
- Mobilio and WOAD (Nat. Commun. 2025): BVI user studies.
- Schöller et al.: the constant-velocity baseline for pedestrian forecasting.
- Dauner et al., CoRL 2023: open-loop and closed-loop scores disagree.
- BBeep (CHI 2019): warning lead time and passing distance.
- Accident-anticipation work that reports lead time at matched recall.

## 3. Task and data

- **SANPO-Real**, `camera_head/left` (ZED-M, 15 FPS, 70–78° horizontal FOV, fx 1367–1391 px),
  processed every 3rd frame (5 Hz, 0.2 s per step). Depth is SANPO's stereo depth. Camera poses
  come from `fixed_camera_poses.csv` (OpenCV axes, Z-up world, NOT_READY rows interpolated).
- **Split.** SANPO's official split, on the 364 sessions with tracked objects: 252 train /
  41 validation (from official train) / 71 test. Everything is fitted or chosen on train or
  validation. Test is scored once.
- **Perception.** YOLO26n fine-tuned to 17 hazard classes (`yolo_training.md`), ByteTrack, and the
  median SANPO depth inside each box. Only tracked objects are labelled: the untracked depth
  sweep's pseudo-objects were 99.7% ground (`methodology.md` §4.9).
- **Arrival label (corridor).** An object *arrives* if, within 20 steps (4 s), it is inside the
  walking corridor: 0 < forward ≤ 3 m along the direction of travel (the 1 s camera displacement),
  and |lateral| ≤ 0.75/2 m + the class half-width. The test set has 71 sessions, 19 with arrivals,
  102 arriving tracks, and a positive rate of 0.140 over (frame, object) pairs. Sensitivity
  variants: width 0.75 / 1.5 m, horizon 13 / 20 / 25 steps (`dataset/GRU/relabel.py`).
- **Opportunity window.** Time from first sighting to arrival. Median 2.7 s; 25% of arrivals are
  first seen < 1 s before arriving, and 29% at least 4 s before. Lead time is bounded by this
  window, so we report lead conditional on it.

## 4. System

One frame at 5 Hz, in order (`architecture.md` §5–6):

1. **Fusion.** A constant-velocity Kalman filter per track on the ground plane, with head motion
   removed using the camera pose. The noise model is fitted by one-step predictive likelihood on
   train sessions, with Student-t measurement noise (ν = 4) **[PENDING: final numbers after the
   merge]**.
2. **When: arrival predictors.** GRU over 8 steps (1.6 s) of per-object features. The physics rule
   is time until the fused path enters the corridor. The Kalman predictor samples the state
   covariance to give P(arrive).
3. **How bad: severity.** A per-class severity from mass, raised to γ.
4. **What: scene context** (§4.1 below), as multipliers on the priority score.
5. **Safety envelope.** An object in the path with TTC ≤ 1 s always triggers an override,
   whatever the learned parts say.
6. **Announcement manager.** At most one phrase at a time. Overrides preempt. It applies a 4 s
   cooldown, a 1.5× escalation, a 1.25× switch margin, and drops stale messages (> 0.6 s).
7. **Narration.** Scripted templates ("Car close, left.") played from a pre-rendered clip library.
   There is no language model and no runtime speech synthesis.

### 4.1 Walkable-surface segmentation

- **Labels.** SANPO's 31 panoptic classes are mapped to 8: walkable, crosswalk, road, curb,
  stairs, non-walkable ground, obstacle and sky.
- **Model.** YOLO26s-sem at 1024 px, FP16 Core ML, run asynchronously at 2–5 Hz.
- **Training data.** All 237 SANPO-Real sessions with masks, on either camera: train 150 / val 39
  / test 48 sessions, with the replay's validation and test sessions held out.
- **Per track.** The ground under the object is the majority ground class in the object's image
  columns at its depth (±15%, ≥ 0.3 m). This needs only bearing and distance. Three flags follow:
  - *in path*: the object and the walker are both on walkable ground;
  - *near a boundary*: the object's ground mixes walkable and unsafe ground;
  - *other surface*: the walker is on walkable ground and the object is on road or non-walkable
    ground.

  Declared multipliers: ×1.5, ×1.25 → **×1.0** (§6.3), and ×0.5.
- **Per scene.** *Edge ahead*: scanning up the image centre, the first row where walkable ground
  gives way to curb, stairs, road or non-walkable ground, with its depth. It drives a separate
  "Curb ahead." alert (`src/segmentation/edge_alert.py`):
  - announced kinds: curb, stairs and road;
  - within 5 m;
  - confirmed in 2 consecutive segmentation frames;
  - one alert per episode;
  - re-armed after 2 s with no edge.

## 5. Evaluation protocol

- **Arrival ranking (held-out).**
  - Metrics: AUROC and AUPRC over test (frame, object) pairs.
  - Learned models are averaged over 3 seeds.
  - Paired session-bootstrap differences against the GRU (1,000 draws), with Holm correction
    within each stratum (static vs. moving).
- **System replay.** Every validation and test session is replayed frame by frame through each
  engine.
  - *Warned*: share of arrivals announced in the 4 s before arrival.
  - *False alert*: an announcement whose object does not arrive within 4 s.
  - Each engine's knob is chosen on validation as the setting with the most warned arrivals within
    an alert budget (2, 5 or 10 false alerts/min), then scored once on test.
  - We also report switches/min and the Kaplan–Meier share of arrivals not yet warned by time
    since first sighting.
- **Context ablation.** Context-on vs. context-off arms on the same sessions, as paired
  session-bootstrap differences.
- **Context flag validity.** P(arrive | flag) / P(arrive | not flag), Mantel–Haenszel-adjusted over
  distance bins (0/3/6/10 m), with session-bootstrap CIs. Any revision is fitted on train sessions
  and checked once on validation + test.
- **Segmentation.**
  - mIoU and per-class IoU.
  - **False-safe rate**: unsafe ground predicted walkable, over all unsafe-ground pixels.
  - Road-as-walkable rate.
  - Curb/stairs boundary F at ±3 px.
  - Each per camera, with session-bootstrap CIs.
- **Edge alerts.** No SANPO annotation marks "stepped off the curb", so the reference is the
  mask-implied *step-off*: the ground just ahead turns from walkable to curb, stairs or road. We
  report:
  - alerts/min;
  - the share of step-offs with an alert in the 10 s before, and the lead time;
  - the share of alerts followed by a step-off. This is a lower bound on precision: walkers stop,
    turn and walk along edges.

## 6. Results

### 6.1 Arrival forecasting: the baseline ladder

**[PENDING: the corrected-label re-run (teammate).]** Shape of the result on the stale labels
(`v2-kalman-lidar`, `methodology.md` §4.14):
- the GRU (AUROC ≈ 0.88) ties GBT, the fitted Kalman predictor and nearest-in-corridor;
- it beats the constant-velocity model, the declared Kalman filter, TTC and K0 (Holm p < 0.05).

If the tie holds, the learned model's claim moves from accuracy to **stability** and **budgeted
recall** (§6.2, §7).

### 6.2 System replay at frozen alert budgets

Source: `evaluation/benchmarks/threat_engine_replay/corridor_2026_10_17/`. 71 test sessions,
56 arrivals; declared Kalman noise. **[PENDING: replace with the fitted-tracker re-run.]**

| Engine (budget 5 false alerts/min) | knob | warned | false alerts/min | precision | switches/min |
|---|---:|---|---|---|---|
| v2, GRU | 0.8 | 0.32 [0.22, 0.55] | 4.55 [2.24, 7.23] | 0.54 [0.35, 0.68] | 6.76 [2.17, 12.20] |
| v2, GRU, no camera poses | 0.95 | 0.25 [0.13, 0.47] | 5.93 [2.99, 9.38] | 0.37 [0.23, 0.50] | 6.07 [1.79, 11.17] |
| nearest in corridor | 0.8 | 0.25 [0.16, 0.43] | 5.66 [3.04, 8.34] | 0.39 [0.23, 0.50] | 5.79 [1.86, 10.32] |
| v2, physics | — | meets no budget on validation (≥ ~17 false alerts/min) | | | |
| v1 (K0 routing) | — | meets no budget; at 50% recall: 88.7 false alerts/min | | | |

- At 10 false alerts/min, the GRU warns about 0.41 [0.28, 0.76].
- Kaplan–Meier, budget 5: 66% of arrivals are still unwarned 4 s after first sighting with the
  GRU, against 78% for nearest-in-corridor.
- No engine meets a 2 false alerts/min budget on validation.

### 6.3 Scene context: oracle bound with SANPO's own masks

Data: 47 replay sessions have head-camera masks (32 train / 5 val / 10 test), giving 5,682
(frame, object) observations. Source: `evaluation/benchmarks/scene_context/`.

| Flag | declared × | share | RR | RR, distance-adjusted |
|---|---:|---:|---|---|
| in path | 1.5 | 0.68 | 3.91 [1.73, 12.95] | 1.56 [0.98, 4.17] |
| near a boundary | 1.25 | 0.36 | 0.44 [0.17, 0.71] | **0.83 [0.41, 0.99]** |
| other surface | 0.5 | 0.21 | 0.29 [0.06, 0.67] | 0.70 [0.20, 1.06] |

- **"Near a boundary" points the wrong way.** Objects at a surface edge arrive less often, because
  most of them stand beside the walker's path at the sidewalk edge rather than in it.
- **Train-only revision.** On the 32 train sessions we tested a motion-qualified rule: near a
  boundary *and* closing on the walker's centre line at ≥ 0.3 m/s. It did not raise the arrival
  rate (distance-adjusted RR 0.80 [0.21, 1.15]), and neither did motion alone (1.00 [0.76, 1.29];
  bearing in the camera frame includes head turns).
- **Decision.** The multiplier was set to neutral (×1.0). On held-out validation + test the flag's
  RR is 1.04 [0.12, 6.22], consistent with neutral.
- **Replay** (10 test sessions, 20 arrivals; GRU at its validation-chosen threshold). Context on
  vs. off: warned +0.05 [0.00, +0.07] and false alerts/min +0.50 [0.00, +1.35] with the neutral
  boundary weight; with ×1.25 the false-alert change was 0.00. Twenty arrivals can't settle this;
  §6.6 is the powered test.

### 6.4 Edge-ahead alerts (oracle)

Source: `evaluation/benchmarks/edge_alerts/gt_2026_10_17/` and `gt_sustained_2026_10_17/`. Data:
53 head-masked sessions, 26.6 min at 2.5 Hz, with SANPO's masks as the class map.

- **Rate.** The rule fires 0.79 [0.49, 1.12] times per minute.
- **Warned.** 29% [13, 47] of mask-implied step-offs are warned, with a median lead of 1.2 s. This
  rises to 35% [14, 59] when one-frame mask flicker is excluded (sensitivity, chosen post hoc).
- **Stairs.** All 3 stairs step-offs are warned.
- **Road.** Most road step-offs are missed. Either the edge is only seen beyond 5 m, or no edge is
  in the centre band before the walker turns onto the road.
- **Followed.** About half of alerts are followed by a step-off within 10 s; the rest are edges
  the walker stopped at, turned from or walked along.
- **Not tuned.** The rule wasn't tuned on these sessions. The obvious levers (range, band width,
  confirmation) need a train-only sweep, and the model-mask re-run (§6.6) gives the
  model-vs-oracle gap.

### 6.5 Segmentation model

**[PENDING: `v2-seg-training`.]** Table: mIoU, false-safe rate, road-as-walkable, and curb/stairs
boundary F, per camera, test split (48 sessions), with session-bootstrap CIs. Compare against the
SENSATION-DS road-as-sidewalk rate.

### 6.6 Scene context with the model's masks

**[PENDING: needs §6.5.]** `tools/build_scene_context.py --source <best.pt> --all-sessions`, then
the replay with `--context-dir` over all 71 test sessions: context on vs. off at each budget,
paired. Repeat the edge-alert evaluation with the model's masks, against the oracle numbers in
§6.4.

### 6.7 Depth-realism bound · 6.8 On-device latency · 6.9 Announcement order

**[PENDING: teammate.]** Each comes with its own `methodology.md` section on `v2-kalman-lidar`:
- LiDAR-degraded depth: Δ for every headline metric;
- iPhone 17 Pro Max median / p95 / p99 by stage, a sustained run with the thermal timeline, and the
  latency-aware replay;
- VLM referee protocol v2: order swap, text-only control, human ceiling, Gwet's AC1 gate.

## 7. Discussion (points to make)

- **What learning buys.** The GRU may tie geometry on AUROC (§6.1), but at a fixed alert budget it
  warns about more arrivals with fewer false alerts than nearest-in-corridor (§6.2), and it ranks
  more stably. If the corrected-label ladder confirms a tie, say so plainly and make stability and
  budgeted recall the claim.
- **The opportunity window caps lead time.** A quarter of arrivals are first seen < 1 s before
  arriving, so no engine can warn about them in time. Report lead time conditional on the window,
  never unconditionally.
- **Alert budgets matter more than one F1 point.** The physics engine is competitive on ranking but
  can't get under ~17 false alerts/min at the system level.
- **Context rules must be tested, not assumed.** Two of three plausible rules hold; one is reversed.
  The oracle check costs nothing once masks exist, and it should come before any learned context
  feature.
- **Edges are a separate channel.** Most step-off events have no tracked object, so an object-only
  engine never announces them.

## 8. Limitations

- **Open-loop replay**, not deployment. Warned arrivals are not avoided ones (Dauner et al.). No
  user study in the workshop version.
- **Depth** comes from SANPO's stereo, not phone LiDAR; it is an upper bound, partly measured by
  §6.7. Labels and inputs share one depth source.
- **Head-mounted camera.** A phone would more likely be chest-worn. Chest masks exist for 130
  sessions but would need Stage 1 re-run on that stream.
- **Ego-motion.** Camera poses pass trajectory-level checks, but the object-level reconciliation is
  inconclusive, so the no-pose arm is reported alongside.
- **Declared constants.** Severity γ, behaviour multipliers, context multipliers and announcer
  timing are declared, not fitted. The near-boundary revision is the one exception, fitted on
  train and declared.
- **Edge alerts** are scored against a mask-implied step-off, not an annotated event.
- **Only tracked objects** are scored; the untracked depth sweep is excluded.

## 9. Conclusion

[Write last.]

---

## Figures and tables to produce

| # | Content | Source | Generator |
|---|---|---|---|
| F1 | System diagram (one frame, 5 Hz) | `architecture.md` §5 | by hand |
| F2 | Opportunity-window histogram (test) | replay `metrics.json` | `evaluation/generate_report_figures.py` (add) |
| F3 | Operating curves: warned vs. false alerts/min, all engines, budgets marked | replay CSV | same (add) |
| F4 | Kaplan–Meier "not yet warned", budget 5 | replay `metrics.json` | same (add) |
| F5 | Context-flag risk ratios, raw vs. distance-adjusted, train vs. held-out | `scene_context/*/metrics.json` | same (add) |
| F6 | Segmentation qualitative examples: curb, stairs, road vs. sidewalk; oracle vs. model | `v2-seg-training` | **[PENDING]** |
| T1 | Baseline ladder | `baseline_ladder/` | **[PENDING]** |
| T2 | Replay at budgets | §6.2 | — |
| T3 | Segmentation scores | `segmentation/` | **[PENDING]** |
| T4 | On-device latency | teammate | **[PENDING]** |
