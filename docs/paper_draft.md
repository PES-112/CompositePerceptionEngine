# Paper draft: hazard-arrival forecasting and alert policy for blind pedestrians, on SANPO

Working draft, started 2026-10-17. Each result is either **final** (with its source file) or marked
**[PENDING: what, owner]**. Methods detail lives in `methodology.md`; this file is the paper's text
and the checklist of what is still missing. Framing follows `reports/CPE v2 next steps.md`: an
open-loop benchmark of alert policies, aimed at a CVPR/ICCV accessibility workshop or a WACV
applications paper; the ASSETS version needs the listening study.

**Depth-decoding bug and head-camera rerun.** Until 2026-10-17 every SANPO depth map was read as a
scrambled 960×960 array, and the 2026-10-18 rebuild used the chest camera's fx. Stage 1 was re-run
with the head camera's fx on correct depth (2026-10-20, `methodology.md` §4.18), and the numbers
below come from that rerun unless the status table says otherwise.

**Claim policy.** The replay is open-loop: the recorded walker never heard an alert. We claim
predictive accuracy against named baselines, precision and recall at frozen alert budgets,
opportunity-normalised lead time, announcement stability and measured on-device latency, all on
SANPO's own depth. We do **not** claim "safer", "reduces collisions", "usable by blind people" or
that any alert rate is acceptable.

---

## Status of each result

| § | Result | State | Source / owner |
|---|---|---|---|
| 6.1 | Baseline ladder on the corridor label | final (head-camera rerun; methodology §4.18) | `baseline_ladder/…_corridor_w0.75_d3_h20/` |
| 6.2 | System replay at alert budgets | final (head-camera rerun; methodology §4.18) | `threat_engine_replay/corridor_headfx/` |
| 6.3 | Scene context, oracle (SANPO masks) | final (head-camera rerun; near-boundary decided on train; methodology §4.19) | `scene_context/gt_headfx_*`, `threat_engine_replay/oracle_context_headfx_2026_10_20/` |
| 6.4 | Edge-ahead alerts, oracle | final (correct depth; rule kept by a train-only sweep) | `edge_alerts/gt_depthfix_*` |
| 6.5 | Segmentation model (YOLO26n-sem @640) | final (scored once on test) | `segmentation/cpe_yolo26n_sem_walkable_test/` |
| 6.6 | Edge alerts with the model's masks | final (correct depth) | `edge_alerts/cpe_yolo26n_sem_walkable_depthfix_*` |
| 6.6 | Scene context with the model's masks | **[PENDING: re-run on the head-camera rerun]**; the 2026-10-18 runs are superseded | Mac (Days 2–3) |
| 6.7 | Optional: phone-LiDAR range check (SANPO depth cut to 5 m) | optional; supplementary only | teammate |
| 6.8 | On-device latency, latency-aware replay | **[PENDING: iPhone run]** (sideloaded build, `ios/CPETiming/RUN_ON_IPHONE.md`) | teammate |
| 6.9 | Announcement order (VLM referee, protocol v2) | **[PENDING: GPU-served VLMs]** | teammate |
| — | English audio clip library, clip durations vs. the announcer's 1.2 s | **[PENDING]**; translation is future work | `v2-narration` (Days 1–4) |
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
time-to-collision envelope bounds every learned decision. The GRU (AUROC 0.911) ties
gradient-boosted trees and beats every geometric baseline. At five false alerts per minute, the
learned engine warns about 0.35 of arrivals at precision 0.55, against 0.23 at 0.39 for the
strongest geometric baseline, though it switches announcements more often. Few warnings come early:
0.14 of arrivals are warned at least 0.2 s ahead. Walkable-surface segmentation adds scene context;
with SANPO's own masks as an oracle, one of three declared context rules points the wrong way, and
we report it, and context does not change the alerts measurably. A trained segmentation model's
edge-ahead ("Curb ahead.") alerts match the oracle's. [PENDING: model context.] All models run on
an iPhone 17 Pro Max in [PENDING] ms (p95). We release labels, code and the replay harness.

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
  1. An **arrival benchmark on SANPO-Real** (official split, 421 sessions with tracked objects):
     corridor arrival labels, opportunity windows and a replay harness, released.
  2. A **baseline ladder** for arrival forecasting: K0, TTC, constant-velocity, Kalman (fitted
     noise), nearest-in-corridor, logistic regression, gradient-boosted trees, and GRUs with 1, 4
     and 8 history steps. We report ties plainly.
  3. **Alert-policy evaluation at frozen budgets**: warned share, false alerts per minute and
     announcement switches, with operating points chosen on validation and
     opportunity-normalised lead time (Kaplan–Meier).
  4. **Scene context from walkable-surface segmentation**, with an oracle bound from SANPO's masks.
     It includes a negative result: the "near a boundary" rule points the wrong way.
  5. **On-device measurement** on an iPhone 17 Pro Max.

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
- **Split.** SANPO's official split, on the 421 sessions with tracked objects: 288 train /
  48 validation (from official train) / 85 test. Everything is fitted or chosen on train or
  validation. Test is scored once.
- **Perception.** YOLO26n fine-tuned to 17 hazard classes (`yolo_training.md`), ByteTrack, and the
  median SANPO depth inside each box. Only tracked objects are labelled: the untracked depth
  sweep's pseudo-objects were 99.7% ground (`methodology.md` §4.9).
- **Arrival label (corridor).** An object *arrives* if, within 20 steps (4 s), it is inside the
  walking corridor: 0 < forward ≤ 3 m along the direction of travel (the 1 s camera displacement),
  and |lateral| ≤ 0.75/2 m + the class half-width. The test set has 85 sessions, 24 with arrivals,
  6,628 objects, and a positive rate of 0.161 over (frame, object) pairs; the replay counts 79
  arrivals (tracks first seen before entering the zone). Sensitivity variants: width 0.75 / 1.5 m,
  horizon 13 / 20 / 25 steps (`dataset/GRU/relabel.py`).
- **Opportunity window.** Time from first sighting to arrival. Median 2.6 s; 24% of arrivals are
  first seen < 1 s before arriving, and 33% at least 4 s before. Lead time is bounded by this
  window, so we report lead conditional on it.

## 4. System

One frame at 5 Hz, in order (`architecture.md` §5–6):

1. **Fusion.** A constant-velocity Kalman filter per track on the ground plane, with head motion
   removed using the camera pose. The noise model is fitted on train sessions by 1 s-ahead
   predictive likelihood. Validation chose the Student-t update (ν = 4, gate 60) over the
   declared values (log-likelihood −3.09 vs −3.51).
2. **When: arrival predictors.** GRU over 8 steps (1.6 s) of per-object features. The physics rule
   is time until the fused path enters the corridor. The Kalman predictor samples the state
   covariance to give P(arrive).
3. **How bad: severity.** A per-class severity from mass, raised to γ.
4. **What: scene context** (§4.1 below), as multipliers on the priority score.
5. **Safety envelope.** An object in the path with TTC ≤ 1 s always triggers an override,
   whatever the learned parts say.
6. **Announcement manager.** At most one phrase at a time. Overrides preempt. Declared defaults:
   a 4 s cooldown, a 1.5× escalation, a 1.25× switch margin, and stale messages dropped after
   0.6 s; the replay chooses cooldown and switch margin on validation together with the threshold
   (§5). Segmentation edge alerts ("Curb ahead.", §4.1) have their own lane, below overrides and
   above object threats, and wait up to 1.2 s (one clip). The replay does not include them; they
   are scored on their own (§6.4).
7. **Narration.** Scripted templates ("Car close, left.") played from a pre-rendered clip library.
   There is no language model and no runtime speech synthesis.

### 4.1 Walkable-surface segmentation

- **Labels.** SANPO's 31 panoptic classes are mapped to 8: walkable, crosswalk, road, curb,
  stairs, non-walkable ground, obstacle and sky.
- **Model.** YOLO26n-sem at 640 px, FP16 Core ML (argmax in the graph), run asynchronously at
  2–5 Hz.
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
    an alert budget (2, 5 or 10 false alerts/min), then scored once on test. For the pose-based
    GRU and Kalman engines and nearest-in-corridor, the knob includes the announcer's cooldown and
    switch margin.
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

Source: `evaluation/benchmarks/baseline_ladder/sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20/`
(`methodology.md` §4.18). 85 test sessions, 6,628 objects, positive rate 0.161. Learned methods
are averaged over 3 seeds. Differences are against the GRU, with paired session-bootstrap 95% CIs
and Holm correction (30 tests per stratum).

| Method | AUROC | AUPRC | ΔAUROC vs GRU | Holm p |
|---|---:|---:|---|---:|
| GBT, with history | 0.917 | 0.753 | +0.006 [−0.005, +0.014] | 1.00 |
| GRU, attention | 0.916 | 0.739 | +0.005 [−0.008, +0.012] | 1.00 |
| GBT, current step | 0.912 | 0.745 | +0.001 [−0.009, +0.009] | 1.00 |
| **GRU (8 steps)** | **0.911** | **0.743** | — | — |
| logistic regression | 0.888 | 0.622 | −0.024 [−0.039, −0.010] | 0.022 |
| nearest | 0.886 | 0.621 | −0.025 [−0.041, −0.006] | 0.085 |
| constant velocity (2 steps) | 0.880 | 0.704 | −0.031 [−0.045, −0.017] | < 0.001 |
| Kalman, declared noise | 0.877 | 0.662 | −0.035 [−0.052, −0.016] | 0.001 |
| nearest in corridor | 0.873 | 0.663 | −0.039 [−0.064, −0.019] | 0.005 |
| Kalman, fitted noise | 0.864 | 0.689 | −0.048 [−0.069, −0.022] | 0.001 |
| TTC | 0.669 | 0.388 | −0.243 [−0.340, −0.195] | < 0.001 |
| K0 (v1 score) | 0.537 | 0.168 | −0.375 [−0.439, −0.323] | < 0.001 |

- **The learned models tie each other.** The GRU with 1, 4 or 8 history steps, the attention GRU
  and GBT are within 0.006 AUROC; no difference survives Holm.
- **They beat every geometric baseline.** All CIs for the Kalman, constant-velocity,
  nearest-in-corridor, TTC and K0 baselines exclude 0. `nearest` is below the GRU on AUROC (Holm p
  0.085) and AUPRC (−0.122, Holm p < 0.001).
- **The fitted Kalman does not beat the declared one** (AUROC −0.013 [−0.022, +0.001], Holm p
  0.30), although it wins on validation likelihood (§4). In the replay it still cuts false alerts
  (§6.2).
- **Static classes** (463 objects): simple rules rank them at least as well as the GRU
  (nearest-in-corridor +0.082 [+0.002, +0.147]); nothing survives Holm.

So the learned model's advantage is over geometry, not over other learned models. Choosing the GRU
over GBT is a deployment decision, not an accuracy one.

### 6.2 System replay at frozen alert budgets

Source: `evaluation/benchmarks/threat_engine_replay/corridor_headfx/` (`methodology.md` §4.18).
85 test sessions, 79 arrivals; fitted Kalman noise; operating points chosen on 48 validation
sessions. Knob = threshold / cooldown (s) / switch margin / stale (s).

| Engine (budget 5 false alerts/min) | knob | warned | warned ≥ 0.2 s ahead | false alerts/min | precision | switches/min |
|---|---:|---|---|---|---|---|
| v2, GRU | 0.6 / 8 / 1 / 0.6 | 0.35 [0.23, 0.52] | 0.14 [0.06, 0.25] | 5.62 [3.44, 8.09] | 0.55 [0.40, 0.69] | 8.31 [3.72, 13.65] |
| nearest in corridor | 0.95 / 2 / 1 / 0.6 | 0.23 [0.11, 0.35] | 0.10 [0.02, 0.19] | 4.93 [3.09, 7.01] | 0.39 [0.25, 0.52] | 5.28 [2.23, 8.84] |
| v2, Kalman; v2, GRU without camera poses | — | meet only the 10/min budget on validation | | | | |
| v2, physics | — | meets no budget on validation (≥ 14.0 false alerts/min) | | | | |
| v1 (K0 routing) | — | meets no budget (≥ 22.9); at 50% recall: 99.8 false alerts/min | | | | |

- **The GRU warns more, and more precisely,** than the strongest geometric baseline at the same
  budget, with overlapping CIs. It switches announcements more often (8.3 vs 5.3 per minute).
- **Lead is the limit.** Only 0.04 of arrivals are warned ≥ 1 s ahead, although 76% are first
  seen ≥ 1 s ahead. Tuning on lead (`early_budget_*`) picks the same points.
- **At 10 false alerts/min**, the GRU warns 0.46 [0.36, 0.64] (7.2 false/min, precision 0.53) and
  the Kalman engine 0.38 [0.25, 0.57] (9.8 false/min, precision 0.43).
- **Fitted vs. declared Kalman noise**, physics engine at fixed thresholds: false alerts fall by
  3.1–5.7 per minute and precision rises by 0.04–0.05 (CIs exclude 0); recall falls by 0.03–0.11.
- **Kaplan–Meier, budget 5:** 64% of arrivals are still unwarned 4 s after first sighting with the
  GRU, against 78% for nearest-in-corridor.
- **No engine meets 2 false alerts/min** on validation. Validation predicts test false alerts well
  at 5/min (4.9–5.6 on test).

### 6.3 Scene context: oracle bound with SANPO's own masks

Data: 53 replay sessions have head-camera masks (35 train / 8 val / 10 test), giving 5,898
(frame, object) observations. Sources: `evaluation/benchmarks/scene_context/gt_headfx_*_2026_10_20/`
(`methodology.md` §4.19). RR = P(arrive | flag) / P(arrive | not flag), Mantel–Haenszel-adjusted
over distance bins, session-bootstrap 95% CI.

| Flag | declared × | share | RR | RR, distance-adjusted |
|---|---:|---:|---|---|
| in path | 1.5 | 0.75 | 1.11 [0.59, 2.14] | 1.32 [0.93, 2.09] |
| near a boundary | 1.25 | 0.18 | 0.41 [0.23, 0.59] | **0.69 [0.42, 0.87]** |
| other surface | 0.5 | 0.11 | 0.61 [0.22, 1.02] | 0.89 [0.38, 1.22] |

- **"Near a boundary" points the wrong way.** Objects at a surface edge arrive less often, because
  most of them stand beside the walker's path at the sidewalk edge rather than in it.
- **The other two rules are not supported either.** "In path" points the right way but its CI
  includes 1; "other surface" is near 1, so its ×0.5 has no support here. No revision was
  declared for them, so they stay as declared.
- **Train-only revision of "near a boundary"**, with the decision rule fixed before looking: keep
  a motion-qualified flag only if its distance-adjusted RR on train is above 1, otherwise make the
  flag neutral. On the 35 train sessions (4,145 observations), the flag gives 0.60 [0.23, 0.83],
  and near a boundary *and* closing on the walker's centre line at ≥ 0.3 m/s gives 0.51 [0.00,
  0.84]. Motion alone gives 0.86 [0.59, 0.99] (bearing in the camera frame includes head turns).
- **Decision.** The multiplier is neutral (×1.0). On held-out validation + test (1,753
  observations) the flag's RR is 0.80 [0.38, 3.37], consistent with neutral.
- **Replay** (`threat_engine_replay/oracle_context_headfx_2026_10_20/`; 10 test sessions, 27
  arrivals; GRU, each arm at its validation-chosen knob). At budget 5 both arms choose threshold
  0.4. Context on vs. off: warned ±0.00, false alerts/min +1.49 [0.00, +3.86], precision −0.04
  [−0.20, +0.02]. At a fixed threshold of 0.6, context gives 0.26 warned and 3.0 false/min, against
  0.15 and 2.5 without it. With 75% of objects in path, context mostly scales scores up, like a
  lower threshold. Twenty-seven arrivals can't separate the two; §6.6 is the powered test.

### 6.4 Edge-ahead alerts (oracle)

Source: `evaluation/benchmarks/edge_alerts/gt_depthfix_2026_10_19/`, `_sustained_`,
`_trainsweep_`. Data: 53 head-masked sessions, 26.6 min at 2.5 Hz, with SANPO's masks as the class
map, on correct depth.

- **Rate and recall.** The rule fires 0.68 [0.40, 1.01] times per minute. It warns 18%
  [6, 33] of mask-implied step-offs, with a median lead of 0.8 s; 19% excluding one-frame flicker.
- **Stairs.** All 3 stairs step-offs are warned.
- **Road.** Only 12% of road step-offs are warned. The usual miss is an edge first seen 6–10 m
  ahead that never gets confirmed inside the 5 m range: a head-mounted camera sees no ground closer
  than about 2.6 m, so the window is about 1.5 s.
- **Trade-off (train-only sweep).** Recall can be bought with alerts: 0.18 at 0.7 alerts/min,
  0.34 at 1.1, 0.58 at 2.2 and 0.76 at 3.2. The declared ≤ 1/min budget keeps the 5 m rule. Where
  on that curve a blind walker wants to be is a question for the listening study, not for us.

### 6.5 Segmentation model

YOLO26n-sem from Cityscapes weights, imgsz 640, batch 8, trained on the M4 (MPS). It
early-stopped at epoch 8 (patience 5); best = epoch 3 by validation mIoU (0.570), after which
validation loss rose. Scored once on the test split: 48 sessions, 3,850 human-annotated frames,
session-bootstrap 95% CIs. Source: `evaluation/benchmarks/segmentation/cpe_yolo26n_sem_walkable_test/`.

| | frames | mIoU | false-safe | road as walkable | walkable as unsafe | boundary F curb | boundary F stairs |
|---|---:|---|---|---:|---:|---:|---:|
| all | 3,850 | **0.615** [0.540, 0.641] | **0.044** [0.029, 0.062] | 0.173 | 0.056 | 0.295 | 0.062 |
| chest | 2,561 | 0.604 | 0.058 | 0.163 | 0.053 | 0.305 | 0.067 |
| head | 1,289 | 0.554 | 0.026 | 0.222 | 0.067 | 0.283 | — (no stairs) |

- **Definitions.** False-safe = unsafe-ground pixels (road, curb, stairs, non-walkable) predicted
  walkable or crosswalk, over all unsafe-ground pixels. Road as walkable = ground-truth road pixels
  predicted walkable or crosswalk, over all ground-truth road pixels (SENSATION-DS's
  road-as-sidewalk error, with crosswalk counted as walkable). Boundary F at ±3 px.
- **Per class (IoU).** walkable 0.83, crosswalk 0.59, road 0.49, curb 0.24, stairs 0.47,
  non-walkable 0.69, obstacle 0.74, sky 0.87. Head test frames contain no stairs, so head mIoU
  includes a 0 for stairs.
- **Weak spots.** Thin edges: curb boundary F 0.30, stairs 0.06. About one road pixel in six is
  called walkable. **[PENDING: SENSATION-DS's published rate, with its definition, for comparison.]**
- **Model choice (deviation, declared).** The plan declared before training was to try YOLO26s-sem at
  640 if test scores showed the model to be the bottleneck. We kept n instead:
  1. validation mIoU peaked at epoch 3 and validation loss then rose, so the model was
     overfitting; more capacity on the same data is unlikely to help;
  2. thin edges depend on input resolution more than on model size, and s at 640 sees the same
     pixels;
  3. choosing a second model after seeing test scores would make the final model test-informed.

  The follow-up is n at a larger imgsz on a GPU machine, not s at 640.
- **On device.** Exported to Core ML (FP16, argmax in the graph, 3.1 MB); pixel agreement with
  PyTorch 0.99–1.00 on 4 test frames. On-iPhone latency: **[PENDING: §6.8]** (Ultralytics reports
  4.9 ms for YOLO26n-sem on iPhone 17 Pro).

### 6.6 Scene context and edge alerts with the model's masks

The model's masks (YOLO26n-sem @640, §6.5) replace SANPO's.

**Scene context, model.** **[PENDING: re-run on the head-camera rerun (Mac, Day 2–3): flags on
the oracle's 18 val/test sessions and on all 133 val/test sessions; context on vs. off replay on all
85 test sessions.]** The 2026-10-18 run (scrambled depth, chest-camera fx) found that the model
reproduced the oracle's flags and that context did not change the alerts measurably; those
numbers are superseded and not reported.

**Edge-ahead alerts, model vs. oracle** (correct depth; `edge_alerts/cpe_yolo26n_sem_walkable_depthfix_*`
and `gt_depthfix_*`). Step-offs are taken from SANPO's masks for both (`--reference gt`), so both
are graded against the same events. 43 of the 53 head-masked sessions are in the segmentation
training split, so only the 10 official-test sessions are a fair score for the model:

| | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---|---:|
| oracle, test | 10 | 0.60 | 7 | 0.14 (1/7) | 0.33 |
| **model, test** | 10 | 0.80 | 7 | **0.14 (1/7)** | 0.50 |
| oracle, test, sustained | 10 | 0.60 | 4 | 0.00 (0/4) | 0.00 |
| **model, test, sustained** | 10 | 0.80 | 4 | **0.00 (0/4)** | 0.00 |
| oracle, all 53 | 53 | 0.68 [0.40, 1.01] | 45 | 0.18 [0.06, 0.33] | 0.33 |
| model, all 53 (43 trained on) | 53 | 0.56 [0.31, 0.87] | 45 | 0.20 [0.08, 0.36] | 0.60 |

- **The model's edge alerts match the oracle's.** On held-out sessions both warn 1 of 7 step-offs
  and none of the 4 sustained ones. Over all 53 sessions the model warns 0.20 against the oracle's
  0.18, and more of its alerts are followed by a step-off (0.60 vs 0.33). Seven step-offs are too
  few for a CI.
- **The limit is the rule's geometry, not the masks.** On correct depth the oracle itself warns
  only 18% (§6.4): edges are seen beyond 5 m and pass out of view before they are confirmed. The
  2026-10-18 finding that the model fell far below the oracle came from scrambled depth.
- **Thin edges are still weak in pixels** (curb boundary F 0.30, stairs 0.06, road as walkable
  0.17; §6.5). Follow-up (not run): n at a larger imgsz on a GPU machine. s at 640 would not help
  (§6.5).

### 6.7 Optional: phone-LiDAR range · 6.8 On-device latency · 6.9 Announcement order

All headline results use SANPO's depth as is. **[PENDING: teammate.]** Each comes with its own
`methodology.md` section on `v2-kalman-lidar`:
- optional, supplementary: SANPO depth cut to the phone LiDAR's ~5 m range, showing how much of
  each headline metric survives. It's a what-if for a phone build, not part of the main results;
- iPhone 17 Pro Max median / p95 / p99 by stage, a sustained run with the thermal timeline, and the
  latency-aware replay;
- VLM referee protocol v2: order swap, text-only control, human ceiling, Gwet's AC1 gate.

## 7. Discussion (points to make)

- **What learning buys.** Learned models beat geometry on ranking (+0.024 to +0.375 AUROC over
  the geometric baselines, §6.1) and tie each other. At a 5/min budget the GRU warns more arrivals
  at higher precision than nearest-in-corridor (§6.2), but it switches more often, so stability is
  not a claim we can make.
- **Few warnings are early.** At 5 false alerts/min, 0.14 of arrivals are warned ≥ 0.2 s ahead
  and 0.04 ≥ 1 s ahead, though three-quarters are visible ≥ 1 s ahead. The predictors rarely put
  an object above threshold early. Training for early probability mass is the next step.
- **The opportunity window caps lead time.** A quarter of arrivals are first seen < 1 s before
  arriving, so no engine can warn about them in time. Report lead time conditional on the window,
  never unconditionally.
- **Alert budgets matter more than one AUROC point.** The constant-velocity and Kalman predictors
  trail the GRU by only 0.03–0.05 AUROC (§6.1), but the physics engine can't get under ~14 false
  alerts/min at the system level (§6.2); the fitted
  Kalman ties the declared one on ranking yet cuts 3–6 false alerts/min in the replay.
- **Context rules must be tested, not assumed.** Of three plausible rules, one is reversed and the
  other two are not supported at this sample size. The oracle check costs nothing once masks
  exist, and it should come before any learned context feature.
- **Edges are a separate channel.** Most step-off events have no tracked object, so an object-only
  engine never announces them.

## 8. Limitations

- **Open-loop replay**, not deployment. Warned arrivals are not avoided ones (Dauner et al.). No
  user study in the workshop version.
- **Depth** comes from SANPO's stereo throughout. A phone's LiDAR reaches only ~5 m, so the
  results describe a system with SANPO-quality depth. The optional §6.7 check shows what a 5 m
  limit would cost. Labels and inputs share one depth source.
- **Head-mounted camera.** A phone would more likely be chest-worn. Chest masks exist for 130
  sessions but would need Stage 1 re-run on that stream.
- **Ego-motion.** Camera poses pass trajectory-level checks, but the object-level reconciliation is
  inconclusive, so the no-pose arm is reported alongside.
- **Declared constants.** Severity γ, behaviour multipliers, context multipliers, escalation and
  staleness are declared, not fitted. The announcer's cooldown and switch margin are chosen on
  validation with each threshold. The near-boundary multiplier was set by a rule declared before
  looking and applied to train sessions only.
- **Edge alerts** are scored against a mask-implied step-off, not an annotated event. Only 10
  held-out sessions (7 step-offs) score the model's edge alerts.
- **Segmentation.** Thin edges are weak: curb boundary F 0.30 and stairs 0.06 (±3 px). Road as
  walkable is 0.173: ground-truth road pixels predicted walkable or crosswalk, over all
  ground-truth road pixels. YOLO26n-sem was kept at 640 px for the training budget on an M4; s was
  not tried (§6.5).
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
| T1 | Baseline ladder | `baseline_ladder/…_corridor_w0.75_d3_h20/` | §6.1 table |
| T2 | Replay at budgets | `threat_engine_replay/corridor_headfx/` | §6.2 table |
| T3 | Segmentation scores | `segmentation/cpe_yolo26n_sem_walkable_test/` | §6.5 table |
| T4 | On-device latency | teammate | **[PENDING]** |
