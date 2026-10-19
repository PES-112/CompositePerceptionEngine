# Paper draft: hazard-arrival forecasting and alert policy for blind pedestrians, on SANPO

Working draft, started 2026-10-17. Each result is either **final** (with its source file) or marked
**[PENDING: what, owner]**. Methods detail lives in `methodology.md`; this file is the paper's text
and the checklist of what is still missing. Framing follows `reports/CPE v2 next steps.md`: an
open-loop benchmark of alert policies, aimed at a CVPR/ICCV accessibility workshop or a WACV
applications paper; the ASSETS version needs the listening study.

**Depth-decoding bug (found 2026-10-17, segmentation side fixed 2026-10-19).** Every SANPO depth
map was read as a scrambled 960×960 array. Any number below that depends on depth is superseded
until it is re-run; the status table says which.

**Claim policy.** The replay is open-loop: the recorded walker never heard an alert. We claim
predictive accuracy against named baselines, precision and recall at frozen alert budgets,
opportunity-normalised lead time, announcement stability and measured on-device latency, all on
SANPO's own depth. We do **not** claim "safer", "reduces collisions", "usable by blind people" or
that any alert rate is acceptable.

---

## Status of each result

| § | Result | State | Source / owner |
|---|---|---|---|
| 6.1 | Baseline ladder on the corridor label | **[PENDING: head-camera Stage-1 rebuild]**; methodology §4.16–4.17 are provisional | teammate (Day 2) |
| 6.2 | System replay at alert budgets | **[PENDING: head-camera rebuild]**; every replay before it used scrambled depth | teammate (Day 2) |
| 6.3 | Scene context, oracle (SANPO masks) | **[PENDING: redo on correct depth]**: the 2026-10-17 numbers and the near-boundary decision are superseded | segmentation (Day 2) |
| 6.4 | Edge-ahead alerts, oracle | final (correct depth; rule kept by a train-only sweep) | `edge_alerts/gt_depthfix_*` |
| 6.5 | Segmentation model (YOLO26n-sem @640) | final (scored once on test) | `segmentation/cpe_yolo26n_sem_walkable_test/` |
| 6.6 | Scene context and edge alerts with the model's masks | **[PENDING: redo on correct depth]**: the 2026-10-18 runs are superseded | Mac (Days 1–3) |
| 6.7 | Optional: phone-LiDAR range check (SANPO depth cut to 5 m) | optional; supplementary only | teammate |
| 6.8 | On-device latency, latency-aware replay | **[PENDING: TestFlight run]** | teammate |
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
time-to-collision envelope bounds every learned decision. At five false alerts per minute, the
learned engine warns about [PENDING: final] of arrivals against [PENDING] for the strongest
geometric baseline, with [PENDING] fewer announcement switches. Walkable-surface segmentation adds
scene context; with SANPO's own masks as an oracle, one of three declared context rules points the
wrong way, and we report it. A trained segmentation model reproduces the oracle's context flags
but does not change the alerts measurably, and its edge-ahead alerts fall well short of the
oracle's. All models run on an iPhone 17 Pro Max in [PENDING] ms (p95). We
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

> **Superseded (depth-decoding bug).** Numbers in this section that use depth were computed on scrambled depth maps and are being re-run (`pending_work.md`, Integration). The segmentation model's pixel scores don't use depth and stand.


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

> **Superseded (depth-decoding bug).** Numbers in this section that use depth were computed on scrambled depth maps and are being re-run (`pending_work.md`, Integration). The segmentation model's pixel scores don't use depth and stand.


The model's masks (YOLO26n-sem @640, §6.5) replace SANPO's. The 112 replay validation and test
sessions were never used in segmentation training. Sources: `scene_context/model_val_test_*_2026_10_18/`,
`threat_engine_replay/model_context_2026_10_18/`, `edge_alerts/cpe_yolo26n_sem_walkable_*_2026_10_18/`.

**Flags, model vs. oracle on the same 15 sessions** (5 val + 10 test, 1,561 observations; RR
distance-adjusted, session-bootstrap 95% CI):

| Flag | × | oracle (SANPO masks) | model |
|---|---:|---|---|
| in path | 1.5 | 2.64 [0.00, 7.64] | 2.68 [2.20, 10.34] |
| near a boundary | 1.0 | 1.04 [0.12, 6.22] | 1.15 [0.14, 5.97] |
| other surface | 0.5 | 0.38 [0.00, 1.67] | 0.37 [0.00, 0.44] |

The model reproduces the oracle's flags: same shares (in path 0.68 vs 0.72) and the same
directions. Over all 112 sessions (6,748 observations, which only the model can cover), the
effects shrink: in path 1.12 [0.78, 1.55], near a boundary 1.04 [0.61, 1.58], other surface 0.89
[0.40, 1.69]. None is significant there.

**Replay, context on vs. off** (all 71 test sessions, 56 arrivals, paired; GRU at each arm's
validation-chosen knob; declared tracker, **[PENDING: re-run with the fitted one]**). At `budget_10`: warned +0.04 [−0.06, +0.08], false alerts/min +0.41
[−0.30, +1.44], precision −0.01 [−0.07, +0.04]. At `budget_2` and `budget_5` the context arm has no
setting that meets the rule on validation. **Context does not change the alerts measurably.**

**Edge-ahead alerts, model vs. oracle.** Step-offs are taken from SANPO's masks for both
(`--reference gt`), so both are graded against the same events. 43 of the 53 head-masked sessions
are in the segmentation training split, so only the 10 official-test sessions are a fair score
for the model:

| | sessions | alerts / min | step-offs | warned | followed |
|---|---:|---:|---:|---|---:|
| oracle, test | 10 | 0.80 | 7 | 0.43 (3/7) | 0.50 |
| **model, test** | 10 | 1.00 | 7 | **0.14 (1/7)** | 0.40 |
| oracle, test, sustained | 10 | 0.80 | 4 | 0.50 (2/4) | 0.25 |
| **model, test, sustained** | 10 | 1.00 | 4 | **0.00 (0/4)** | 0.00 |
| oracle, all 53 | 53 | 0.79 [0.49, 1.12] | 45 | 0.29 [0.13, 0.47] | 0.48 |
| model, all 53 (43 trained on) | 53 | 0.83 [0.49, 1.22] | 45 | 0.22 [0.10, 0.37] | 0.45 |

- **The model's edge alerts fall far below the oracle's on held-out sessions.** It warns about 1
  of 7 step-offs, against 3 of 7 for the oracle, and 0 of 4 sustained ones. It alerts slightly more
  often, but fewer alerts lead to a step-off. Seven step-offs are too few for a CI. The result is
  consistent with the weak thin-edge scores (curb boundary F 0.30, stairs 0.06, road as walkable
  0.17).
- **Self-referenced** (step-offs from the model's own masks), the model scores 0.22 warned on 50
  step-offs; on test, 0.12 of 8. Grading the model against its own masks hides the gap, so the
  GT-referenced rows are the result.
- **Follow-up (not run).** n at a larger imgsz on a GPU machine, since thin edges depend on input
  resolution. s at 640 would not help (§6.5).

### 6.7 Optional: phone-LiDAR range · 6.8 On-device latency · 6.9 Announcement order

All headline results use SANPO's depth as is. **[PENDING: teammate.]** Each comes with its own
`methodology.md` section on `v2-kalman-lidar`:
- optional, supplementary: SANPO depth cut to the phone LiDAR's ~5 m range, showing how much of
  each headline metric survives. It's a what-if for a phone build, not part of the main results;
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
- **Depth** comes from SANPO's stereo throughout. A phone's LiDAR reaches only ~5 m, so the
  results describe a system with SANPO-quality depth. The optional §6.7 check shows what a 5 m
  limit would cost. Labels and inputs share one depth source.
- **Head-mounted camera.** A phone would more likely be chest-worn. Chest masks exist for 130
  sessions but would need Stage 1 re-run on that stream.
- **Ego-motion.** Camera poses pass trajectory-level checks, but the object-level reconciliation is
  inconclusive, so the no-pose arm is reported alongside.
- **Declared constants.** Severity γ, behaviour multipliers, context multipliers and announcer
  timing are declared, not fitted. The near-boundary revision is the one exception, fitted on
  train and declared.
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
| T1 | Baseline ladder | `baseline_ladder/` | **[PENDING]** |
| T2 | Replay at budgets | §6.2 | — |
| T3 | Segmentation scores | `segmentation/cpe_yolo26n_sem_walkable_test/` | §6.5 table |
| T4 | On-device latency | teammate | **[PENDING]** |
