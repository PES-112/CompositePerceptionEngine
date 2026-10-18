# CPE System Design (v2)

**Status:** adopted 2026-10-14. This replaces the v1 design, in which an SLM-1 cognitive layer
was arbitrated by a physics judge and narrated by SLM-2. v1 is preserved in git history (this file at
commit `a6e37a7`).

Legend used throughout: ✅ built and tested · 🟡 built offline, not yet in the runtime pipeline ·
⬜ planned.

---

## 1. One-line pitch

A phone-based hazard alert for blind pedestrians. It ranks nearby objects by **when** they will
reach the walker, **how bad** a collision would be, and **what** the scene says about them. It keeps
every learned decision inside a **deterministic physics safety envelope**, and it is evaluated
end-to-end by replaying the SANPO dataset.

## 2. Design principles

1. **Physics is the safety floor.** An object about to arrive (TTC ≤ 1 s, in the walking path)
   always triggers an alert, whatever the learned models say. The models can only reorder what
   comes after that.
2. **Each component answers a different question.**
   - **When:** the GRU arrival predictor.
   - **How bad:** physics class severity.
   - **What:** walkable-surface segmentation.

   Components that answer the same question add little. Our ensemble results show this: combining
   four arrival predictors gained only +0.023 AUROC.
3. **No language model at runtime.** Every runtime model is small, fast and testable. Narration is
   scripted.
4. **Target device: iPhone 17 Pro Max.** It's evaluated by **replaying SANPO**, not by a live
   deployment (§4).
5. **No human ground truth required.** Labels come from the walker's measured future (did the object
   actually come close?). The one judgement call, "which object should be announced first", is
   judged by a blinded referee calibrated against a small human set (§8).

## 3. What changed from v1

| v1 component | v2 replacement | Why |
|---|---|---|
| SLM-1 cognitive layer (Qwen3-1.7B, ~500 ms, ~1 GB) | GRU arrival predictor + segmentation context + fused priority score | An LLM is slow, large and unverifiable in a safety loop. Learned arrival models beat K0 by +0.32 AUROC |
| K0 kinetic score (`severity · v² / d`) as the ranking | K0's **severity** term kept as "how bad"; **arrival** comes from the GRU | As an arrival predictor K0 scores AUROC 0.57, barely above chance |
| Routing by K0 thresholds (ignore / cognitive / reflex) | TTC override + one fused priority ranking | One ranking replaces two lanes that both ranked the same objects |
| Physics Verification "judge" (SLM vs physics, PPO reward) | **Safety envelope**: keeps the TTC override rule; SLM arbitration and RL reward retired | Nothing left to arbitrate once there is no SLM |
| Narrator SLM-2 (Phi-4-mini) | Scripted templates + pre-rendered audio clips | Deterministic, zero latency, easy to translate |
| "Sensor fusion" box (never implemented) | Per-track Kalman filter on the ground plane, head motion removed using camera pose | Head turns currently look like object motion to every model |
| Depth-column obstacle sweep (no ground removal) | Same sweep, with walkable ground masked out by segmentation | 99.7% of encounter labels were the ground ahead of the walker |
| Jetson Orin Nano 8GB target (4× analytical proxy) | iPhone 17 Pro Max; models measured on the phone itself | The phone is the actual target, and measuring it costs nothing |

## 4. Simulation vs. deployment

All results come from replaying SANPO-Real sessions, and the demo replays pre-recorded video that
already carries depth. We **do not** run a depth model and **do not** run a live instance. **Depth is
SANPO's own depth, used as is, everywhere**: no engine, label or replay degrades, truncates or
replaces it. The phone's own LiDAR is not used. The product's depth comes from an **external LiDAR
sensor**, which is not modelled yet. The right-hand column states how each input would be obtained
in the product; none of it is implemented.

| Input | In the simulation (what we run) | In the product (stated, not built) |
|---|---|---|
| RGB video | SANPO `camera_head/left` frames, 2208×1242 | Phone camera (AVFoundation) |
| Metric depth | SANPO `depth_maps/` (dataset stereo depth) | External LiDAR sensor (model and interface not decided) |
| Ego-motion | SANPO `fixed_camera_poses.csv` (implemented; object-level reconciliation inconclusive, §9) | Not decided (phone IMU / Core Motion, or the sensor's own pose) |
| Camera intrinsics | SANPO `description.json`, per session: 15 FPS, fx 1910–1993 px (51–60° HFOV) | Phone camera intrinsics |
| Model execution | PyTorch on the development host | Core ML on the Apple Neural Engine; **latency measured on the device** (§7) |
| Audio | Synthesised bytes, logged | Pre-rendered clips played locally |

**Statement for the paper.** Depth is SANPO's recorded depth, used as is. How the external LiDAR's
range and noise compare with it is unknown until that sensor is chosen, and is stated as a
limitation, not simulated.
It can be bounded by re-running the evaluation with SANPO depth truncated at 5 m and downsampled
(optional experiment, `pending_work.md`).

## 5. Architecture

```text
 SANPO replay (sim)  /  iPhone sensors (deployment)
 ┌───────────────────────────────────────────────────────────────┐
 │ RGB frames  ·  metric depth  ·  camera pose / IMU  ·  intrinsics│
 └──────┬───────────────┬──────────────────┬─────────────────────┘
        │               │                  │
        ▼               │                  │
 ┌──────────────┐       │                  │
 │ 1 PERCEPTION │ ✅    │                  │
 │ YOLO26n v3   │       │                  │
 │ + ByteTrack  │◄──────┘ depth sampled    │
 └──────┬───────┘   in each box            │
        ▼                                  ▼
 ┌──────────────────────────────────────────────┐      ┌──────────────────────┐
 │ 2 SENSOR FUSION & TRACKING  ✅               │      │ 3 SCENE CONTEXT  ⬜   │
 │ Kalman filter per track, ground-plane (x, z) │◄─────│ walkable-surface     │
 │ head motion removed using camera pose        │ mask │ segmentation, 2–5 Hz │
 │ → position, velocity, uncertainty            │      │ road / curb / sidewalk│
 └──────┬───────────────────────────────────────┘      └──────────┬───────────┘
        ▼                                                         │
 ┌───────────────────────────────────────────────────────────────▼──────────┐
 │ 4 THREAT ENGINE                                                           │
 │  SAFETY ENVELOPE ✅  TTC ≤ 1 s in the walking cone → OVERRIDE (bypasses all)│
 │  WHEN     ✅  GRU arrival predictor → P(arrives ≤ 4 s), expected time       │
 │  HOW BAD  ✅  class severity (mass law, physics.py), compressed by γ        │
 │  WHAT     ⬜  context: on my path? on the road? near the curb? (interface ✅)│
 │  FUSION   ✅  priority = P(arrive) × severity^γ × context → threat ranking   │
 └──────┬────────────────────────────────────────────────────────────────────┘
        ▼
 ┌──────────────────────────────┐    ┌──────────────────────────────────────┐
 │ 5 ANNOUNCEMENT MANAGER  ✅    │───►│ 6 NARRATION & AUDIO                   │
 │ cooldown, hysteresis,        │    │ scripted templates ✅                 │
 │ override preempts, drop stale│    │ clip list ✅ · rendered clips ⬜       │
 └──────────────────────────────┘    │ critical lane: English + cached clip ✅│
                                     └──────────────────────────────────────┘
```

## 6. Components

### 6.1 Perception ✅

- **Model.** YOLO26n v3, fine-tuned to a 17-class hazard taxonomy: mAP50 0.641, mAP50-95 0.459.
- **Tracking.** ByteTrack assigns persistent track IDs.
- **Depth.** Sampled from the central patch of each box, masking out closer objects that overlap it.
- **Depth checks.** Detections are re-scored for size and plausibility against depth.
- **Unlabelled obstacles.** A depth sweep of the lower frame catches obstacles YOLO missed.

Code: `src/perception_stack/`. Training and data: `yolo_training.md`.

v2 change: the obstacle sweep will ignore pixels that segmentation labels as walkable ground (§6.3).
This removes the "ground ahead" pseudo-objects that dominated the encounter labels.

### 6.2 Sensor fusion & tracking ✅ (noise fitted on train; ego-motion on SANPO unvalidated)

| | |
|---|---|
| Input | Per track: depth and box bearing; per frame: camera pose (SANPO; the product's pose source is not decided) |
| Method | Constant-velocity Kalman filter per track in ground-plane coordinates (x sideways, z forward); before each update, rotate tracks by the camera's own motion |
| Output | Smoothed position, velocity (signed, so retreat is visible) and an uncertainty, per track |
| Replaces | Least-squares closing speed over 5 depth samples (`physics.compute_velocity`); the unimplemented `scene_stable` flag |
| Why | Head turns currently look like every object moving sideways. Velocity is differentiated depth and is noisy. The fused priority can down-weight uncertain tracks |
| Cost | Under 1 ms per frame |
| Code | `src/threat_engine/fusion.py` (`GroundTracker`, `EgoPose`, `sanpo_session_poses`); tests in `tests/test_threat_engine.py` (head turn ≠ object motion, retreat is negative, coasting through missing depth, outlier gating) |
| Noise model | Fitted on official-train sessions by one-step predictive likelihood (`tools/fit_kalman_noise.py`, methodology §4.13, refitted on correct depth §4.16): Student-t update (ν = 4), depth σ = 0.058 + 0.053·d m, acceleration σ 2.93 m/s² (static classes 2.04), bearing σ 0.37°. `NoiseModel.declared()` keeps the 2026-10-14 hand-set values |
| Uncertainty out | `FusedTrack.rel_cov` (4×4 relative position/velocity covariance) feeds the Kalman arrival predictor (`kalman_arrival.py`: P(enter the corridor) over 128 samples) |
| Status | Built, fitted and unit-tested. On correct depth the fit does **not** help: ladder AUROC −0.016 [−0.023, −0.004] vs. the declared filter, replay unchanged, 1 s forecast error higher (methodology §4.16). A multi-step fitting objective is the next try. Ego-motion removal stays an ablation (§9) |

### 6.3 Scene context: walkable-surface segmentation ⬜

- **Model.** A small segmentation network (SegFormer-B0 class or a YOLO segmentation variant).
- **Rate.** Runs asynchronously at 2–5 Hz, because ground layout changes slowly.
- **Training data.** SANPO's own labels (`labelmap.json`): `road`, `curb`, `sidewalk`,
  `crosswalk`, `stairs`, `paved trail`, `other walkable surface`, `inaccessible surface`. No new
  annotation is needed.

| Output | Level | Used by |
|---|---|---|
| surface under the object (road / sidewalk / crosswalk / other) | per track | fusion context, GRU input |
| in walking path (inside the walkable corridor ahead) | per track | fusion context |
| near a boundary (close to curb or road edge) | per track | fusion context (about to cross) |
| distance to the nearest walkable edge ahead (curb, stairs, drop-off) | per scene | its own scripted alert |
| walkable-ground mask | per frame | obstacle sweep (§6.1) |

### 6.4 Threat engine

**Safety envelope ✅.** If an object's time-to-collision is ≤ 1.0 s **and** it is inside the forward
walking cone (±30°), the result is an immediate override alert. No learned component can suppress
or delay it. It runs on fused (Kalman-smoothed, signed) velocity from §6.2 and only on confirmed
tracks; a track may still fire for 2 predicted frames while its depth is missing. Code:
`src/threat_engine/envelope.py`; the edge cases (threshold ties, zero/negative closing speed,
missing depth, unconfirmed tracks) are unit-tested.

**When ✅.** The GRU arrival predictor (`models/GRU/`) reads each track's last 8 processed frames:
- distance, closing speed and bearing, plus their per-step changes;
- detection confidence, box size and track age;
- class.

It outputs a distribution over arrival-time slices, which gives P(arrives within the 4 s horizon)
and an expected arrival time. K0 is deliberately **not** an input. At runtime,
`GRUArrivalPredictor` (`src/threat_engine/priority.py`) encodes each track with the same function
the dataset uses (`dataset.GRU.samples.history_features`; parity is unit-tested). Without a
checkpoint, `PhysicsArrivalPredictor` (time until the fused relative path enters the zone inside the
cone) takes its place. The plain GRU is used, not the stacked ensemble: held-out, the stack adds
nothing significant and needs both GRUs.

**How bad ✅.** Class severity from `physics.py`, derived from real-world mass:
`behaviour(c) · (mass(c)/70 kg)^0.5`, with explicit weights for trip hazards (stairs, pothole, puddle,
crosswalk). λ = 0.5 and the behaviour multipliers are declared design choices, not measured results.

**What ⬜ (interface ✅).** A context multiplier computed from §6.3: ×1.5 "in my path", ×1.25 "near a
boundary", ×0.5 "on another surface" (a car on the road while I'm on the sidewalk). The
`SceneContext` interface and multipliers exist; the segmentation model that fills them doesn't.

**Fusion ✅.**

```text
priority(object) = P(arrives within 4 s) × severity(class)^γ × context(surface, in path, boundary)
ties broken by expected arrival time (sooner first)
below the ignore threshold → not announced
```

γ = 0.25 (declared) keeps "how bad" from swamping "when": a bus counts 1.9× a person instead of
13×. The ignore threshold is chosen on SANPO's official **validation** sessions by
`tools/replay_threat_engine.py`, then applied once to test. This is "Level 2" in
`models/Ensemble/README.md`.

### 6.5 Announcement manager ✅

Decides **whether** to speak, separately from **what** is most dangerous:
- **Cooldown per track:** don't repeat the same object within N seconds unless its priority rises
  sharply.
- **Hysteresis:** switch the announced object only if the new one beats it by a margin. This is
  where the GRU's lower flicker (0.59 vs. 0.73) pays off.
- **Override preempts:** a safety-envelope alert interrupts any narration in progress.
- **Drop stale:** a message that can't play within its time budget is discarded, not queued.

Code: `src/threat_engine/announcer.py` (cooldown 4 s, escalation ×1.5, switch margin ×1.25,
override repeat 2 s, stale after 0.6 s; declared values). `src/threat_engine/engine.py` composes
§6.2–6.5 per frame, and `src/pipeline/orchestrator.V2Pipeline` adds narration.

### 6.6 Narration & audio

| Part | Status | Notes |
|---|---|---|
| Scripted templates (`src/narration/templates.py`) | ✅ | "Motorcycle fast from your left." Deterministic, ~0 ms |
| Critical lane: override → English + cached clip only, never translation or model speech | ✅ | Enforced in code (`src/narration/pipeline.py`) |
| Pre-rendered clip library | list ✅ · audio ⬜ | `templates.all_phrases()` enumerates every phrase (classes × 5 directions × 7 forms); rendering it **offline** with a high-quality voice is pending. Zero runtime speech synthesis on the phone |
| Distance buckets ("close", "5 meters", "10 meters", "20 meters", "far") | ✅ | Keeps the clip set finite |
| Segmentation phrases ("Curb ahead.", "Crosswalk ahead.") | ⬜ | Needs §6.3 |
| Translation | ✅ phrase table · optional | Pre-translated phrases; no runtime translation model |
| Spatial audio (pan the clip toward the object) | ⬜ optional | Free on iOS |

## 7. Latency budget (iPhone 17 Pro Max)

| Path | Rate | Budget | Evidence today |
|---|---|---|---|
| Perception → safety envelope → cached clip | every processed frame | **p95 < 50 ms** | GB10 native p95 7.83 ms (measured). The legacy Jetson analytical proxy gave 34.31 ms (simulated, retired). **Not yet measured on the iPhone** |
| Sensor fusion (Kalman) | every frame | < 1 ms | Estimate (a few matrix operations per track) |
| GRU + fusion score | every frame | < 2 ms | Estimate (~30–65k parameters) |
| Segmentation | 2–5 Hz, async | must not delay perception | Not measured |
| Narration | per announcement | ~0 ms (template + pre-rendered clip) | Template measured at ~0 ms |

On-device numbers come from `ios/CPETiming`. This SwiftUI timing app runs YOLO26n v3 and the
arrival GRU (both FP16 Core ML) on bundled SANPO frames and recorded SANPO features; the phone's own
camera and LiDAR aren't used. It's built by
GitHub Actions on macOS and installed through TestFlight, so no team Mac is needed. The replay then
delays each alert by the measured latency (`--latency-csv`, "latency-aware replay"), so the reported
timeline matches the device. Status: the harness, the GRU export and the replay option are built.
The phone run hasn't happened yet: it needs an Apple Developer account and the phone
(`ios/CPETiming/README.md`).

## 8. Evaluation design

- **Data.** SANPO-Real, 462 curated sessions (`simulation/datasets/sanpo/valid_streams.json`),
  on **SANPO's official session split**: 374 train / 88 test
  (`simulation/datasets/sanpo/official_split.json`); validation is carved from train.
  - Everything is tuned on train only: GRU, segmentation, fusion weights, thresholds and
    announcement parameters.
  - The test set is scored once.
- **Open-loop.** The recorded walker never reacts to our alerts. We can measure whether a warning
  *would have* come in time, not whether anyone avoided anything.
- **Labels without humans.** An object "arrives" if it later comes within the hazard distance inside
  the forward cone, taken from measured future depth.

| Level | Metrics | Baselines |
|---|---|---|
| Arrival ranking | AUROC, arrival_order, soonest_top1, flicker rate (session-bootstrap 95% CIs) | K0, TTC, cone-restricted TTC, nearest object, nearest-in-cone |
| System (replay) | warning lead time before arrival, missed arrivals, alerts per minute, announcement switches per minute, all with on-device latency added | K0-routed v1 pipeline |
| Announcement order | blinded referee win rate on frames where v2 and a baseline disagree; 3 local VLM referees calibrated against a human-labelled subset (Cohen's κ) | K0, nearest-in-cone |
| Ablations | segmentation context on/off; ego-motion removal on/off; history length 8/16/20 | the full v2 engine |

Statistical procedure, metric definitions and referee protocol: `methodology.md`.

**Results so far (official test split).**
- Arrival ranking: `evaluation/benchmarks/heldout_official_split/official462_2026_10_14/report.md`.
- System replay: `evaluation/benchmarks/threat_engine_replay/official_2026_10_14/report.md`.
- Both are summarised with numbers in `progress.md` §6–7.

## 9. Known data issues

Resolved 2026-10-14 (`tools/audit_sanpo_sessions.py`, all 462 sessions):

1. **Frame rate ✅.** Every session is 15 FPS; the existing CSVs assumed 30. Stored speeds are exactly
   2× too high and one processed step is 0.2 s, so the 20-step horizon is **4 s** and the 8-step
   history **1.6 s**. Rankings and step-based labels are unaffected; Stage 1 now reads the real fps.
2. **Field of view ✅.** HFOV is 51–60° per session, not 70°. Stage 1 now uses the session's fx for a
   pinhole bearing; the replay converts old rows. One consequence: with the real FOV, a ±30° cone
   covers almost the whole image, so the arrival label's cone barely restricts anything. In effect
   the label is "comes within 3 m while in view".
3. **Train/test split ✅.** SANPO ships one (`sanpo-real/splits/`): 374 / 88 of our sessions.

Open:

0. **SANPO depth was decoded wrong in Stage 1 (found 2026-10-17; loader fixed; everything rebuilt
   2026-10-18, methodology §4.16).**
   Each `depth_maps/*.float16.gz` file is a 2-value header [720, 1280] followed by a 720×1280 image.
   The loader read the header as padding and reshaped the data as 960×960. That scrambles each map into
   three shifted, interleaved strips: vertical structure roughly survives, but a box samples the wrong
   part of the scene. Example: a person 1.3 m away read 6.4 m; another at 6 m read 21 m.
   - **Affected:** every Stage-1 CSV, the GRU datasets (inputs *and* arrival labels), the trained GRUs,
     the fitted Kalman noise (its "0.7 m box-depth noise" is mostly this), and every ladder and replay
     result. The comparisons are internally consistent but measure scrambled depth.
   - **Second bug, also fixed:** the depth-sweep pseudo-boxes (`obs_*`, whole lower-frame columns)
     were used as occluders and masked out real objects in front of them. The same person then read
     18.8 m.
   - **Fix:** `depth_loader.decode_sanpo_depth` (tested) and the occluder filter in
     `pipeline.py`. Rebuilding is `pending_work.md` P0.0.

4. **Ground pseudo-objects.** The obstacle sweep has no ground removal, so 99.7% of the original
   encounter labels were ground. Results use tracked objects only. Segmentation masking (§6.3) is
   the real fix; 53 sessions have SANPO masks.
5. **Camera poses vs. box depth.** SANPO's world up axis differs by session and visual tracking drops
   out in many sessions; both are handled. Still, neither pose file makes static objects stationary
   in the world frame, so ego-motion removal is unvalidated on SANPO (an ablation in the replay).
6. **Labels and inputs share one depth source.** "Predicts arrival" means "predicts what the depth
   will measure". Decision (2026-10-17): SANPO's `depth_maps/` are used as is everywhere; the
   `zed_depth_maps/` split below is not used. (Was: optionally use SANPO's real-time `zed_depth_maps/` as the model input and the
   offline `depth_maps/` for labels.

## 10. Limitations to state in the paper

- **Replay, not deployment.** Evaluation is open-loop. No user study was run, and no live device
  was used.
- **Depth.** All results use SANPO's recorded depth as is (§4). The product's external LiDAR is not
  modelled; its range and noise relative to SANPO depth are unknown until the sensor is chosen.
- **Lead time.** Even on SANPO depth, at the alert budgets at most 0.19 of arrivals are warned at
  least 0.2 s ahead, and ≤ 0.07 at least 1 s ahead. The headline "warned" counts announcements on
  the arrival frame itself (methodology §4.15–4.16).
- **Alert budgets.** On correct depth the engines speak ~20 times a minute at the chosen points,
  and the validation-chosen `budget_5` point gives 8.5 false alerts/min on test (methodology §4.16).
- **Declared choices.** λ and the behaviour multipliers are declared design choices.
- **Camera position.** The camera is head-mounted (SANPO). A phone would more likely be worn on
  the chest; SANPO's `camera_chest/` stream exists but is unused.

## 11. Code map: current modules in v2

| Module | v2 role |
|---|---|
| `src/perception_stack/` | Perception (§6.1). Bearing and the size check use the session's real fx; v2 velocity comes from the Kalman tracker |
| `src/threat_engine/` | **v2 engine:** fusion (§6.2), safety envelope and fused priority (§6.4), announcement manager (§6.5) |
| `src/threat_prioritizer/`, `src/cognitive_layer/stub.py`, `src/physics_verification/`, `src/reflex_layer/` | The **v1 chain** (K0 routing → stub → judge), kept as the replay baseline. `NarratorEvent` (physics_verification) is still narration's input type |
| `src/pipeline/orchestrator.py` | `V2Pipeline` (v2 engine + narration) and `process_frame` (v1 baseline) |
| `src/narration/` | Narration (§6.6). Keep; add the clip library |
| `models/GRU/`, `models/Ensemble/`, `dataset/GRU/` | "When" component (§6.4): training, held-out evaluation (`models/Ensemble/heldout.py`), official-split datasets (`dataset/GRU/resplit.py`) |
| `tools/audit_sanpo_sessions.py`, `tools/replay_threat_engine.py` | Data audit (§9) and the end-to-end replay (§8); the replay's `--latency-csv` adds measured on-device latency; depth is always SANPO's, as is |
| `tools/fit_kalman_noise.py` | Fits the fusion noise model (§6.2) on official-train sessions |
| `src/threat_engine/kalman_arrival.py` | Kalman arrival predictor: P(enter the corridor) over the fused track's covariance (replay arm `v2_kalman`, ladder rung `cv_kalman`) |
| `evaluation/kinetic_ablation.py`, `evaluation/vlm_referee.py` | Metric definitions and the referee (protocol v2: numbered boxes, order swap, text-only control, AC1 / α), reused for v2 evaluation |
| `ios/CPETiming/`, `training/scripts/export_gru_coreml.py`, `.github/workflows/ios-timing.yml` | On-device latency harness (§7) |

## 12. Open decisions

1. **Segmentation model:** SegFormer-B0 class vs. a YOLO segmentation variant. Decide by on-iPhone
   latency and mIoU on SANPO masks.
2. **Fusion form:** the fixed product above vs. a small fitted model. The product is easier to
   explain and test.
3. **Prediction horizon, hazard distance and cone.** Currently 20 steps (4 s) and 3 m. With the real
   FOV the ±30° cone covers nearly the whole image; a lateral corridor (|x| < ~1 m on the ground
   plane) would be a more meaningful "in my path" test.
4. **Head vs. chest camera** for the paper's main results.
5. **Announcement trade-off.** In the replay, v2 is precise but quiet: it warns about 26% of arrivals
   at 17 alerts/min. Where to sit on that curve is a product decision; tune it on train/val only.
