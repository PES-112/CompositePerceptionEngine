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

All results come from replaying SANPO-Real sessions. We **do not** run a depth model and **do not**
run a live instance. Depth is read from SANPO. The right-hand column states how each input would be
obtained on the phone; none of it is implemented.

| Input | In the simulation (what we run) | On iPhone 17 Pro Max (stated, not built) |
|---|---|---|
| RGB video | SANPO `camera_head/left` frames, 2208×1242 | Rear wide camera (AVFoundation / ARKit) |
| Metric depth | SANPO `depth_maps/` (dataset stereo depth) | LiDAR scene depth via ARKit: low resolution (~256×192), reliable to ~5 m |
| Ego-motion | SANPO `camera_poses.csv` | ARKit camera pose / Core Motion gyroscope |
| Camera intrinsics | SANPO `description.json` (e.g. 58.4° horizontal FOV) | ARKit camera intrinsics, per frame |
| Model execution | PyTorch on the development host | Core ML on the Apple Neural Engine; **latency measured on the device** (§7) |
| Audio | Synthesised bytes, logged | Pre-rendered clips played locally |

**Statement for the paper.** Depth comes from the dataset, so results are an upper bound on what
phone depth would allow. The LiDAR's shorter range (~5 m vs. SANPO's tens of metres) is the main gap.
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
 │ 2 SENSOR FUSION & TRACKING  ⬜               │      │ 3 SCENE CONTEXT  ⬜   │
 │ Kalman filter per track, ground-plane (x, z) │◄─────│ walkable-surface     │
 │ head motion removed using camera pose        │ mask │ segmentation, 2–5 Hz │
 │ → position, velocity, uncertainty            │      │ road / curb / sidewalk│
 └──────┬───────────────────────────────────────┘      └──────────┬───────────┘
        ▼                                                         │
 ┌───────────────────────────────────────────────────────────────▼──────────┐
 │ 4 THREAT ENGINE                                                           │
 │  SAFETY ENVELOPE ✅  TTC ≤ 1 s in the walking path → OVERRIDE (bypasses all)│
 │  WHEN     🟡  GRU arrival predictor → P(arrives ≤ horizon), expected time   │
 │  HOW BAD  ✅  class severity (mass law, physics.py)                         │
 │  WHAT     ⬜  context: on my path? on the road? near the curb?               │
 │  FUSION   ⬜  priority = P(arrive) × severity × context → threat ranking     │
 └──────┬────────────────────────────────────────────────────────────────────┘
        ▼
 ┌──────────────────────────────┐    ┌──────────────────────────────────────┐
 │ 5 ANNOUNCEMENT MANAGER  ⬜    │───►│ 6 NARRATION & AUDIO                   │
 │ cooldown, hysteresis,        │    │ scripted templates ✅                 │
 │ override preempts, drop stale│    │ pre-rendered clip library ⬜          │
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

### 6.2 Sensor fusion & tracking ⬜

| | |
|---|---|
| Input | Per track: depth and box bearing; per frame: camera pose (SANPO) or ARKit pose / gyroscope (phone) |
| Method | Constant-velocity Kalman filter per track in ground-plane coordinates (x sideways, z forward); before each update, rotate tracks by the camera's own motion |
| Output | Smoothed position, velocity (signed, so retreat is visible) and an uncertainty, per track |
| Replaces | Least-squares closing speed over 5 depth samples (`physics.compute_velocity`); the unimplemented `scene_stable` flag |
| Why | Head turns currently look like every object moving sideways. Velocity is differentiated depth and is noisy. The fused priority can down-weight uncertain tracks |
| Cost | Under 1 ms per frame |

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

**Safety envelope ✅ (rule exists; cone restriction ⬜).** If an object's time-to-collision is
≤ 1.0 s **and** it is inside the forward walking cone, the result is an immediate override alert. No
learned component can suppress or delay it. Today's rule (`src/physics_verification/`,
`src/threat_prioritizer/events.py`) does not yet check the cone, and its velocity is unsmoothed.
v2 adds both, using the fused tracks from §6.2.

**When 🟡.** The GRU arrival predictor (`models/GRU/`) reads each track's last 8 processed frames:
- distance, closing speed and bearing, plus their per-step changes;
- detection confidence, box size and track age;
- class.

It outputs a distribution over arrival-time slices, which gives P(arrives within the horizon) and an
expected arrival time. K0 is deliberately **not** an input. The stacked ensemble (`models/Ensemble/`)
combines both GRUs with the nearest-in-cone and TTC rules.

**How bad ✅.** Class severity from `physics.py`, derived from real-world mass:
`behaviour(c) · (mass(c)/70 kg)^0.5`, with explicit weights for trip hazards (stairs, pothole, puddle,
crosswalk). λ = 0.5 and the behaviour multipliers are declared design choices, not measured results.

**What ⬜.** A context multiplier computed from §6.3. It goes up for "in my path" and "crossing the
boundary"; it goes down for "on the road while I'm on the sidewalk".

**Fusion ⬜.**

```text
priority(object) = P(arrives within horizon) × severity(class) × context(surface, in path, boundary)
ties broken by expected arrival time (sooner first)
below a priority threshold → ignored
```

This is "Level 2" in `models/Ensemble/README.md`. The fusion weights are fitted or tuned on SANPO
**training** sessions only.

### 6.5 Announcement manager ⬜

Decides **whether** to speak, separately from **what** is most dangerous:
- **Cooldown per track:** don't repeat the same object within N seconds unless its priority rises
  sharply.
- **Hysteresis:** switch the announced object only if the new one beats it by a margin. This is
  where the GRU's lower flicker (0.59 vs. 0.73) pays off.
- **Override preempts:** a safety-envelope alert interrupts any narration in progress.
- **Drop stale:** a message that can't play within its time budget is discarded, not queued.

### 6.6 Narration & audio

| Part | Status | Notes |
|---|---|---|
| Scripted templates (`src/narration/templates.py`) | ✅ | "Motorcycle fast from your left." Deterministic, ~0 ms |
| Critical lane: override → English + cached clip only, never translation or model speech | ✅ | Enforced in code (`src/narration/pipeline.py`) |
| Pre-rendered clip library | ⬜ | About 17 classes × 5 directions × 3–4 urgency levels per language, rendered **offline** with a high-quality voice. Zero runtime speech synthesis on the phone |
| Distance buckets ("close", "2 meters", "5 meters") | ⬜ | Keeps the clip set finite |
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

On-device numbers come from exporting each model to Core ML and running it on the iPhone 17 Pro Max,
on SANPO frames. The replay then delays each alert by the measured latency ("latency-aware replay"),
so the reported timeline matches the device. Neither step has been done yet (`pending_work.md`).

## 8. Evaluation design

- **Data.** SANPO-Real, 462 curated sessions (`simulation/datasets/sanpo/valid_streams.json`),
  split **by session** into train and test.
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

## 9. Known data issues (must be resolved before final numbers)

1. **Frame rate.** SANPO-Real reports **15 FPS** (checked on one session; verify all 462).
   - `tools/stream_sanpo_perception.py` defaults to `--fps 30`.
   - If the CSVs were built at 30, every closing speed is 2× too high, and the "2 s" horizon is 4 s.
2. **Field of view.** SANPO reports a **58.4°** horizontal FOV. `compute_bearing()` assumes 70°, so
   every bearing is ~20% too wide. Use per-session intrinsics instead.
3. **Ground pseudo-objects.** The obstacle sweep has no ground removal, so 99.7% of the original
   encounter labels were ground. Current GRU results use tracked objects only. Segmentation masking
   (§6.3) is the real fix.
4. **Labels and inputs share one depth source.** "Predicts arrival" means "predicts what the depth
   will measure". Optionally, use SANPO's real-time `zed_depth_maps/` as the model input and the
   offline `depth_maps/` for labels.

## 10. Limitations to state in the paper

- **Replay, not deployment.** Evaluation is open-loop. No user study was run, and no live device
  was used.
- **Depth.** Depth comes from SANPO, not phone LiDAR (§4). Results are an upper bound for the
  LiDAR's range and resolution.
- **Declared choices.** λ and the behaviour multipliers are declared design choices.
- **Camera position.** The camera is head-mounted (SANPO). A phone would more likely be worn on
  the chest; SANPO's `camera_chest/` stream exists but is unused.

## 11. Code map: current modules in v2

| Module | v2 role |
|---|---|
| `src/perception_stack/` | Perception (§6.1). Keep. Velocity moves to the Kalman tracker; bearing should use real intrinsics |
| `src/threat_prioritizer/events.py` | `ThreatEvent` contract. Keep; routing becomes override + fused priority |
| `src/reflex_layer/reflex.py` | Safety envelope path. Keep |
| `src/physics_verification/physics_verification.py` | Rule 1 (override) becomes the safety envelope. Rules 2–3 and the PPO reward constants are SLM-only; retire when the fused ranker lands |
| `src/cognitive_layer/stub.py` | Placeholder. **Replace** with the fused ranker; `src/pipeline/orchestrator.py` call site stays |
| `src/pipeline/orchestrator.py` | Per-frame composition. Keep |
| `src/narration/` | Narration (§6.6). Keep; add the clip library |
| `src/shared/fact_sheet.py`, `src/perception_stack/fact_sheet_builder.py`, `tools/run_fact_sheets.py` | SLM-1 prompt format; no v2 consumer. Retire once nothing imports them |
| `models/GRU/`, `models/Ensemble/`, `dataset/GRU/` | "When" component (§6.4) |
| `evaluation/kinetic_ablation.py`, `evaluation/vlm_referee.py` | Metric definitions and the referee, reused for v2 evaluation |

## 12. Open decisions

1. **Segmentation model:** SegFormer-B0 class vs. a YOLO segmentation variant. Decide by on-iPhone
   latency and mIoU on SANPO masks.
2. **Fusion form:** the fixed product above vs. a small fitted model. The product is easier to
   explain and test.
3. **Prediction horizon and hazard distance** after the FPS fix. Currently 20 frames and 3 m for
   tracked objects.
4. **Head vs. chest camera** for the paper's main results.
