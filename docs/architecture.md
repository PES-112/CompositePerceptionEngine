# Composite Perception Engine Architecture

> [!NOTE]
> Corrected to match the diagram provided. Physics Verification is **downstream of SLM-1** — it judges SLM-1's semantic output against the Reflex Layer's raw kinetic score, then selects events to narrate.

---

## 1. Corrected System Architecture

```
┌──────────────────────────────────────┐
│             INPUT LAYER              │
│  Primary Camera                      │
│  Egocentric Depth Data (from dataset)│
│  Gyroscope / 360 Camera              │
└──────────────┬───────────────────────┘
               │
               ▼
       ┌───────────────┐
       │  Sensor Fusion │
       └───────┬────────┘
               │
               ▼
  ┌────────────────────────────┐
  │      Perception Stack      │
  │  YOLO26 Nano + Tracking    │
  │  + Egocentric Depth Map    │
  └────────────┬───────────────┘
               │
               ▼
       ┌───────────────┐
       │    Threat      │
       │  Prioritizer   │  ← Computes raw TTC + Kinetic Scores
       └──┬─────────┬──┘
          │         │
       Low Risk   High Risk / Contextual
          │         │
          ▼         ├──────────────────────────┐
  ┌───────────┐     │                          │
  │  Ignore   │     ▼                          ▼
  │ Objects   │ ┌──────────────┐   ┌─────────────────────────┐
  └───────────┘ │ Reflex Layer │   │     Cognitive Layer      │
                │ Deterministic│   │ SLM-1 (Qwen3/Qwen2.5) │
                │ Physics      │   │   Semantic Evaluation    │
                │ TTC < 1.0s   │   │   of Scene Context       │
                └──────┬───────┘   └────────────┬────────────┘
                       │                        │
                OVERRIDE SIGNAL          Normal Flow
                       │                        │
                       └──────────┬─────────────┘
                                  ▼
                     ┌────────────────────────┐
                     │   Physics Verification  │
                     │      (The Judge)        │
                     │                         │
                     │  Compares:              │
                     │  SLM-1 semantic eval    │
                     │  vs Raw Kinetic Score   │
                     │                         │
                     │  → Selects NarratorEvent│
                     └────────────┬────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  Narrator SLM-2  │
                        └────────┬────────┘
                                 │
                         ┌───────┴────────┐
                         │   (Optional)   │
                         ▼               ▼
              ┌──────────────────┐  ┌───────────────┐
              │ Indic Language   │  │  Audio Output │
              │ Translation      │  │ Cached/local │
              └──────────┬───────┘  └───────┬───────┘
                         └──────────────────┘
                                  ▲
                     ┌────────────┘
                     │
              ┌──────────────┐
              │   System     │
              │  Heartbeat   │  ← Periodic ambient updates
              └──────────────┘
```

**Key correction:** Physics Verification is *downstream* of SLM-1. It arbitrates between two independent intelligence streams.

---

## 2. Component Validation

### 2.1 Input Layer & Dataset Strategy

The system relies on three primary datasets for simulation and training:

| Dataset | Role in CPE |
|---|---|
| **SANPO / KITTI** | Provides **ground-truth egocentric depth**. The simulation extracts distance `d` directly from the dataset rather than predicting it via a neural network. This ensures the Physics Verification layer has perfect deterministic grounding. |
| **UASOL** | Provides "chaos" real-world unstructured sidewalk footage to stress-test the optical flow stability checks. |
| **HEADSUP** | Provides pedestrian intent labels (e.g., "looking at phone", "distracted") which feed into the cognitive layer's semantic context engine. |

> [!NOTE]
> Training relies heavily on the depth extraction from datasets. The depth `d` is used to calculate the raw Kinetic Score, which serves as the **unbiased reward signal teacher** for the SLM-1 agent during PPO training.

---

### 2.2 Perception Stack

YOLO26 Nano + ByteTrack + Depth Map overlay. Outputs per object:
- `track_id`, `class`, `bbox`
- Metric distance `d` (**extracted from dataset depth map at bbox centroid**)
- Velocity `v = Δd / Δt` across frames

**Edge Optimization (YOLO Fine-Tuning):**
The YOLO26n model is fine-tuned on a domain-specific hazard dataset (potholes, bollards, stairs, crosswalks, poles, puddles, dogs, and benches) rather than standard COCO classes. This provides critical semantic labels for navigation hazards at the edge.

The fine-tuning entry point is `training/scripts/train_yolo26n_hazards.py`, backed by `training/configs/cpe_hazard_classes.yaml` and documented in `docs/yolo_training.md`. It keeps the YOLO26n nano backbone as the student model, reduces the detector head from COCO's broad 80-class taxonomy to 17 CPE navigation classes, freezes early layers by default, and supports deferred edge export via `training/scripts/export_yolo26n_edge.py` using modern Ultralytics `quantize=16` or `quantize=8` arguments. This retains navigation-relevant COCO classes such as `dog` and `bench`, and adds non-COCO hazards such as `pole`, `bollard`, `stairs`, `crosswalk`, `pothole`, and `puddle` without moving to a larger model family.

The current preferred detector is the v3-from-base checkpoint. v2 continued from the prior CPE checkpoint and improved several custom hazards but caused retained-class degradation, especially `truck`; v3 restarted from base `models/yolo/base_yolo26n/yolo26n.pt`, reused the cleaned 17-class dataset, and repaired most retention issues while improving mAP50-95. Evaluation now includes all-class held-out mAP plus retained COCO-class comparison against base YOLO using class-name remapping to avoid COCO-ID/CPE-ID mismatches.

**Compute Reduction (N-Frame Skip):**
To achieve 30+ FPS on edge devices, the system employs an N-frame skip (e.g., every 3rd frame). Between detections, the system relies on depth-guided ByteTrack interpolation, yielding a ~67% compute reduction without sacrificing safety.

Real-time readiness should be validated with `tools/benchmark_edge_realtime.py`, which measures streaming YOLO26n + ByteTrack latency, optional depth loading, depth post-processing, and effective FPS under the same `frame_step` setting used by the perception loop. The current v3 detector-only GB10 benchmark has p95 total latency of `9.65 ms` per processed frame at `frame_step=3`, below both the `100 ms` processed-frame budget for 30 FPS input and the `<50 ms` reflex-path budget for detector/tracker work. A SANPO-Real smoke test on valid stream `0xCqEk5hjEvrygxu26MZkieSv45D_gaJ` measured p95 `9.04 ms` in preloaded RGB+depth mode, while raw dataset replay measured p95 `51.54 ms` because PNG disk read/decode dominated; live edge testing should use sensor/in-memory frames or preloaded mode rather than treating compressed-dataset I/O as detector latency.

**Error Analysis & Dataset Gap Analysis Workflow:**
To identify blind spots in the YOLO detection stack and prioritize fine-tuning efforts, the engine includes a dedicated gap analysis workflow (`notebooks/sanpo_yolo_gap_analysis.ipynb`) and a batch evaluation utility (`tools/gap_analysis_experiments.py`):
- **Depth-only Candidate Region Detection:** It identifies regions in the depth map within customizable navigation ranges that have no corresponding YOLO detections.
- **Whitelist-agnostic Analysis:** The system can evaluate gaps across all 80 standard COCO classes (rather than a constrained subset) to check for general perception failures.
- **Multi-Range Evaluation:** Supports running parallel comparisons across multiple depth ranges (e.g., Immediate 0.5m-2.0m, Mid-range 0.5m-4.0m, Standard 0.5m-6.0m, and Far-range 2.0m-6.0m) to optimize the system configuration.
- **Hard Example Logging:** Instead of exporting heavy images to disk, it flags challenging frames and saves their metadata (`session_id` and list of `frame_id`s) to range-specific JSON files (`hard_examples.json`) for downstream annotation workflows.
- **Comparison & Recommendations:** Compiles frame stats, candidate regions, and generates a structured comparison report (`distance_metric_comparison.md`) to guide target dataset curation.

---

### 2.3 Threat Prioritizer

Routes objects based on the runtime `ThreatEvent` contract in `src/threat_prioritizer/events.py`. Stage 1 perception rows already contain YOLO class, track ID, confidence, bounding box, depth-derived distance, closing velocity, and bearing. The threat prioritizer enriches each row with TTC and kinetic score, then emits one route per object.

```
Tracked perception row
  → PerceivedObject(distance, velocity, bearing, K, TTC)
  → ThreatEvent(route=ignore|cognitive|reflex, priority, reason)

K = Class_severity × v^γ / max(d,ε)   ← K0; γ and severity under ablation
TTC = d / V_closing, when V_closing > 0

TTC <= 1.0s or K >= HIGH_THRESHOLD → Reflex Layer (<50ms)
K >= LOW_THRESHOLD                  → Cognitive Layer (~500ms)
near static hazard                  → Cognitive Layer
else                                → Ignore
```

Offline SANPO/perception CSVs can be converted into this event stream with `tools/build_threat_events.py`, which writes non-ignore `ThreatEvent` JSONL plus a route/class summary.

> [!NOTE]
> K0 (`sev × v² / max(d,ε)`) is the production kinetic score and the only research-derived
> formula. The former K1–K5 candidates were dummies and have been removed: measurement showed
> K1/K2/K5 are *exactly* rank-identical to 1/TTC within a class (ρ = 1.0000) and K3 is ρ = 0.9998,
> so they were five labels for two behaviours. The earlier claim that "K4 is the leading candidate"
> had no benchmark behind it and has been withdrawn. K0 is to be defended by **ablation** of its own
> terms and by **complementarity with SLM-1**, not by a contest against strawmen — see
> `docs/kinetic_score_opinion.md` for the evaluation strategy and
> `docs/kinetic_score_opinion.md` §10 for the ground-truth decision record.
>
> **Class severity is derived, not hand-tuned (2026-08-19).** `severity(c) = behaviour(c) ×
> (mass(c)/70 kg)^λ` with `λ = 0.5`, because collision consequence scales with delivered energy.
> λ = 0.5 (severity ∝ √mass) keeps a real destructive bias — car 4.6× a person, bus 13×, bus 2.8× a
> car — without λ = 1's 171× runaway, which lets a distant bus outrank an imminent pedestrian.
> λ is **frozen at 0.5** (2026-08-25) and no longer swept: with Tier-B blinded human labelling dropped, nothing in the evaluation plan can adjudicate between λ values, so λ and the behaviour multipliers are declared design choices reported as a limitation, not measured results. A sparse behaviour multiplier covers hazards mass cannot see (erratic motion,
> trip height, head-height mounting), and four massless trip hazards (stairs, pothole, puddle,
> crosswalk) keep explicit weights outside the mass law. Bounding-box size enters `kinetic_score()`
> as an optional `size_exponent` term on apparent size `A_px/d`, **disabled by default**: box area
> is ~99% predictable from class + depth by projective geometry, so it is redundant for labelled
> objects and is carried as an ablation arm rather than a production term.

> [!IMPORTANT]
> Both tracks can fire simultaneously for different objects. A speeding car goes Reflex; a jaywalker goes Cognitive. Physics Verification merges both.

---

### 2.4 Reflex Layer (< 50ms, Deterministic)

The Reflex Layer bridge is implemented in `src/reflex_layer/reflex.py`. It consumes `ThreatFrame` objects, selects only `ThreatEvent(route="reflex")`, converts them to `ReflexResult`, and calls `PhysicsVerification` without waiting for SLM-1.

- Computes TTC precisely for high-K or fast-closing objects
- `IF TTC <= 1.0s` → fires **OVERRIDE SIGNAL** to Physics Verification
- `IF K >= HIGH_THRESHOLD` but TTC is above override threshold → uses physics fallback as a reflex-path warning candidate
- Bypasses SLM-1 entirely for hard real-time behavior
- No neural network in this path
- Covered by `tests/test_reflex_layer.py` for override, non-reflex suppression, and physics fallback behavior

---

### 2.5 Cognitive Layer — SLM-1

**Inputs:** Symbolic Fact Sheet (structured JSON) from the Threat Prioritizer

The Fact Sheet is the sole runtime input to SLM-1. All fields are derivable from a phone camera at inference time:

```json
{
  "frame_id": 142,
  "scene_stable": true,
  "objects": [
    {
      "track_id":      "t_07",
      "object_class":  "car",
      "bearing":       "ahead",
      "bearing_deg":   2.1,
      "distance_m":    8.5,
      "velocity_ms":   4.2,
      "ttc_s":         2.02,
      "kinetic_score": 8.74,
      "route":         "cognitive"
    }
  ]
}
```

**Excluded from runtime FactSheet (training-data only):**
- `intent_label` — from HEADSUP dataset; not available from phone camera at inference. Used only to enrich SFT training examples so the SLM learns to infer intent from scene context.
- `hallucination_filtered` — internal pipeline flag; not a semantic input.

**Bearing derivation:** `bearing_deg = ((cx_px - frame_width/2) / (frame_width/2)) × (hfov_deg/2)`. No extra hardware needed — computed from YOLO bbox centroid pixel alone. Phone HFoV default: 70°.

**Task:** Produce a semantic evaluation of the scene:

```json
{
  "primary_threat_id": "t_07",
  "reason": "Car 8.5m ahead closing at 4.2m/s — highest kinetic risk on direct path.",
  "scene_state": "vehicle_approaching",
  "confidence": 0.92,
  "future_confirmed": true
}
```

SLM-1 reasons about **trajectory, proximity urgency, and semantic context** — not just kinetic score. This is what makes it complementary to the physics layer.

---

### 2.6 Physics Verification — The Judge ✅

Arbitrates SLM-1 semantic vs. Reflex kinetic score. Selects the [NarratorEvent](file:///e:/capstone/code/src/physics_verification/physics_verification.py#39-59).

```
Adjudication rules:
  1. OVERRIDE active (TTC < 1.0s)
       → Bypass SLM-1. Direct alarm to Narrator.

  2. SLM-1 primary_threat == highest-K object
       → High confidence. Send with full context.

  3. SLM-1 diverges from highest-K object
       → Conflict: weight by (K_score × semantic_confidence)
       → Log divergence → RL reward signal.

  4. SLM-1 says safe, K_score is high
       → Hallucination detected. Override with physics.
```

This prevents both SLM hallucinations **and** physics false positives (e.g. fast car in adjacent lane, no actual crossing risk).

---

### 2.7 Narrator SLM — SLM-2

**Input:** [NarratorEvent](file:///e:/capstone/code/src/physics_verification/physics_verification.py#39-59) JSON from Physics Verification
**Output:** Short, clear natural language warning

```
{class: motorcycle, dist: 6m, bearing: left, v: 8m/s}
→ "Motorcycle fast from your left."
```

**Optional:** Indic Language Translation via IndicTrans2 (~+75ms latency, acceptable for narration path).

---

### 2.8 System Heartbeat

Fires every 5–8 seconds when no threat is detected. Feeds Audio Output directly — does NOT pass through Physics Verification. Prevents dead silence in low-risk environments.

---

## 3. RL Reward Loop

Physics Verification grades SLM-1 at every cycle:

```
+100  : SLM-1 primary_threat == highest-K object
+50   : SLM-1 catches semantic threat missed by kinetics
-200  : SLM-1 misses object with K > HIGH_THRESHOLD
-500  : SLM-1 says safe while OVERRIDE is active
```

Physics layer is the teacher. No human annotation needed.

---

## 4. Strengths

| | |
|---|---|
| Depth from dataset | Grounded, error-free physics |
| Threat Prioritizer routing | SLM-1 never processes irrelevant objects |
| Physics-Semantic arbitration | Neither intelligence stream acts alone |
| Override path | Hard safety guarantee, SLM-latency independent |
| Heartbeat | No dead silence; ambient awareness |
| Indic Translation | First-class accessibility feature |

---

## 5. Open Questions

| Question | Recommendation |
|---|---|
| `LOW_THRESHOLD` value? | Start `K < 0.5` (static/far); calibrate from dataset |
| `HIGH_THRESHOLD` value? | `TTC < 1.5s` OR `K > 5.0`; tune via simulation |
| SLM-1 context length? | 2-sec buffer ≈ ~60 frames; keep JSON compact |
| Indic model? | **IndicTrans2** (AI4Bharat) — 200M params, 22 languages, edge-ready |
| Heartbeat interval? | 5–8s when `K_max < LOW_THRESHOLD` across all objects |

---

## 6. Hardware and Deployment Constraints

The canonical training-host specifications, edge-target status, simulation assumptions, and latency budgets are maintained in [`hardware_targets.md`](./hardware_targets.md). The repository currently has measurements from the NVIDIA GB10 training host and an analytical Jetson Orin Nano 8GB proxy; it does not yet have measurements from physical edge hardware.

The architecture therefore treats runtime formats as target-dependent:

| Component | Current development runtime | Intended edge runtime |
|---|---|---|
| YOLO26n v3 | PyTorch CUDA on GB10 | TensorRT or ONNX FP16/INT8 |
| Reflex and Physics Verification | Python reference implementation | Native or optimized local runtime |
| Cognitive SLM-1 | Not integrated | Quantized local runtime after profiling |
| Critical audio | Not integrated | Deterministic templates and cached clips |

---

## 7. SLM Recommendations (Mobile Edge)

| Role | Model | Params | INT4 Size | Why |
|---|---|---|---|---|
| **SLM-1** (Cognitive) | **Qwen3-1.7B non-thinking mode** | 1.7B | INT4 target | Newer default; strict JSON output without thinking-mode latency |
| **SLM-1 fallback** | Qwen2.5-1.5B-Instruct | 1.5B | INT4 target | Use if Qwen3 runtime or quantization is blocked |
| **Narration** | Template-first critical warnings; Phi-4-mini optional | 0-4B | runtime-dependent | Deterministic for safety-critical alerts; richer SLM narration only off the reflex path |
| **Indic Translation** | IndicTrans2 distilled + phrase table | ~200M | ~200MB | Phrase table for critical alerts; model translation for non-critical narration |

**Memory note:** keep only the detector and reflex/audio critical path resident by default; load SLM narration and translation only if the target edge profile has measured headroom.

---

## 8. Extracting RL Rewards from Datasets (PPO Workflow)

The most elegant part of CPE is how it uses datasets to **train SLM-1 without human labeling**. Here is exactly how PPO (Proximal Policy Optimization) applies to your architecture:

### 1. The RL Environment (Simulation)
- **State ($S_t$):** A frame from SANPO/UASOL processed by YOLO + Depth Dataset into a structured JSON Fact Sheet (list of objects, distances, velocities, intents).
- **Action ($A_t$):** SLM-1 outputs a `primary_threat_id` and a `reason`.
- **Teacher (The Judge):** The Python [PhysicsVerification](file:///e:/capstone/code/src/physics_verification/physics_verification.py#68-162) script deterministically calculates the raw Kinetic Score ($K$) for every object using the dataset depth.

### 2. The Reward Signal ($R_t$)
At every frame, the Physics Judge compares the SLM-1 chosen action ($A_t$) against the raw physics reality:
- **+100 Reward:** SLM-1's chosen target perfectly matches the object with the highest $K$ score.
- **-500 Penalty (Fatal):** SLM-1 claims the scene is safe, but the Physics Judge calculates $TTC < 1.0s$ for an approaching vehicle.
- **+50 Semantic Bonus:** SLM-1 picks a pedestrian looking at a phone (from HEADSUP labels) heading into the path, overriding a faster but non-colliding background car.

### 3. PPO Update Step
```python
ppo_trainer.step(
    queries=[fact_sheet_json],    # The scene state
    responses=[slm_output],       # SLM-1's action
    scores=[calculated_reward]    # Awarded by the Physics Judge
)
```
This loop runs thousands of times across the SANPO and UASOL video frames. **SLM-1 gradually learns the laws of physics** (e.g., that high velocity + low distance = danger) simply by trying to maximize the reward given by the deterministic judge.

---

## 9. Offline Training & Knowledge Distillation (Teacher-Student)

To bridge the gap between rich semantic understanding and edge-device constraints (battery, memory, latency), the system uses a **Knowledge Distillation** pipeline for SLM-1.

### 1. Offline Generation (The Teacher)
Massive VLM models (e.g., Grounding DINO, GPT-4V, YOLOv10x) process thousands of street-view images offline. These slow, heavy models generate perfect pseudo-labels, bounding boxes, and rich natural-language descriptions of the scene (e.g., "Deep pothole immediately ahead at foot level, requiring a full stop").

### 2. Edge Fine-Tuning (The Student)
The edge-deployed **SLM-1 (primary: Qwen3-1.7B non-thinking mode; fallback: Qwen2.5-1.5B)** is fine-tuned on this generated dataset. It learns to map the extremely lightweight, edge-generated "Fact Sheet" (YOLO class + Depth + Kinetic Score) to the rich semantic understanding demonstrated by the Teacher model.

This allows the edge system to mimic the reasoning capabilities of a 100B+ parameter model while executing in ~500ms on a mobile NPU.

---

## 10. Minor Architecture Recommendations

To refine the architecture without making major structural changes:

1. **Depth Extrapolation Smoothing:** Since dataset depth points can sometimes be sparse or noisy at bounding box edges, apply a **Kalman Filter** to the `d` values extracted from the dataset before calculating `v = Δd / Δt`. This prevents jitter in the Kinetic Score from causing false reflex overrides.
2. **Dynamic Heartbeat:** Instead of a fixed 5-second interval, let the System Heartbeat scale inversely with scene complexity. (e.g., empty room = 10s, crowded but safe sidewalk = 4s).
3. **Intent-Gating:** Only pass HEADSUP intent labels to SLM-1 for objects within a 15-meter radius. Processing intent for distant background pedestrians wastes SLM token context window.

---

## 11. Technology Audit: Model and Runtime Choices

The rationale behind the model choices in §7's table, plus the tracking-stack and orchestration
decisions. Formerly a standalone `roadmap.md`; merged here since it is elaborating on the same
components this file already specifies. Current implementation status of each component:
`progress.md`; backlog: `pending_work.md`.

### 11.1 Perception: YOLO26n v3 + depth-guided tracking — keep

YOLO26n is the right detector family because the system is edge-latency constrained, not
leaderboard-mAP constrained (§2.2). The v3 checkpoint already passes the simulated Jetson
detector/reflex budget across 10 SANPO valid streams (`hardware_targets.md`). Moving to a larger
detector now would likely increase latency and integration risk before proving the current model is
inadequate. The next optimization should be deployment format, not model size: export v3 to ONNX
FP16/INT8, then TensorRT FP16/INT8 on actual Jetson/NVIDIA hardware, and compare PyTorch vs.
ONNX/TensorRT latency and accuracy on the same SANPO 10-session set. Use Ultralytics `quantize=16` /
`quantize=8` — the legacy `half=True` / `int8=True` flags are deprecated aliases. For INT8, use
representative calibration data from `training/configs/cpe_hazard_classes.yaml` or SANPO-derived
frames.

**Tracking:** the Ultralytics tracking path uses ByteTrack. Keep it for prototype stability; evaluate
TrackTrack or BoT-SORT only after baseline end-to-end metrics exist — switching trackers before an
end-to-end baseline would confound any latency or accuracy comparison.

### 11.2 SLM-1 Cognitive Layer — Qwen3-1.7B non-thinking, Qwen2.5-1.5B fallback

Qwen3-1.7B over Qwen2.5-1.5B as primary: newer, Apache-2.0, multilingual, 32K context window, and
supports explicit thinking/non-thinking modes. For real-time CPE cognition, use **non-thinking mode**
to avoid unnecessary chain-of-thought latency. Qwen2.5-1.5B remains the fallback if Qwen3 runtime
support or quantization causes deployment friction.

```text
SLM-1 primary: Qwen3-1.7B, quantized GGUF/ONNX where practical, non-thinking mode
SLM-1 fallback: Qwen2.5-1.5B-Instruct, INT4/GGUF
Output: strict JSON semantic assessment over the perception fact sheet
Latency budget: soft real-time, <=500 ms
```

Do not let SLM-1 sit in the hard safety path — its output is advisory and must be arbitrated by
Physics Verification (§2.6).

### 11.3 SLM-2 Narration Layer — template-first, Phi-4-mini optional

Phi-3 Mini is acceptable but no longer the best default; prefer Phi-4-mini-instruct for narration if
memory permits, otherwise template-first narration for the prototype. Phi-4-mini-instruct is newer
than Phi-3 Mini with stronger multilingual/reasoning support, but at ~3.8B/4B class it may be too
heavy if SLM-1, translation, and TTS are also resident. The narrator has a constrained job — turn a
verified event into a short warning — where a deterministic template layer may outperform an SLM on
latency, safety, and consistency for the first demo.

> [!NOTE]
> **Phase 1 implemented (2026-08-27):** `src/narration/templates.py` — pure Python, no ML
> dependency, consumes `NarratorEvent` directly and reproduces the worked example above exactly
> (`"Motorcycle fast from your left."`). Phase 2 (Phi-4-mini) is still not started.

```text
SLM-2 phase 1: deterministic/template narrator for critical warnings
SLM-2 phase 2: Phi-4-mini-instruct only for non-critical richer narration
SLM-2 fallback: Phi-3-mini-4k-instruct ONNX/GGUF if Phi-4 runtime is too heavy
```

This keeps the demo safe: critical alerts should be short, deterministic, and immediate.

### 11.4 Translation — IndicTrans2 distilled

Keep IndicTrans2 distilled models for the prototype; revisit IndicTrans3 only after local
availability/runtime is validated. IndicTrans2 supports the 22 scheduled Indic languages with
distilled 200M/320M variants suitable for edge-style testing — the 1B variants may improve quality
but are harder to fit alongside detector + SLMs + TTS. Translation must not block critical reflex
alerts; critical alerts should use pre-translated phrase templates where possible.

```text
Critical path: pre-translated alert phrase table
Non-critical narration: IndicTrans2 distilled En-Indic model
Fallback: English-only output if translation exceeds latency budget
```

> [!NOTE]
> **Partially implemented (2026-08-27):** `src/narration/translation.py` has real, tested
> `PhraseTableTranslator` (critical path) and `EnglishFallbackTranslator` implementations, plus a
> correct-API `IndicTrans2Translator` adapter that has **not** been run against real model weights
> yet (`IndicTransToolkit`/`transformers` are installed and importable; the
> `ai4bharat/indictrans2-en-indic-dist-200M` checkpoint itself was not downloaded in this session —
> see the module docstring for the smoke-test command to run once it is).

### 11.5 TTS / Audio — two-tier strategy

Do not lock the prototype to FastSpeech2 only. FastSpeech2 is older but still viable if a working
local voice is already available; for edge deployment, Piper-style ONNX voices are attractive where
language coverage is acceptable because they are fast and designed for local inference; for Indic
naturalness, AI4Bharat models such as IndicF5 / Indic Parler TTS may produce better quality but are
heavier and should be benchmarked before entering the real-time path.

```text
Critical path: beep/haptic + cached short spoken clips
Fast local TTS candidate: Piper/ONNX if target language voice exists
Indic quality candidate: AI4Bharat IndicF5 or Indic Parler TTS for non-critical narration only after latency testing
Fallback: FastSpeech2 if it is already integrated and meets latency
```

> [!NOTE]
> **Implemented and measured (2026-08-27):** `src/narration/tts.py` has a real, tested
> `CachedClipTTS` (critical path — a cache miss falls back to a beep in ~0ms, never blocks on a
> model) and a real, tested `PiperTTS` backend. Measured on this dev machine (CPU only, no GPU),
> voice `en_US-lessac-low`: one-time model load ~444 ms, per-utterance synthesis 30–68 ms for the
> four representative `NarratorEvent` phrasings (`tools/benchmark_narration_latency.py`,
> `evaluation/benchmarks/narration_latency/report.md`). No formal TTS latency budget exists yet
> anywhere in `docs/` to grade this against — that benchmark script flags the gap; add one to
> `hardware_targets.md` once a target device measurement exists. FastSpeech2 and the Indic voices
> (IndicF5 / Indic Parler-TTS) remain unimplemented, per the "benchmark before entering the
> real-time path" guidance above.

### 11.6 Orchestration architecture — explicit queues and budgets

Keep the asynchronous dual-track architecture, but implement it with explicit queues and budgets
rather than vague asyncio glue:

| Lane | Budget | Components | Behavior |
|---|---:|---|---|
| Perception/reflex | `<50 ms p95` detector/reflex budget | YOLO26n v3, tracker, depth, TTC/K-score | Never waits for SLM/translation/TTS |
| Cognitive | `<=500 ms` soft budget | SLM-1 semantic assessment | Advisory only |
| Narration | best effort, interruptible | template/SLM-2, translation, TTS | Dropped or shortened if stale |
| Audio emergency | immediate | beep/haptic/cached clips | Preempts all other audio |

Structured event queues: `PerceptionFrame -> TrackedObjects -> ThreatEvent -> VerifiedEvent ->
AudioCommand`. The Physics Verification layer is the arbitration boundary; it must reject stale or
hallucinated semantic output.

---

## 12. Historical Implementation Plan (superseded)

A 14-day day-by-day plan was drafted after YOLO26n v3 training and SANPO edge benchmarking, before
work diverged (after roughly Day 4) to prioritize the kinetic-score defense (`methodology.md` §4,
`ablation_guide.md`) ahead of the Cognitive Layer. Kept here as a historical record of intent; the
current backlog is `pending_work.md`, not this plan.

| Days | Focus | Status at time of divergence |
|---|---|---|
| 1–2 | Detector export (ONNX/TensorRT) and runtime baseline comparison | Not run; export script exists (`training/scripts/export_yolo26n_edge.py`) but no exported artifact is committed |
| 3–4 | Perception-to-Physics event contract (`ThreatEvent` schema, builder, unit tests) | Done — `src/threat_prioritizer/events.py`, `tools/build_threat_events.py` |
| 5–6 | Reflex Layer + audio emergency path | Reflex Layer done first-pass (`src/reflex_layer/reflex.py`); audio emergency command layer still pending |
| 7–8 | SLM-1 Cognitive prototype | Not started — work diverged here into the kinetic-score ablation instead |
| 9–10 | Physics Verification arbitration rules + divergence logging | Partial — judge logic exists and is unit-tested against the Reflex Layer only |
| 11–12 | Narration, translation, TTS | Not started |
| 13–14 | Full pipeline edge simulation + demo packaging | Not started — no full SANPO replay has been run yet |
