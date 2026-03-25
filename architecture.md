# Architecture Validation: Argus-REI / CPE (Corrected)

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
                │ Deterministic│   │   SLM-1 (Qwen/Phi-3)    │
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
              │ Translation      │  │  (FastSpeech2)│
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

---

### 2.3 Threat Prioritizer

Routes objects based on raw Kinetic Score:

```
K = (W_mass · Class_severity) × V²_closing / max(d, ε)

K < LOW_THRESHOLD  → Ignore (static/far objects)
K > HIGH_THRESHOLD → Reflex Layer (hard real-time, < 50ms)
else               → Cognitive Layer (soft real-time, ~500ms)
```

> [!IMPORTANT]
> Both tracks can fire simultaneously for different objects. A speeding car goes Reflex; a jaywalker goes Cognitive. Physics Verification merges both.

---

### 2.4 Reflex Layer (< 50ms, Deterministic)

- Computes TTC precisely for high-K objects
- `IF TTC < 1.0s` → fires **OVERRIDE SIGNAL** to Physics Verification
- Bypasses SLM-1 entirely — hard real-time guarantee
- No neural network in this path

---

### 2.5 Cognitive Layer — SLM-1

**Inputs:** YOLO detections + 2-sec trajectory history + depth data + intent labels

**Task:** Produce a semantic evaluation of the scene:

```json
{
  "primary_threat": "track_007",
  "reason": "Motorcycle accelerating from left at 8m/s",
  "secondary": "track_012",
  "scene_state": "crossing_intersection"
}
```

SLM-1 reasons about **intent, trajectory, and context** — not just kinetic score. This is what makes it complementary to the physics layer.

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

## 6. Mobile Edge Hardware (Snapdragon 8 Gen 3)

| Resource | Limit | Notes |
|---|---|---|
| **SLM budget** | ≤ 2.6 GB (SLM-1 + SLM-2) | After YOLO + TTS |
| **NPU** | Hexagon (45 TOPS) | INT4/INT8 LLM inference |
| **Reflex latency** | < 50ms | Pure deterministic, no SLM |
| **Cognitive latency** | ~500ms | SLM-1 + Physics Verification |
| **Deployment format** | GGUF Q4_K_M (llama.cpp + QNN) | Targets Hexagon NPU |

---

## 7. SLM Recommendations (Mobile Edge)

| Role | Model | Params | INT4 Size | Why |
|---|---|---|---|---|
| **SLM-1** (Cognitive) | **Qwen2.5-1.5B-Instruct** | 1.5B | ~900MB | Best reasoning/size ratio; strong JSON output |
| **SLM-2** (Narrator) | **Phi-3-Mini-4K-Instruct** | 3.8B | ~2.2GB | Best fluency; ONNX Mobile + QNN EP supported |
| **SLM-1 fallback** | Gemma-3-1B-IT | 1.0B | ~600MB | MediaPipe native Android integration |
| **Indic Translation** | IndicTrans2 | ~200M | ~200MB | 22 Indian languages, edge-designed |

**Total:** ~3.5GB. Swap SLM-2 to 2-bit quant (~1.1GB) if memory-constrained.

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

## 9. Minor Architecture Recommendations

To refine the architecture without making major structural changes:

1. **Depth Extrapolation Smoothing:** Since dataset depth points can sometimes be sparse or noisy at bounding box edges, apply a **Kalman Filter** to the `d` values extracted from the dataset before calculating `v = Δd / Δt`. This prevents jitter in the Kinetic Score from causing false reflex overrides.
2. **Dynamic Heartbeat:** Instead of a fixed 5-second interval, let the System Heartbeat scale inversely with scene complexity. (e.g., empty room = 10s, crowded but safe sidewalk = 4s).
3. **Intent-Gating:** Only pass HEADSUP intent labels to SLM-1 for objects within a 15-meter radius. Processing intent for distant background pedestrians wastes SLM token context window.
