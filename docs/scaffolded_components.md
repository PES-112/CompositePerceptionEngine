# Scaffolded Components — Status & Specs

These components have their module directories created (`__init__.py` present) but contain no implementation yet. All are planned for Phase 2 and beyond.

---

## Reflex Layer — `src/reflex_layer/`

**Phase:** 2
**Role:** Deterministic, hard real-time path (< 50ms). Receives high-K objects from the Threat Prioritizer. Computes TTC precisely. If `TTC < 1.0s`, fires an OVERRIDE signal to Physics Verification, bypassing SLM-1 entirely.

**Planned logic:**
```
TTC = distance_m / max(velocity_ms, ε)
IF TTC < 1.0s → emit ReflexResult(override=True)
ELSE          → emit ReflexResult(override=False, kinetic_score=K)
```
No neural network in this path — pure physics.

---

## Cognitive Layer / SLM-1 — `src/cognitive_layer/`

**Phase:** 2 (integration), Phase 1 already produces training data for it.
**Role:** Semantic evaluation of the scene. Receives YOLO detections + 2-sec trajectory history + depth data (+ optional intent labels from HEADSUP). Produces `SemanticEval` JSON.

**Planned model:** `Qwen2.5-1.5B-Instruct` fine-tuned with LoRA adapter (trained in Phase 1 SFT, then Phase 3 PPO).

**Target latency:** ~500ms (soft real-time).

**Example output:**
```json
{
  "primary_threat": "track_007",
  "reason": "Motorcycle accelerating from left at 8m/s",
  "secondary": "track_012",
  "scene_state": "crossing_intersection"
}
```

---

## Threat Prioritizer — `src/threat_prioritizer/`

**Phase:** 2
**Role:** Routes Perception Stack output to the correct downstream path based on raw Kinetic Score K.

```
K < LOW_THRESHOLD  (0.5)  → Ignore
K > HIGH_THRESHOLD (5.0)  → Reflex Layer
else                       → Cognitive Layer (SLM-1)
```

Both tracks can fire simultaneously (e.g. speeding car → Reflex, nearby pedestrian → Cognitive).

---

## Narrator SLM-2 — `src/narrator_slm/`

**Phase:** 3
**Role:** Receives `NarratorEvent` from Physics Verification. Generates a short (≤10 word) verbal navigation warning.

**Planned model:** `Phi-3-Mini-4K-Instruct` (3.8B, INT4 ~2.2GB). Chosen for fluency and ONNX Mobile + QNN support.

**Example:**
```
Input : {class: motorcycle, dist: 6m, bearing: left, v: 8m/s}
Output: "Motorcycle fast from your left."
```

---

## Sensor Fusion — `src/sensor_fusion/`

**Phase:** 2
**Role:** Merges primary camera feed + egocentric depth data + gyroscope / 360-camera inputs before passing to the Perception Stack.

---

## Sensor Interface — `src/sensor_interface/`

**Phase:** 2–4
**Role:** Hardware abstraction layer for camera, gyro, and depth sensor I/O. Replaced by dataset file I/O during Phase 1.

---

## Indic Translation — `src/indic_translation/`

**Phase:** 4
**Role:** Optional post-processing step after SLM-2 narration. Translates output to any of 22 Indian languages.

**Planned model:** `IndicTrans2` (AI4Bharat, ~200M params, ~200MB). +75ms latency — acceptable for narration path.

---

## Audio Output — `src/audio_output/`

**Phase:** 4
**Role:** Text-to-speech using FastSpeech2. Receives final narration string (possibly translated) and emits audio.

---

## System Heartbeat — `src/system_heartbeat/`

**Phase:** 4
**Role:** Fires ambient audio updates every 5–8 seconds when no threat is detected. Prevents dead silence in low-risk environments. Feeds Audio Output directly — bypasses Physics Verification.

Dynamic interval planned: empty scene = 10s, crowded-but-safe sidewalk = 4s.
