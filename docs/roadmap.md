# CPE Prototype Roadmap and Technology Audit

This roadmap reflects the current repository state after YOLO26n v3 training, SANPO valid-stream intake, and simulated edge latency benchmarking.

## Current Status

| Area | Current state | Decision |
|---|---|---|
| Detector | YOLO26n v3 from base checkpoint trained on the 17-class CPE hazard taxonomy | Keep as the detector for the prototype; do not move to a larger detector unless SANPO failure analysis proves a recall gap |
| Detector checkpoint | `training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt` | Preferred checkpoint |
| Real-time benchmark | 10 SANPO-Real valid streams, Jetson Orin Nano 8GB simulation profile | Avg simulated p95 latency `34.31 ms`; worst-session p95 `41.96 ms`; passes `<50 ms` detector/reflex budget |
| SANPO data | `valid_streams.json` maps to `sanpo-real/<session_id>/camera_head/left` RGB/depth branches | Use SANPO-Real for latency and physics simulation; use SANPO-Synthetic if segmentation-mask correctness checks are needed |
| Tracking | Ultralytics tracking path currently uses ByteTrack | Keep ByteTrack for prototype stability; evaluate TrackTrack or BoT-SORT only after baseline end-to-end metrics are collected |
| Depth handling | SANPO depth maps may be native `960 x 960`, not RGB resolution | Fixed by inferring depth shape and scaling RGB boxes into depth coordinates |

## Technology Audit

### 1. Perception: YOLO26n v3 + Depth-Guided Tracking

**Verdict: keep.**

YOLO26n is the right detector family for the prototype because the system is edge-latency constrained, not leaderboard-mAP constrained. The v3 checkpoint already passes the simulated Jetson detector/reflex budget across 10 SANPO valid streams. Moving to a larger detector now would likely increase latency and integration risk before proving the current model is inadequate.

Next optimization should be deployment format, not model size:

1. Export v3 to ONNX FP16/INT8 for portability.
2. Export v3 to TensorRT FP16/INT8 on actual Jetson/NVIDIA target hardware.
3. Compare PyTorch vs ONNX/TensorRT latency and accuracy on the same SANPO 10-session set.

Important export note: use Ultralytics `quantize=16` or `quantize=8`; legacy `half=True` / `int8=True` are deprecated aliases. For INT8, use representative calibration data from `training/configs/cpe_hazard_classes.yaml` or SANPO-derived frames.

### 2. SLM-1 Cognitive Layer

**Roadmap update: replace Qwen2.5-1.5B as the default with Qwen3-1.7B in non-thinking mode. Keep Qwen2.5-1.5B as fallback.**

Rationale:

- Qwen3-1.7B is newer, Apache-2.0, multilingual, has a 32K context window, and supports explicit thinking/non-thinking modes.
- For real-time CPE cognition, use **non-thinking mode** to avoid unnecessary chain-of-thought latency.
- Qwen2.5-1.5B remains a reasonable fallback if Qwen3 runtime support or quantization causes deployment friction.

Prototype target:

```text
SLM-1 primary: Qwen3-1.7B, quantized GGUF/ONNX where practical, non-thinking mode
SLM-1 fallback: Qwen2.5-1.5B-Instruct, INT4/GGUF
Output: strict JSON semantic assessment over the perception fact sheet
Latency budget: soft real-time, <=500 ms
```

Do not let SLM-1 sit in the hard safety path. Its output is advisory and must be arbitrated by Physics Verification.

### 3. SLM-2 Narration Layer

**Roadmap update: Phi-3 Mini is acceptable but no longer the best default. Prefer Phi-4-mini-instruct for narration if memory permits; otherwise use template-first narration for the prototype.**

Rationale:

- Phi-4-mini-instruct is newer than Phi-3 Mini and has stronger multilingual/reasoning support.
- Phi-4-mini is roughly 3.8B/4B class, so it may be too heavy if SLM-1, translation, and TTS are also resident.
- The narrator has a constrained job: turn a verified event into a short warning. A deterministic template layer may outperform an SLM on latency, safety, and consistency for the first demo.

Prototype target:

```text
SLM-2 phase 1: deterministic/template narrator for critical warnings
SLM-2 phase 2: Phi-4-mini-instruct only for non-critical richer narration
SLM-2 fallback: Phi-3-mini-4k-instruct ONNX/GGUF if Phi-4 runtime is too heavy
```

This keeps the demo safe: critical alerts should be short, deterministic, and immediate.

### 4. Translation

**Verdict: keep IndicTrans2 distilled models for the prototype; revisit IndicTrans3 only after local availability/runtime is validated.**

Rationale:

- IndicTrans2 supports the 22 scheduled Indic languages and has distilled 200M/320M variants suitable for edge-style testing.
- The 1B variants may improve quality but are harder to fit alongside detector + SLMs + TTS.
- Translation must not block critical reflex alerts. Critical alerts should use pre-translated phrase templates where possible.

Prototype target:

```text
Critical path: pre-translated alert phrase table
Non-critical narration: IndicTrans2 distilled En-Indic model
Fallback: English-only output if translation exceeds latency budget
```

### 5. TTS / Audio

**Roadmap update: do not lock the prototype to FastSpeech2 only. Use a two-tier audio strategy.**

Rationale:

- FastSpeech2 is older but still viable if a working local voice is already available.
- For edge deployment, Piper-style ONNX voices are attractive where language coverage is acceptable because they are fast and designed for local inference.
- For Indic naturalness, AI4Bharat models such as IndicF5 / Indic Parler TTS may produce better quality but are heavier and should be benchmarked before entering the real-time path.

Prototype target:

```text
Critical path: beep/haptic + cached short spoken clips
Fast local TTS candidate: Piper/ONNX if target language voice exists
Indic quality candidate: AI4Bharat IndicF5 or Indic Parler TTS for non-critical narration only after latency testing
Fallback: FastSpeech2 if it is already integrated and meets latency
```

### 6. Orchestration Architecture

**Verdict: keep asynchronous dual-track architecture, but implement with explicit queues and budgets rather than vague asyncio glue.**

Required runtime lanes:

| Lane | Budget | Components | Behavior |
|---|---:|---|---|
| Perception/reflex | `<50 ms p95` detector/reflex budget | YOLO26n v3, tracker, depth, TTC/K-score | Never waits for SLM/translation/TTS |
| Cognitive | `<=500 ms` soft budget | SLM-1 semantic assessment | Advisory only |
| Narration | best effort, interruptible | template/SLM-2, translation, TTS | Dropped or shortened if stale |
| Audio emergency | immediate | beep/haptic/cached clips | Preempts all other audio |

Use structured event queues:

```text
PerceptionFrame -> TrackedObjects -> ThreatEvent -> VerifiedEvent -> AudioCommand
```

The Physics Verification layer is the arbitration boundary; it must reject stale or hallucinated semantic output.

## Revised 14-Day Plan

### Days 1-2: Detector Export and Runtime Baseline

Owner: Vision/System

- Export v3 `best.pt` to ONNX FP16.
- Export v3 to TensorRT FP16/INT8 on target Jetson/NVIDIA hardware when available.
- Run `tools/benchmark_edge_realtime.py` on the SANPO 10-session set for PyTorch vs exported runtime.
- Save all metrics under `evaluation/benchmarks/sanpo_edge_realtime/`.

Deliverable: runtime comparison table with p50/p95/p99 latency and pass/fail against `<50 ms`.

### Days 3-4: Perception-to-Physics Event Contract

Owner: Vision + Physics

Status: implemented as a first pass in `src/threat_prioritizer/events.py` with `tools/build_threat_events.py` and unit tests.

- Freeze the JSON schema for tracked detections, distance, velocity, TTC, and K-score.
- Add a deterministic `ThreatEvent` builder from Stage 1 rows.
- Unit-test TTC and K-score with synthetic frame sequences.
- Next: run it on real SANPO perception CSVs and tune `LOW_K_THRESHOLD`, `HIGH_K_THRESHOLD`, and near-static hazard distance.

Deliverable: `ThreatEvent` JSONL from SANPO streams.

### Days 5-6: Reflex Layer and Audio Emergency Path

Owner: Physics + Audio

Status: Reflex Layer bridge implemented first-pass in `src/reflex_layer/reflex.py`; audio emergency command layer is still pending.

- Convert `ThreatEvent(route="reflex")` into `ReflexResult` and Physics Verification `NarratorEvent`.
- Implement hardcoded beep/haptic/cached-clip command for high-risk events.
- Ensure the reflex path does not call SLM-1, SLM-2, translation, or neural TTS.
- Measure event-to-audio-command latency separately from detector latency.

Deliverable: reflex alerts generated from SANPO replay.

### Days 7-8: SLM-1 Cognitive Prototype

Owner: Cognitive

- Prototype Qwen3-1.7B non-thinking mode with strict JSON output.
- Benchmark against Qwen2.5-1.5B fallback if Qwen3 runtime is too heavy.
- Feed only compact fact sheets, not raw images.

Deliverable: semantic JSON assessment with latency histogram.

### Days 9-10: Physics Verification

Owner: Physics + Cognitive

- Implement arbitration rules: reflex override, SLM agreement, SLM disagreement, stale semantic response rejection.
- Log divergences for future training/reward use.

Deliverable: verified event stream and divergence report.

### Days 11-12: Narration, Translation, and TTS

Owner: Narration + Audio

- Start with deterministic warning templates.
- Add IndicTrans2 distilled translation for non-critical narration.
- Add TTS only after measuring cached-clip and template latency.
- Keep critical alerts pre-translated/cached.

Deliverable: audio-command path with critical/non-critical behavior separated.

### Days 13-14: Edge Simulation and Demo Packaging

Owner: All

- Run the full pipeline on the 10 SANPO valid streams.
- Compare GB10 native, Jetson simulation profile, and any actual edge hardware results.
- Record demo with frame view, detections, verified event logs, and audio commands.

Deliverable: reproducible demo plus metrics table.

## Immediate Next Step

Detector export/runtime comparison is available through `training/scripts/export_yolo26n_edge.py`, but full-system work now moves to the **perception-to-physics event contract** so SLM, reflex, and audio components can plug into one stable runtime stream.

Concrete task list:

1. Add an export utility that exports the v3 checkpoint to ONNX/TensorRT with modern Ultralytics `quantize` arguments.
2. Run ONNX FP16 export on the lab machine if dependencies are available.
3. If ONNX Runtime GPU is unavailable on ARM, save the PyTorch checkpoint as the current runtime baseline and defer TensorRT engine export to actual Jetson/NVIDIA target hardware.
4. Re-run the 10-session SANPO benchmark with each available runtime and update the aggregate comparison.
