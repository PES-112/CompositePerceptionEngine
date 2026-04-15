# Changelog — System Architecture (Argus-REI / CPE)

> Tracks high-level design decisions, layer additions, and architectural corrections across all development sessions.

---

## [Phase 1 Active] — 2026-03-25 to 2026-03-30 | Offline Data Curation

**Goal:** Generate a high-quality JSONL training dataset from SANPO and UASOL video sequences that enables supervised fine-tuning of SLM-1 (Qwen2.5-1.5B-Instruct).

### Completed in This Phase

| Component | Status |
|---|---|
| Perception Stack (Stage 1) | ✅ Complete — streaming, nth-frame skip, depth-guided filtering |
| Fact Sheet Builder (Stage 2) | ✅ Complete — K₀ + K₊₂ JSONL output |
| Physics Verification (Judge) | ✅ Complete — arbitration logic, RL rewards |
| Download / Prepare Tools | ✅ Complete |
| Reflex Layer | 🟡 Scaffolded (module exists, logic pending Phase 2) |
| Cognitive Layer | 🟡 Scaffolded (SLM-1 not integrated yet) |
| Narrator SLM | 🟡 Scaffolded |
| Audio Output | 🟡 Scaffolded |
| Sensor Fusion | 🟡 Scaffolded |
| Indic Translation | 🟡 Scaffolded |

---

## 2026-03-30 | Performance Overhaul

**Session:** *Optimizing CPE Perception Pipeline Performance*

### Architectural Changes

#### Streaming Generator Architecture
The pipeline was refactored from a batch-accumulate-then-write pattern to a streaming generator pattern. `run_perception_stream()` yields per-frame rows; `StreamingCSVWriter` writes them immediately. This fundamentally decouples memory usage from dataset size.

#### Occlusion-Aware Depth Sampling
Introduced front-to-back ordering of detections + cumulative occluder list passed to `median_depth_in_box()`. This prevents the physically incorrect pattern of farther objects appearing closer because a foreground object's surface bleeds into their depth sample region.

#### Grid Obstacle Sweep
Replaced the single-ROI unlabeled obstacle detector with a 5-column grid sweep covering the lower 60% of each frame. YOLO bounding boxes are masked out before scanning so only truly unaccounted obstacles trigger detections.

#### Nth-Frame Skip Strategy
Added `frame_step` parameter throughout the pipeline + CLI. For SANPO at 30fps: `frame_step=3` reduces processing to ~10fps equivalent (~67% compute reduction) while maintaining sufficient temporal resolution for velocity estimation.

---

## 2026-03-27 | Occlusion Problem Discovered

**Session:** *YOLO isn't detecting the pole in front of the person*

### Architectural Decision

**Problem Identified:** YOLO's COCO training set has no category for standalone poles, bollards, fences, or infrastructure obstacles. These objects are near-field collision hazards for visually impaired pedestrians that YOLO will never detect.

**Decision:** Do not attempt to train or fine-tune YOLO to detect these. Instead, use the **depth map directly** to detect any physical close-range obstacle not already accounted for by a YOLO detection. This keeps YOLO's inference fast and purpose-limited, while adding a complementary depth-only sweep for the objects it structurally cannot identify.

---

## 2026-03-25 → 2026-03-27 | Argus-REI Architecture Finalized

**Session:** *Implementing Argus-REI Perception Pipeline* / *Documenting Perception Pipeline Workflow*

### Key Architectural Decisions

#### Physics Verification is Downstream of SLM-1 (Correction)

Early diagrams incorrectly placed Physics Verification as a parallel layer to SLM-1. **Correction:** Physics Verification is *downstream* of both the Reflex Layer and Cognitive Layer. It acts as a judge that arbitrates between two independent intelligence streams:
- Reflex Layer: deterministic physics (TTC < 1.0s → OVERRIDE, bypasses SLM-1)
- Cognitive Layer: SLM-1 semantic evaluation (intent, context, trajectory)

#### Dual-SLM Architecture

| Model | Role | Size (INT4) | Target Hardware |
|---|---|---|---|
| **SLM-1** (Qwen2.5-1.5B-Instruct) | Cognitive Layer — threat identification | ~900MB | Hexagon NPU (QNN) |
| **SLM-2** (Phi-3-Mini-4K-Instruct) | Narrator — verbal warning generation | ~2.2GB | Hexagon NPU (QNN) |

Total SLM budget: ~3.5GB. Fits within Snapdragon 8 Gen 3 NPU allocation after YOLO + TTS overhead.

#### Physics-as-Teacher for RL (No Human Labeling)

The most important architectural insight: SLM-1 is trained via PPO using Physics Verification as the reward signal. The physics layer is the *teacher*. No human annotation is needed because ground-truth threat labels are derived deterministically from dataset depth + velocity. SLM-1 learns causal physical reasoning by maximizing kinetic-score-aligned rewards across thousands of SANPO/UASOL frames.

#### Two-Stage Perception Flow

```
Stage 1 — Perception
  RGB + Depth → YOLO + Tracking + Physics → CSV

Stage 2 — Fact Sheets
  CSV → Kinetic Score (K₀ + K₊₂) → JSONL for SFT
```

---

## 2026-03-25 | Initial Architecture Design

**Session:** *Architecting Dual-SLM Navigation System*

### Original System Design

The CPE (Composite Perception Engine) was designed as a neuro-symbolic navigation assistant for blind pedestrians with these primary architectural goals:

1. **Dual-track perception:** Deterministic physics (Reflex) + semantic SLM reasoning (Cognitive) operating independently with arbitration.
2. **Future Kinetic Grounding:** Look-ahead dataset buffers to reward SLM-1 for causal semantic reasoning during PPO training.
3. **Phase 1 priority:** Build the data pipeline first — offline SANPO/UASOL curation with YOLO + depth + velocity + intent, then SFT of SLM-1.
4. **Mobile-edge deployment target:** Snapdragon 8 Gen 3 via LoRA adapters + GGUF quantization.

### Datasets

| Dataset | Role |
|---|---|
| **SANPO** | Primary — ground-truth egocentric depth, 80% of training data |
| **UASOL** | Secondary — unstructured real-world sidewalk, stress-tests optical flow stability |
| **HEADSUP** | Pedestrian intent labels (distracted, looking at phone) → SLM-1 context |

---

## File Reference
- **Architecture doc:** [`architecture.md`](../../architecture.md)
- **Implementation plan:** [`implementation_plan.md`](../../implementation_plan.md)
- **Phase 1 plan:** [`phase_1_breakdown.md`](../../phase_1_breakdown.md)
