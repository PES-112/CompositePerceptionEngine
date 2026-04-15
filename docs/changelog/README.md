# CPE — Component Changelog Index

This folder documents every architectural change made to the **Composite Perception Engine (Argus-REI)** across development sessions. Each file tracks one component's evolution with dated entries, before/after rationale, and links to affected source files.

---

## Component Changelogs

| Component | File | Status | Last Updated |
|---|---|---|---|
| Perception Stack — Depth Loader | [perception_stack__depth_loader.md](./perception_stack__depth_loader.md) | 🟢 Active | 2026-03-30 |
| Perception Stack — YOLO Tracker | [perception_stack__yolo_tracker.md](./perception_stack__yolo_tracker.md) | 🟢 Active | 2026-03-30 |
| Perception Stack — Pipeline | [perception_stack__pipeline.md](./perception_stack__pipeline.md) | 🟢 Active | 2026-03-30 |
| Perception Stack — Physics | [perception_stack__physics.md](./perception_stack__physics.md) | 🟢 Active | 2026-03-30 |
| Perception Stack — CSV Writer | [perception_stack__csv_writer.md](./perception_stack__csv_writer.md) | 🟢 Active | 2026-03-30 |
| Perception Stack — Fact Sheet Builder | [perception_stack__fact_sheet_builder.md](./perception_stack__fact_sheet_builder.md) | 🟢 Active | 2026-03-25 |
| Physics Verification | [physics_verification.md](./physics_verification.md) | 🟡 Scaffolded | 2026-03-25 |
| Tools — run_perception | [tools__run_perception.md](./tools__run_perception.md) | 🟢 Active | 2026-03-30 |
| Tools — run_fact_sheets | [tools__run_fact_sheets.md](./tools__run_fact_sheets.md) | 🟢 Active | 2026-03-25 |
| Architecture (System-Wide) | [architecture.md](./architecture.md) | 🟢 Active | 2026-03-25 |

---

## Conversation Sessions That Generated Changes

| Date | Session Summary | Components Affected |
|---|---|---|
| **2026-03-25** | Architecting Dual-SLM Navigation System — designed the overall CPE architecture with Reflex/Cognitive/Physics Verification layers. Initialized all source modules. | All (initial design) |
| **2026-03-25 → 2026-03-27** | Implementing Argus-REI Perception Pipeline — integrated YOLO26n + ByteTrack, depth-based obstacle avoidance, and CSV fact sheets for SLM-1 fine-tuning. | `perception_stack`, `physics_verification`, `tools` |
| **2026-03-27** | Documenting Perception Pipeline Workflow — produced Stage 1 → Stage 2 workflow summary for presentation. | Architecture docs |
| **2026-03-27** | Occlusion Problem — identified that YOLO misses unlabeled foreground obstacles (poles). Added depth-based `detect_unlabeled_obstacle()`. | `pipeline.py` |
| **2026-03-30** | Performance Overhaul — ported centre-patch depth extraction, occlusion masking, depth-guided NMS, grid-based obstacle sweep, nth-frame skip, streaming generator, and tensor batch physics from notebook to production. | 8 files in `perception_stack`, `tools` |

---

## Architecture Overview (Current as of 2026-03-30)

```
Input (Camera + Depth + IMU)
        │
        ▼
  Sensor Fusion
        │
        ▼
  Perception Stack  ──────────────────────────────────────────────────────
  ├── depth_loader.py        (SANPO / UASOL depth format handling)
  ├── yolo_tracker.py        (YOLO26n + ByteTrack + depth-guided NMS)
  ├── pipeline.py            (Stage 1 orchestrator — streaming generator)
  ├── physics.py             (Bearing, velocity, kinetic score — scalar + tensor)
  ├── csv_writer.py          (Canonical CSV schema + StreamingCSVWriter)
  └── fact_sheet_builder.py  (Stage 2 — K₀ + K₊₂ → JSONL for SLM-1 SFT)
        │
  Threat Prioritizer
  ├── Low K  → Ignored
  ├── High K → Reflex Layer (TTC < 1.0s → OVERRIDE)
  └── Mid K  → Cognitive Layer (SLM-1 semantic eval)
        │
  Physics Verification (The Judge)
        │
  Narrator SLM-2  →  [Indic Translation]  →  Audio Output
        │
  System Heartbeat (ambient, 5–8s)
```
