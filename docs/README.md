# CPE Documentation Index

**Project:** Composite Perception Engine (Argus-REI)
**Goal:** Neuro-symbolic pedestrian navigation assistant for visually impaired users.
**Current Phase:** Phase 1 — Offline Data Curation + SLM-1 Supervised Fine-Tuning

---

## 📋 Feed These to an LLM (In This Order)

To bring an LLM up to speed on the entire project, feed the files in this order:

| # | File | What It Covers |
|---|---|---|
| 1 | [`architecture.md`](./architecture.md) | Full system design — layers, data flow, RL loop, hardware targets, SLM model choices |
| 2 | [`phase_1_breakdown.md`](./phase_1_breakdown.md) | 4-phase roadmap + deep-dive on Phase 1 steps (datasets → YOLO → depth → JSONL → SFT) |
| 3 | [`perception_stack.md`](./perception_stack.md) | Every module in `src/perception_stack/` — current implementation, APIs, design decisions |
| 4 | [`physics_verification.md`](./physics_verification.md) | The Judge — adjudication rules, RL reward structure, data classes |
| 5 | [`scaffolded_components.md`](./scaffolded_components.md) | What's planned but not implemented (Reflex, Cognitive, Narrator, Audio, etc.) |
| 6 | [`changes.csv`](./changes.csv) | Chronological log of every feature added or changed |
| 7 | [`walkthrough.md`](./walkthrough.md) | Detailed before/after code diffs from the Phase 1 performance overhaul |

---

## 🗂️ Document Descriptions

### Core Architecture
- **[`architecture.md`](./architecture.md)** — The canonical system diagram with all layers, dataset strategy, RL reward loop, threat prioritizer routing formula, SLM memory budget, and hardware specs for Snapdragon 8 Gen 3 deployment.

### Planning
- **[`phase_1_breakdown.md`](./phase_1_breakdown.md)** — 4-phase project milestones (25%/50%/75%/100%) + granular Phase 1 step-by-step guide from dataset download through SFT on Colab.
- **[`implementation_plan.md`](./implementation_plan.md)** — Technical plan produced during the performance overhaul (bottlenecks identified, proposed changes, verification plan).

### Component Docs
- **[`perception_stack.md`](./perception_stack.md)** — Single doc for the entire `src/perception_stack/` package. Covers `depth_loader`, `yolo_tracker`, `pipeline`, `physics`, `csv_writer`, `fact_sheet_builder`, CLI tools, and the full data flow from raw video to `train.jsonl`.
- **[`physics_verification.md`](./physics_verification.md)** — `src/physics_verification/` — The Judge module. Adjudication rules, RL rewards, integration status.
- **[`scaffolded_components.md`](./scaffolded_components.md)** — All components that are planned/scaffolded but not yet implemented: Reflex Layer, Cognitive Layer/SLM-1, Threat Prioritizer, Narrator SLM-2, Sensor Fusion, Indic Translation, Audio Output, System Heartbeat.

### Change Tracking
- **[`changes.csv`](./changes.csv)** — Running log of every feature added. Columns: `date, component, feature, status, files_changed, notes`. Add a new row whenever a change is made.
- **[`walkthrough.md`](./walkthrough.md)** — Detailed technical walkthrough with full before/after code diffs for the 2026-03-30 performance overhaul (8 files changed).

---

## 🏗️ Current Implementation Status

| Component | Phase | Status |
|---|---|---|
| `perception_stack` — Stage 1 (RGB+Depth → CSV) | 1 | ✅ Complete |
| `perception_stack` — Stage 2 (CSV → JSONL) | 1 | ✅ Complete |
| `physics_verification` — Judge logic | 1 | ✅ Complete |
| SLM-1 SFT in Colab | 1 | 🔲 Pending (needs full SANPO dataset run) |
| `reflex_layer` | 2 | 🟡 Scaffolded |
| `cognitive_layer` | 2 | 🟡 Scaffolded |
| `threat_prioritizer` | 2 | 🟡 Scaffolded |
| `narrator_slm` (SLM-2) | 3 | 🟡 Scaffolded |
| PPO training loop | 3 | 🔲 Not started |
| `audio_output` (FastSpeech2) | 4 | 🟡 Scaffolded |
| `indic_translation` (IndicTrans2) | 4 | 🟡 Scaffolded |
| `system_heartbeat` | 4 | 🟡 Scaffolded |
| Edge export (GGUF + QNN) | 4 | 🔲 Not started |

---

## How to Add to `changes.csv`

When making a new change, append a row in this format:
```
YYYY-MM-DD,<component>,<feature description>,<done|in-progress|planned>,<files changed>,<brief note>
```

Example:
```
2026-04-15,Perception Stack,Kalman filter on depth velocity,done,"src/perception_stack/physics.py","Smooths jittery d values"
```
