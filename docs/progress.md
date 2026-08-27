# Composite Perception Engine Progress

Living checklist for where the project currently stands. Update this whenever a major component,
dataset, model, benchmark, or integration step changes. For a presentation-oriented narrative of the
same state, see `progress_presentation.md`; for the full backlog, see `pending_work.md`.

## Current Focus

Close out the kinetic-score evidence (blinded referee + human calibration on the already-exported
disagreement frames — needs the remote box, see `pending_work.md` §1) and start the real SLM-1
adapter — the runtime bridge, narration/TTS layer, and a synthetic full-pipeline replay are now in
place (`src/pipeline/orchestrator.py`, `tools/run_full_pipeline_demo.py`) as the place for a real
Cognitive Layer to plug into.

## Pipeline Checklist

| Stage | Status | Current artifact / notes |
|---|---|---|
| Repository structure and docs hygiene | Done | `.agents/AGENTS.md`, `README.md`, `docs/CHANGELOG.md` maintenance rules are active. |
| Hardware and edge-target documentation | Done | `docs/hardware_targets.md` separates observed GB10 specs from the analytical Jetson profile and pending physical-device validation. |
| Dataset intake | Done for YOLO v3 | Roboflow manifest and SANPO valid-stream metadata are in place. |
| YOLO26n hazard detector | Done for current prototype | Preferred checkpoint: `training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt`. mAP50 0.641 / mAP50-95 0.459. |
| Detector validation | Done for v1/v2/v3 comparison | Version comparison artifacts live under `evaluation/benchmarks/yolo26n_version_comparison/`. |
| SANPO edge latency benchmark | Done for detector/reflex budget proxy | 10-session Jetson Orin Nano 8GB simulation avg p95 `34.31 ms`, worst p95 `41.96 ms`. Simulated, not physical hardware. |
| Kinetic score K0 defined and defended | Done | `severity(c) · v² / max(d, ε)`; K1–K5 dummies deleted; see `docs/methodology.md` §4. |
| Kinetic score ablation (label-free metrics) | Done | 139 sessions, 19,402 frames, 2026-08-26. `no-velocity`/`ttc` lose; `linear`/`no-severity`/`size`/λ tie. `docs/ablation_guide.md` §6. |
| Kinetic score blinded referee adjudication | **Not started** | 219 disagreement frames exported and idle; VLM + human calibration not run. `docs/pending_work.md` §1. |
| Depth smoothing filter | **Not started** | Flagged as a blocker for trusting K0 under real depth noise. `docs/pending_work.md` §1.4. |
| Edge detector export | Ready, deferred | `training/scripts/export_yolo26n_edge.py` exists; run after runtime components stabilize. |
| ThreatEvent contract | Done first pass | `src/threat_prioritizer/events.py` converts perception rows to `ignore`, `cognitive`, or `reflex` events. |
| Reflex Layer bridge | Done first pass | `src/reflex_layer/reflex.py` converts reflex events into Physics Verification narrator events. |
| Cognitive SLM-1 adapter | Rule-based stub done, real model not started | `src/cognitive_layer/stub.py` — same `SemanticEval` contract a real model will fill; wired into the runtime bridge below, not just standalone. |
| Runtime orchestrator (Threat Prioritizer → Cognitive → Physics Verification → Narration in one call) | Done first pass, synthetic data | `src/pipeline/orchestrator.py` — closes a real gap: `reflex_layer/reflex.py` only ever called Physics Verification for reflex-route events, so cognitive-only frames previously produced no alert at all regardless of Cognitive Layer output. |
| Physics Verification integration | Done for the composed path (stub semantics), real-model integration pending | Arbitrates stub `SemanticEval` vs. physics today via the orchestrator above; swapping in a real SLM-1 needs no changes here, only to `stub_semantic_eval()`'s replacement. Stale-response handling still not built. |
| Narration/audio command layer | Phase 1 done | `src/narration/templates.py` + `pipeline.py` — deterministic template narrator, structurally enforces override events skip translation/model-TTS. |
| Translation/TTS runtime | Phase 1 done, model backends partial | `src/narration/translation.py` + `tts.py` — phrase-table/English-fallback/cached-clip done; PiperTTS done and latency-measured (30–68 ms/utterance, CPU); IndicTrans2 adapter coded but not run against real weights. See `pending_work.md` §4. |
| Full pipeline replay | Done on synthetic data; real SANPO replay not started | `tools/run_full_pipeline_demo.py` runs Perception row → Threat Prioritizer → Cognitive (stub) → Physics Verification → Narration end-to-end on a 6-frame synthetic scenario. Re-run over real SANPO CSVs once available (`pending_work.md` §5). |
| Physical edge hardware validation | Not started | No physical Jetson (or other edge device) has been benchmarked yet. |
| Component export packaging | Deferred | Export detector, SLM, translation, and TTS only after Python pipeline behavior is stable. |

## Immediate Next Steps

Full prioritized backlog: `docs/pending_work.md`. Top 3 right now:

1. Run the VLM referee (`evaluation/vlm_referee.py`) plus human calibration on the 219 already-exported
   kinetic-score disagreement frames — the fastest path to a complete, citable result.
2. ~~Wire one full replay end-to-end~~ — done on synthetic data (`tools/run_full_pipeline_demo.py`);
   re-run on real SANPO CSVs once available.
3. Start the Cognitive Layer (SLM-1) adapter — `src/cognitive_layer/stub.py` is exactly where a real
   model replaces the rule-based placeholder; longest lead time of the remaining components.

## Useful Commands

Build threat events from a perception CSV:

```bash
.venv/bin/python tools/build_threat_events.py \
  --csv path/to/perception.csv \
  --out evaluation/logs/threat_events.jsonl \
  --summary evaluation/logs/threat_events_summary.json
```

Run unit tests for the current runtime bridge:

```bash
.venv/bin/python -m unittest tests.test_threat_prioritizer tests.test_reflex_layer -v
```

Regenerate result figures/tables from current benchmark artifacts:

```bash
.venv/bin/python evaluation/generate_report_figures.py
```
