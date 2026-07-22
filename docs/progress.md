# Composite Perception Engine Progress

Living checklist for where the project currently stands. Update this whenever a major component, dataset, model, benchmark, or integration step changes.

## Current Focus

Build the runtime bridge from perception into deterministic safety logic before adding SLM/TTS exports.

## Pipeline Checklist

| Stage | Status | Current artifact / notes |
|---|---|---|
| Repository structure and docs hygiene | Done | `.agents/AGENTS.md`, `README.md`, `docs/CHANGELOG.md` maintenance rules are active. |
| Hardware and edge-target documentation | Done | `docs/hardware_targets.md` separates observed GB10 specs from the analytical Jetson profile and pending physical-device validation. |
| Dataset intake | Done for YOLO v3 | Roboflow manifest and SANPO valid-stream metadata are in place. |
| YOLO26n hazard detector | Done for current prototype | Preferred checkpoint: `training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt`. |
| Detector validation | Done for v1/v2/v3 comparison | Version comparison artifacts live under `evaluation/benchmarks/yolo26n_version_comparison/`. |
| SANPO edge latency benchmark | Done for detector/reflex budget proxy | 10-session Jetson Orin Nano 8GB simulation avg p95 `34.31 ms`, worst p95 `41.96 ms`. |
| Edge detector export | Ready, deferred | `training/scripts/export_yolo26n_edge.py` exists; run after runtime components stabilize. |
| ThreatEvent contract | Done first pass | `src/threat_prioritizer/events.py` converts perception rows to `ignore`, `cognitive`, or `reflex` events. |
| Reflex Layer bridge | Done first pass | `src/reflex_layer/reflex.py` converts reflex events into Physics Verification narrator events. |
| Cognitive SLM-1 adapter | Not started | Next major component: strict JSON semantic output for cognitive-route events. |
| Physics Verification integration | Partial | Existing judge works with Reflex Layer; still needs cognitive SLM response integration and stale-response handling. |
| Narration/audio command layer | Not started | Start with deterministic templates/cached critical alerts before neural TTS. |
| Translation/TTS runtime | Not started | IndicTrans2/phrase table and local TTS should remain off the reflex path. |
| Full SANPO replay | Not started | Need end-to-end run: perception → threat events → reflex/cognitive → verified event → audio command. |
| Component export packaging | Deferred | Export detector, SLM, translation, and TTS only after Python pipeline behavior is stable. |

## Immediate Next Steps

1. Add a Cognitive Layer SLM stub/adapter that consumes `ThreatEvent(route="cognitive")` and emits `SemanticEval` JSON.
2. Add stale-response and latency-budget handling between Cognitive Layer and Physics Verification.
3. Add deterministic narration/audio command generation for `NarratorEvent`, starting with reflex override alerts.
4. Run a SANPO replay through `tools/build_threat_events.py` and the Reflex Layer to tune thresholds.

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
