# Progress Summary — Presentation Reference

Purpose-built for turning directly into slides. For the day-to-day completion checklist, see
`progress.md`; for what's left, see `pending_work.md`; for the reasoning behind each method, see
`methodology.md`.

---

## 1. One-line pitch

CPE is a real-time hazard-alerting system for visually impaired pedestrians: a deterministic,
physics-grounded perception path guarantees a hard latency bound on critical threats, with an
advisory semantic layer designed to add context without ever gating safety.

## 2. System architecture (slide: 1 diagram)

```
Camera + Depth + Gyro
   -> Perception Stack (YOLO26n + ByteTrack + Depth)
   -> Threat Prioritizer (TTC + Kinetic Score)
        - Low risk    -> Ignore
        - High risk   -> Reflex Layer (deterministic, <50ms)  --------\
        - Contextual  -> Cognitive Layer (SLM-1, ~500ms, NOT YET BUILT) --> Physics Verification (Judge)
                                                                            -> Narrator SLM-2 (NOT YET BUILT)
                                                                            -> Translation / Audio (NOT YET BUILT)
```

Full diagram and component contracts: `architecture.md`.

## 3. What's built and validated today

| Component | Status | Evidence |
|---|---|---|
| SANPO dataset intake (462 valid real sessions, GCS streaming) | Done | `sanpo_dataset.md` |
| Hazard dataset augmentation (Roboflow, 17-class taxonomy) | Done | `yolo_training.md` |
| Empirical class-gap analysis (depth-vs-detection) | Done | `sanpo_dataset.md` §"Gap Analysis Workflow" |
| YOLO26n hazard detector (v1 → v2 → v3) | Done, v3 preferred | §4 below |
| Detector retention regression testing vs. base YOLO | Done | §4 below |
| Kinetic score formula K0 derived and defended | Done | §5 below |
| K0 term-by-term ablation (139 sessions, 19.4K frames) | Done | §5 below |
| Edge latency — native GB10 measurement | Done (measured) | §6 below |
| Edge latency — Jetson Orin Nano 8GB proxy | Done (simulated, not physical) | §6 below |
| Threat Prioritizer → Cognitive (stub) → Physics Verification → Narration, one runtime call | Done, synthetic data | `src/pipeline/orchestrator.py`, `tools/run_full_pipeline_demo.py` |
| Cognitive Layer (SLM-1) | Rule-based stub only — **real model not started** | `pending_work.md` |
| Physics Verification full integration | Done for the composed path above; real-model swap-in still pending | `pending_work.md` |
| Narration / translation / TTS — phase 1 (templates, phrase-table, cached-clip, Piper) | Done and latency-measured | `pending_work.md` §4, `evaluation/benchmarks/narration_latency/report.md` |
| Narration / translation / TTS — model backends (Phi-4-mini, IndicTrans2 weights, IndicF5/FastSpeech2) | **Not started** | `pending_work.md` §4 |
| Full end-to-end replay | Done on synthetic data; **real SANPO replay not started** | `tools/run_full_pipeline_demo.py`, `pending_work.md` §5 |
| Physical edge hardware validation | **Not started** | `pending_work.md` |

## 4. Key result: hazard detector

Three fine-tuning iterations, same held-out test split. v3 is the current preferred checkpoint.

| Version | mAP50 | mAP50-95 | What changed |
|---|---:|---:|---|
| v1 | 0.651 | 0.432 | Hazard-only baseline; retained COCO classes not evaluated |
| v2 | 0.599 | 0.424 | Full 17-class data, but regressed retained classes (esp. `truck`) |
| **v3** | **0.641** | **0.459** | Restarted from base checkpoint on the same data — repaired most v2 regression |

**Finding worth presenting on its own slide:** continuing fine-tuning from an already domain-adapted
checkpoint (v1→v2) caused catastrophic forgetting of retained COCO classes; restarting from the
original base checkpoint on identical data (v3) fixed most of it while keeping the hazard-class gains.
This is a training-protocol result, not just a data result.

Remaining known weakness: `dog` retention is still below base YOLO after v3.

Figures: `evaluation/benchmarks/figures/yolo_version_map.png`,
`evaluation/benchmarks/figures/yolo_per_class_map50.png`,
`evaluation/benchmarks/figures/yolo_retention_f1_delta.png`.

## 5. Key result: kinetic score ablation

**The question:** does every term of `K0 = severity(class) · v² / distance` actually earn its place,
or are some of them decoration? Rather than a multi-formula popularity contest (the earlier K1–K5
approach, discarded — see `methodology.md` §4.2), each term is individually knocked out and the
ranking-quality impact is measured on 139 real SANPO sessions (19,402 scored frames).

| Arm | Verdict | Headline number |
|---|---|---|
| no-velocity (sanity check) | **Loses decisively** | Rank stability collapses to 0.28–0.52 (from ~1.0) |
| ttc (plain time-to-collision) | **Loses decisively** | 4× worse encounter-detection rate than K0 |
| linear (`v¹` instead of `v²`) | Ties with K0 | Identical on every metric |
| no-severity (drop class weighting) | Ties with K0 | Identical on every metric |
| size (add bbox-size term) | Ties with K0 | Identical on every metric |
| λ = 0.25 / λ = 1.0 (mass exponent) | Ties with K0 | Identical on every metric |

**Headline finding:** velocity and the K0 formula structure are the only elements that measurably
drive ranking quality. Class severity, the `v²` exponent, bounding-box size, and the mass exponent λ
all tie — kinematics dominate every reordering on this corpus. This is a real, reportable finding
(not a null result to hide): it means those terms should be defended as declared design choices with
explicit limitations, not oversold as empirically necessary.

**Still open:** whether the tied terms matter to a *human's* judgment of "top threat" — the 219
frames where arms disagreed have been exported for blinded VLM/human referee adjudication but that
step has not been run yet (`pending_work.md`).

Figure: `evaluation/benchmarks/figures/kinetic_ablation_metrics.png`. Full table: `ablation_guide.md` §6.

## 6. Key result: edge latency

| Profile | Avg p95 latency | Budget (50 ms) | Status |
|---|---:|---|---|
| Native NVIDIA GB10 (measured) | 7.83 ms | Pass | Measured |
| Jetson Orin Nano 8GB (4× compute scale + 3 ms overhead proxy) | 34.31 ms (worst session: 41.96 ms) | Pass, all 10/10 sessions | **Simulated, not physical hardware** |

**Framing for slides:** this passes a conservative analytical proxy, not a physical device — the
single most important caveat to state out loud, since it's the most likely question from an audience
familiar with edge deployment. Physical Jetson validation is the top item in `pending_work.md`.

Figure: `evaluation/benchmarks/figures/edge_latency_sessions.png`.

## 7. Suggested slide order

1. Problem + one-line pitch (§1)
2. Architecture diagram, explicitly marking what's built vs. not (§2–3)
3. Detector result + the v2→v3 training-protocol finding (§4)
4. Kinetic score: why not a 6-formula contest, what ablation showed instead (§5)
5. Edge latency: measured vs. simulated, framed honestly (§6)
6. What's next, in priority order (`pending_work.md` §1–2)

## 8. Numbers to have ready for Q&A

- 462 curated SANPO valid sessions; 139 (30%, seeded) used in the ablation study.
- 19,402 scored frames, 219 disagreement frames pending referee adjudication.
- 17-class hazard taxonomy; detector v3 mAP50 0.641 / mAP50-95 0.459.
- Reflex-path hard budget: 50 ms p95. Detector-only GB10 measured p95: 9.65 ms. Simulated Jetson
  worst-session p95: 41.96 ms.
- Zero human-labeled ground truth used anywhere in the ablation to date — all results in §5 are
  label-free; the one tier that requires labels (blinded referee) has not run yet.
