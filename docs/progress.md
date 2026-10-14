# CPE Progress

Where the project stands, written so each section maps to one or two slides. The design itself is
`architecture.md`; what's left is `pending_work.md`. Last updated 2026-10-14.

---

## 1. One-line pitch

A phone-based hazard alert for blind pedestrians. It ranks nearby objects by **when** they'll reach
the walker, **how bad** a collision would be, and **what** the scene says about them. Every learned
decision sits inside a deterministic physics safety envelope. It's evaluated by replaying the SANPO
dataset, with the iPhone 17 Pro Max as the target device.

## 2. The design pivot (2026-10-14)

| Before (v1) | Now (v2) | Why |
|---|---|---|
| SLM-1 language model picks the main threat | GRU arrival predictor + segmentation context + fused score | Faster, smaller, testable; GRU beats K0 by +0.32 AUROC |
| K0 kinetic score ranks threats | K0's severity term kept as "how bad" only | K0 predicts arrival barely above chance (AUROC 0.57) |
| Physics "judge" arbitrates SLM vs physics | Physics **safety envelope**: TTC ≤ 1 s always alerts | No SLM left to arbitrate |
| Narrator SLM-2 | Scripted phrases + pre-rendered audio | Deterministic, zero latency |
| Jetson Orin Nano (simulated) | iPhone 17 Pro Max, models measured on the phone | The real target, at no cost |

Full design and diagram: `architecture.md` §3, §5.

## 3. Status at a glance

| Component | Status | Evidence |
|---|---|---|
| SANPO intake (462 curated sessions, streamed from the public bucket) | ✅ Done | `sanpo_dataset.md` |
| Hazard dataset (Roboflow, 17 classes) | ✅ Done | `yolo_training.md` |
| YOLO26n hazard detector v3 | ✅ Done | §4 |
| Perception stack (YOLO + ByteTrack + depth) | ✅ Done | `src/perception_stack/` |
| K0 kinetic score + term ablation (139 sessions) | ✅ Done; K0 now demoted to severity only | §5 |
| GRU arrival predictors (plain and attention), 462 sessions | 🟡 Done offline, not in the runtime yet | §6 |
| Stacked ensemble | 🟡 Done offline | §6 |
| Safety envelope (TTC override) | ✅ Rule built and tested; cone restriction pending | `src/reflex_layer/`, `tests/test_reflex_layer.py` |
| Runtime chain, prioritizer → (stub) ranking → judge → narration | ✅ Synthetic data only | `tools/run_full_pipeline_demo.py` |
| Scripted narration + critical audio lane | ✅ Done, measured at ~0 ms | `src/narration/` |
| Sensor fusion (Kalman, head-motion removal) | ⬜ Not started | `pending_work.md` P1 |
| Walkable-surface segmentation | ⬜ Not started | `pending_work.md` P1 |
| Fused priority score (when × how bad × what) | ⬜ Not started | `pending_work.md` P1 |
| Real SANPO end-to-end replay | ⬜ Not started | `pending_work.md` P2 |
| On-iPhone latency | ⬜ Not started | `pending_work.md` P3 |

## 4. Result: hazard detector

Same held-out test split for all three versions.

| Version | mAP50 | mAP50-95 | What changed |
|---|---:|---:|---|
| v1 | 0.651 | 0.432 | Hazard-only baseline |
| v2 | 0.599 | 0.424 | Continued from v1 on 17 classes; forgot retained COCO classes (esp. `truck`) |
| **v3** | **0.641** | **0.459** | Restarted from the base checkpoint on the same data; repaired most forgetting |

**Slide-worthy finding:** continuing fine-tuning from an already adapted checkpoint caused
catastrophic forgetting. Restarting from the base model on identical data fixed it. This is a
training-protocol result, not a data result. Known weakness: `dog` is still below base YOLO.

Figures: `evaluation/benchmarks/figures/yolo_version_map.png`, `yolo_per_class_map50.png`,
`yolo_retention_f1_delta.png`.

## 5. Result: what's in the K0 kinetic score

K0 = `severity(class) · v² / distance`. Each term was knocked out one at a time on 139 SANPO sessions
(19,402 frames), scored with label-free ranking metrics and session-bootstrap 95% CIs.

| Arm | Verdict |
|---|---|
| no-velocity | **Loses**: rank stability collapses to 0.26–0.52 (from ~1.0). Velocity matters |
| linear (`v` instead of `v²`) | Ties K0 |
| no-severity | Ties K0 |
| size (add box size) | Ties K0 |
| λ = 0.25 / 1.0 (mass exponent) | Struck: not reproducible from current code |

**Takeaway:** kinematics drive the ranking; severity, `v²` and box size don't measurably change it on
this corpus. A later finding qualifies the ablation's encounter results. 99.7% of its encounter
labels were the ground just ahead of the walker. With those removed, K0, TTC and no-velocity all tie.
So the old claim that "TTC is 4× worse" was an artifact and should not be presented.

## 6. Result: learned arrival prediction (the new "when")

**Task:** for each tracked object, will it come within 3 m in the forward cone within the horizon,
and how soon? Data: 462 SANPO sessions (364 with tracked objects, 109 with arrivals), 5-fold
session-level cross-validation, 3 seeds.

| Method | AUROC ↑ | arrival_order ↑ | flicker ↓ |
|---|---:|---:|---:|
| K0 | 0.571 | 0.585 | 0.660 |
| TTC | 0.656 | 0.612 | 0.676 |
| nearest object in the forward cone | 0.871 | 0.681 | 0.727 |
| **GRU** | 0.887 | 0.676 | **0.590** |
| GRU + attention | 0.893 | 0.666 | 0.655 |
| **Stacked ensemble** | **0.894** | **0.688** | — |

**Slide-worthy findings:**
1. **K0 is not an arrival predictor** (AUROC 0.57). That is what motivated the pivot.
2. **The GRU clearly beats physics formulas:** +0.32 AUROC over K0, +0.23 over TTC, all CIs exclude 0.
3. **A one-line geometric rule nearly matches it.** "Nearest object in the cone" ties the GRU on
   arrival order. The limit is the input data, not the model. That's why v2 adds head-motion removal
   and scene context instead of a bigger model.
4. **The GRU's top pick is much steadier** (flicker 0.59 vs. 0.73), which means fewer contradictory
   announcements.
5. **Attention across objects adds nothing measurable** once the model size is matched.

Reports: `evaluation/benchmarks/gru_encounter_eval/run_2026_09_29_arrival462/report.md`,
`evaluation/benchmarks/ensemble_eval/arrival462_2026_09_29/report.md`.

## 7. Result: latency so far

| What | Number | Status |
|---|---|---|
| YOLO26n v3 + tracking + depth, 10 SANPO sessions, GB10 server | avg p95 **7.83 ms**, worst 9.74 ms | Measured (server, not phone) |
| Same, legacy Jetson Orin Nano analytical proxy (4× + 3 ms) | avg p95 34.31 ms | Simulated; **retired** with the phone pivot |
| Scripted narration template | ~0 ms | Measured |
| Piper speech synthesis (CPU), if live synthesis were used | 30–68 ms per phrase | Measured; v2 uses pre-rendered clips instead |
| iPhone 17 Pro Max | — | **Not measured yet**, next step |

Budget for the alert path: p95 < 50 ms.

## 8. Known issue to mention honestly

SANPO-Real reports **15 FPS** and a **58.4°** field of view (checked on one session). The pipeline
assumes 30 FPS and 70°. If the CSVs were built at 30 FPS, speeds are 2× too high and the "2 s"
horizon is 4 s. Comparisons between methods stay fair, since all of them saw the same data, but the
physical units need fixing. This is the top item in `pending_work.md`.

## 9. Suggested slide order

1. Problem + pitch (§1)
2. Design pivot: why we dropped the language model (§2)
3. Architecture diagram (`architecture.md` §5), built vs. planned marked
4. Simulation vs. deployment: SANPO replay standing in for the iPhone (`architecture.md` §4)
5. Detector result + training-protocol finding (§4)
6. K0 ablation → "K0 isn't an arrival predictor" (§5, §6 finding 1)
7. GRU / ensemble results and the geometric-rule ceiling (§6)
8. Latency so far + on-device plan (§7)
9. Known issues and limitations (§8, `architecture.md` §9–10)
10. Next steps (`pending_work.md` "Next three")

## 10. Numbers for Q&A

- 462 curated SANPO-Real sessions; 139 (seeded 30%) in the K0 ablation; all 462 in the GRU runs.
- 19,402 scored frames in the ablation.
- 17-class detector, v3 mAP50 0.641 / mAP50-95 0.459.
- GRU AUROC 0.887 vs. K0 0.571, TTC 0.656, nearest-in-cone 0.871; ensemble 0.894.
- GRU flicker 0.59 vs. nearest-in-cone 0.73.
- GRU size ~30k parameters (attention model ~64k); checkpoints 130–260 KB.
- Alert-path budget p95 < 50 ms; measured server p95 7.83 ms; iPhone not yet measured.
- Zero human labels used so far; the referee + human calibration step hasn't run.
