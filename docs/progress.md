# CPE Progress

Where the project stands, written so each section maps to one or two slides. The design itself is
`architecture.md`; what's left is `pending_work.md`. Last updated 2026-10-14 (evening).

---

## 1. One-line pitch

A phone-based hazard alert for blind pedestrians. It ranks nearby objects by **when** they'll reach
the walker, **how bad** a collision would be, and **what** the scene says about them. Every learned
decision sits inside a deterministic physics safety envelope. It's evaluated by replaying the SANPO
dataset, with the iPhone 17 Pro Max as the target device.

## 2. The design pivot (2026-10-14)

| Before (v1) | Now (v2) | Why |
|---|---|---|
| SLM-1 language model picks the main threat | GRU arrival predictor + segmentation context + fused score | Faster, smaller, testable; GRU beats K0 by +0.32 AUROC held-out |
| K0 kinetic score ranks threats | K0's severity term kept as "how bad" only | K0 predicts arrival barely above chance (AUROC 0.57) |
| Physics "judge" arbitrates SLM vs physics | Physics **safety envelope**: TTC ≤ 1 s in the walking cone always alerts | No SLM left to arbitrate |
| Narrator SLM-2 | Scripted phrases + pre-rendered audio | Deterministic, zero latency |
| Jetson Orin Nano (simulated) | iPhone 17 Pro Max, models measured on the phone | The real target, at no cost |

Full design and diagram: `architecture.md` §3, §5.

## 3. Status at a glance

| Component | Status | Evidence |
|---|---|---|
| SANPO intake (462 curated sessions, streamed from the public bucket) | ✅ Done | `sanpo_dataset.md` |
| SANPO audit: real fps / intrinsics / official split / mask coverage, all 462 sessions | ✅ Done | §8, `tools/audit_sanpo_sessions.py` |
| Hazard dataset (Roboflow, 17 classes) | ✅ Done | `yolo_training.md` |
| YOLO26n hazard detector v3 (the default everywhere) | ✅ Done | §4 |
| Perception stack (YOLO + ByteTrack + depth), real per-session fps and focal length | ✅ Done | `src/perception_stack/` |
| K0 kinetic score + term ablation (139 sessions) | ✅ Done; K0 demoted to severity only | §5 |
| GRU arrival predictors, held-out on SANPO's official test split | ✅ Done | §6 |
| Sensor fusion (Kalman per track, ground plane, ego-motion removal) | ✅ Built + unit-tested; ego-motion unvalidated on SANPO | §8, `src/threat_engine/fusion.py` |
| Safety envelope (TTC ≤ 1 s, walking cone, fused velocity) | ✅ Built + edge cases tested | `src/threat_engine/envelope.py` |
| Fused priority (when × how bad^γ × what) with runtime GRU | ✅ Built; "what" waits for segmentation | `src/threat_engine/priority.py` |
| Announcement manager (cooldown, hysteresis, preemption, stale drop) | ✅ Built + tested | `src/threat_engine/announcer.py` |
| End-to-end SANPO replay, v1 vs v2, official test split | ✅ Done | §7 |
| Scripted narration, distance buckets, phrase list for pre-rendering | ✅ Done | `src/narration/` |
| Walkable-surface segmentation | 🟨 Data, context, metrics, oracle check built; model not trained yet | `src/segmentation/`, `pending_work.md` P1.1 |
| Pre-rendered audio clips | ⬜ Phrase list ready, audio not rendered | `pending_work.md` P4 |
| On-iPhone latency | ⬜ Not started (Core ML export option added) | `pending_work.md` P3 |

Tests: 51 unit tests, all passing (`python -m unittest discover -s tests`).

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

**Task:** for each tracked object, will it come within 3 m in the forward cone within **4 s**, and
how soon? Input: the object's last 1.6 s (8 steps at 5 Hz). Trained on SANPO's **official train**
sessions; scored once on its **71 official test** sessions (25 with arrivals, 4,721 objects); 3 seeds.

| Method | AUROC ↑ | arrival_order ↑ | flicker ↓ |
|---|---:|---:|---:|
| K0 | 0.571 | 0.563 | 0.656 |
| TTC | 0.661 | 0.597 | 0.680 |
| TTC in the forward cone (new, fair physics baseline) | 0.765 | 0.683 | 0.673 |
| nearest object in the forward cone | 0.875 | 0.711 | 0.714 |
| **GRU** | **0.888** | 0.697 | **0.561** |
| GRU + attention | 0.892 | 0.670 | 0.645 |
| Stacked ensemble (fitted on validation) | 0.893 | 0.708 | 0.628 |

Paired differences, GRU minus baseline (95% session-bootstrap CI): K0 **+0.317** [+0.219, +0.384],
TTC **+0.227** [+0.154, +0.279], cone TTC **+0.123** [+0.075, +0.151], nearest-in-cone +0.013
[−0.018, +0.046] (tie).

**Slide-worthy findings:**
1. **K0 is not an arrival predictor** (AUROC 0.57). That is what motivated the pivot.
2. **The GRU beats every physics formula**, including the fair cone-restricted TTC; all CIs exclude 0.
3. **It generalises:** held-out AUROC 0.888 matches 5-fold cross-validation (0.887).
4. **A one-line geometric rule ties it on accuracy.** "Nearest object in the cone" is within the CI.
   The limit is the input data, not the model.
5. **The GRU's top pick is much steadier** (flicker 0.56 vs 0.71), which means fewer contradictory
   announcements; §7 shows this at system level.
6. **Attention across objects doesn't help.** It is significantly *worse* on arrival order
   (−0.028 [−0.044, −0.003]); the ensembles add nothing significant held-out.

Reports: `evaluation/benchmarks/heldout_official_split/official462_2026_10_14/report.md` (held-out),
`evaluation/benchmarks/gru_encounter_eval/run_2026_09_29_arrival462/report.md` (cross-validation).

## 7. Result: end-to-end replay, v1 vs v2 (system level)

Every official test session replayed frame by frame through each engine (71 sessions, 88 arrivals).
Each v2 ignore threshold was chosen on 41 validation sessions first.

| Engine | Warned about arrivals ↑ | Precision ↑ | Alerts / min | Overrides / min | Topic switches / min ↓ |
|---|---:|---:|---:|---:|---:|
| v1 (K0 routing, as built) | 0.65 | 0.24 | 119 | 39 | 61 |
| v1 + v2 announcement manager | 0.53 | 0.22 | 49 | 27 | 40 |
| v2, physics arrival | 0.30 | 0.27 | 29 | 6.5 | 22 |
| **v2, GRU arrival** | 0.26 | **0.59** | **17** | **6.5** | **12** |

Precision = the announced object is within 3 m in the cone now or within 4 s. CIs in the report.

**Slide-worthy findings:**
1. **v1 is unusable as a product:** it talks twice a second and three of four alerts are about
   objects that never come close.
2. **v2 with the GRU is 2.5× more precise and 7× quieter**, and changes topic 5× less often.
3. **The cost is recall:** v2 warns about 26% of arrivals vs v1's 65%, at a validation-tuned
   threshold. Where to sit on this trade-off is the next tuning question (`pending_work.md`).
4. **Lead time is ~0 s for every engine:** objects usually first appear already near the 3 m zone.
   A tighter zone or longer range is needed before lead time is a meaningful metric.

Report: `evaluation/benchmarks/threat_engine_replay/official_2026_10_14/report.md`.

## 8. Data facts established today (all 462 sessions)

- **15 FPS everywhere; the old CSVs assumed 30.** Recovered from the data itself: stored speed =
  10.0 × the per-step depth slope (IQR 9.99–10.01), i.e. 10 Hz assumed for what is 5 Hz. Stored speeds
  are 2× too high and every old "2 s" is really 4 s. Rankings and step-based labels are unaffected,
  so every metric above stands. Stage 1 now reads the real rate.
- **The head camera's real field of view is 70–78°** (ZED-M), close to the 70° assumed. Stage 1 now
  uses each session's focal length. (The 2026-10-14 audit read the chest ZED 2i, 51–60°, by mistake;
  fixed 2026-10-17 and the corridor labels rebuilt.) The ±30° "cone" covers most of the image.
- **SANPO has an official split**: 374 train / 88 test of our sessions. All final numbers use it.
- **Segmentation masks:** 53 sessions on the head camera (43 train, 10 test) and 130 on the chest
  camera (102 train, 28 test; 10,840 human-annotated frames).
- **Camera poses don't reconcile with box depth yet.** Ego-motion removal is built and unit-tested
  but unvalidated on SANPO; it is reported as an ablation (`architecture.md` §9).

## 9. Latency so far

| What | Number | Status |
|---|---|---|
| YOLO26n v3 + tracking + depth, 10 SANPO sessions, GB10 server | avg p95 **7.83 ms**, worst 9.74 ms | Measured (server, not phone) |
| Same, legacy Jetson Orin Nano analytical proxy (4× + 3 ms) | avg p95 34.31 ms | Simulated; **retired** with the phone pivot |
| Threat engine (fusion + GRU + priority + announcer), whole replay | 71 test sessions × 7 engines + 41 val sessions × 30 settings in 38 s total on a laptop CPU | Measured (CPU, not phone) |
| Scripted narration template | ~0 ms | Measured |
| iPhone 17 Pro Max | — | **Not measured yet**, next step |

Budget for the alert path: p95 < 50 ms. Real-time budget per processed frame: 200 ms (5 Hz).

## 10. Suggested slide order

1. Problem + pitch (§1)
2. Design pivot: why we dropped the language model (§2)
3. Architecture diagram (`architecture.md` §5), built vs. planned marked
4. Simulation vs. deployment: SANPO replay standing in for the iPhone (`architecture.md` §4)
5. Detector result + training-protocol finding (§4)
6. K0 ablation → "K0 isn't an arrival predictor" (§5, §6 finding 1)
7. Held-out GRU results and the geometric-rule ceiling (§6)
8. System replay: v1 vs v2, precision vs recall (§7)
9. Data audit: what we found and fixed (§8)
10. Latency so far + on-device plan (§9)
11. Limitations (`architecture.md` §10) and next steps (`pending_work.md` "Next three")

## 11. Numbers for Q&A

- 462 curated SANPO-Real sessions; official split 374 train / 88 test (71 test sessions have tracked
  objects); 139 (seeded 30%) in the K0 ablation.
- 15 FPS, stride 3 → 5 Hz processing; 4 s arrival horizon; 1.6 s GRU history; 3 m hazard zone.
- 17-class detector, v3 mAP50 0.641 / mAP50-95 0.459.
- Held-out GRU AUROC 0.888 vs K0 0.571, TTC 0.661, cone TTC 0.765, nearest-in-cone 0.875.
- Replay: v2 precision 0.59 vs v1 0.24; 17 vs 119 alerts/min; recall 0.26 vs 0.65.
- GRU size ~31k parameters (attention model ~64k); checkpoints ~130–260 KB.
- Alert-path budget p95 < 50 ms; measured server p95 7.83 ms; iPhone not yet measured.
- Zero human labels used so far; the referee + human calibration step hasn't run.
