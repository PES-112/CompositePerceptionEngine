# CPE Progress

Where the project stands, written so each section maps to one or two slides. The design itself is
`architecture.md`; what's left is `pending_work.md`. Last updated 2026-11-02.

Sections 4–8 are the 2026-10-14 snapshot and are kept for the slide story. The current numbers
(head-camera rerun, IDD fine-tunes, conformal budgets) are in `methodology.md` §4.18–4.26 and
`CHANGELOG.md`. §3, §9 and §11 are current.

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
| Walkable-surface segmentation (YOLO26n-sem, SANPO + IDD) | ✅ Trained; on the phone with a road bias. Context has no measurable replay effect (§4.19) | `methodology.md` §4.18, §4.26 |
| Surface narration, "Curb ahead", "Obstacle ahead", stairs from height | ✅ Built + tested; edge memory tried, no gain | `methodology.md` §4.23–4.25 |
| Conformal alert budgets (CRC) | ✅ Held on test at 5/min for every arm | `methodology.md` §4.23 |
| IDD (Indian street) fine-tunes: segmenter + detector v4 | ✅ Done; phone runs v4 + v3 for static classes | `methodology.md` §4.26 |
| Pre-rendered audio clips | 🟨 Phone speaks with the system voice (`Speech.swift`), stereo-panned; clip library not rendered | `pending_work.md` P4 |
| On-iPhone model latency (YOLO26n v3 + GRU, `ios/CPETiming`) | ✅ Measured 2026-10-21, iPhone 17 Pro Max | §9 |
| CPEWalk iPhone app (whole chain on device) | ✅ Builds; golden parity test passes; one field walk (2026-10-27) | `ios/CPEWalk/README.md` |
| Whole-app per-frame latency (CPEWalk, two detectors + segmenter) | ⬜ Not recorded; the app only shows it on screen | §9, `pending_work.md` 2 |

Tests: 119 Python unit tests, all passing (`python -m unittest discover -s tests`, 2026-11-02), plus
the Swift golden-session and field-fix tests in `ios/CPEWalk/Tests`.

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
| iPhone 17 Pro Max: YOLO26n v3 + GRU (Core ML FP16), 30 min sustained | median **32.0 ms**, p95 **36.9 ms** per frame (models alone p95 8.6 ms) | **Measured on the phone** (§9.1) |
| iPhone 17 Pro Max: whole CPEWalk app (two detectors, segmenter, fusion, engine) | — | **Not recorded yet** (§9.2) |

Budget for the alert path: p95 < 50 ms. Real-time budget per processed frame: 200 ms (5 Hz).

### 9.1 On-device performance (iPhone 17 Pro Max, measured 2026-10-21)

`ios/CPETiming` on an iPhone 17 Pro Max (iPhone18,2, iOS 26.6.2). It runs the YOLO26n v3 detector and
the arrival GRU (both Core ML FP16) on 300 bundled SANPO test frames and recorded SANPO features,
paced at 15 FPS, 60 warm-up frames excluded. Two 1,500-frame benchmark runs and one 30-minute
sustained run (27,000 frames). Raw CSVs and the generated report:
`evaluation/benchmarks/ondevice_latency/iphone17promax_2026_10_21/`.

Sustained 30-minute run, per frame:

| Stage | Median ms | p95 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|
| JPEG decode + resize (`pre`) | 24.10 | 28.91 | 30.23 | 37.11 |
| Detector, YOLO26n v3 (`det`) | 5.47 | 5.73 | 5.96 | 9.23 |
| Reading detector output (`det_post`) | 2.01 | 2.41 | 2.62 | 4.79 |
| Arrival GRU (`gru`) | 0.47 | 0.53 | 0.58 | 2.08 |
| **Total** | **31.95** | **36.87** | **38.25** | **42.58** |
| Models only (det + det_post + gru) | 7.94 | 8.59 | — | — |

The two benchmark runs agree to within 0.2 ms at the median (total median 32.26 and 32.28 ms, p95
35.78 and 35.89 ms).

| What | Result |
|---|---|
| Frames over the 50 ms alert-path budget | 0 of 27,000 |
| Headroom at 5 Hz processing (200 ms per frame) | p95 uses 18% of the budget |
| Throttling over 30 min | None: p95 per 5-min window 36.3–37.3 ms, no upward trend |
| Thermal state | Nominal for 100% of frames |
| Battery | 85% at start and end. iOS reports the level coarsely and the charging state was not logged, so no battery claim is made |

Reading the numbers:
- **Most of the time is decoding the bundled JPEGs** (24 ms of 32 ms). The live app gets camera
  pixel buffers from ARKit, so it has no JPEG decode; it has a resize and the other stages below
  instead.
- **The models are cheap:** detector + GRU fit in under 9 ms at p95.
- **Scope:** two models on SANPO frames, not the whole app. Not included: the second (IDD) detector,
  the segmenter (Ultralytics reports 4.9 ms for YOLO26n-sem on iPhone 17 Pro, not measured by us),
  ByteTrack, LiDAR depth lookup, Kalman fusion, the engine, speech.

### 9.2 Whole-app performance (CPEWalk) — not recorded yet

Since 2026-10-28 every frame runs two detectors (v4 IDD + v3 legacy), so the CPETiming numbers
above are a lower bound for the app. Since 2026-11-02 CPEWalk logs every processed frame to a CSV:
the time of each stage, the total, the thermal state, the battery level and the charging state
(`ios/CPEWalk/README.md`, "Timing log"). This build is installed on the iPhone 17 Pro Max. No walk
has been recorded with it yet, so there are no numbers here. After the next walk, summarise the CSV
with `ios/CPETiming/analyze_timing.py`.

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
10. On-device performance on the iPhone 17 Pro Max (§9)
11. Limitations (`architecture.md` §10) and next steps (`pending_work.md` "Next three")

## 11. Numbers for Q&A

- 462 curated SANPO-Real sessions; official split 374 train / 88 test (71 test sessions have tracked
  objects); 139 (seeded 30%) in the K0 ablation.
- 15 FPS, stride 3 → 5 Hz processing; 4 s arrival horizon; 1.6 s GRU history; 3 m hazard zone.
- 17-class detector, v3 mAP50 0.641 / mAP50-95 0.459.
- Held-out GRU AUROC 0.888 vs K0 0.571, TTC 0.661, cone TTC 0.765, nearest-in-cone 0.875.
- Replay: v2 precision 0.59 vs v1 0.24; 17 vs 119 alerts/min; recall 0.26 vs 0.65.
- GRU size ~31k parameters (attention model ~64k); checkpoints ~130–260 KB.
- Alert-path budget p95 < 50 ms; measured server p95 7.83 ms; iPhone 17 Pro Max detector + GRU p95
  36.9 ms per frame including JPEG decode (8.6 ms for the models), flat over 30 min, thermal nominal.
  Whole-app latency not recorded yet.
- Zero human labels used so far; the referee + human calibration step hasn't run.
