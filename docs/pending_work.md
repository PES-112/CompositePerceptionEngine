# Pending Work

Everything not yet done for the v2 design (`architecture.md`), in priority order. When an item is
finished, move it to `progress.md` and log it in `CHANGELOG.md`. Last updated 2026-10-14 (evening).

Priority key: **P0** blocks correct numbers · **P1** builds the v2 engine · **P2** evaluation ·
**P3** iPhone · **P4** narration · **P5** paper.

---

## Next three

1. **Measure on the iPhone (P3.1–P3.2).** It's the one headline claim with no number yet. The
   timing app, the GRU export and the CI workflow exist (`ios/CPETiming/README.md`). What's left: an
   Apple Developer account, the App Store Connect secrets, a green CI run, and the phone.
2. **Make warnings earlier (P2.2).** Operating points are now chosen on `warned_before` and the
   announcer was swept (done 2026-10-19, methodology §4.17). Neither creates lead: ≤ 0.22 of
   arrivals are warned ≥ 0.2 s ahead. The predictors are the limit. Next: a GRU objective that
   rewards early probability (time-to-arrival-weighted loss or a "≥ 1 s ahead" target) and a
   rising-P(arrive) announcement trigger.
3. **Walkable-surface segmentation (P1.1).** Trained (YOLO26n-sem @640 on the M4, 2026-10-18) and
   re-run with its masks. Context: no measurable effect. Edge alerts: well below the oracle on
   held-out sessions (1 of 7 step-offs warned vs 3 of 7). Next: n at a larger imgsz on a GPU
   machine (not started).

## Integration and the 4-day plan (2026-10-19 to 2026-10-22)

`v2-threat-engine` is the integration branch.

| Branch | Owner | State (2026-10-20) |
|---|---|---|
| `v2-kalman-lidar` | teammate | Depth-decoder fix, fitted Kalman, latency replay, and the head-camera Stage-1 rerun (methodology §4.18). Contains all of `v2-threat-engine`, so it fast-forwards. |
| `v2-segmentation` | segmentation | Contains `v2-seg-training` and `v2-kalman-lidar` up to the head-camera rerun (`767dd88`; conflicts resolved, segmentation renumbered §4.18 → §4.19), plus the depth fix for the segmentation scripts (`a5f7b4e`) and the model's edge alerts on correct depth. PR #7. |
| `v2-seg-training` | Mac (M4) | Retired, folded into `v2-segmentation`. |
| `v2-narration` | narration | English clip library; translation is future work. |

Order:
1. Fast-forward `v2-threat-engine` to `v2-kalman-lidar`.
2. Merge PR #7. It now merges cleanly.
3. ~~When the head-camera rebuild lands, merge `v2-threat-engine` into `v2-kalman-lidar`, then
   fast-forward again.~~ The rebuild landed on `v2-kalman-lidar` itself, so step 1 covers it; it
   was merged into `v2-segmentation` on 2026-10-20. Scene context and context replays read bearings from the Stage-1 datasets, so they run on
   `v2-segmentation` from this merge on (fresh `data/sanpo_scene_context_v2/`).
4. `v2-narration` merges via its own PR.

**Superseded by the depth-decoding bug** (every SANPO depth map was read as a scrambled 960×960
array until 2026-10-17; `architecture.md` §9). Kept for the record:
- `scene_context/oracle_gt_*`, `gt_train_*`, `gt_val_test_*`, `model_val_test_*`;
- `threat_engine_replay/oracle_context_*`, `model_context_*`;
- `edge_alerts/*` before `*_depthfix_*`;
- the near-boundary decision (×1.25 → ×1.0), which must be redone on train sessions;
- the cached tracks in `data/sanpo_scene_context/` and `data/sanpo_edge_track/` (the `_v2` roots
  replace them).

Unaffected: segmentation training, its test scores, and the Core ML export (no depth involved).

| Day | Teammate | Segmentation (DeviousDrops) | Mac (M4) | Narration |
|---|---|---|---|---|
| 1 (10-19) | Fast-forward `v2-threat-engine`; start the head-camera rebuild | Merge + depth fix; oracle edge alerts on correct depth | Model edge alerts on correct depth | Push branch; phrase review; renderer |
| 2 | Rebuild in: ladder, Kalman, replay | Oracle scene context; near-boundary decision on train; oracle replay | Model scene context, val + test (overnight) | Render; clip durations vs `CLIP_S` |
| 3 | iPhone timing if the account is ready; latency-aware replay; review the announcer PR | "Curb ahead." through the announcer; merge narration; paper numbers | Context on/off replay; model vs oracle; docs | Edge phrases, loader, tests; PR |
| 4 | Final methodology numbers | Paper freeze; merge to `main` | Paper figures (F2–F6) | Latency and size numbers |

---

## Done on 2026-10-17 (branch `v2-kalman-lidar`)

- Kalman noise fitted on train (Student-t); Kalman arrival predictor; replay before/after
  (methodology §4.13).
- Baseline ladder on the corridor label (methodology §4.14).
- Lead-based recall columns in the replay (methodology §4.15). Depth stays SANPO's, as is; the
  LiDAR-range simulation was removed.
- On-device timing harness, GRU Core ML export and the latency-aware replay. This covers P3.1 (the
  GRU part), P3.3 and the tooling for P3.2.
- VLM referee protocol v2 (P2.4 tooling) and the BLV listening-study plan.

## Done on 2026-10-14 (moved to `progress.md`)

- P0 data audit: 15 FPS everywhere (old CSVs assumed 30; units corrected), real intrinsics in
  Stage 1, SANPO's official split adopted, mask coverage counted.
- Held-out GRU / ensemble evaluation on the official test split, plus the cone-restricted TTC
  baseline.
- v2 threat engine: Kalman fusion, safety envelope with cone + fused velocity and edge-case tests,
  fused priority with runtime GRU, announcement manager.
- End-to-end SANPO replay of v1 vs v2 with system metrics.
- Narration distance buckets and the pre-render phrase list; Core ML export option; SLM-1
  fact-sheet code retired.

## P0. Data correctness

0. **~~Re-run Stage 1 with head-camera fx and the fixed size check~~ (done 2026-10-20, methodology
   §4.18).** The 2026-10-18 rebuild used the chest camera's fx and a mis-scaled depth lookup
   (architecture §9 items 8–9). Re-run with `notebooks/colab_rebuild.ipynb` (`RUN_TAG = "headfx"`;
   `tools/rebuild_local.py` does the same off Colab); the viewer threshold was re-picked. Follow-up:
   the fitted Kalman no longer beats the declared one on the test ladder (no credible difference).
   History of the first rebuild:
   **~~Rebuild everything on correctly decoded depth~~ (done 2026-10-18, methodology §4.16).** The
   SANPO depth loader read each 720×1280 map as a scrambled 960×960 array, and depth-sweep boxes
   masked real objects (`architecture.md` §9 item 0). Both were fixed in code (2026-10-17), then:
   1. Re-run Stage 1 on all 462 sessions (`tools/stream_sanpo_perception.py`; GPU box).
   2. Rebuild the GRU datasets and corridor labels (`dataset/GRU/build.py`, `resplit.py`,
      `relabel.py`).
   3. Retrain the GRUs (3 seeds per variant).
   4. Refit the Kalman noise (`tools/fit_kalman_noise.py`).
   5. Re-run the ladder, the replays and the before/after comparison.

   Steps 1–5 ran as one pass of `notebooks/colab_rebuild.ipynb`. The viewer's legacy-input shim is
   gone. Follow-ups it raised:
   - ~~**The fitted Kalman no longer helps.**~~ Fixed 2026-10-19: refitted on 1 s-ahead likelihood
     (methodology §4.17); a units bug in the label and replay tools was fixed at the same time.
   - ~~**Alert budgets.**~~ Re-tuned on `warned_before` with an announcer sweep (§4.17). Validation
     still under-predicts test false alerts (≤ 5 → 6.0–7.4 per minute).
   - **Not finished:** the Student-t half of the 1 s Kalman fit (stopped twice by low memory).
1. **~~Rebuild Stage-1 CSVs at the correct rate~~ (done by the P0.0 rebuild).** Existing CSVs work once converted (×0.5
   speed, pinhole bearing), and every ranking metric is unaffected. A rebuild with the updated
   `tools/stream_sanpo_perception.py` gives clean units for the paper and is needed anyway for
   ground masking (P1.1) and the chest camera (P2.6). Needs the GPU box (ultralytics).
2. **Reconcile camera poses with box depth.** With either `camera_poses.csv` or
   `fixed_camera_poses.csv`, near static objects move more per step in the world frame than in the
   camera frame. Check, in order: pose-to-frame timing (are poses per RGB frame of the *left*
   camera?), the camera's position vs. the left lens, pose drift during `NOT_READY` stretches, and
   whether the depth maps are z-depth or range. Until resolved, ego-motion removal stays an ablation.

## P1. Build the rest of the v2 engine

1. **Walkable-surface segmentation (§6.3).** Done: 8-class taxonomy, dataset builder (237 masked
   sessions), `SceneContext` per track + edge ahead, pixel metrics (false-safe rate, boundary F),
   oracle replay and flag-association analysis. To do, in order:
   - ~~Train and score the model.~~ Done 2026-10-18 (`v2-seg-training`): YOLO26n-sem @640, test
     mIoU 0.615 [0.540, 0.641], false-safe 0.044 [0.029, 0.062], curb / stairs boundary F 0.30 /
     0.06 (methodology §4.19).
   - ~~Model context over all 112 val/test sessions and the powered replay.~~ Done 2026-10-18:
     context on vs off, warned +0.04 [−0.06, +0.08] at `budget_10`, no measurable effect.
   - **Next:** n at a larger imgsz (≥ 1024) on a GPU machine, for curb and stair edges. Then re-run
     the edge alerts and score the 10 official-test sessions.
   - ~~Decide the multipliers from train sessions only if the declared ones are contradicted.~~
     Done 2026-10-17: near a boundary ×1.25 → ×1.0 (train-only check; methodology §4.19).
   - Stage 1 on the chest stream for the 130 chest-masked curated sessions (lab server), to get an
     oracle bound on 28 test sessions instead of 10.
   - Edge-ahead alert rule and its oracle evaluation: done 2026-10-17
     (`src/segmentation/edge_alert.py`, `tools/evaluate_edge_alerts.py`). Left: play it through the
     announcement manager as a normal-lane phrase (threat-engine owner) and the phrases themselves
     (`v2-narration`). Re-scored with the model's masks 2026-10-18: far below the oracle.
   - Mask flat ground out of `detect_unlabeled_obstacles` (live pipeline only; the replay uses
     tracked objects).
2. **"In my path" as a lateral corridor, not a cone.** With the real FOV the ±30° cone covers nearly
   the whole image. Use |lateral| < ~1 m on the ground plane (fusion already provides it) for the
   envelope and the label; re-run the held-out evaluation.
3. **GRU inputs from fusion.** Retrain the GRU on fused ground-plane features (signed closing speed,
   lateral speed, position uncertainty) and segmentation features (surface under the object, from
   `src/segmentation/context.py`). Compare against
   the current GRU on the official split.
4. **Retire the v1 chain** once the replay comparison is in the paper: `src/threat_prioritizer/`
   routing, `src/cognitive_layer/stub.py`, Physics Verification rules 2–3 and the PPO reward
   constants. Keep `NarratorEvent` (move it next to narration).

## P2. Evaluation (official test split)

1. **Ablations on the replay:** segmentation context on/off; ego-motion on/off (already reported,
   pending P0.2); GRU history 8 / 16 / 20 steps; γ (severity exponent) 0 / 0.25 / 0.5 / 1.
2. **Operating-point curve.** Sweep ignore threshold, cooldown and switch margin on validation;
   report warned-vs-precision and warned-vs-alerts/min curves for v1 and v2 (see "Next three").
3. **Lead time that means something.** Most objects first appear near the 3 m zone, so lead time
   is ~0 s for every engine. Report lead time for a tighter zone (1.5 m) and per class.
4. **Referee on announcement order.**
   - Blinded 3-VLM referee (`evaluation/vlm_referee.py`) on frames where v2's top pick differs from
     K0's and from nearest-in-cone's.
   - Plus a 100–150-frame human calibration set with Cohen's κ.
   - Needs GPU-served local VLMs. Run guide: `ablation_guide.md` §4.
5. **Dropped: depth realism bound.** Depth stays SANPO's, as is. The product uses an external
   LiDAR, so an iPhone-LiDAR simulation doesn't apply. Revisit once the sensor is chosen.
6. **Optional: chest camera.** Re-run Stage 1 on SANPO `camera_chest/` for the more phone-like
   mounting.

## P3. iPhone 17 Pro Max

1. **Export to Core ML.**
   - YOLO26n v3: `python training/scripts/export_yolo26n_edge.py --formats coreml --quantize 16`
     (option added; run it on a Mac).
   - The GRU: trace `models/GRU/GRU` and convert with `coremltools` (script not written yet).
   - The segmentation model, once it exists.
2. **Measure on the phone.** Run each model on the iPhone 17 Pro Max over SANPO frames, then report
   median/p95 latency and a sustained (several-minute) run to expose thermal throttling. Xcode's
   Core ML performance report on a connected device is the simplest route. It **needs a Mac with
   Xcode**; confirm one is available.
3. **Latency-aware replay.** Delay each alert in `tools/replay_threat_engine.py` by the measured
   on-device latency.
4. **Optional demo app.** An on-device replay of one SANPO session: bundled frames + depth, models
   run live, audio clips played. Only needed for a live demo; P3.2 alone is enough for the paper.
5. **Update the retired Jetson tooling.** `tools/benchmark_edge_realtime.py` still carries the
   Jetson proxy profile. Keep it for the server measurement and drop the proxy from reporting.

## P4. Narration

1. Render the English clip library offline: `templates.all_phrases(classes)` lists every phrase;
   render each with a high-quality voice and load them into `CachedClipTTS`. **Translation (Hindi,
   Kannada) is future work** (decided 2026-10-19). The phrase-table code stays, but no translated
   library is built for the paper.
2. Add scripted segmentation phrases ("Curb ahead.", "Crosswalk ahead.") once P1.1 exists.
3. Optional: stereo panning toward the object's bearing.

## P5. Paper

Draft: `paper_draft.md` (2026-10-17). It holds the text, the results that are final, a status table
of every pending result with its owner, and the figure list.

1. Rewrite `methodology.md` §4 around the held-out and replay results (§5–6 already describe the
   procedures).
2. Limitations (`architecture.md` §10): replay not deployment, dataset depth, declared λ / γ /
   behaviour multipliers / announcer constants, head-mounted camera, unvalidated ego-motion.
3. Regenerate figures (`evaluation/generate_report_figures.py`); add held-out, replay and on-device
   latency figures.

## Carried over from v1 (still relevant, lower priority)

- **Detector:** `dog` retention is still below base YOLO, so it needs more data. Gap-analysis
  findings haven't fed a v4 round.
- **Detector evaluation** has no uncertainty estimate: no multi-seed runs, no CIs on mAP.
- **Behaviour multipliers** in `physics.py` have no literature or data behind them. Ground them or
  state them as a limitation.
- **Stratified severity check** (`evaluation/kinetic_ablation_stratified.py`). Still the only test
  of whether severity matters in the rare frames it was designed for. It now informs γ.

## Dropped with the v2 pivot

| Item | Why dropped |
|---|---|
| SLM-1 adapter, SFT data, PPO training, fact-sheet prompt format | Language model removed from the runtime |
| Narrator SLM-2 (Phi-4-mini), live speech synthesis on device | Replaced by scripted phrases + pre-rendered clips |
| IndicTrans2 runtime translation | Phrases are pre-translated |
| Physical Jetson validation, Jetson 4× proxy | Target is now the iPhone, measured directly |
| K0-specific referee on the 219 ablation disagreement frames, top-3 VLM check | K0 no longer ranks; the referee now judges v2's announcement order (P2.4) |
