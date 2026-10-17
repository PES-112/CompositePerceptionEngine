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
2. **Re-choose operating points on lead, not `warned` (P2.2).** `warned` counts announcements on the
   arrival frame. At the current budgets only 0.08–0.14 of arrivals are warned ≥ 0.2 s ahead, and
   ≤ 0.06 ≥ 1 s ahead (methodology §4.15). Pick budgets on `warned_before`, and sweep the announcer
   (cooldown, switch margin, stale drop) as well as the threshold.
3. **Walkable-surface segmentation (P1.1).** It's the missing "what" and the real fix for the
   ground pseudo-objects. 53 sessions have SANPO masks (43 train / 10 test).

---

## Done on 2026-10-17 (branch `v2-kalman-lidar`)

- Kalman noise fitted on train (Student-t); Kalman arrival predictor; replay before/after
  (methodology §4.13).
- Baseline ladder on the corridor label (methodology §4.14).
- LiDAR depth-realism bound at 5 / 4 / 3 m, plus lead-based recall columns (methodology §4.15). This
  covers P2.5.
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

1. **Rebuild Stage-1 CSVs at the correct rate (optional).** Existing CSVs work once converted (×0.5
   speed, pinhole bearing), and every ranking metric is unaffected. A rebuild with the updated
   `tools/stream_sanpo_perception.py` gives clean units for the paper and is needed anyway for
   ground masking (P1.1) and the chest camera (P2.6). Needs the GPU box (ultralytics).
2. **Reconcile camera poses with box depth.** With either `camera_poses.csv` or
   `fixed_camera_poses.csv`, near static objects move more per step in the world frame than in the
   camera frame. Check, in order: pose-to-frame timing (are poses per RGB frame of the *left*
   camera?), the camera's position vs. the left lens, pose drift during `NOT_READY` stretches, and
   whether the depth maps are z-depth or range. Until resolved, ego-motion removal stays an ablation.

## P1. Build the rest of the v2 engine

1. **Walkable-surface segmentation (§6.3).**
   - Fine-tune a small model on SANPO-Synthetic + the 43 annotated real train sessions (road, curb,
     sidewalk, crosswalk, stairs, walkable / inaccessible surface); report mIoU on the 10 test ones.
   - Fill `SceneContext` (in path, near boundary, other surface) per track; add the scene-level
     "distance to walkable edge" alert.
   - Mask walkable ground out of `detect_unlabeled_obstacles`.
2. **"In my path" as a lateral corridor, not a cone.** With the real FOV the ±30° cone covers nearly
   the whole image. Use |lateral| < ~1 m on the ground plane (fusion already provides it) for the
   envelope and the label; re-run the held-out evaluation.
3. **GRU inputs from fusion.** Retrain the GRU on fused ground-plane features (signed closing speed,
   lateral speed, position uncertainty) and, once available, segmentation features. Compare against
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
5. **Optional: depth realism bound.** Re-run with SANPO depth truncated at 5 m and downsampled to
   approximate iPhone LiDAR.
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

1. Render the clip library offline: `templates.all_phrases(classes)` lists every phrase; render each
   per language with a high-quality voice and load them into `CachedClipTTS`.
2. Add scripted segmentation phrases ("Curb ahead.", "Crosswalk ahead.") once P1.1 exists.
3. Optional: stereo panning toward the object's bearing.

## P5. Paper

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
