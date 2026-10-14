# Pending Work

Everything not yet done for the v2 design (`architecture.md`), in priority order. When an item is
finished, move it to `progress.md` and log it in `CHANGELOG.md`. Last updated 2026-10-14.

Priority key: **P0** blocks correct numbers · **P1** builds the v2 engine · **P2** evaluation ·
**P3** iPhone · **P4** narration · **P5** paper.

---

## Next three

1. **Fix the frame-rate / field-of-view question (P0.1–P0.2).** Every physical number depends on it.
2. **Build sensor fusion + the fused priority score (P1.1, P1.3)** and swap it in for the cognitive
   stub. That gives a complete v2 engine without waiting for segmentation.
3. **Run the real SANPO replay end-to-end (P2.1)** with v2 vs. the K0-routed v1 pipeline on the test
   split.

---

## P0. Data correctness

1. **Verify the frame rate.** SANPO-Real reported 15 FPS on the one session checked, but
   `tools/stream_sanpo_perception.py` defaults to `--fps 30`.
   - Read `description.json` for all 462 sessions; it's public, metadata only.
   - Confirm which `--fps` the existing Stage-1 CSVs used.
   - If they were built at 30, rebuild the CSVs at the right value, then rebuild the GRU dataset and
     retrain.
2. **Use real camera intrinsics.** SANPO reports a 58.4° horizontal FOV. `compute_bearing()` assumes
   70°, and `yolo_tracker.py` assumes a 960 px focal length. Read both from `description.json` per
   session.
3. **Fix the SANPO train/test split.** There's no split file at the bucket root; check the SANPO
   GitHub/TFDS loader for an official one.
   - Otherwise fix a session-level split once and commit it next to `valid_streams.json`.
   - The current 5-fold cross-validation has seen every session; keep it for development only.
4. **Check segmentation-mask coverage.** Count how many of the 462 real sessions include
   `segmentation_masks/`. If few, train segmentation on SANPO-Synthetic plus the annotated real
   sessions, and state it.

## P1. Build the v2 threat engine

1. **Sensor fusion & tracking (`architecture.md` §6.2).**
   - One Kalman filter per track, in ground-plane coordinates.
   - Remove head motion using `camera_poses.csv`.
   - Output a signed velocity and an uncertainty.
   - Replaces `physics.compute_velocity` and the long-pending "depth smoothing filter".
   - Unit-test it against synthetic tracks with a known head turn.
2. **Walkable-surface segmentation (§6.3).**
   - Train a small model on SANPO labels (road, curb, sidewalk, crosswalk, stairs, walkable /
     inaccessible surface) and report mIoU.
   - Derive the per-track features (surface, in path, near boundary) and the scene feature
     (distance to the walkable edge).
   - Mask walkable ground out of `detect_unlabeled_obstacles`.
3. **Fused priority score (§6.4).**
   - `P(arrive) × severity × context`, plus the ignore threshold.
   - Replaces `src/cognitive_layer/stub.py` behind the same orchestrator call.
   - Fit or tune it on train sessions only.
4. **Safety envelope upgrade.** Restrict the TTC override to the forward walking cone and feed it
   fused (smoothed) velocity. Add the edge-case tests the old backlog asked for: missing depth
   mid-track, exact threshold ties, simultaneous override and ranked objects, zero/negative velocity.
5. **GRU inputs from fusion.** Retrain the GRU on ground-plane, head-motion-free features (+ the
   segmentation features once available). Compare against the current GRU on the same split.
6. **Announcement manager (§6.5).** Per-track cooldown, hysteresis, override preemption, and
   dropping stale messages.
7. **Retire SLM-only code** once the fused ranker is in:
   - Physics Verification rules 2–3 and the PPO reward constants;
   - `src/shared/fact_sheet.py`, `src/perception_stack/fact_sheet_builder.py`,
     `tools/run_fact_sheets.py`.

## P2. Evaluation (SANPO replay, test split)

1. **End-to-end replay** of the full v2 engine over real Stage-1 CSVs
   (`build_threat_frames_from_csv` → orchestrator).
   - System metrics: warning lead time, missed arrivals, alerts per minute, announcement switches
     per minute.
   - Compare against the K0-routed v1 pipeline.
2. **Fairer physics baseline.** Add cone-restricted, smoothed TTC to `models/GRU/evaluate.py`
   baselines and re-score saved runs. No retraining is needed.
3. **Ablations:**
   - segmentation context on/off;
   - head-motion removal on/off;
   - GRU history length 8 / 16 / 20 frames.
4. **Referee on announcement order.**
   - Blinded 3-VLM referee (`evaluation/vlm_referee.py`) on frames where v2's top pick differs from
     K0's and from nearest-in-cone's.
   - Plus a 100–150-frame human calibration set with Cohen's κ.
   - Needs GPU-served local VLMs. Run guide: `ablation_guide.md` §4.
5. **Optional: depth realism bound.** Re-run with SANPO depth truncated at 5 m and downsampled to
   approximate iPhone LiDAR. This bounds how much the "dataset depth" assumption flatters the
   results.
6. **Optional: chest camera.** Re-run Stage 1 on SANPO `camera_chest/` for the more phone-like
   mounting.

## P3. iPhone 17 Pro Max

1. **Export to Core ML.**
   - YOLO26n v3 (Ultralytics `format=coreml`; `training/scripts/export_yolo26n_edge.py` only offers
     onnx / engine / openvino / torchscript today, so add `coreml` to its `--formats` choices);
   - the segmentation model;
   - the GRU (`coremltools`).
2. **Measure on the phone.** Run each model on the iPhone 17 Pro Max over SANPO frames, then report
   median/p95 latency and a sustained (several-minute) run to expose thermal throttling. Xcode's
   Core ML performance report on a connected device is the simplest route. It **needs a Mac with
   Xcode**; confirm one is available.
3. **Latency-aware replay.** Delay each alert in the P2.1 replay by the measured on-device latency.
4. **Optional demo app.** An on-device replay of one SANPO session: bundled frames + depth, models
   run live, audio clips played. This is only needed for a live demo; P3.2 alone is enough for the
   paper's latency claim.
5. **Update the retired Jetson tooling.** `tools/benchmark_edge_realtime.py` still carries the
   Jetson proxy profile. Keep it for the server measurement and drop the proxy from reporting.

## P4. Narration

1. Pre-render the clip library offline: classes × directions × urgency × distance buckets, per
   language.
2. Bucket distances in `templates.py` ("close", "2 meters", "5 meters").
3. Add scripted segmentation phrases ("Curb ahead.", "Crosswalk ahead.", "Car pulling out, left.").
4. Optional: stereo panning toward the object's bearing.

## P5. Paper

1. Rewrite `methodology.md` around the v2 engine once P1–P2 produce results (§1 and §5 already
   updated to the v2 framing).
2. The limitations section (`architecture.md` §10): replay not deployment, dataset depth, declared
   λ / behaviour multipliers, head-mounted camera.
3. Regenerate figures (`evaluation/generate_report_figures.py`) after the FPS fix; add GRU/ensemble
   and on-device latency figures.

## Carried over from v1 (still relevant, lower priority)

- **Detector:** `dog` retention is still below base YOLO, so it needs more data. Gap-analysis
  findings haven't fed a v4 round.
- **Detector evaluation** has no uncertainty estimate: no multi-seed runs, no CIs on mAP.
- **Behaviour multipliers** in `physics.py` have no literature or data behind them. Ground them or
  state them as a limitation.
- **Stratified severity check** (`evaluation/kinetic_ablation_stratified.py`). Still the only test
  of whether severity matters in the rare frames it was designed for. It now informs the "how bad"
  term.

## Dropped with the v2 pivot

| Item | Why dropped |
|---|---|
| SLM-1 adapter, SFT data, PPO training | Language model removed from the runtime |
| Narrator SLM-2 (Phi-4-mini), live speech synthesis on device | Replaced by scripted phrases + pre-rendered clips |
| IndicTrans2 runtime translation | Phrases are pre-translated |
| Physical Jetson validation, Jetson 4× proxy | Target is now the iPhone, measured directly |
| K0-specific referee on the 219 ablation disagreement frames, top-3 VLM check | K0 no longer ranks; the referee now judges v2's announcement order (P2.4) |
