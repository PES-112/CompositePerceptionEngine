# Pending Work

Everything not yet done for the v2 design (`architecture.md`), in priority order. When an item is
finished, move it to `progress.md` and log it in `CHANGELOG.md`. Last updated 2026-10-14 (evening).

Priority key: **P0** blocks correct numbers · **P1** builds the v2 engine · **P2** evaluation ·
**P3** iPhone · **P4** narration · **P5** paper.

---

## Next three

1. **Measure on the iPhone (P3.1–P3.2).** It's the one headline claim with no number yet. Needs a
   Mac with Xcode; the Core ML export option is already in place.
2. **Pick the announcement operating point (P2.2).** v2 is precise but quiet (26% of arrivals warned
   at 17 alerts/min). Sweep the announcer and threshold on validation sessions and present the
   precision/recall curve, not one point.
3. **Walkable-surface segmentation (P1.1).** Data, context, metrics and the oracle check are built;
   what's left is training YOLO26s-sem on the lab server and re-running the replay with its masks.

## Integration: merge order (2026-10-17)

`v2-threat-engine` is the integration branch. Four branches feed it:

| Branch | Owner | Base | State |
|---|---|---|---|
| `v2-segmentation` | segmentation | all of `v2-threat-engine` | merges with no conflicts: **merge first** |
| `v2-kalman-lidar` | teammate | `a5fd0aa`, **without the camera-intrinsics fix** (`4fe42a6` → `430234a`) | must merge `v2-threat-engine` and re-run |
| `v2-seg-training` | Mac (M4) | `v2-segmentation` | training; results only |
| `v2-narration` | narration | `v2-threat-engine` | `src/narration/` only |

1. Merge `v2-segmentation` → `v2-threat-engine` (PR).
2. `v2-kalman-lidar` merges `v2-threat-engine`.
   - Conflicts are additive. In `tools/replay_threat_engine.py`, keep both the context arms and the
     latency delays: one docstring paragraph each, the `SceneContext` and `KalmanArrivalPredictor`
     imports, `session_contexts()` and `load_latencies()`, `--context-dir` and `--latency-*`, and
     both `contexts` and `delays_for(...)` in the scoring loop.
   - The docs conflicts (README, CHANGELOG, architecture, pending_work) are side-by-side sections.
   - Then **re-run everything label-dependent on the corrected labels**: the Kalman noise fit, the
     baseline ladder, the LiDAR bound and the corridor replay. The ladder on that branch reports 20
     test sessions with arrivals and a positive rate of 0.169; the corrected labels give 19 and
     0.140.
3. `v2-seg-training` merges `v2-threat-engine` before its replay step, so the model-context replay
   uses the fitted tracker.
4. `v2-narration` merges last; it only touches `src/narration/`, its tests and §6.6.

---

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

1. **Walkable-surface segmentation (§6.3).** Done: 8-class taxonomy, dataset builder (237 masked
   sessions), `SceneContext` per track + edge ahead, pixel metrics (false-safe rate, boundary F),
   oracle replay and flag-association analysis. To do, in order:
   - On the lab server: `python tools/build_sanpo_segmentation.py --out data/sanpo_walkable_seg`,
     then `training/scripts/train_yolo26_sem_walkable.py` and `evaluate_yolo26_sem_walkable.py
     --split test`. Report mIoU, false-safe rate and curb/stairs boundary F with CIs, per camera.
   - `tools/build_scene_context.py --source <best.pt> --all-sessions` (all 41 val / 71 test sessions,
     not just the 15 with head masks), then the replay with `--context-dir`: the powered on/off test.
   - ~~Decide the multipliers from train sessions only if the declared ones are contradicted.~~
     Done 2026-10-17: near a boundary ×1.25 → ×1.0 (train-only check; methodology §4.14).
   - Stage 1 on the chest stream for the 130 chest-masked curated sessions (lab server), to get an
     oracle bound on 28 test sessions instead of 10.
   - Edge-ahead alert rule and its oracle evaluation: done 2026-10-17
     (`src/segmentation/edge_alert.py`, `tools/evaluate_edge_alerts.py`). Left: play it through the
     announcement manager as a normal-lane phrase (threat-engine owner), the phrases themselves
     (`v2-narration`), and re-scoring with the model's masks.
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
