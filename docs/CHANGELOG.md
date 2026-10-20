# Changelog

All notable changes to the Composite Perception Engine (CPE) project will be documented in this file.

## [2026-10-20, severity check] - K0 / severity literature check and a γ = 0 ablation

### Results (methodology §4.20)
- K0 on the corridor ladder: AUROC 0.537 (static 0.353); stays retired as a ranking.
- Replay with severity off (`severity_gamma0_headfx_2026_10_20`): `budget_5` warned 0.34 at 6.14
  false/min vs 0.35 at 5.71 with γ = 0.25; curves overlap. Severity stays declared, reported as a
  limitation; post-freeze replacement: Δv-based harm (Geisslinger et al. 2023).

### Added
- `tools/replay_threat_engine.py --severity-gamma` (default `SEVERITY_GAMMA`, unchanged results).
- `tools/session_viewer.html`: alerts spoken during playback (browser TTS), for the demo video.

## [2026-10-20, model scene context] - Model flags, agreement and replay on the head-camera rerun

### Results (methodology §4.19; paper §6.6)
- YOLO26n-sem @640 context for all 133 val/test sessions (5,829 frames, CPU, `.venv-seg` with
  Ultralytics 8.4.178), `data/sanpo_scene_context_v2/cpe_yolo26n_sem_walkable/`.
- Agreement with the oracle on the 18 head-masked val/test sessions: walker surface 0.967, object
  surface 0.908; κ 0.72 / 0.51 / 0.58 (in path / near a boundary / other surface).
- Distance-adjusted RR, all 133: in path 1.11 [0.82, 1.30], near a boundary 0.59 [0.39, 0.82],
  other surface 0.86 [0.61, 1.25]. Multipliers unchanged (near a boundary already ×1.0).
- Replay, context on − off (85 test sessions, 79 arrivals; `model_context_headfx_2026_10_20`): at
  `budget_5`, warned +0.01 [−0.13, +0.08], false alerts/min +0.52 [0.00, +1.02]; at `budget_10`,
  warned −0.06 [−0.19, −0.01]. No measurable effect; the context arm picks a higher threshold.

### Changed
- `tools/build_scene_context.py`: model sources use the worker pool too (prediction serialised by
  a lock); about 4× faster on CPU, where downloads dominate.

## [2026-10-20, TestFlight trigger] - Ship the timing app to TestFlight by tag

### Changed
- `.github/workflows/ios-timing.yml`: pushing a tag `ios-testflight*` archives, signs and uploads to
  TestFlight. The manual **Run workflow** button only appears for workflows on the default branch
  (`main`), which this one isn't on. Branch pushes still build unsigned when `ios/**` or the export
  scripts change (they pass on `v2-threat-engine` with the head-camera GRU).
- `ios/CPETiming/README.md`: setup step 5 uses the tag.
- `.github/workflows/ios-timing.yml`: every run also uploads an unsigned `.ipa` artifact, so the app can
  be installed with a free Apple ID through Sideloadly (no paid account, no Mac; 7-day installs).
  README section added.

## [2026-10-20, oracle scene context] - Flags, near-boundary decision and replay on the head-camera rerun

### Results (methodology §4.19)
- Oracle context for the 53 head-masked sessions (35 / 8 / 10), `data/sanpo_scene_context_v2/gt/`.
- Near a boundary, train only, distance-adjusted RR 0.60 [0.23, 0.83] (approaching variant 0.51
  [0.00, 0.84]): under the declared rule it stays ×1.0. Held-out check 0.80 [0.38, 3.37].
- In path 1.32 [0.93, 2.09] and other surface 0.89 [0.38, 1.22] on all 53; multipliers unchanged.
- Oracle replay, context on − off (10 test sessions, 27 arrivals; `oracle_context_headfx_2026_10_20`):
  at `budget_5`, warned ±0, false alerts/min +1.49 [0.00, +3.86]. The `early_budget_5` gain
  (+0.19 warned) comes from a different validation-chosen threshold, not from context.

## [2026-10-20, integration] - Head-camera rerun merged into segmentation

Branch `v2-segmentation` (PR #7).

### Merged
- `v2-kalman-lidar` up to `767dd88` (head-camera Stage-1 rerun). Conflicts in methodology and this
  changelog only; both sides kept.

### Changed
- Both branches had added a methodology §4.18. The head-camera numbers keep §4.18; segmentation is
  renumbered **§4.19** (references in architecture, pending_work and `priority.py` updated).
- Scene-context flags and context replays read bearings from the Stage-1 datasets, so they are
  re-run on the head-camera datasets; edge alerts use only masks and depth and stand.

### Results
- Model edge alerts on correct depth (`edge_alerts/cpe_yolo26n_sem_walkable_depthfix_*`, EssKae1234;
  step-offs from SANPO's masks): 0.56 alerts/min, 0.20 [0.08, 0.36] of 45 step-offs warned, 0.60
  of alerts followed by a step-off (oracle: 0.68, 0.18, 0.33). Official test: 1 of 7 warned, 0 of 4
  sustained.

## [2026-10-20, head-camera results] - Stage-1 rerun ingested; current numbers in methodology §4.18

### Changed
- Datasets, GRU runs (4 variants × 3 seeds), Kalman fit, ladder and before/after comparison
  replaced by the head-camera Colab rerun (`RUN_TAG = "headfx"`; 421 walks with samples, 288/48/85
  official split). The replay (`threat_engine_replay/corridor_headfx/`) and before/after comparison
  were run locally after the Colab GPU quota ran out.
- `src/threat_engine/fusion.py`: `NoiseModel` defaults from the 1 s-ahead fit, which chose the
  Student-t update (ν = 4, gate 60) on validation (−3.09 vs −3.51 declared).
- `tools/build_session_viewer.py`: threshold 0.6 with announcer cooldown 8 s, switch margin 1,
  stale 0.6 s (v2_gru `early_budget_5` on the rerun).
- Docs: methodology §4.18 (current numbers; §4.13–4.17 superseded), architecture §6.2 and §10,
  pending_work P0.0 closed.

### Found
- GRU AUROC 0.911 ties GBT and attention and beats every rule and Kalman baseline (all CIs exclude 0).
- The fitted Kalman no longer beats the declared one on the ladder (AUROC −0.013 [−0.022, +0.001],
  Holm p 0.30), contradicting §4.17; in the replay it still cuts 3.1–5.7 false alerts/min.
- At ≤ 5 false/min, v2_gru warns 0.35 of arrivals, 0.14 ≥ 0.2 s ahead and 0.04 ≥ 1 s ahead,
  precision 0.55; only v2_gru and nearest-in-corridor meet that budget on validation.

## [2026-10-19, local rebuild] - Run the Stage-1 rerun off Colab

### Added
- `tools/rebuild_local.py`: `notebooks/colab_rebuild.ipynb` as one resumable script (Windows or Linux,
  CUDA GPU; `--allow-cpu` otherwise). Same commands and outputs; Stage-1 CSVs already in
  `data/processed/sanpo_real_462/` are reused, finished Part-B steps leave markers in
  `data/rebuild_local/.done/`, and the results are zipped as `cpe_rebuild_<run-tag>_results.zip`.
  Written because Colab's free GPU quota ran out after 360 of 462 walks.
- `notebooks/colab_rebuild.ipynb`: the GPU check applies only while Stage 1 or GRU training is left,
  so a CPU runtime can finish the replay, Kalman comparison and zip when the GPU quota runs out.

## [2026-10-19, edge alerts on correct depth] - Oracle re-run and a train-only rule sweep

### Results (`edge_alerts/gt_depthfix_*`; methodology §4.19)
- 0.68 alerts/min; 18% [6, 33] of step-offs warned (was 29% on scrambled depth); median lead
  0.8 s. All 3 stairs step-offs warned; 12% of road step-offs warned.
- The main miss: an edge first seen 6–10 m ahead and never confirmed inside 5 m.
- Train-only sweep of range × confirmation, at ≤ 1 alert/min declared beforehand: the 5 m /
  2-frame rule stays. The curve runs from 0.18 warned at 0.7/min to 0.76 at 3.2/min.

### Added
- `tools/evaluate_edge_alerts.py`: `--alert-m`, `--confirm-frames`, `--sweep` (chosen on
  official-train sessions only).
## [2026-10-19, edge announcer] - "Curb ahead." through the announcement manager

Branch `v2-edge-announcer` (PR into `v2-segmentation`, for the threat-engine owner's review).

### Added
- `AnnouncementManager.step(..., edge=None)`: an edge lane below overrides and above object threats.
  - Edges wait up to `EDGE_STALE_S` = 1.2 s (one clip), against 0.6 s for objects.
  - An edge that arrives in the same frame as an override is kept and plays afterwards.
  - `Announcement.distance_m` is set for edges.
- `ThreatEngine.step(..., edge=None)` and `V2Pipeline.step(..., edge=None)`. `EngineFrame.narrator_event()`
  turns an edge into a `NarratorEvent(reason="edge ahead", bearing 0)`.
- `tests/test_edge_announcement.py` (4 tests; 95 pass, 2 skipped).

## [2026-10-19, integration] - Segmentation branches merged with the depth fix; 4-day plan

Branch `v2-segmentation` (PR #7).

### Merged
- `v2-seg-training` (fast-forward) and `v2-kalman-lidar` into `v2-segmentation`. Conflicts kept both
  sides: in `tools/replay_threat_engine.py`, the context arms together with the latency delays,
  announcer sweep and early-budget rules; README, CHANGELOG and pending_work side by side.

### Fixed
- `tools/build_scene_context.py` used the removed scrambling depth decoder: it now uses
  `decode_sanpo_depth`, and maps bearing to an image column through fx for pinhole-unit datasets.
- New cache roots `data/sanpo_scene_context_v2/` and `data/sanpo_edge_track_v2/`, so old tracks
  are never reused.
- `audit_sanpo_sessions.fetch` retries read timeouts and dropped connections.
- Test for the bearing-to-column mapping (91 pass, 2 skipped).

### Changed
- The segmentation methodology section is renumbered §4.14 → **§4.18**: both branches had added a
  §4.14.
- Superseded (scrambled depth): every scene-context, context-replay and edge-alert result before
  `*_depthfix_*`, and the near-boundary ×1.0 decision. They are marked in methodology §4.19 and
  `paper_draft.md`. The segmentation model's pixel scores and the Core ML export stand.
- Translation is future work. P4 covers the English clip library only.
- `pending_work.md`: merge order and the 4-day plan (2026-10-19 → 22).

## [2026-10-19, head camera] - Merge the intrinsics fix; fix the size check's depth lookup; Stage 1 to be re-run

### Merged
- `v2-threat-engine` into `v2-kalman-lidar`: the teammate's head-camera intrinsics fix
  (`tools/audit_sanpo_sessions.camera_details`, `tools/stream_sanpo_perception.py`,
  `session_audit.json` with head fx and chest-camera masks). Data and model artifacts keep this
  branch's correct-depth versions; all are regenerated by the Stage-1 re-run.

### Fixed
- `src/perception_stack/yolo_tracker.depth_rescore`: the centre depth for the size check and NMS is
  read at the box centre scaled to the depth map (`scale_point_to_depth`), not at frame pixels.
  Test: `tests/test_depth_rescore.py` (fails without the fix).
- `dataset/GRU/build.py`: `csv_units` refuses CSVs whose sidecar fx isn't the head camera's (±2%);
  test in `tests/test_dataset_units.py`.

### Found
- The 2026-10-18 Colab rebuild streamed with the chest camera's fx (it ran before the merge). On one
  test walk, the fixed Stage 1 keeps 465 detections vs 426 (48 new, 9 lost), and mean |bearing| rises
  12.4° → 17.1°. Bearings can't be patched after the fact because fx also gates detections, so
  Stage 1 is re-run. Methodology §4.16–4.17 numbers are provisional until then.

### Changed
- `notebooks/colab_rebuild.ipynb`: `RUN_TAG` (fresh Drive folders and zip name, so old CSVs are never
  reused); B2 fits `--objective ahead` and keeps it only if it beats the declared values on
  validation; B4 replays with `--announcer-sweep v2_gru v2_kalman nearest_in_corridor`.
- Docs: architecture §9 items 8–9; pending_work P0.0 reopened; methodology §4.17 marked provisional.

## [2026-10-19, units fix + lead tuning] - Bearings converted once; Kalman fit on 1 s; tuned on early warnings

### Fixed
- **Units after the rebuild.** The rebuilt CSVs store pinhole bearings at the true rate, but
  `dataset/GRU/relabel.py`, `models/GRU/benchmark.py` (rule baselines and tabular position features),
  `tools/replay_threat_engine.py`, `tools/fit_kalman_noise.py` and `tools/compare_kalman_noise.py`
  converted them again as legacy ±35° bearings (10–15% too central) and the replay halved speeds.
  `dataset/GRU/samples.py`: `PINHOLE_UNITS`, `legacy_units(manifest)`. `build.py` records
  `source.units` from the stream sidecars. The eight rebuilt manifests are marked. Tests:
  `tests/test_dataset_units.py` (3).
- The six corridor labels relabelled; every corridor GRU run retrained (4 variants × 3 seeds).

### Added
- `tools/fit_kalman_noise.py --objective ahead`: 1 s-ahead predictive log-likelihood using the
  filter's propagated covariance -> `kalman_noise_fit/fit_pose_ahead.json` (Gaussian half only;
  the Student-t half was stopped twice by low memory).
- `tools/replay_threat_engine.py`: `early_budget_{2,5,10}` rules (maximise `warned_before`);
  `--announcer-sweep ARM ...` tunes cooldown / switch margin / stale drop with the threshold;
  `knob_str`; a Kaplan-Meier table for `early_budget_5`. Test: `OperatingRuleTests`.

### Changed
- `src/threat_engine/fusion.py`: `NoiseModel` defaults are the 1 s-ahead fit (acceleration σ 0.50,
  static 2.14, depth σ 0.033 + 0.263·d, bearing σ 0.28°, Gaussian, gate 11.62), chosen on validation
  1 s log-likelihood (−3.08 vs −3.43 declared vs −3.57 one-step fit).
- `tests/test_kalman_latency.py`: the Student-t test sets ν = 4 and compares against a Gaussian
  update instead of a fixed distance.
- `tools/build_session_viewer.py`: default threshold 0.5 (v2_gru's `early_budget_5` point).
- Reports regenerated: the corridor ladder (+ `ref_cv_kalman_declared/`), `kalman_noise_fit/`.
  New replay: `threat_engine_replay/corridor_tuned/` (`corridor_depthfix/` is superseded).
- Docs: methodology §4.17 (new), the §4.16 note; architecture §6.2, §9 item 7, §10; pending_work.

### Results (test, scored once; methodology §4.17)
- Ladder: GRU 0.905. It ties attention, GBT and the fitted Kalman (0.893); the 1 s Kalman beats
  the declared one by +0.017 [+0.009, +0.026] (Holm p 0.003).
- Replay: v2_gru at `early_budget_5`: warned 0.37 [0.26, 0.57], ≥ 0.2 s ahead 0.19 [0.11, 0.36],
  ≥ 1 s ahead 0.06, 7.0 false/min (chosen for ≤ 5 on val), precision 0.58. Announcer tuning adds
  ≤ 0.03 `warned_before` on test; the predictors limit lead.

## [2026-10-18, depth rebuild] - Everything rebuilt on correct SANPO depth

### Changed
- Stage 1 re-run on all 462 sessions with the fixed depth decoder (`notebooks/colab_rebuild.ipynb`,
  Colab T4). Rebuilt: `dataset/GRU/sanpo_real_462_*` (arrival dataset, official split, six corridor
  labels), the corridor GRU runs (`corridor_h8` for gru and gru_attention, `corridor_hist4`,
  `corridor_hist1`; 3 seeds each), `evaluation/benchmarks/kalman_noise_fit/`, the corridor ladder
  (+ `ref_cv_kalman_declared/`). New replay: `threat_engine_replay/corridor_depthfix/`.
- More sessions have tracked objects in reach: 287 / 47 / 83 train / val / test (was 252 / 41 / 71).
- `src/threat_engine/fusion.py`: `NoiseModel` defaults refitted. Depth σ 0.058 + 0.053·d m (was
  0.30 + 0.22·d), acceleration σ 2.93 m/s² (static 2.04), bearing σ 0.37°, initial speed σ 0.90 m/s.
- `tools/build_session_viewer.py`: `ViewerGRU` (legacy ±35° bearings, 2× speeds) removed; the viewer
  uses `GRUArrivalPredictor` on the new checkpoint. Default threshold 0.8 (v2_gru's validation
  `budget_5` point).
- Docs: methodology §4.16 (new) with all results; §4.13–4.15 marked superseded; architecture §6.2,
  §9 item 0, §10; pending_work P0.0, P0.1 and the next three.

### Results (test, scored once; methodology §4.16)
- Ladder: GRU 0.909 AUROC. It ties GBT and the other GRU variants and beats every dynamics rule,
  including both Kalman rungs and nearest-in-corridor (Holm p ≤ 0.001).
- **Contradiction:** the refitted Kalman is better on one-step likelihood but worse on the ladder
  (−0.016 [−0.023, −0.004] AUROC vs. declared) and on 1 s forecasts, with no replay change.
- Replay: v2_gru at `budget_5` warns 0.35 [0.26, 0.53] of arrivals at 8.5 [5.2, 12.1] false/min
  (chosen for ≤ 5 on validation), precision 0.56. Only 0.12 are warned ≥ 0.2 s ahead.

## [2026-10-18] - Walkable-surface model trained (YOLO26n-sem @640, M4) and re-run with its masks

Branch `v2-seg-training` (from `v2-segmentation`, which it merged at `879dbe7`).

### Added
- `training/runs/cpe_yolo26n_sem_walkable/`: YOLO26n-sem @640, batch 8, from Cityscapes weights,
  on MPS. Early-stopped at epoch 8 (6.3 h); best.pt = epoch 3 (val mIoU 0.570). Speed pilot and
  the declared choice: `segmentation/pilot_mps_2026_10_17/`.
- Test, scored once (48 sessions): mIoU 0.615 [0.540, 0.641], false-safe 0.044 [0.029, 0.062],
  road as walkable 0.173 (over ground-truth road pixels), boundary F curb 0.30 / stairs 0.06.
- `weights/best.mlpackage`: Core ML FP16, argmax in the graph, 3.1 MB.
- **Superseded (scrambled depth).** Model scene context for the 112 replay val/test sessions; `scene_context/model_val_test_*`,
  `threat_engine_replay/model_context_2026_10_18/`, `edge_alerts/cpe_yolo26n_sem_walkable_*`.
  - Context flags match the oracle on its 15 sessions.
  - Context on vs. off in the replay: warned +0.04 [−0.06, +0.08], not measurable.
  - Held-out edge alerts warn about 1 of 7 SANPO-mask step-offs; the oracle warns about 3 of 7.
- `tools/evaluate_edge_alerts.py --reference gt`: the step-offs come from SANPO's masks while the
  alerts come from the model. Before this, a model was graded against its own masks. 1 test
  (72 total).

### Decided
- Kept n; did not train s@640. A deliberate deviation from the declared follow-up, because:
  - validation overfit after epoch 3;
  - thin edges need resolution, not capacity;
  - picking a second model from test scores would make the choice test-informed.

  The follow-up is n at a larger imgsz on a GPU machine. Not started.

### Notes
- `data/` and the venv live in `*.nosync` folders, so iCloud Drive doesn't sync them. Syncing
  slowed training about 2×.
- `audit_sanpo_sessions.fetch` retries only on HTTPError. Read timeouts were handled with a
  shell retry loop (the file isn't on this branch's list).
- The `.gitignore` rule `data/` also matches a Core ML package's `Data/` folder on macOS. The
  package files were force-added.

## [2026-10-17, Colab rebuild] - One notebook redoes everything on correct depth

### Added
- `notebooks/colab_rebuild.ipynb`: the P0.0 rebuild on a free Colab T4. Part A runs Stage 1 on all
  462 walks (`tools/stream_sanpo_perception.py`, 300 frames, stride 3) with CSVs written to Google
  Drive; Part B rebuilds the arrival dataset, official split and six corridor labels, refits the
  Kalman noise and writes it into `NoiseModel`'s defaults, retrains the four corridor GRU variants
  (3 seeds), and re-runs the ladder, the replay (`corridor_depthfix`) and the Kalman before/after.
  Finished walks and steps are kept on Drive and skipped, so a disconnect costs one walk or one
  step. Results come back as one zip; docs and numbers are updated locally after it.

## [2026-10-17, depth bug] - SANPO depth maps were decoded wrong; loader and occlusion fixed

### Found
- **Every SANPO depth map was misread.** `depth_maps/*.float16.gz` holds a [720, 1280] header and then
  a 720×1280 image (921,602 values). `depth_loader._infer_sanpo_shape` took the header for padding and
  reshaped from the start as 960×960 (also 921,600 values), scrambling each map into three shifted,
  interleaved copies. Found in the session viewer: a woman ~1.3 m in front of the camera was
  labelled 18.8 m. All Stage-1 distances, the GRU datasets and labels, the models, the fitted Kalman
  noise and every result built on them used scrambled depth.
- **Depth-sweep pseudo-boxes masked real objects.** `obs_*` boxes (whole lower-frame columns) were
  passed as occluders to `median_depth_in_box`, so a close person standing in front of one was
  sampled from background (6.4 m → 18.8 m in the same example).

### Fixed
- `src/perception_stack/depth_loader.py`: `decode_sanpo_depth()` reads the [H, W] header; headerless
  files fall back to known shapes; unknown sizes raise. The woman now reads 1.28 m.
- `src/perception_stack/pipeline.py`: `obs_*` rows are no longer occluders.
- `tools/gap_analysis_experiments.py`: uses the same decoder (it was 2 values off).
- `tests/test_depth_loader.py` (4 tests). Suite: 66 pass.

### Not yet done
- Rebuilding Stage-1 CSVs, the datasets, the models and the results (`pending_work.md` P0.0). Existing
  numbers stay in the repo but are provisional.

## [2026-10-17, viewer] - Session viewer for checking the engine by eye

### Added
- `tools/build_session_viewer.py` + `tools/session_viewer.html`: runs Stage 1 (YOLO26n v3 + ByteTrack +
  SANPO depth as is) and the v2 engine (Kalman, envelope, GRU, announcer) on one SANPO walk. It
  writes a local HTML viewer to `data/viewer/<session>/` (git-ignored) showing:
  - every box with distance, P(arrive) and priority, plus safety overrides and the spoken phrase;
  - the corridor ground truth (which objects enter the path within 4 s);
  - a timeline of alerts (true / late / false) and arrivals (warned early / at arrival / missed).

  Arrivals follow the replay's rule (seen at least one frame before arriving). Default: the
  official-test walk with the most arrivals, 30 s, threshold 0.6, poses on.

## [2026-10-17, decision] - SANPO depth used as is everywhere; no iPhone LiDAR

Team decision: the product's depth comes from an **external LiDAR sensor**, and the demo replays
pre-recorded video that already carries depth. The phone's own LiDAR is not used. SANPO's recorded
depth (`depth_maps/`) is used as is by every engine, label and replay.

### Removed
- `tools/replay_threat_engine.py`: `lidarize()`, `--lidar-range`, `--lidar-seed` (the iPhone-LiDAR
  degradation added earlier today) and its tests. The results folders
  `corridor_kalman_2026_10_17_lidar{3,4,5}m` are removed too; their findings are superseded.
- `ios/CPETiming`: the ARKit / LiDAR `sceneDepth` load, the camera permission and the ARKit
  device-capability requirement. Benchmark blocks no longer alternate ARKit on/off. CSV columns
  `ar_on` and `ar_fps` are dropped, and `analyze_timing.py` is updated to match.

### Kept
- The lead-based recall columns (`warned_before`, `warned_lead_ge1`) and their finding, measured on
  SANPO depth: at the alert budgets only 0.08–0.14 of arrivals are warned ≥ 0.2 s ahead
  (methodology §4.15, rewritten).
- Latency-aware replay (`--latency-csv`), the Kalman work and the ladder: unchanged, all on SANPO
  depth.

### Changed
- Docs: `architecture.md` §4 (product column: external LiDAR, not modelled; pose source not decided),
  §6.2, §7, §9 item 6 (`zed_depth_maps/` not used), §10, §11; `methodology.md` §1, §4.15, §5.1 and
  the drafting prompts; `pending_work.md` (depth-realism bound dropped); `README.md`;
  `ios/CPETiming/README.md`.
- `tests/test_kalman_lidar.py` renamed to `tests/test_kalman_latency.py`. Suite: 62 pass.

## [2026-10-17, later] - LiDAR depth-realism bound; lead-based recall

### Found
- **Most "warned" arrivals are warned on the arrival frame itself.** At every arm's `budget_5` /
  `budget_10` point on SANPO depth, only 0.08–0.14 of arrivals are warned ≥ 0.2 s ahead and ≤ 0.06
  ≥ 1 s ahead. For example, `v2_gru` at `budget_5`: warned 0.32, ≥ 0.2 s ahead 0.08 [0.00, 0.11],
  ≥ 1 s ahead 0.00. Operating points were picked on `warned`, which rewards this.
- **LiDAR range:** no measurable cost at 5 m or 4 m. At 3 m, warnings ≥ 1 s ahead go to 0.00 for every
  arm. A 5 m range leaves a median 1.2 s opportunity window (SANPO 2.6 s), and 81% of arrivals are
  seen before they arrive (21% at 3 m).

### Added
- `tools/replay_threat_engine.py`: `lidarize()` and `--lidar-range` / `--lidar-seed` (range cut, 5%
  dropouts, distance/eccentricity noise, closing speed re-fitted over 6 steps; labels stay on SANPO
  depth); new metrics `warned_before` (lead ≥ 0.2 s) and `warned_lead_ge1` (lead ≥ 1 s) in every
  operating-point table.
- Replays `corridor_kalman_2026_10_17_lidar{5,4,3}m`; `corridor_kalman_2026_10_17` regenerated with
  the new columns.

### Changed
- `.github/workflows/ios-timing.yml`: torch 2.7.0 and coremltools 9.0 pinned. The first CI run failed
  at the export step; the newest torch is untested with coremltools.
- Docs: `methodology.md` §4.15 (new), `architecture.md` §10, `pending_work.md` (next three, done
  2026-10-17).

## [2026-10-17] - Kalman noise fitted on train, Kalman arrival predictor, corridor baseline ladder, on-device timing harness, referee v2

Branch `v2-kalman-lidar` (from `v2-threat-engine` a5fd0aa).

### Found
- **The fusion filter's noise model was ~10× too optimistic.** Box-median SANPO depth has a robust
  σ ≈ 0.7 m even under 3 m and heavy tails; the declared σ = 0.05 + 0.01·d² gave a mean NIS of 17.5 on
  validation (2 = consistent) and rejected 26% of measurements, so tracks kept restarting and fused
  velocities were mostly noise. That was the main cause of `v2_physics` meeting no alert budget.
- **On the corridor label, the learned models no longer beat simple baselines significantly.** GRU AUROC
  0.875 vs nearest-in-corridor 0.851 (Δ −0.024 [−0.040, +0.002], Holm p 0.43), GBT on the current
  step 0.865, fitted Kalman 0.862 (Δ −0.013 [−0.039, +0.024]). History length doesn't matter
  (hist1 ≈ hist4 ≈ h8). The GRU's edge is in stability and the replay operating point, not AUROC.
- With the fitted filter, **no-pose arms do as well or better than pose arms**: ego-motion removal stays
  an ablation.
- `nearest_in_corridor` in the replay depends on the tracker's positions; with the fitted (smoother)
  filter it warns 0.05 of arrivals at `budget_5` (was 0.24).

### Added
- `src/threat_engine/fusion.py`: `NoiseModel` (fitted defaults; `NoiseModel.declared()` = the old values),
  Student-t measurement update, class-dependent acceleration noise for static classes,
  `FusedTrack.rel_cov` (4×4 relative covariance, optional field; interface otherwise unchanged),
  `GroundTracker(innovations=[])` recording for fitting.
- `src/threat_engine/kalman_arrival.py`: `KalmanArrivalPredictor`, P(enter the corridor within the
  horizon) over 128 fixed samples of the track's covariance.
- `tools/fit_kalman_noise.py` → `evaluation/benchmarks/kalman_noise_fit/fit_pose.json` (train-only
  predictive-likelihood fit). `tools/compare_kalman_noise.py` → `replay_before_after_test.{md,json}`
  (paired before/after with session-bootstrap CIs).
- Replay arms `v2_kalman`, `v2_kalman_nopose`; report
  `evaluation/benchmarks/threat_engine_replay/corridor_kalman_2026_10_17/`.
- `models/GRU/benchmark.py`: `cv_kalman` is now the fitted filter scored by `KalmanArrivalPredictor`;
  `cv_kalman_declared` keeps the old rung. Corridor ladder report under
  `evaluation/benchmarks/baseline_ladder/sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20/`
  (+ `ref_cv_kalman_declared/`).
- `tools/replay_threat_engine.py`: `--latency-ms` / `--latency-csv` (latency-aware replay: warned and
  lead use the heard time) and `--lidar-range` (see the next entry).
- On-device timing: `training/scripts/export_gru_coreml.py` (GRU → Core ML, fixed shapes, softmax in
  graph; graph conversion checked on Windows, saving needs macOS/Linux), `ios/CPETiming/` (SwiftUI +
  XcodeGen app timing detector + GRU per stage with ARKit/LiDAR load, `prepare_bundle.py`,
  `analyze_timing.py`, TestFlight how-to), `.github/workflows/ios-timing.yml` (macOS build, optional
  TestFlight upload). Not yet run on a phone.
- `evaluation/vlm_referee.py` protocol v2 and `docs/listening_study_plan.md` (committed separately).
- Tests: `tests/test_kalman_lidar.py` (14). Suite: 67 pass.

### Changed
- Fusion defaults: depth σ 0.30 + 0.22·d m, acceleration σ 0.35 m/s² (static 0.098), bearing σ 0.55°,
  initial speed σ 0.85 m/s, Student-t ν = 4, gate NIS 60. Two fusion tests now pin
  `NoiseModel.declared()` (hard gate) or use a longer track (coasting).
- `docs/related_work.md`: SANPO is Waghmare et al., WACV 2025; Mobilio's venue marked unconfirmed.
- Docs: `methodology.md` §4.7 (referee v2), §4.13 (rewritten: fitted noise, results), new §4.14
  (corridor ladder), §5.1 (timing harness, latency-aware replay); `architecture.md` §6.2, §7, §11;
  `ablation_guide.md` §4; `README.md` structure.

## [2026-10-17, evening] - Near-boundary revision, edge-ahead alerts, paper draft, merge order

Branch `v2-segmentation` (PR #7 into `v2-threat-engine`).

### Changed
- `CONTEXT_NEAR_BOUNDARY` 1.25 → **1.0** (`src/threat_engine/priority.py`). The decision rule was
  fixed in advance and checked on train sessions only. "Near a boundary and approaching" (≥ 0.3 m/s
  toward the centre line) didn't raise the arrival rate either: distance-adjusted RR 0.80
  [0.21, 1.15]. Held-out: 1.04 [0.12, 6.22]. Oracle replay re-run
  (`oracle_context_neutral_boundary_2026_10_17/`).
- `evaluation/context_association.py`: `--splits`, an *approaching* control flag, and the combined
  flag adjusted over distance × motion.

### Added
- `context.approach_speed`; `src/segmentation/edge_alert.py` (the "Curb ahead." rule);
  `tools/evaluate_edge_alerts.py`. Oracle results on 53 sessions: 0.79 alerts/min, 29% of step-offs
  warned (35% excluding one-frame flicker), all stairs warned, most road step-offs missed. Not tuned.
- 6 tests (71 total).
- `docs/paper_draft.md`: text, claim policy, final vs. pending results with owners, figure list.
- `docs/pending_work.md`: merge order. `v2-kalman-lidar` predates the camera-intrinsics fix and
  must re-run its label-dependent results after merging.

## [2026-10-17, later] - Walkable-surface segmentation: data, scene context, metrics, oracle check

Branch `v2-segmentation` (from `v2-threat-engine`).

### Added
- `src/segmentation/labels.py`: SANPO 31 classes → 8 CPE classes + ignore; mask decoding.
- `src/segmentation/context.py`: walker surface, surface under each object (from bearing + depth),
  `SceneContext` flags, edge ahead (class + distance), flat-ground mask for the obstacle sweep.
- `evaluation/segmentation_metrics.py`: mIoU, false-safe rate, road-as-walkable, curb/stairs
  boundary F. `tests/test_segmentation.py` (12 tests).
- `tools/build_sanpo_segmentation.py`: Ultralytics semantic dataset from all 237 masked SANPO-Real
  sessions (train 150 / val 39 / test 48 sessions; 18,787 human-annotated frames).
- `training/scripts/train_yolo26_sem_walkable.py`, `evaluate_yolo26_sem_walkable.py` (YOLO26s-sem,
  imgsz 1024, FP16 Core ML; run on the lab server).
- `tools/build_scene_context.py`: per-frame context for replay sessions from SANPO masks (oracle) or
  a trained model. `tools/replay_threat_engine.py --context-dir`: context-on arms with paired
  differences. `evaluation/context_association.py`: flag vs arrival-label risk ratios.

### Found
- Oracle (SANPO masks, 47 head-masked sessions): "in path" and "other surface" point the declared
  way (distance-adjusted RR 1.56 [0.98, 4.17] and 0.70 [0.20, 1.06]); **"near a boundary" points the
  wrong way** (0.83 [0.41, 0.99]) against its declared ×1.25. Not retuned (test sessions included).
- Oracle replay, 10 test sessions / 20 arrivals: GRU warned +0.05 [0.00, +0.07] with context, same
  false-alert rate. Underpowered.

## [2026-10-17] - Camera intrinsics fix (head ZED-M, not chest ZED 2i)

### Fixed
- **The audit and Stage 1 read the wrong camera's intrinsics.** `description.json` lists two cameras;
  `session_camera_details` is parallel to `session_camera_location` = `[camera_chest, camera_head]`.
  Element 0 (the chest ZED 2i: fx 1910–1993 px, 51–60° HFOV) was used, but every stream is
  `camera_head`, a ZED-M (fx 1367–1391 px, **70–78°**). Pinhole bearings and lateral offsets were
  ~30% too small. `tools/audit_sanpo_sessions.camera_details()` picks by location;
  `tools/stream_sanpo_perception.py` uses it.
- Corridor datasets relabelled (all six w/h variants): test arriving tracks 118 → 102, train
  208 → 169. Corridor GRU runs, the baseline ladder and the replay from `a5fd0aa` are stale until rerun.
- The 70° linear bearing in the old CSVs was close to right; the ±30° cone still covers most of the
  image and is ~3.5 m wide at 3 m, so the case for the corridor label stands.

### Added
- The audit records chest-camera intrinsics and masks: masks on 53 curated sessions' head camera
  (43 train / 10 test) and 130 sessions' chest camera (102 / 28; 8,411 + 2,429 human-annotated
  frames, the rest propagated).

## [2026-10-14, later] - Data Audit, Held-Out Results, v2 Threat Engine Built and Replayed

Branch `v2-threat-engine` (from `main` after `contextual_evaluation` was fast-forwarded into it).

### Found
- **All 462 sessions are 15 FPS; the Stage-1 CSVs assumed 30.** Recovered from the data: stored
  speed = 10.0 × per-step depth slope (IQR 9.99–10.01). Speeds are 2× too high, one processed step
  is 0.2 s, the 20-step horizon is 4 s and the 8-step history 1.6 s. Rankings and step-based labels
  are unaffected, so earlier metrics stand; time units in docs are corrected throughout.
- **Real HFOV is 51.4–60.1°** (fx 1910–1993 px), not 70°. With it, the ±30° label cone spans almost
  the whole image.
- **SANPO ships an official split** (`sanpo-real/splits/`): 374 train / 88 test of our sessions.
  53 sessions have segmentation masks; all have camera poses.
- **SANPO camera poses don't reconcile with box depth:** the world up axis varies by session and
  tracking drops out; even after handling both, static objects move more in the world frame than in
  the camera frame. Ego-motion removal is reported as an ablation only.

### Added
- `tools/audit_sanpo_sessions.py` → `simulation/datasets/sanpo/session_audit.json`,
  `official_split.json`.
- `dataset/GRU/resplit.py` and dataset `sanpo_real_462_tracked_d3_h8_current_m32_official`.
- `models/GRU/evaluate.py`: `TTC_in_cone` baseline (in-cone first, then TTC, then distance).
- `models/Ensemble/heldout.py`; GRU and GRU_attention runs `official462_2026_10_14_seed{0,1,2}`;
  report `evaluation/benchmarks/heldout_official_split/official462_2026_10_14/`. GRU AUROC 0.888
  held-out; beats K0 / TTC / cone TTC; ties nearest-in-cone; attention significantly worse on
  arrival order.
- `src/threat_engine/`: `fusion.py` (Kalman per track on the ground plane, ego-motion via pose,
  signed closing speed, time-to-zone, CPA, gating, coasting; `sanpo_session_poses`), `envelope.py`
  (TTC ≤ 1 s in the walking cone on fused velocity), `priority.py` (P(arrive) × severity^γ × context;
  `GRUArrivalPredictor`, `PhysicsArrivalPredictor`), `announcer.py` (preemption, hysteresis,
  cooldown/escalation, stale drop), `engine.py`.
- `src/pipeline/orchestrator.V2Pipeline`; `tools/run_full_pipeline_demo.py` runs v2 by default
  (`--v1` for the baseline) and falls back to cached clips without `piper-tts`.
- `tools/replay_threat_engine.py` and report `evaluation/benchmarks/threat_engine_replay/official_2026_10_14/`:
  v2 + GRU precision 0.59 vs v1 0.24, 17 vs 119 alerts/min, recall 0.26 vs 0.65.
- `tests/test_threat_engine.py` (23 tests). Suite: 51 tests, all pass.
- `src/narration/templates.py`: distance buckets (5 / 10 / 20 m, far) and `all_phrases()` (the clip
  library to pre-render).

### Changed
- `tools/stream_sanpo_perception.py` reads fps, fx and HFOV per session from `description.json`
  (`--fps` now only overrides) and records them in `<session>.frames.json`.
- `physics.compute_bearing(focal_px=...)` gives a pinhole bearing; `run_perception_stream(focal_px=...)`
  and `YoloTracker(focal_px=...)` use the real focal length.
- `evaluate_yolo26n_hazards.py`, `compare_yolo26n_retention.py`, `gap_analysis_experiments.py`
  default to the finetuned 17-class v3 checkpoint, like the runtime.
- `export_yolo26n_edge.py`: `--formats coreml` (copies the `.mlpackage` directory).
- `dataset/GRU/samples.history_features` extracted so the dataset and the runtime GRU share one
  encoder (output verified byte-identical to the old code).
- `src/perception_stack/__init__.py` loads `YoloTracker` / `run_perception*` lazily, so the threat
  engine and its tests import without ultralytics (`test_reflex_layer` runs again).
- Docs: `architecture.md` (statuses, §9 data issues, code map, open decisions), `progress.md`
  (rewritten with held-out and replay results), `pending_work.md` (rewritten), `methodology.md`
  (§2.1 audit and split, new §4.11–4.13, drafting prompts), `sanpo_dataset.md`, `ablation_guide.md`,
  `yolo_training.md`, `README.md`, `docs/README.md`, `dataset/README.md`, `models/GRU/README.md`,
  `models/Ensemble/README.md`.

### Removed
- `src/shared/fact_sheet.py`, `src/perception_stack/fact_sheet_builder.py`,
  `tools/run_fact_sheets.py` (SLM-1 prompt format; no remaining consumer).

## [2026-10-14] - v2 Architecture Adopted; Docs Consolidated; Dead Code Removed

The team adopted a v2 design. The SLM-1 cognitive layer and the narrator SLM-2 are dropped. Threat is
now ranked by **when** (GRU arrival predictor), **how bad** (class severity) and **what**
(walkable-surface segmentation, planned), inside a deterministic TTC safety envelope. Narration is
scripted and pre-rendered. The target device is an iPhone 17 Pro Max. Evaluation is SANPO replay
only, with SANPO depth standing in for phone LiDAR; no depth model, no live deployment.

### Changed
- `docs/architecture.md` rewritten as the v2 design doc. It covers:
  - components and diagram;
  - simulation vs. iPhone deployment;
  - latency budget and evaluation design;
  - known data issues, limitations, and a code map from v1 modules to v2 roles.

  v1 is at commit `a6e37a7`.
- `docs/progress.md` rewritten as a presentation-ready status doc (absorbs
  `progress_presentation.md`). Corrected two stale claims:
  - the λ arms are struck, not tied;
  - TTC's "4× worse encounter recall" was a depth-blob artifact.
- `docs/pending_work.md` rewritten around v2 (P0 data correctness → P5 paper), with a "dropped with
  the v2 pivot" table.
- `docs/methodology.md`:
  - §1 reframed for v2;
  - §2.1 adds the FPS / FOV caveat;
  - §4 intro explains K0's demotion to the severity term;
  - §4.9 no longer cites TTC's blob-driven encounter result;
  - §5 replaces the Jetson proxy with on-device measurement plus latency-aware replay;
  - Appendix A.1, A.2, A.3, A.4, A.5, A.6, A.7 updated for v2 (A.4/A.5 are now the threat engine and
    the safety envelope).
- `docs/README.md`, `README.md`: v2 pipeline, maps and project tree.
- `docs/sanpo_dataset.md`: new "Camera Facts and Label Map" section; Jetson profile marked retired.
- `docs/related_work.md`, `docs/ablation_guide.md`, `docs/yolo_training.md`: references repointed;
  SLM/RL citation sections marked v1-only.
- Code comments only, no behaviour change: references to removed docs and old `architecture.md`
  sections repointed in:
  - `src/narration/*`, `src/pipeline/orchestrator.py`, `src/cognitive_layer/stub.py`,
    `src/perception_stack/physics.py`;
  - `evaluation/kinetic_ablation*.py`, `evaluation/vlm_referee.py`;
  - `tools/benchmark_narration_latency.py`, `tools/run_full_pipeline_demo.py`.

  `evaluation/benchmarks/narration_latency/report.md` and `kinetic_score_eval/README.md` were
  updated to match.

### Found
- SANPO-Real `description.json` reports **15 FPS** and a **58.4°** horizontal FOV (one session
  checked). The pipeline assumes 30 FPS and 70°. Logged as `pending_work.md` P0.
- SANPO `labelmap.json` includes `road`, `curb`, `sidewalk`, `crosswalk`, `stairs` and walkable /
  inaccessible surface classes, so the segmentation plan needs no new annotation.

### Removed
- **Docs**, content merged (all retrievable from git history at `a6e37a7`):
  - `docs/progress_presentation.md` → `progress.md`;
  - `docs/hardware_targets.md` → `architecture.md` §7 and `methodology.md` §5.1;
  - `docs/kinetic_score_opinion.md` → condensed in `methodology.md` §4.
- **Dead code:**
  - `training/rewards/reward_function.py`: SLM-1 PPO reward, referenced fact-sheet fields that no
    longer exist;
  - `training/rl_agent/__init__.py`: empty; the PPO scripts were never written;
  - `tools/manual_score_inspector.py`: built on the deleted K1–K5 formulas;
  - `tools/download_sample_data.py` and `tools/prepare_sample_data.py`: superseded by
    `stream_sanpo_perception.py`;
  - `notebooks/CPE_Preprocessing.ipynb`: unreferenced;
  - `colab/sanpo_kinetic_replay_colab/` and `notebooks/sanpo_kinetic_replay_colab.ipynb`: old JSONL
    replay; the two copies had already diverged.

## [2026-09-29] - Full 462-Session Run of the Arrival GRUs and Ensemble

The download of all 462 curated SANPO-Real sessions finished, and the arrival-time experiment and
ensemble were re-run on the full set. Write-up:
`evaluation/benchmarks/gru_encounter_eval/run_2026_09_29_arrival462/report.md`.

### Changed conclusions (vs. the 387-session run)
- **GRU vs. nearest-in-cone, AUROC:** +0.007 (tie) → **+0.016 [+0.002, +0.029]**, a small real
  gain. Arrival order is still a tie.
- **Attention vs. size-matched GRU, AUROC:** +0.010 (significant) → **+0.007 [−0.000, +0.016]**,
  a tie. On full data, scene context adds nothing measurable.

### Unchanged
- GRU vs. K0: +0.32 AUROC, +0.09 arrival order. GRU vs. TTC: +0.23 / +0.06.
- The stacked ensemble is best at encounter prediction (AUROC 0.894, +0.023 over nearest-in-cone).
- No model beats nearest-in-cone on arrival order.

### Added
- Stage-1 CSVs for all 462 sessions in `data/processed/sanpo_real_462/`, which is gitignored.
- Dataset `dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_f5_s0/` (1.0 MB).
- Run sets `arrival462_2026_09_29` (both architectures) and `arrival462h96_2026_09_29`.
- Ensemble results in `evaluation/benchmarks/ensemble_eval/arrival462_2026_09_29/`.

## [2026-09-29] - Arrival-Time GRUs with Cross-Validation (387 SANPO Sessions)

The GRUs can now rank tracked objects by **how soon** they will come within a hazard distance. They
were evaluated with 5-fold session-level cross-validation on 387 of the 462 curated SANPO-Real
sessions. Full write-up: `evaluation/benchmarks/gru_encounter_eval/run_2026_09_29_arrival/report.md`.

### Found
- **K0 and TTC.** On "will this tracked object come within 3 m in front of you in 2 s", the GRU
  beats K0 (AUROC +0.31 [+0.25, +0.35]) and TTC (+0.23). It also ranks objects by arrival better
  than both (+0.08 / +0.06, CIs exclude 0).
- **Nearest-object rules.** The GRU ties the plain "nearest object in the forward cone" rule on
  both AUROC and arrival order. Its advantage is a steadier top pick: flicker 0.59 vs. 0.73.
- **Attention.** Cross-object attention adds +0.010 AUROC [+0.002, +0.021]. That survives a
  size-matched control (plain GRU at hidden 96), but there is no gain on arrival order or on
  interaction frames, and flicker is worse.
- **K0 as an arrival ranking.** It is barely above chance at predicting who comes close (AUROC
  0.57). It should not be the only ranking used to decide what gets announced first.

### Added
- **Dataset builder options.**
  - `--hazard-m` sets the encounter distance; 1.5 m stays identical to the ablation, which is
    unit-tested.
  - `--folds K` builds session-level K-fold cross-validation, stratified by encounter count and
    stored as `all.npz`.
  - Per-object `tte` (frames until arrival) and `ttc` arrays.
- **Model and training options.**
  - `models/GRU/train.py --time-bins K` adds an arrival-time output (K slices of the 2 s horizon
    plus "not within it"), trained with cross-entropy. It is shared by both architectures through
    `ThreatHead(n_outputs)`.
  - Training loops over folds.
  - Each run saves `test_outputs.npy`.
- **Evaluation.**
  - `models/GRU/evaluate.py` adds a TTC baseline and two metrics: `soonest_top1` and
    `arrival_order` (pairwise concordance).
  - Flicker now follows the announced (how-soon) pick; this is identical for yes/no models.
- **Summaries.**
  - `models/GRU/summarize.py` pools cross-validation folds, so every session is scored once.
  - It reports paired session-bootstrap differences, model vs. each baseline and attention vs. GRU.
  - Run names are matched exactly.
- **Tests.** 16 in total, covering arrival times, folds, the cross-validation round trip, arrival
  pairs, score conversion, loss targets and output shape.
- **Data and runs.**
  - Built datasets `ablation_30pct_tracked_d3_h8_current_m32_f5_s0` and
    `sanpo_real_387_tracked_d3_h8_current_m32_f5_s0`.
  - Run sets `arrival387_2026_09_29` and `arrival387h96_2026_09_29`.
  - The preliminary `arrival139_2026_09_28` runs were kept local, not committed. They can be
    reproduced from `dataset/GRU/ablation_30pct_tracked_d3_h8_current_m32_f5_s0`.

- `models/Ensemble/ensemble.py` implements Level 1 of the ensemble plan. It combines the GRU,
  attention GRU, nearest-in-cone and TTC arrival scores in two ways:
  - **Rank averaging**, which fits nothing.
  - **A softmax-regression stacker** fitted out-of-fold. A test proves fold F never sees its own
    labels.
  - Run on 387 sessions:
    - The stacked ensemble is the best "will it come close" predictor so far: AUROC 0.889, +0.019
      [+0.008, +0.032] over nearest-in-cone, and +0.012 over the GRU.
    - It still only **ties** the nearest-in-cone rule on arrival order (+0.001 [−0.010, +0.012]).
  - Results: `evaluation/benchmarks/ensemble_eval/arrival387_2026_09_29/`.
  - Level 2 (blending arrival with K0 severity for the announcement order) is still planned.
  - 3 new tests: `tests/test_ensemble.py`.
- `models/GRU/summarize.py` now exposes `run_dirs()` and `pooled_runs()` for reuse. Its output is
  unchanged, which was verified.

### Verified
- Re-running `summarize` and `evaluate` on the 2026-09-28 fixed-split runs reproduces their numbers
  exactly after the refactor.

### Changed
- `docs/methodology.md` §4.10 covers the arrival-time task, the cross-validation protocol, paired
  statistics and results.
- `dataset/README.md` and `models/GRU/README.md` document the new options, metrics and latest
  results.

## [2026-09-28] - GRU First Real Run; Encounter Label Found to Be Dominated by Depth Blobs

Regenerated the 139-session ablation CSVs locally, built the GRU datasets, and trained both
architectures (3 seeds each). Full write-up:
`evaluation/benchmarks/gru_encounter_eval/run_2026_09_28/report.md`.

### Found
- **99.7% of the ablation's encounter labels are untracked `obs_*` depth blobs.**
  `detect_unlabeled_obstacles` has no ground removal, so these are mostly the ground ahead of the
  walker. They are re-IDed every frame, so K0 is always 0 for them, and only ~36% of encounter
  frames are winnable by any method.
- **With blobs removed, K0, no-velocity and TTC tie on encounter_top1** (0.305 / 0.306 / 0.310, CIs
  roughly [0.18, 0.43]). The ablation's TTC and no-velocity encounter results are artifacts of which
  formula picks the ground blob. Flicker and rank-stability results were not re-checked.
- **GRU results.**
  - All objects: both GRUs learn exactly "nearest object" (AUROC 1.000).
  - Tracked only: GRU AUROC 0.905 ± 0.054, attention 0.879 ± 0.022, nearest 0.937, K0 0.481.
  - Neither beats nearest, and attention adds nothing measurable.
  - The tracked-only data is too thin to be conclusive (7 validation positives, 45 test encounter
    frames).
- **Correction.** The encounter horizon is 20 frames at 10 Hz = **2 s**, not "20 source frames".
  The CSVs renumber strided frames consecutively. Docs and docstrings corrected.

### Added
- `dataset/GRU/build.py --tracked-only`, which drops the `obs_*` blobs (`samples.UNTRACKED_PREFIX`),
  with a test (9 tests total).
- Built datasets `dataset/GRU/ablation_30pct_h8_current_m32_s0/` and
  `dataset/GRU/ablation_30pct_tracked_h8_current_m32_s0/` (1.5 MB and 0.4 MB).
- 12 trained runs under `models/GRU/{GRU,GRU_attention}/runs/{all,tracked}_2026_09_28_seed{0,1,2}/`.
- `models/GRU/summarize.py`, which gives mean ± std across seeds next to the baselines.
- `evaluation/encounter_blob_check.py`, which recomputes the ablation's encounter_top1 with and
  without the blobs.

### Changed
- `docs/methodology.md` - new §4.9 open caveat 2 (blob-dominated encounter label); §4.10 results.
- `README.md`, `dataset/README.md`, `models/GRU/README.md` - new files and the blob explanation.

## [2026-09-28] - Learned Encounter Predictors: GRU vs. GRU + Cross-Object Attention

New branch `contextual_evaluation` (from `kinetic-score-evaluation`). Adds two small learned models
that predict the §4.6 encounter label from each object's recent track history, to test whether
scene context (objects seeing each other) improves threat prediction over K0 and over a per-object
model. Implemented and unit-tested; **not yet run on real SANPO CSVs**.

### Added
- `models/GRU/GRU/model.py` - Architecture 1: per-object GRU; each object scored from its own 0.8 s
  history.
- `models/GRU/GRU_attention/model.py` - Architecture 2: same GRU plus one residual self-attention
  block across the objects in a frame (no positional encoding, since fact sheets are K-sorted;
  training-time context dropout).
- `models/GRU/layers.py` - shared encoder and head, so the attention block is the only difference
  between the two.
- New top-level `dataset/` folder (model-ready datasets, one subfolder per model family; see
  `dataset/README.md`). `dataset/GRU/` is deliberately **one dataset shared by both GRU
  architectures** - separate datasets would add a second difference besides the attention layer.
  - `dataset/GRU/samples.py` - Stage-1 per-session CSVs -> per-frame samples. Reuses
    `evaluation/kinetic_ablation.py`'s `prepare()` and `encounters()` directly (no
    re-implementation); `kinetic_score` is deliberately excluded from the inputs. Stores splits as
    compressed `.npz` (plain arrays, no pickle).
  - `dataset/GRU/build.py` - builds `dataset/GRU/<name>/{train,val,test}.npz` + `manifest.json`
    (settings, source-CSV fingerprint, session split, label stats) once; warns if a file exceeds
    GitHub's 50 MB warning size.
- `models/GRU/train.py`, `models/GRU/evaluate.py`, `models/GRU/registry.py` - one shared training
  loop and evaluation, reading a built dataset (`--dataset`). Runs record the dataset fingerprint and
  re-evaluation refuses a different dataset. Per-object AUROC/average precision and per-frame
  encounter_top1/flicker_rate with the ablation's exact definitions and session bootstrap, reported
  on all frames and on an interaction slice, against K0, nearest-object, and nearest-in-cone
  baselines.
- `models/GRU/README.md`, `tests/test_gru_models.py` - 8 tests: labels match the ablation, history
  padding, `.npz` save/load round trip, disjoint session split, padding invariance, object-order
  invariance, only the attention model sees other objects, no NaNs with single-object frames.
- `models/__init__.py`, `dataset/__init__.py` so `python -m ...` resolves to this repo's packages.

### Verified
- End-to-end train -> checkpoint -> standalone re-evaluation on synthetic session CSVs (not SANPO):
  both architectures learn the task and the standalone evaluator reproduces the training-time
  metrics. Synthetic numbers are a pipeline check only and are not reported anywhere.

### Changed
- `docs/methodology.md` - new §4.10 (design, protocol, baselines, capacity control, caveats) and a
  matching bullet in the A.6 drafting prompt.
- `README.md` - project structure lists `models/GRU/` and `dataset/`.
- `docs/architecture.md` unchanged: these models are an offline evaluation experiment and do not
  enter the runtime pipeline yet.

## [2026-08-27] - Merged Real Ablation Run Artifacts; Docs Reconciled Against Ground Truth

Merged `origin/kinetic-score-evaluation` (ishaan's commit `d9f4330`, "Add run_2026_08_26 artifacts;
strike unreproducible lambda arms") into this branch, then updated every doc that had been written
with caveats pending this exact data.

### Merged
- `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/{report.md,metrics.json,disagreements.json,disagreements_key.json}`
  - the real DGX Spark run outputs, previously only a hand-transcribed summary table in this repo.
- One conflict in `docs/ablation_guide.md` (both branches had independently edited the same λ-arm
  section) resolved by combining rather than picking a side: ishaan's more precise root-cause
  explanation (commit `e045310` monkey-patched `SEVERITY_LAMBDA` per arm) and cleaner strikethrough
  table treatment, plus this branch's dilution-effect explanation, `kinetic_ablation_stratified.py`
  pointer, and §7 Troubleshooting section (which predates his branch point).

### Verified (not just merged)
- Cross-checked the merged `report.md`/`metrics.json` against `ablation_guide.md`'s summary table:
  every point estimate matches exactly, and the CI bounds behind every "TIES" verdict genuinely
  overlap (e.g. K0 and `linear` flicker rate both report the identical 95% CI `[0.964, 0.983]`) -
  confirming the tie verdicts were not just eyeballed from the missing data.

### Changed
- `docs/methodology.md` §4.9 - removed the two now-resolved caveats (λ numbers "flagged unverified",
  raw artifacts "never committed"); replaced with a "Resolved 2026-08-27" note stating what was
  verified and how. One caveat remains open (the stratified-check dilution question).
- `docs/pending_work.md` §1 - items 1-2 (push artifacts, reconcile λ) marked done with what was
  verified; added the per-arm disagreement-frame breakdown now visible from the committed
  `disagreements_key.json` (`no-velocity` 60, `ttc` 60, `size` 51, `linear` 26, `no-severity` 11,
  `lam=0.25` 6, `lam=1.0` 5); fixed a stale cross-reference (depth-smoothing filter was cited as
  "§1.4", actually "§1.7" after earlier renumbering); updated "Suggested next actions".
- `docs/progress.md` - kinetic-score-ablation checklist row updated from "λ tie" to "λ struck, raw
  data committed and audited".
- `evaluation/generate_report_figures.py` - `plot_kinetic_ablation_metrics()` and the results-summary
  table now read from the real `metrics.json` (with genuine 95% CI error bars on the chart) instead of
  the hand-transcribed `ablation_summary.csv`, which is kept only as a fallback for older checkouts.
  Struck λ arms are plotted with a distinct color/legend entry rather than hidden. Added a new
  "disagreement frames exported per arm" table to `results_summary.md`, sourced from
  `disagreements_key.json`.

No formula or threshold in `src/perception_stack/physics.py` was changed - `pending_work.md` still
explicitly advises against that until the stratified check and blinded referee (items 3-4) run.

## [2026-08-27] - First Full-Pipeline Runtime Composition (Synthetic Data)

Closed the integration gap `pending_work.md` §5 called "the single most valuable integration
milestone remaining": every runtime component built so far — Threat Prioritizer, Cognitive Layer,
Physics Verification, Narration — is now called together, in order, in one process, and actually
produces audio-ready output. Runs on synthetic data only; real SANPO replay is still open.

### Added
- `src/cognitive_layer/stub.py` - rule-based, non-ML stand-in for SLM-1: picks the highest-K object
  among a frame's cognitive-route events as the semantic primary threat, with a canned reason string.
  Explicitly documented as a placeholder to be replaced, not extended, once a real model exists - the
  exact point where `SemanticEval` should come from a trained model instead.
- `src/pipeline/orchestrator.py` - `process_frame()`/`process_frames()`, the composition that did not
  exist anywhere in the repo before this: builds the physics ranking from every non-ignore object in
  a frame (reflex- and cognitive-route alike), gets a `SemanticEval` from the Cognitive Layer,
  adjudicates once via `PhysicsVerification`, and feeds any resulting `NarratorEvent` into
  `NarrationPipeline`. **Real gap found and fixed:** `src/reflex_layer/reflex.py` only ever calls
  `PhysicsVerification` for reflex-route events by design ("Cognitive events are intentionally left
  for the SLM path") - nothing before this module built that other half. A frame containing only
  cognitive-route objects (e.g. a near-static pothole with nothing fast-closing present) previously
  produced no `PhysicsVerification` call and no narrator event at all, regardless of Cognitive Layer
  output. `orchestrator.py`'s self-check specifically regression-tests against this.
- `tools/run_full_pipeline_demo.py` - runs a 6-frame synthetic scenario (a car closing at a constant
  1.5 m/s alongside a static pothole) through the full chain and prints per-frame routing, the
  semantic pick, whether an alert fired, which narration lane it used, and the resulting text.
  Distances were chosen by actually running `physics.kinetic_score()` first, not guessed, to produce
  a genuine ignore -> cognitive -> reflex escalation. Confirmed live: the override frame stays in
  English even when `--lang hi` is passed and a matching phrase-table entry exists for it - the
  critical lane's translation bypass (`architecture.md` §11.6) working as designed, not a missing
  translation.
- **Finding from running the demo:** with real (not guessed) kinetic-score values, a `car` object
  moving at a slow constant 1.5 m/s crosses `LOW_K_THRESHOLD` (cognitive routing) at roughly 20 m,
  driven mostly by its class severity weight rather than proximity or closing speed - relevant to the
  threshold-calibration item in `pending_work.md`'s refinement backlog, since it suggests the routing
  may be more sensitive to vehicle classes than the thresholds were likely eyeballed against.

### Changed
- `docs/pending_work.md` §5, `docs/progress.md`, `docs/progress_presentation.md`, `README.md` -
  updated from "not started" to reflect the above; §5 keeps "real SANPO replay" explicitly open since
  this milestone used synthetic data only.

No existing runtime component (`threat_prioritizer`, `reflex_layer`, `physics_verification`,
`narration`) was modified - this entry is new, additive composition code on top of all of them.

## [2026-08-27] - Narration Layer Phase 1: Template Narrator, Translation/TTS Adapters, Literature Review

Built the end-of-pipeline narration/translation/TTS layer — the parts of the roadmap that don't
depend on the SANPO ablation data — plus a verified external-literature reference doc and a
refinement backlog from a critical pass over what's already built.

### Added
- `src/narration/` — new module: `templates.py` (deterministic template narrator, no ML dependency,
  reproduces `architecture.md` §2.7's worked example exactly), `translation.py`
  (`PhraseTableTranslator`, `EnglishFallbackTranslator` real and tested; `IndicTrans2Translator`
  correct-API adapter, code complete but not run against real model weights), `tts.py`
  (`CachedClipTTS` real and tested; `PiperTTS` real, tested, and latency-measured), `pipeline.py`
  (`NarrationPipeline` — structurally enforces that an override event never reaches translation or
  model-backed TTS, per the orchestration-lane rule in `architecture.md` §11.6). All four modules
  have a runnable `--self-check` (invoke as `python -m src.narration.<module> --self-check`).
- `tools/benchmark_narration_latency.py` - standalone latency benchmark, no SANPO data or GPU needed.
  Measured on this dev machine (CPU only): template generation ~0 ms, Piper voice load ~444 ms
  (one-time), Piper synthesis 30-68 ms/utterance for four representative `NarratorEvent` phrasings.
  Found and flagged a real documentation gap: no formal TTS latency budget exists anywhere in
  `docs/` (only 50 ms reflex / 500 ms cognitive are defined).
- `models/tts/piper/en_US-lessac-low.onnx` (+ config) - downloaded Piper voice used for the above.
- `piper-tts`, `IndicTransToolkit` added to `requirements.txt` (installed and verified importable
  this session; the IndicTrans2 model weights themselves were not downloaded).
- `docs/related_work.md` - ~20 externally-verified paper citations (title/authors/year/venue checked
  against a live search, not recalled from memory), organized by which CPE component or design
  decision each grounds: positioning (Dakopoulos & Bourbakis 2010 ETA survey; Liu & Slade 2026
  "Mobilio" smartphone nav app - the closest published comparable system), perception (ByteTrack,
  YOLO26 technical overview, Depth Anything, Grounding DINO), the SANPO dataset paper, kinetic-score
  grounding in real traffic-safety literature (Rosen & Sander 2009/2011 pedestrian impact-speed risk
  curves - independent empirical support for the `v^2` design choice beyond CPE's own architectural
  argument), evaluation methodology (Bradley-Terry 1952, Cohen's kappa 1960, Zheng et al. 2023
  LLM-as-judge), SLM/distillation (Hinton et al. 2015, Qwen3, Phi-4-mini), RL (Schulman et al. 2017
  PPO; Bai et al. 2022 Constitutional AI/RLAIF as the conceptual anchor for Physics-Verification's
  human-label-free reward design), translation (IndicTrans2), and TTS (FastSpeech2, VITS underlying
  Piper, IndicF5/Indic Parler-TTS).
- `docs/pending_work.md` §10 "Refinement backlog" - a critical pass distinct from "not started" work:
  7 items where something already produces results but the result or code has a specific, named gap
  (detector eval has no uncertainty quantification; reflex/cognitive routing thresholds are still
  uncalibrated defaults despite the ablation corpus now existing to calibrate them against; the
  Jetson simulation's 4.0x scaling factor has no cited basis; safety-critical test coverage is thin
  at 9 tests total; `BEHAVIOUR_MULTIPLIER` has no grounding unlike mass now does; bearing uses a fixed
  uncalibrated 70 degree HFoV; the VLM referees' "three distinct families" independence is assumed,
  not checked).

### Changed
- `docs/architecture.md` §11.3-11.5 - added implementation-status notes with the measured numbers
  above; §11.5 in particular now states FastSpeech2/IndicF5/Indic Parler-TTS remain unimplemented per
  the existing "benchmark before entering the real-time path" guidance, now that Piper has been
  benchmarked and it passed.
- `docs/pending_work.md` §4, `docs/progress.md`, `README.md` (SLM Stack table + project tree) -
  updated from "not started" to reflect the phase-1 status above.

No formula, threshold, or existing pipeline component was changed - this entry is new, additive
narration-layer code plus documentation/research, gated the same way everything else in this repo is
gated: implemented-and-tested is stated separately from implemented-but-unverified.

## [2026-08-27] - Ablation Methodology Audit: Stratified Severity Check, λ Inconsistency Flagged

A critical review of the 2026-08-26 kinetic-score ablation run's methodology surfaced two real gaps
before its "severity/λ/size tie with K0" conclusions should be treated as settled.

### Added
- `evaluation/kinetic_ablation_stratified.py` - re-runs `kinetic_ablation.py`'s label-free metrics
  (imported, not reimplemented) restricted to frames containing a severity-differentiated,
  kinematically-close pair of objects - the specific scenario severity's design intent targets. Also
  reports the coverage number (what fraction of the corpus even qualifies) and the K0-vs-arm argmax
  disagreement rate on the stratified subset vs. the full corpus. Rationale: a corpus-wide tie on
  `no-severity` cannot distinguish "severity never matters" from "severity matters only in a minority
  of frames diluted into 19,402" - this isolates that minority directly. Self-check passes; has not
  yet been run against real data (needs `data/processed/ablation_30pct/` on the remote box that ran
  the original ablation).
- `evaluation/kinetic_ablation.py` - `session_metrics()` now accepts an optional pre-computed `groups`
  argument so the stratified script reuses the exact same metric math instead of a second
  implementation that could drift out of sync. Purely additive; default behavior (no `groups` passed)
  is unchanged, verified against the existing `--self-check`.

### Changed
- `docs/ablation_guide.md` - added a `[!WARNING]` before the §6 results table: the `lam=0.25`/
  `lam=1.0` columns are not reproducible from the currently committed `kinetic_ablation.py`, whose
  lambda-sweep plumbing was removed in commit `0e6e0d7` on 2026-08-25 - the day *before* this run's
  Step 4 timestamp. Both arms' verdicts changed from "TIES" to "unverified" pending reconciliation.
  Also documented that no raw run artifacts (`metrics.json`, `report.md`, `disagreements*.json`) were
  ever committed for this run, so the tie verdicts for every arm are not yet independently auditable
  from this repo - only the transcribed summary table exists. Added the stratified-check command to
  the "Next steps" section.
- `docs/methodology.md` §4.9 - added the same three caveats (dilution risk, λ inconsistency, missing
  raw artifacts) so the paper-ready methods reference doesn't overstate what the corpus-wide ties
  currently prove.
- `docs/pending_work.md` §1 - rewritten as a precise ordered list: push the missing run artifacts from
  the remote box -> reconcile the λ inconsistency -> run the stratified check -> run the VLM referee ->
  human calibration -> top-3 VLM check -> depth smoothing filter. Added an explicit instruction not to
  change `physics.py`'s formula or weights based on the current evidence alone.
- `README.md` - added `evaluation/kinetic_ablation_stratified.py` to the project tree.

No formula or weight in `src/perception_stack/physics.py` was changed - this entry is entirely
methodology/tooling/documentation, pending the remote-box steps in `pending_work.md` §1.

## [2026-08-27] - Documentation Reorganization, Methodology/Progress/Backlog Docs, Result Figures

### Added
- `docs/methodology.md` - consolidated, paper-ready methods reference covering dataset construction, detector training/evaluation protocol, kinetic-score formulation and ablation design, edge-latency simulation methodology, and the experimental-design principles applied throughout (never grading a method against itself, relative/forced-choice judgments in place of absolute scores without ground truth, session-level bootstrapping, etc). Synthesizes `architecture.md`, `kinetic_score_opinion.md`, `decisions.md`, `ablation_guide.md`, `yolo_training.md`, and `hardware_targets.md` into one drafting-ready document without duplicating their full reasoning trails.
- `docs/progress_presentation.md` - presentation-shaped summary of current progress: what's built vs. simulated vs. not started, the v2->v3 detector training-protocol finding, the kinetic-score ablation headline result, the measured-vs-simulated edge latency split, a suggested slide order, and a Q&A number sheet.
- `docs/pending_work.md` - explicit, prioritized backlog (kinetic-score referee/human-calibration run, Cognitive Layer, Physics Verification completion, narration/translation/TTS, full SANPO replay, physical edge validation, detector loose ends, export packaging, paper-writing dependencies) with a top-3 recommended next actions section.
- `evaluation/generate_report_figures.py` - regenerates every result figure and a markdown table digest from the current committed benchmark artifacts (`evaluation/benchmarks/yolo26n_version_comparison/`, `.../sanpo_edge_realtime/ten_session_v3_jetson_orin_nano_8gb/`, `.../kinetic_score_eval/`). Output: `evaluation/benchmarks/figures/` (6 PNGs + `results_summary.md`), safe to delete and regenerate.
- `evaluation/benchmarks/kinetic_score_eval/run_2026_08_26/ablation_summary.csv` - machine-readable transcription of the ablation table already published in `docs/ablation_guide.md` §6 (2026-08-26 run, 139 sessions), so it can be plotted; no new data, just a queryable form of the existing published numbers.
- `matplotlib` added to `requirements.txt` for the new figure-generation script.

### Changed
- `docs/progress.md` - refreshed; the previous version predated the kinetic-score ablation work and did not reflect it. Now includes the ablation result, the blinded-referee gap, and the depth-smoothing blocker, and points to `pending_work.md` for the full backlog instead of duplicating it.
- `docs/decisions.md` - status header corrected: the ground-truth-strategy decision was already resolved 2026-08-18 in the body of the entry, but the header still read "Undecided" - fixed the inconsistency and clarified that resolved decisions stay in this file permanently as historical record rather than being deleted once closed.
- `docs/roadmap.md` - added a status note clarifying that actual work after Day 4 diverged from the 14-day plan to prioritize the kinetic-score defense; the day-by-day schedule is historical intent, not a current plan.
- `docs/README.md` - documentation map reorganized into grouped sections (current status / how the system works / research methodology and evidence / datasets and training / history) instead of one flat table, with the new docs and `evaluation/benchmarks/figures/` added to the artifact conventions and maintenance rules.
- `README.md` - project tree updated with the new docs and `evaluation/generate_report_figures.py` / `evaluation/benchmarks/figures/`.

No existing document content was deleted; all changes above are additive or status/consistency corrections.

### Removed / merged (same-day follow-up)
- `docs/verification_guide.md` - merged into `docs/ablation_guide.md` as new §5b (top-3 VLM agreement check via `topk_threat_validation.py`, which `ablation_guide.md` did not previously document) and §7 (troubleshooting). The two guides covered the same commands from different angles; content preserved, file count reduced.
- `docs/decisions.md` - merged into `docs/kinetic_score_opinion.md` as new §10 (formal problem statement, the three-tier ground-truth options table, and the closed decision record). The header inconsistency noted earlier in this entry is now moot since the file no longer exists standalone.
- `docs/sanpo_gap_analysis.md` - merged into `docs/sanpo_dataset.md` as a new "Gap Analysis Workflow" section; the two files covered adjacent SANPO workflows (intake/benchmarking vs. gap analysis) that belong under one dataset doc.
- `docs/roadmap.md` - merged into `docs/architecture.md` as new §11 (Technology Audit: Model and Runtime Choices - the rationale behind §7's SLM/translation/TTS table, plus tracking and orchestration decisions) and §12 (Historical Implementation Plan, superseded by `pending_work.md`).
- `docs/research_paper_prompts.md` - merged into `docs/methodology.md` as a new Appendix of LLM paper-drafting prompts, updated to reflect the current (not yet fully implemented) state of the Cognitive Layer and the pending referee step.
- All cross-references to the five removed files were updated across `docs/`, `README.md`, and `.agents/AGENTS.md`. No content was deleted in these merges, only relocated and, where two files said the same thing, deduplicated. Net effect: 17 files in `docs/` -> 12.

## [2026-08-25] - v3 Detector Default, Frozen Mass Exponent, Enforced 30% Ablation Sample

### Changed
- `src/perception_stack/yolo_tracker.py` - `DEFAULT_MODEL` now points at the finetuned CPE hazard checkpoint v3 (`training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt`) instead of base `models/yolo/base_yolo26n/yolo26n.pt`. The base COCO checkpoint does not contain the CPE hazard classes at all, so any caller that did not pass `model_path` explicitly was silently detecting a different taxonomy than the one the kinetic score's severity table is written against. The path is now absolute (derived from the module location) so the default survives being run from any working directory.
- `src/perception_stack/physics.py` - `SEVERITY_LAMBDA` **frozen at 0.5**; the value is unchanged, but it is now documented as a declared design choice rather than a swept parameter. Dropping Tier-B blinded human labelling from the evaluation plan removed the only tier that could adjudicate between lambda values - the label-free metrics score arrival time, which lambda barely moves. The same applies to `BEHAVIOUR_MULTIPLIER`, which is retained by decision.
- `evaluation/kinetic_ablation.py` - removed the `lam=0.25` and `lam=1.0` arms and the `lam` plumbing in `_score_k()`, which no longer mutates `physics.SEVERITY_LAMBDA`. Six arms remain: K0, linear, no-severity, no-velocity, size, ttc.
- `evaluation/kinetic_ablation.py` - the run is now **restricted to the seeded sample** of `simulation/datasets/sanpo/valid_streams.json` via `--valid-streams`, `--session-fraction` (default 0.30) and `--sample-seed`, rather than scoring whatever CSVs happen to sit in `--csv-dir`. A stale session CSV from an earlier run would otherwise widen the corpus silently and invalidate the session-level bootstrap CIs. Off-sample CSVs are skipped and sampled sessions with no CSV yet are reported as a warning. `--sample-seed` is deliberately separate from `--seed` (bootstrap) so changing one cannot silently change the other. Pass `--session-fraction 1.0` to score every session on disk.
- `tools/download_sanpo_valid_streams.py` - now owns `sample_sessions()` / `sampled_session_ids()` and the `DEFAULT_SESSION_FRACTION` (0.30) and `DEFAULT_SAMPLE_SEED` constants. Moved here from `tools/stream_sanpo_perception.py` because Stage 1 and the ablation must agree on the subset, and this module is stdlib-only so the analysis side can import it without pulling in ultralytics/torch. 139 of the 462 valid streams are selected at the default fraction and seed.
- `tools/stream_sanpo_perception.py` - imports the shared sampler and constants instead of defining its own copy; the `DEFAULT_MODEL` comment claiming YoloTracker defaults to the base checkpoint is corrected.
- `docs/ablation_guide.md`, `docs/architecture.md`, `docs/research_paper_prompts.md` - lambda documented as frozen and reported as a stated limitation; the two lambda arms removed from the arm table.

### Testing
- `python src/perception_stack/physics.py` self-check passes with lambda unchanged at 0.5 (car 4.63x, bus 13.09x a person).
- `evaluation/kinetic_ablation.py` run end-to-end over six synthetic per-session CSVs named after real sampled session IDs plus one off-sample CSV: the off-sample file is skipped, the 133 not-yet-processed sampled sessions are warned about, all six arms score, and `report.md` / `disagreements.json` are written.

## [2026-08-19b] - Local VLM Referees, Mass Exponent Raised, Ablation Run Guide

### Changed
- `src/perception_stack/physics.py` - `SEVERITY_LAMBDA` raised from 0.25 to **0.5** (severity proportional to sqrt(mass)). 0.25 compressed a 171x mass range to 3.6x, which under-weighted the destructive-mass bias the score exists to express; 0.5 gives car 4.6x a person, bus 13x, bus 2.8x a car, without lambda=1's 171x runaway. The self-check now bounds the pure mass-law spread on *both* sides (5x < spread < 60x) rather than only capping it, so a future edit cannot silently flatten mass away either. Note the original compression argument was overstated: with v^2 in the score, kinematics dominate severity in most reorderings, so lambda mostly breaks ties between objects of similar motion.
- `evaluation/vlm_referee.py` - referees are now **local models**, not hosted APIs. The three vendor-specific client functions collapse into one `ask_local()` that speaks the OpenAI-compatible `/chat/completions` shape over stdlib urllib, which vLLM, Ollama, llama.cpp `llama-server`, LM Studio and SGLang all serve. No API keys, no client SDK dependency, no frame leaving the machine. Defaults are three different pretraining lineages (Qwen2.5-VL-7B, InternVL3-8B, Gemma-3-12B) on ports 8001-8003; `--endpoint NAME=URL` and `--model NAME=ID` retarget them, so a single-GPU box can run the three referees sequentially against one port. `temperature=0` so a re-run reproduces the ballot.
- `evaluation/kinetic_ablation.py` - added `lambda = 0.25` and `lambda = 1.0` ablation arms. `_score_k()` swaps `physics.SEVERITY_LAMBDA` and restores it rather than duplicating the severity law, so the arms cannot drift from production.
- `tools/stream_sanpo_perception.py` - frame downloads now run through a thread pool (`--download-workers`, default 8). One HTTPS GET per frame makes the fetch latency-bound, not bandwidth-bound; failures are re-raised so a half-downloaded session fails loudly instead of producing a short CSV.
- `requirements.txt` - dropped the commented-out hosted-VLM SDKs; the referee needs no client library.

### Added
- `docs/ablation_guide.md` - the run guide: what each ablation arm asks, environment setup and self-checks, the Stage-1 streaming run (sampling, disk, resume, the stride/fps trap), the label-free metrics pass, serving three local VLMs (including the one-GPU sequential path), the blinded judging run, the human calibration subset with kappa interpretation thresholds, and how to write up each of the three possible per-arm outcomes.

### Testing
- `evaluation/vlm_referee.py --self-check` now stands up a throwaway `http.server`, calls `ask_local()` against it, and asserts the request path, model, `temperature`, base64 image part and text part - the payload shape is the one thing that silently breaks when a serving stack is swapped.

## [2026-08-19] - Mass-Derived Severity, 30% SANPO Ablation Pipeline, Multi-VLM Referee

### Changed
- `src/perception_stack/physics.py` - class severity is now **derived from real-world mass** instead of hand-tuned: `severity(c) = behaviour(c) x (mass(c) / 70 kg) ** lambda`, lambda = 0.25, exposed as `class_severity()` with `CLASS_SEVERITY` materialised from it. lambda = 0.25 compresses the 171x class mass range to ~5x severity spread; raw mass (lambda=1) gives ~600x and lets a distant bus outrank an imminent pedestrian. Fitting the previous hand-tuned table against log mass recovers lambda ~ 0.18 (R^2 = 0.50), so the mass law restates beliefs already encoded there without the arbitrariness. A sparse `BEHAVIOUR_MULTIPLIER` covers hazards mass cannot express (motorcycle/dog erratic motion, bollard/pole trip height, traffic light/stop sign mounted above head height), and four massless trip hazards (stairs, pothole, puddle, crosswalk) keep explicit weights in `STATIC_HAZARD_SEVERITY` outside the mass law rather than being fudged into it.
- `src/perception_stack/physics.py` - `kinetic_score()` gained keyword-only `gamma`, `size_exponent` and `bbox_area_px` parameters. The defaults (`gamma=2.0`, `size_exponent=0.0`) reproduce shipped K0 exactly; the ablation drives the same production function rather than maintaining a second implementation that can drift. The size term uses apparent size `A_px / d` normalised by `SIZE_REFERENCE_PX_PER_M` and is **off by default**: bounding-box area is ~99% predictable from class + metric depth by projective geometry, so it is redundant for labelled objects and is carried as an ablation arm, not a production term.

- `tools/visualize_stage1.py` - dropped its private six-class severity table and now calls the production `kinetic_score()`. A visualiser that ranks objects differently from the pipeline it visualises is worse than no visualiser.

### Added
- `tools/stream_sanpo_perception.py` - runs Stage 1 over a seeded 30% sample of the 462 SANPO valid streams **streamed directly from the public GCS bucket**, with no local copy of SANPO. One session's frames are pulled to scratch, converted to a per-session CSV, then deleted, so peak disk stays at one session regardless of run length. Resumable (finished sessions are skipped; CSVs are written to `.partial` and renamed only on success) and failure-isolated, for unattended `nohup` runs on a shared SSH box. Writes `<session>.frames.json` mapping each CSV `frame_idx` back to its GCS object so the referee can re-fetch a handful of images later. Frames are strided at download time and `fps` is scaled by the stride - without which every closing velocity would come out `stride`x too fast.
- `evaluation/kinetic_ablation.py` - ablates K0's terms (gamma=1, no severity, no velocity, apparent-size arm, plain time-to-hazard) and scores each variant on the label-free metrics of `kinetic_score_opinion.md` §5 plus the automatic encounter label of §6. All CIs are a percentile bootstrap resampling **whole sessions**, since frames within a session are autocorrelated. Exports the frames where variants pick different top objects as `disagreements.json` (blind) and `disagreements_key.json` (which formula picked what - never shown to a referee).
- `evaluation/vlm_referee.py` - blinded forced-choice referee over those disagreement frames, using three VLMs from three vendors (Claude / Gemini / GPT) because agreement between models sharing training data is a shared prior, not truth. Referees see the annotated RGB frame and a neutral object list in randomised order - no scores, no formula names. Reports per-variant win rates, pairwise Cohen's kappa between referees, and kappa against a human-labelled calibration subset (`--human-template`), which the report marks as MISSING until it exists. Ballots are written per case so an interrupted API run resumes.
- `requirements.txt` - `scipy` (Kendall tau for rank stability) and the optional referee SDKs.

### Testing
- Extended the `physics.py` self-check with the mass law (person = 1.0 by construction, monotone in mass, lambda compresses the mass range below 6x, the behaviour multiplier is load-bearing, unknown classes fall back) and with the new exponents (defaults reproduce K0; the size term stays inert unless `size_exponent` is set).
- Added `--self-check` self-tests to `evaluation/kinetic_ablation.py` (encounter labels exclude a stationary bus and include a closing pedestrian; every velocity-aware variant ranks the closing pedestrian first; the no-velocity arm ranks the parked bus first - the failure it exists to expose; degenerate bootstrap) and `evaluation/vlm_referee.py` (vote parsing rejects hallucinated and unparseable ids rather than guessing, kappa bounds, baseline/variant/neither tallying).

## [2026-08-18] - Kinetic Score Decision: Keep K0, Remove Dummies

### Documentation
- Added `docs/kinetic_score_opinion.md` — recommendation for choosing and defending the kinetic score. Concludes: **keep K0**, whose `v²` term is the only element encoding *consequence* rather than *arrival time* (which the reflex TTC gate at `events.py:217` already covers). Documents how to evaluate with **no ground truth**: six label-free metrics (flicker rate, rank stability under depth perturbation, temporal smoothness, tie rate, complementarity with SLM-1, future self-consistency), automatic "encounter" labels derived from measured future distance, and forced-choice pairwise comparison with a blinded referee restricted to the ~5% of frames where formulas disagree. Rejects multi-VLM polling as primary evidence (polling measures agreement not accuracy; VLM errors are correlated; a still frame carries neither metric depth nor closing velocity) and reframes the VLM as a blinded *referee* calibrated against a human gold set with Cohen's κ reported.

### Changed
- `src/perception_stack/physics.py` — `compute_velocity()` now takes the **least-squares slope** across the whole depth window instead of differencing only the two endpoints, using every sample to cut variance without adding state or tuning knobs. Reverts to endpoint differencing via `use_least_squares=False`. Motivation: velocity is a differentiated depth estimate and `kinetic_score()` squares it, so at 10% depth error K0's score error is +741% versus +190% for a linear-velocity variant. Partially addresses `architecture.md` §10.1, which prescribes a Kalman filter that was never implemented.
- `docs/architecture.md` — **withdrew the unevidenced claim** that "K4 (hybrid momentum + TTC) is the leading candidate." No benchmark was ever run; `evaluation/benchmarks/kinetic_score_eval/` contained only a README.

### Removed
- Deleted `kinetic_score_k1_ttc`, `kinetic_score_k2_linear`, `kinetic_score_k3_quad_distance`, `kinetic_score_k4_hybrid`, `kinetic_score_k5_sigmoid`, `_KINETIC_FORMULAS` and `kinetic_score_all` from `src/perception_stack/physics.py`. K1–K5 were dummies, and measurement showed they were not even independent ones: within a class K1/K2/K5 are *exactly* rank-identical to 1/TTC (Spearman ρ = 1.0000, identical rank vectors over 500 points) and K3 is ρ = 0.9998 — five labels for two behaviours. Cross-class pairwise rank agreement was ≥ 0.945 for every pair except K5. Nothing outside `physics.py` referenced them.
- Deleted `evaluation/kinetic_score_comparison.py` and `evaluation/threat_score_eval.py` — circular by construction (ground truth computed by re-running the formula being scored) with a saturated discriminating metric that passed five of six candidates at its own ρ > 0.7 gate. K0 is now defended by ablation of its own terms; see the rewritten `evaluation/benchmarks/kinetic_score_eval/README.md`.

### Testing
- Added an `assert`-based `_demo()` self-check to `src/perception_stack/physics.py` (`python src/perception_stack/physics.py`) pinning velocity estimation on clean input, the retreating-object clamp, short-history guards, K0 monotonicity in distance/velocity/severity, the `EPSILON` guard, and — the point of the change — that least-squares beats endpoint differencing when one depth sample is corrupted.

## [2026-08-13] - Kinetic Score Ground Truth Decision Record

### Documentation
- Added `docs/decisions.md` — open design-decision log. First entry: "Ground Truth Strategy for Kinetic Score Formula Selection," documenting a circularity problem in `evaluation/threat_score_eval.py` (ground truth is computed by re-running the same formula being scored) and a scale-comparability bug across K0–K5 (K5 is bounded to `[0, severity]` and can never cross `high_k=5.0`, unlike unbounded K0/K3). Records three candidate ground-truth tiers (automatic kinematic "encounter" labels, blinded human judgment, calibrated VLM-as-annotator) and their tradeoffs, pending a team decision. Also flags that `docs/architecture.md`'s claim of "K4 leading candidate" currently has no benchmark evidence behind it (`evaluation/benchmarks/kinetic_score_eval/` contains only a README).

## [2026-08-05] - Manual Kinetic Score Inspector

### Evaluation
- Added `tools/manual_score_inspector.py` — generates a self-contained interactive HTML file for manual frame-by-frame verification of kinetic scores. For each sampled SANPO frame it renders the RGB image with colour-coded bounding boxes (reflex=red, cognitive=orange, ignore=green), a table of all K0–K5 scores per object with routing badges, and the objective ground-truth column (actual depth at T+lookahead). The user can ✅ approve, ❌ flag, or ⏭ skip each frame; judgements persist in `localStorage` and can be exported as a JSON audit file.


### Evaluation
- Added `evaluation/kinetic_score_comparison.py` — benchmark harness for six kinetic score formula candidates (K0–K5). Produces per-formula distribution statistics (mean, P50, P95, P99, IQR), Pearson/Spearman TTC correlation, routing sensitivity (reflex/cognitive/ignore %), monotonicity check, and a ranked Markdown report. Generates Matplotlib plots when available.
- Added `evaluation/threat_score_eval.py` — routing F1/Precision/Recall evaluator. Uses K₊₂ (same track at T+lookahead) as ground truth to score each formula's routing decisions. Primary safety metric is Reflex Recall (missed reflex = safety failure).
- Added `evaluation/benchmarks/kinetic_score_eval/` — dedicated benchmark output folder following the established benchmark folder convention. Contains README with reproduction steps and decision criteria.
- Registered `kinetic_score_eval/` in `evaluation/benchmarks/README.md` index.

### Physics — Additive Formula Candidates (non-breaking)
- Extended `src/perception_stack/physics.py` with five alternative kinetic score functions: `kinetic_score_k1_ttc`, `kinetic_score_k2_linear`, `kinetic_score_k3_quad_distance`, `kinetic_score_k4_hybrid`, `kinetic_score_k5_sigmoid`. All are evaluation-only; the production `kinetic_score()` (K0) is unchanged.
- Added `kinetic_score_all()` dispatcher returning all six formula scores for one object — used by the benchmark harness.

### Shared Data Contract — FactSheet Redesign
- Rewrote `src/shared/fact_sheet.py` with a lean, phone-derivable `DetectedObject` schema:
  - **Removed:** `intent_label` (HEADSUP training-data only; not available at phone inference), `hallucination_filtered` (internal pipeline flag).
  - **Renamed:** `threat_score` → `kinetic_score` (explicit about what it is), `object_id` → `track_id` (aligns with `events.py`), `is_scene_stable` → `scene_stable`.
  - **Added:** `bearing` (human-readable direction string from `bearing_label()`), `route` (Threat Prioritizer lane: reflex/cognitive/ignore).
  - Added `to_slm_user_message()` method — returns the FactSheet as a structured JSON dict for the SLM-1 SFT training record user slot (structured JSON, not flat string).

### Documentation
- Updated `README.md` project structure with new evaluation scripts and `kinetic_score_eval/` benchmark folder.

### Repository Cleanup
- Deleted empty placeholder directories from `src/`, `simulation/`, and `training/` to declutter the workspace and reflect the current state of implementation.
- Updated the `README.md` project structure tree to match the pruned directory state.

## [2026-07-22] - Documentation Ownership and Hardware Targets
### Documentation
- Added `docs/hardware_targets.md` as the source of truth for observed GB10 training-host specs, the planned Jetson Orin Nano 8GB target, runtime budgets, and measured-versus-simulated evidence.
- Moved `Roadmap.md` to `docs/roadmap.md` and normalized the progress checklist path as `docs/progress.md`.
- Renamed the SANPO documents to `docs/sanpo_dataset.md` and `docs/sanpo_gap_analysis.md` so their names match their responsibilities.
- Merged artifact naming and cleanup rules into `docs/README.md`, then removed the overlapping `docs/artifact_structure.md`.
- Updated the main README project tree and documentation index for the consolidated structure.
- Removed the stale Snapdragon-specific hardware target from the architecture and corrected the paper prompt so simulated Jetson results are not described as physical-device measurements.


## [2026-07-22] - Reflex Layer Bridge and Progress Checklist
### Runtime Pipeline
- Added `src/reflex_layer/reflex.py` to convert reflex-route `ThreatEvent`s into `ReflexResult`s and invoke Physics Verification without waiting for cognitive SLM output.
- Updated `src/reflex_layer/__init__.py` exports for the new Reflex Layer bridge.

### Progress Tracking
- Added `docs/progress.md` as a living project checklist covering detector status, dataset intake, ThreatEvent routing, Reflex Layer status, SLM/TTS gaps, and immediate next steps.

### Tests
- Added `tests/test_reflex_layer.py` for TTC override behavior, non-reflex suppression, and high-K physics fallback behavior.

### Documentation
- Updated README structure, architecture notes, roadmap, research prompts, and changelog to reflect the Reflex Layer bridge and current project progress.

## [2026-07-22] - ThreatEvent Routing Contract
### Runtime Pipeline
- Added `src/threat_prioritizer/events.py` with `PerceivedObject`, `ThreatEvent`, `ThreatFrame`, TTC calculation, kinetic-score routing, Physics Verification registry output, and ReflexResult conversion.
- Added `tools/build_threat_events.py` to convert Stage 1 perception CSV files into non-ignore `ThreatEvent` JSONL plus route/class summaries.

### Tests
- Added `tests/test_threat_prioritizer.py` and `tests/__init__.py` with standard-library unit tests for TTC, reflex routing, cognitive near-static hazard routing, ignore routing, and CSV-to-JSONL conversion.

### Documentation
- Updated architecture, roadmap, README structure, and research prompts to document the formal perception-to-physics event contract.

## [2026-07-22] - Roadmap Technology Audit and Edge Export Step
### Planning
- Updated `docs/roadmap.md` with the current detector status, SANPO 10-session latency result, and revised model recommendations: keep YOLO26n v3, prefer Qwen3-1.7B non-thinking mode for SLM-1, use template-first critical narration with optional Phi-4-mini, and keep IndicTrans2 distilled/phrase-table translation for the prototype.

### Tooling
- Added `training/scripts/export_yolo26n_edge.py` and `models/yolo/cpe_yolo26n_hazards_v3_from_base/` to export the preferred v3 checkpoint to ONNX, TensorRT engine, OpenVINO, or TorchScript with modern Ultralytics `quantize` arguments and a model-registry export manifest.
- Updated the training script export path to use `--export-quantize` instead of deprecated INT8 export aliases.

### Documentation
- Updated README, YOLO training docs, training workflow docs, architecture notes, and research-paper prompts to point at the detector export/runtime comparison as the next step.

## [2026-07-22] - README Structure Maintenance Rule
### Documentation
- Added a `.agents/AGENTS.md` rule requiring the main `README.md` Project Structure section to be reviewed and updated whenever new files, folders, scripts, datasets, model registry entries, benchmark artifacts, or major docs are added.
- Updated the main `README.md` project map to include current docs, SANPO valid-stream metadata, YOLO training scripts, edge benchmark tools, model registry folders, and benchmark artifact locations.

## [2026-07-22] - SANPO 10-Session Edge Latency Benchmark
### Evaluation
- Ran v3 over 10 bounded SANPO-Real valid streams with the `jetson_orin_nano_8gb` simulation profile, processing 300 RGB+depth frames total.
- Saved individual metrics and aggregate reports under `evaluation/benchmarks/sanpo_edge_realtime/ten_session_v3_jetson_orin_nano_8gb/`.
- Average simulated p95 latency was `34.31 ms`; worst-session simulated p95 latency was `41.96 ms`, with all sessions passing the `<50 ms` detector/reflex budget.

## [2026-07-22] - Edge Device Simulation Profile
### Evaluation
- Added edge-profile support to `tools/benchmark_edge_realtime.py`, including a conservative `jetson_orin_nano_8gb` simulation profile with `4.0x` latency scaling and `3.0 ms` per-frame sensor/memory overhead.
- Ran the SANPO valid-stream smoke benchmark under the Jetson Orin Nano 8GB profile; simulated p95 total latency was `36.96 ms`, passing the `<50 ms` detector/reflex budget.

## [2026-07-22] - SANPO Valid-Stream Edge Smoke Test
### Evaluation
- Documented the public SANPO GCS bucket layout, valid-stream local convention, label metadata, and bounded download strategy in `docs/sanpo_dataset.md`.
- Added `tools/download_sanpo_valid_streams.py` to download selected paired SANPO-Real RGB/depth frames from `valid_streams.json` without copying whole multi-GB sessions.
- Downloaded valid stream `0xCqEk5hjEvrygxu26MZkieSv45D_gaJ` and saved v3 full RGB+depth edge benchmarks under `evaluation/benchmarks/sanpo_edge_realtime/`.
- Fixed SANPO depth handling for native `960 x 960` `.float16.gz` maps by inferring shape and scaling RGB-space boxes into depth-map coordinates.

## [2026-07-22] - Edge Real-Time Benchmark Utility
### Evaluation
- Added `tools/benchmark_edge_realtime.py` to measure YOLO26n + ByteTrack streaming latency, optional depth loading, depth post-processing, effective FPS, and real-time/reflex budget pass-fail status.
- Saved the v3 detector-only GB10 benchmark under `evaluation/benchmarks/yolo26n_version_comparison/` with p95 total latency of `9.65 ms` per processed frame at `frame_step=3`.

## [2026-07-22] - Local Artifact Relocation
### Project Structure
- Moved the pretrained YOLO26n base checkpoint into `models/yolo/base_yolo26n/yolo26n.pt` and updated training, comparison, perception, notebook, and documentation references.
- Moved SANPO valid stream metadata into `simulation/datasets/sanpo/valid_streams.json` and updated gap-analysis consumers.

## [2026-07-22] - Documentation Cleanup and YOLO Guide Consolidation
### Documentation
- Replaced stale YOLO v1/v2 planning docs with `docs/yolo_training.md` as the current source of truth for dataset roots, checkpoints, training commands, evaluation commands, and version-comparison artifacts.
- Updated the docs index, architecture, artifact-structure guide, research-paper prompts, training README, and benchmark README to reflect the v3-from-base checkpoint and cumulative evaluation results.
- Removed obsolete links to documentation files that no longer exist.

## [2026-07-22] - YOLO26n Version Comparison Artifacts
### Evaluation
- Ran all-class held-out test evaluation for the v3-from-base checkpoint.
- Added cumulative v1/v2/v3 comparison artifacts under `evaluation/benchmarks/yolo26n_version_comparison/`, including overall metrics, per-class mAP tables, and retained-class base-YOLO deltas.

## [2026-07-22] - v3 From-Base Training Setup
### Training
- Removed interrupted YOLO training run folders while preserving successful v1/v2 checkpoints and evaluation artifacts.
- Documented the v3 retention-repair path that starts from base `models/yolo/base_yolo26n/yolo26n.pt`, reuses the cleaned v2_full dataset, and recommends extra Roboflow data for `truck`, `bicycle`, `dog`, `stop sign`, and `bus`.

## [2026-07-20] - AutoBatch Startup Fix
### Training
- Updated the GB10 training script to keep cuDNN benchmark disabled during `batch=-1` AutoBatch search so Ultralytics can probe large batch sizes instead of falling back to batch 16.

## [2026-07-20] - v2 Motorcycle Source Fill
### Training Data
- Added a dedicated `motorcycle` Roboflow detection source and moved the active clean rebuild target to `data/yolo_finetune_v2_full` after auditing that the retained COCO-style source had zero motorcycle positives.

## [2026-07-20] - v2 Missing-Class Source Fill
### Training Data
- Added verified Roboflow detection sources for `crosswalk` and `puddle` so the active 17-class detector is not trained with zero positive examples for those classes.
- Added `training/scripts/rebalance_yolo_splits.py` to move deterministic train examples into validation/test splits for train-only Roboflow sources before final all-class evaluation.

## [2026-07-20] - v2 Balanced Dataset Rebuild
### Training Data
- Fixed Roboflow merge accounting so class image caps are not double-counted while merging multi-source v2 data.
- Added explicit `pole` and `bollard` caps/minimums and pointed the active YOLO data config at `data/yolo_finetune_v2_balanced` for a clean v2 rebuild.

## [2026-07-20] - Roboflow Source Version Verification
### Training Data
- Verified the provided Roboflow Universe source URLs with the Roboflow SDK, updated the v2 manifest to the latest usable versions, and removed the invalid stairs source whose requested version was not found.
- Marked `tiger-dataset/dog-ukbxr` as unavailable for download because the project currently has no generated dataset versions; dog examples remain sourced through the COCO-style retained dataset until another dog dataset is provided.

## [2026-07-20] - v2 Roboflow Source Manifest Update
### Training Data
- Updated the active Roboflow Universe manifest with additional stairs, pothole, dog, bench, and retained COCO-class sources while preserving the existing pole and bollard datasets for retention.
- Tightened Roboflow label intake to accept only 5-column YOLO detection rows so segmentation polygon labels are not accidentally treated as bounding boxes.

## [2026-07-20] - Roboflow Class-Balanced Intake Caps
### Training Data
- Added per-class image caps to `training/scripts/download_roboflow_universe.py` so multiple Roboflow Universe URLs can be supplied per class without letting oversized source datasets dominate v2 fine-tuning.
- Documented the v2 class cap policy and SANPO domain-alignment guidance in `docs/plan_yolo_v2_finetune.md`, and added class limit/minimum examples to the Roboflow manifest template.

## [2026-07-20] - Retained COCO Class Expansion
### Training & Perception
- Added `dog` and `bench` back into the active CPE YOLO taxonomy as retained COCO navigation-obstacle classes, expanding the active detector head to 17 classes.
- Updated training, dataset-download, evaluation, retention-comparison, perception-filtering, physics severity, and planning docs to use retained class IDs 0-10 and custom hazard IDs 11-16.

## [2026-07-20] - CPE Taxonomy Simplification
### Training & Perception
- Reduced the active CPE YOLO hazard taxonomy from 16 to 15 classes by removing the broad head-height obstacle catch-all from training configs, scripts, perception filters, and docs.
- Kept `puddle` separate from `pothole` because it represents a different navigation risk, while noting that v2 data quality should decide whether it remains active long term.

## [2026-07-20] - GB10 YOLO26n Training Throughput Optimization
### Training
- Retuned `training/scripts/train_yolo26n_hazards.py` for the Grace-Blackwell GB10 lab server with RAM dataset caching, AutoBatch defaults, CPU-saturating worker selection, disabled training-loop validation/plots, skipped default export, and startup throughput reporting.
- Updated `docs/plan_yolo_v2_finetune.md` with the GB10 high-throughput profile, the 6.7GB dataset-cache estimate, and the fast v2 command/checklist.

## [2026-07-20] - Project Branding Cleanup
### Documentation & UI Text
- Removed visible legacy project-branding references from repository documentation and the Stage 1 visualization overlay, standardizing project-facing text on Composite Perception Engine (CPE).

## [2026-07-20] - Repository Artifact Cleanup & Model Registry
### Repository Hygiene
- Removed generated Ultralytics plot/preview artifacts from tracked training runs while preserving compact metrics and checkpoint metadata.
- Added artifact conventions, `training/README.md`, `evaluation/benchmarks/README.md`, and `models/yolo/README.md`; the conventions now live in `docs/README.md`.
- Added a local model registry entry under `models/yolo/cpe_yolo26n_hazards_v1/` and compact v1 benchmark summaries under `evaluation/benchmarks/yolo26n_hazards_v1/`.
- Updated `.gitignore` to keep datasets, secrets, binary weights, training plots, and local Roboflow manifests out of version control.

## [2026-07-20] - YOLO26n v2 Fine-Tuning Plan & Retention Comparison
### Training & Evaluation
- Added `docs/plan_yolo_v2_finetune.md`, a checklist-driven plan for continuing from the v1 `best.pt` checkpoint, repairing weak `stairs`/`pothole` performance, and adding retained-class validation data.
- Added `training/scripts/compare_yolo26n_retention.py` to compare a fine-tuned CPE checkpoint against base `models/yolo/base_yolo26n/yolo26n.pt` for retained COCO classes using class-name remapping, while validating the candidate on all active CPE classes.
- Updated `docs/architecture.md`, `docs/plan_yolo_finetune.md`, and `docs/research_paper_prompts.md` to document the v2 staged fine-tuning and retained-class validation workflow.

## [2026-07-20] - YOLO26n Hazard Checkpoint Evaluation
### Training & Evaluation
- Added `training/scripts/evaluate_yolo26n_hazards.py` to validate a fine-tuned checkpoint on `val` or `test`, export per-class metrics, generate a Markdown report, and save prediction previews for the new navigation-hazard classes.
- Documented the held-out test evaluation command in `docs/plan_yolo_finetune.md`.

## [2026-07-15] - Roboflow Universe Dataset Intake for YOLO26n
### Training Data
- Optimized `training/scripts/train_yolo26n_hazards.py` for GPU fine-tuning with CUDA runtime reporting, TF32/cuDNN benchmark flags, multi-GPU device parsing, integer batch handling, AMP controls, cache mode, and optional torch compile.
- Fixed the YOLO26n training preflight by using integer batch sizes and normalizing Roboflow labels to 5-column detection format before training.
- Added `.env` loading to `training/scripts/download_roboflow_universe.py` so SSH users can keep `ROBOFLOW_API_KEY` outside source code while running Roboflow Universe downloads.
- Added `training/scripts/download_roboflow_universe.py` to download selected Roboflow Universe datasets, remap source labels into the CPE hazard taxonomy, and merge YOLO images/labels into `data/yolo_finetune`.
- Added `training/configs/roboflow_universe_sources.example.json` as a manifest template for Universe workspace/project/version sources and class mappings.

### Perception Stack
- Aligned `src/perception_stack/yolo_tracker.py` and `src/perception_stack/physics.py` with the current hazard fine-tuning taxonomy so newly trained hazard classes are not filtered out at inference.

### Documentation
- Updated `docs/plan_yolo_finetune.md` with the SSH Roboflow workflow and minimum/preferred image-count targets for each new hazard class.

## [2026-07-14] - YOLO26n Hazard Fine-Tuning & Multi-Range Gap Analysis
### Tooling & Evaluation
- **Multi-Range Gap Analysis Script**: Added `tools/gap_analysis_experiments.py` to evaluate and compare 4 distinct depth ranges across 20 valid streams using all YOLO classes (whitelist disabled).
- **JSON Hard Example Logging**: Configured the script to save session ID and frame number mappings of hard examples in a range-specific JSON format (`hard_examples.json`) rather than writing large raw frame images to disk.
- **Experimental Reports**: Configured the script to output range-specific CSV summaries and a master comparison report (`distance_metric_comparison.md`) under the `gap_analysis_output/` folder.
- **Workspace Cleanup**: Cleaned up the workspace by deleting legacy visual output PNGs.

### Training
- Added `training/scripts/train_yolo26n_hazards.py`, a lab-GPU training script that fine-tunes `models/yolo/base_yolo26n/yolo26n.pt` on CPE navigation hazards while preserving the nano architecture, freezing early layers, applying rare-class weighting, and optionally exporting INT8 edge artifacts with an export-only mode for finished checkpoints.
- Added `training/configs/cpe_hazard_classes.yaml` with a CPE hazard taxonomy covering retained COCO safety classes plus non-COCO hazards such as `pole`, `bollard`, `stairs`, `crosswalk`, `pothole`, and `puddle`.

### Documentation
- Updated `docs/architecture.md` and `docs/research_paper_prompts.md` to document the multi-range gap analysis capabilities, whitelist removal, and JSON logging format.
- Updated the architecture and research-paper prompts to describe the edge-preserving YOLO26n fine-tuning workflow.

## [2026-07-12] — YOLO Gap Analysis Expansion & Automation
### Tooling & Inference
- **Expanded YOLO Gap Analysis Notebook**: Overhauled `notebooks/sanpo_yolo_gap_analysis.ipynb` with centralized configuration, robust morphological object extraction, 5-panel visualizations, and automated fine-tuning advice reports.
- **GCS-Aware Stream Fallback**: Integrated streaming logic in `SANPOLoader` to dynamically fetch images and depth maps from Google Cloud Storage anonymously when local files are missing.
- **Automated Verification**: Created verification test scripts to execute and validate the notebook code cells under headless conditions.

### Architecture & Documentation
- **Architecture Doc Update**: Documented the new error analysis and dataset gap analysis pipeline in `docs/architecture.md`.
- **Research Prompt Curation**: Added prompt guidelines for writing methodology sections on connected component extraction and hard example mining in `docs/research_paper_prompts.md`.

## [2026-07-10] — Implementation Plans for YOLO Fine-Tuning & Gap Analysis
### Documentation
- **`docs/plan_yolo_finetune.md`**: Detailed implementation plan for fine-tuning YOLO26n. Includes class curation rationale (dropping non-navigation COCO classes while adding `pole`, `stairs`, `crosswalk`, and surface hazards), SANPO panoptic-to-YOLO label conversion pipeline, dataset split strategy, training script outline, and benchmark targets.
- **`docs/sanpo_gap_analysis.md`**: Step-by-step guide for running the gap analysis notebook across SANPO. Documents all config parameters, output folder management (frame sampling to prevent bloat), visualization legend, CSV column definitions, and post-run analysis queries.

## [2026-07-10] — SANPO Analysis & YOLO Class Curation
### Research & Tooling
- **SANPO dataset clarified**: SANPO is a Google Research egocentric navigation dataset (701 real + 1961 synthetic sessions), NOT an Indian dataset. Confirmed 30-class taxonomy covering sidewalks, poles, crosswalks, pedestrians, and cyclists.
- **YOLO class curation**: Removed region-specific classes (autorickshaw); class list finalized to 14 navigation-critical COCO classes plus depth-detected unlabeled blobs.
- **Gap Analysis Notebook**: Created `notebooks/sanpo_yolo_gap_analysis.ipynb` to empirically identify which scene objects are detected by depth but missed by the current YOLO26n COCO whitelist. Outputs per-frame visualizations (3-panel: RGB+YOLO, depth heatmap, gap-only), a frequency table, and a CSV of gap statistics.

## [2026-07-10]
### Architecture & Documentation
- **Knowledge Distillation Strategy**: Added formal documentation to `architecture.md` outlining the Teacher-Student SLM training pipeline for edge constraints.
- **YOLO Domain Fine-tuning**: Added explicit strategy for fine-tuning YOLO26n on street-level hazards rather than generic COCO classes.
- **Documentation Refactoring**: Consolidated `perception_stack.md`, `physics_verification.md`, `phase_1_breakdown.md`, and `scaffolded_components.md` into the master `architecture.md` for a single source of truth.
- **Research Prompts**: Created `research_paper_prompts.md` to aid in generating the academic paper via LLMs.

## [2026-03-30]
### Perception Stack
- **Centre-patch depth sampling**: Added 20% patch to fix bbox edge bleed (`src/perception_stack/depth_loader.py`)
- **Occlusion exclusion masking**: Added `exclude_boxes` param to `median_depth_in_box` (`src/perception_stack/depth_loader.py`)
- **Sparse depth 3x fallback**: Auto-expands patch if no valid pixels found (`src/perception_stack/depth_loader.py`)
- **Class whitelist filter**: `ALLOWED_CLASSES` set to drop non-navigation COCO classes (`src/perception_stack/yolo_tracker.py`)
- **Geometric validation**: Added `is_valid_detection()` for aspect ratio, area, and edge FP checks (`src/perception_stack/yolo_tracker.py`)
- **3-pass depth rescoring**: Added `depth_rescore()` function for size sanity, conf, and depth NMS (`src/perception_stack/yolo_tracker.py`)
- **Grid-based unlabeled obstacle sweep**: Replaces single-ROI scan with 5 columns (`src/perception_stack/pipeline.py`)
- **Front-to-back occlusion ordering**: Added sorting and cumulative occluder list (`src/perception_stack/pipeline.py`)
- **Nth-frame skip**: Added `frame_step` param (e.g., `frame_step=3` for ~67% compute reduction) (`src/perception_stack/pipeline.py`, `tools/run_perception.py`)
- **Streaming generator architecture**: `run_perception_stream` generator yields per-frame, breaking RAM bottleneck (`src/perception_stack/pipeline.py`)
- **tqdm progress bar**: Replaces print every 50 frames (`src/perception_stack/pipeline.py`)
- **StreamingCSVWriter**: Context manager for incremental disk writes paired with generator (`src/perception_stack/csv_writer.py`)
- **Batch tensor physics**: Vectorized functions `batch_compute_bearing` & `batch_kinetic_score` (`src/perception_stack/physics.py`)
- **CLASS_SEVERITY deduplication**: `fact_sheet_builder` now imports from `physics.py` (`src/perception_stack/fact_sheet_builder.py`)

### Tools
- **CLI updates**: Added `--frame_step` arg and streaming API to `run_perception.py`, using `StreamingCSVWriter` (`tools/run_perception.py`)

## [2026-03-27]
### Perception Stack
- **Unlabeled obstacle detection**: Single central-ROI scan for poles/bollards via depth scan (`src/perception_stack/pipeline.py`)

## [2026-03-25]
### Architecture
- **Dual-SLM system design**: Initial CPE architecture finalized with Reflex, Cognitive, and Physics Verification layers (`architecture.md`)

### Perception Stack
- **YOLO26n + ByteTrack integration**: Basic tracking pipeline (`src/perception_stack/yolo_tracker.py`)
- **SANPO + UASOL depth map loading**: Auto-detects `.float16.gz` vs `.png` (`src/perception_stack/depth_loader.py`)
- **Stage 1 pipeline**: RGB+Depth → CSV batch mode with `run_perception()` (`src/perception_stack/pipeline.py`)
- **CSV canonical schema**: `CSV_FIELDS` contract defined and writer implemented (`src/perception_stack/csv_writer.py`)
- **Physics calculations**: Scalar functions for bearing, velocity, and kinetic score (`src/perception_stack/physics.py`)
- **Stage 2 fact sheet builder**: CSV → JSONL with K0 and K+2 lookahead (`src/perception_stack/fact_sheet_builder.py`)

### Physics Verification
- **Adjudication logic**: Added 4-rule judge and RL rewards via `PhysicsVerification` class (`src/physics_verification/physics_verification.py`)

### Tools
- **run_perception.py**: Initial batch API CLI for Stage 1 (`tools/run_perception.py`)
- **run_fact_sheets.py**: Initial implementation for Stage 2 (`tools/run_fact_sheets.py`)
