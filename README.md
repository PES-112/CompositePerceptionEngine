# Composite Perception Engine (CPE)

A phone-based hazard alert for blind pedestrians. It ranks nearby objects by **when** they will reach
the walker, **how bad** a collision would be, and **what** the scene says about them. Every learned
decision sits inside a deterministic physics safety envelope. The target device is an iPhone 17 Pro
Max; evaluation replays the SANPO egocentric dataset.

**Design:** [`docs/architecture.md`](docs/architecture.md) · **Status:**
[`docs/progress.md`](docs/progress.md) · **Backlog:** [`docs/pending_work.md`](docs/pending_work.md)

## Pipeline (v2)

```
SANPO replay: RGB + SANPO depth (as is) + camera pose   (product: phone camera + external LiDAR)
  → Perception        YOLO26n v3 + ByteTrack + depth per box                         ✅
  → Sensor fusion     Kalman track per object, ground plane, head motion removed      ✅ (pose unvalidated on SANPO)
  → Scene context     walkable-surface segmentation (road / curb / sidewalk), 2–5 Hz  ⬜
  → Threat engine
       safety envelope   TTC ≤ 1 s in the walking cone → immediate alert             ✅
       when              GRU arrival predictor (runtime wrapper)                     ✅
       how bad           class severity (mass law), compressed by γ                  ✅
       what              segmentation context                                        ⬜ (interface ✅)
       fusion            priority = P(arrive) × severity^γ × context                 ✅
  → Announcement manager  cooldown, hysteresis, override preempts, drop stale         ✅
  → Narration         scripted phrases → pre-rendered audio clips                     ✅ templates · ⬜ audio
```

✅ built · ⬜ planned. There is no language model at runtime; the v1 SLM cognitive and narrator
layers were dropped on 2026-10-14. The v1 chain (K0 routing → stub → judge) is kept only as the
replay baseline.

**Headline results (SANPO official test split):** GRU arrival AUROC 0.888 vs. K0 0.571 and
cone-restricted TTC 0.765; in end-to-end replay, v2 is 2.5× more precise than v1 (0.59 vs. 0.24) at
7× fewer alerts (17 vs. 119 per minute), but warns about fewer arrivals (26% vs. 65%). Details:
`docs/progress.md` §6–7.

Run the v2 engine on a synthetic scene: `python tools/run_full_pipeline_demo.py` (`--v1` for the
baseline chain).

## Project Structure

```text
CompositePerceptionEngine/
├── .agents/AGENTS.md                # Required docs/changelog upkeep for agents
├── .github/workflows/ios-timing.yml # macOS CI: bundle + Core ML export + build the iOS timing app; optional TestFlight upload
├── docs/                            # See docs/README.md for the full map
│   ├── architecture.md              # v2 system design: components, simulation vs. iPhone, budgets, evaluation, limitations
│   ├── progress.md                  # Status, key results with numbers, slide order, Q&A numbers
│   ├── pending_work.md              # Prioritized backlog (P0 data correctness → P5 paper)
│   ├── methodology.md               # Paper-ready methods + LLM paper-drafting prompts
│   ├── ablation_guide.md            # Stage-1 streaming, K0 ablation, VLM referee + human calibration run guide
│   ├── related_work.md              # Verified citations by component
│   ├── listening_study_plan.md      # Plan: seated BLV listening study (alert preference, tolerable rate); not run
│   ├── sanpo_dataset.md             # SANPO layout, audited camera facts + official split, intake, server latency, gap analysis
│   ├── yolo_training.md             # Detector datasets, checkpoints, commands, metrics
│   └── CHANGELOG.md                 # Dated record of changes
├── src/
│   ├── perception_stack/            # YOLO26n v3 + ByteTrack + depth sampling, physics (severity, pinhole bearing, K0)
│   ├── threat_engine/               # v2: fusion.py (Kalman, train-fitted noise), envelope.py, priority.py (GRU/physics),
│   │                                #     kalman_arrival.py (P(enter corridor) over the track's uncertainty), announcer.py, engine.py
│   ├── pipeline/                    # orchestrator.py — V2Pipeline (v2 + narration) and the v1 process_frame baseline
│   ├── narration/                   # Scripted templates (distance buckets, all_phrases), phrase-table translation, TTS
│   ├── threat_prioritizer/          # v1 baseline: ThreatEvent contract and K0 routing
│   ├── reflex_layer/                # v1 baseline: reflex path
│   ├── physics_verification/        # v1 baseline: judge; NarratorEvent (narration input type)
│   └── cognitive_layer/             # v1 baseline: stub.py ranking
├── models/
│   ├── GRU/                         # "When": arrival predictors (GRU/, GRU_attention/), train/evaluate/summarize,
│   │                                #     benchmark.py (baseline ladder), exports/ (Core ML, generated on macOS)
│   ├── Ensemble/                    # Rank-average and stacked arrival ensembles; heldout.py (official test split)
│   ├── yolo/                        # Detector registry: base_yolo26n, cpe_yolo26n_hazards_v1, ..._v3_from_base
│   └── tts/piper/                   # Piper ONNX voice (live-synthesis benchmark)
├── dataset/
│   └── GRU/                         # samples.py, build.py, resplit.py (official split); built .npz datasets
├── evaluation/
│   ├── kinetic_ablation.py          # K0 term ablation, label-free metrics, encounter labels (reused for v2)
│   ├── kinetic_ablation_stratified.py # Ablation restricted to severity-discriminating frames
│   ├── encounter_blob_check.py      # encounter_top1 with/without untracked depth blobs
│   ├── vlm_referee.py               # Blinded VLM referee, protocol v2: numbered boxes, order swap, text-only control,
│   │                                #     human ceiling, Gwet's AC1 / Krippendorff's α, pre-registered sample
│   ├── topk_threat_validation.py    # K0 top-3 vs. blinded VLMs (v1 check)
│   ├── generate_report_figures.py   # Regenerates evaluation/benchmarks/figures/
│   └── benchmarks/
│       ├── heldout_official_split/  # GRU / ensemble / baselines on SANPO's official test split
│       ├── threat_engine_replay/    # End-to-end replay, v1 vs v2, system metrics (SANPO depth as is)
│       ├── baseline_ladder/         # Held-out arrival ladder per label (cone, corridor): rules, Kalman, tabular, GRUs
│       ├── kalman_noise_fit/        # Kalman noise model fitted on official-train sessions (fit_pose.json)
│       ├── gru_encounter_eval/      # GRU cross-validation runs, one report.md per run
│       ├── ensemble_eval/           # Ensemble reports, paired differences, stacker weights
│       ├── kinetic_score_eval/      # K0 ablation run (2026-08-26)
│       ├── yolo26n_hazards_v1/      # v1 detector evaluation
│       ├── yolo26n_version_comparison/ # v1/v2/v3 accuracy and retention
│       ├── sanpo_edge_realtime/     # Server latency on SANPO (+ retired Jetson proxy)
│       ├── narration_latency/       # Template / Piper latency
│       └── figures/                 # Generated plots + results_summary.md
├── tools/
│   ├── stream_sanpo_perception.py   # Stage 1 over SANPO streamed from the public bucket (per-session fps / fx)
│   ├── audit_sanpo_sessions.py      # FPS / intrinsics / official split / mask coverage for all 462 sessions
│   ├── replay_threat_engine.py      # End-to-end SANPO replay: v1 vs v2 engines; --latency-csv
│   ├── fit_kalman_noise.py          # Fit the Kalman noise model by predictive likelihood on train sessions
│   ├── build_session_viewer.py      # Visual check of one SANPO walk: boxes, engine scores, alerts vs. truth
│   │                                #     (HTML viewer from session_viewer.html, written to data/viewer/)
│   ├── download_sanpo_valid_streams.py # Bounded downloader + the shared seeded session sampler
│   ├── run_perception.py            # Stage 1 on local frames
│   ├── build_threat_events.py       # Stage-1 CSV → ThreatEvent JSONL
│   ├── run_full_pipeline_demo.py    # Runtime chain on synthetic frames
│   ├── benchmark_edge_realtime.py   # Detector/tracker/depth latency on SANPO
│   ├── benchmark_narration_latency.py # Narration/TTS latency
│   ├── gap_analysis_experiments.py  # Depth-vs-YOLO gap analysis
│   └── visualize_stage1.py          # Stage-1 CSV visualisation
├── training/
│   ├── scripts/                     # YOLO download, rebalance, train, evaluate, retention compare, export;
│   │                                #     export_gru_coreml.py (arrival GRU → Core ML)
│   ├── configs/                     # 17-class taxonomy, Roboflow source manifest
│   └── runs/                        # Ultralytics runs and checkpoints (v1, v2, v3)
├── ios/CPETiming/                   # On-device latency harness (SwiftUI, XcodeGen): detector + GRU per stage on SANPO data,
│                                    #     prepare_bundle.py, analyze_timing.py, TestFlight how-to in its README
├── notebooks/                       # sanpo_yolo_gap_analysis.ipynb + gap_analysis_output/
│                                    #     colab_rebuild.ipynb: Stage 1 + datasets + GRUs + Kalman + results on Colab
├── simulation/datasets/sanpo/       # valid_streams.json (462 sessions), session_audit.json, official_split.json
├── tests/                           # Threat engine (fusion, envelope, priority, announcer), Kalman arrival, latency-aware
│                                    #     replay options, v1 routing, GRU, ensemble
├── data/                            # Local datasets (gitignored)
└── requirements.txt
```

## Common commands

```bash
# Audit SANPO metadata (fps, intrinsics, official split, masks) - metadata only, no frames
python tools/audit_sanpo_sessions.py

# Stage 1 over SANPO, streamed (fps and focal length read per session; docs/ablation_guide.md §2)
python tools/stream_sanpo_perception.py --help

# GRU on SANPO's official split, then the held-out report (see models/GRU/README.md)
python -m dataset.GRU.resplit --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_f5_s0
python -m models.GRU.train --arch gru --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official --time-bins 4 --seeds 0 1 2 --run-name official462_2026_10_14
python -m models.Ensemble.heldout --run-name official462_2026_10_14

# End-to-end replay, v1 vs v2 (fetches camera poses into data/sanpo_poses/)
python tools/replay_threat_engine.py --gru-run models/GRU/GRU/runs/official462_2026_10_14_seed0

# Runtime self-checks and tests (no ultralytics needed)
python -m src.pipeline.orchestrator --self-check
python -m unittest discover -s tests

# Detector export; Core ML for the iPhone (run on a Mac)
python training/scripts/export_yolo26n_edge.py --formats coreml --quantize 16
```
