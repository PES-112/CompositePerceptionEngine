# Composite Perception Engine (CPE)

A phone-based hazard alert for blind pedestrians. It ranks nearby objects by **when** they will reach
the walker, **how bad** a collision would be, and **what** the scene says about them. Every learned
decision sits inside a deterministic physics safety envelope. The target device is an iPhone 17 Pro
Max; evaluation replays the SANPO egocentric dataset.

**Design:** [`docs/architecture.md`](docs/architecture.md) · **Status:**
[`docs/progress.md`](docs/progress.md) · **Backlog:** [`docs/pending_work.md`](docs/pending_work.md)

## Pipeline (v2)

```
SANPO replay: RGB + dataset depth + camera pose      (on the phone: camera + LiDAR + ARKit pose)
  → Perception        YOLO26n v3 + ByteTrack + depth per box                         ✅
  → Sensor fusion     Kalman track per object, ground plane, head motion removed      ⬜
  → Scene context     walkable-surface segmentation (road / curb / sidewalk), 2–5 Hz  ⬜
  → Threat engine
       safety envelope   TTC ≤ 1 s in the walking path → immediate alert             ✅
       when              GRU arrival predictor                                       🟡 offline
       how bad           class severity (mass law)                                   ✅
       what              segmentation context                                        ⬜
       fusion            priority = P(arrive) × severity × context                   ⬜
  → Announcement manager  cooldown, hysteresis, override preempts                     ⬜
  → Narration         scripted phrases → pre-rendered audio clips                     ✅ templates
```

✅ built · 🟡 built offline, not in the runtime yet · ⬜ planned. There is no language model at
runtime; the v1 SLM cognitive and narrator layers were dropped on 2026-10-14.

Run the current runtime chain on synthetic frames: `python tools/run_full_pipeline_demo.py` (the
ranking step is still a rule-based stub until the fused score lands).

## Project Structure

```text
CompositePerceptionEngine/
├── .agents/AGENTS.md                # Required docs/changelog upkeep for agents
├── docs/                            # See docs/README.md for the full map
│   ├── architecture.md              # v2 system design: components, simulation vs. iPhone, budgets, evaluation, limitations
│   ├── progress.md                  # Status, key results with numbers, slide order, Q&A numbers
│   ├── pending_work.md              # Prioritized backlog (P0 data correctness → P5 paper)
│   ├── methodology.md               # Paper-ready methods + LLM paper-drafting prompts
│   ├── ablation_guide.md            # Stage-1 streaming, K0 ablation, VLM referee + human calibration run guide
│   ├── related_work.md              # Verified citations by component
│   ├── sanpo_dataset.md             # SANPO layout, camera facts, intake, server latency benchmark, gap analysis
│   ├── yolo_training.md             # Detector datasets, checkpoints, commands, metrics
│   └── CHANGELOG.md                 # Dated record of changes
├── src/
│   ├── perception_stack/            # YOLO26n + ByteTrack + depth sampling, physics (severity, bearing, K0)
│   ├── threat_prioritizer/          # ThreatEvent contract and routing
│   ├── reflex_layer/                # Safety-envelope path (TTC override)
│   ├── physics_verification/        # Override rule (v1 SLM arbitration rules to be retired)
│   ├── cognitive_layer/             # stub.py — placeholder, to be replaced by the fused ranker
│   ├── pipeline/                    # orchestrator.py — per-frame composition through narration
│   ├── narration/                   # Scripted templates, phrase-table translation, cached-clip/Piper TTS
│   └── shared/                      # fact_sheet.py (v1 SLM prompt format; to retire)
├── models/
│   ├── GRU/                         # "When": arrival predictors (GRU/, GRU_attention/), train/evaluate/summarize
│   ├── Ensemble/                    # Rank-average and out-of-fold stacked arrival ensemble
│   ├── yolo/                        # Detector registry: base_yolo26n, cpe_yolo26n_hazards_v1, ..._v3_from_base
│   └── tts/piper/                   # Piper ONNX voice (live-synthesis benchmark)
├── dataset/
│   └── GRU/                         # samples.py + build.py; built .npz datasets shared by both GRU architectures
├── evaluation/
│   ├── kinetic_ablation.py          # K0 term ablation, label-free metrics, encounter labels (reused for v2)
│   ├── kinetic_ablation_stratified.py # Ablation restricted to severity-discriminating frames
│   ├── encounter_blob_check.py      # encounter_top1 with/without untracked depth blobs
│   ├── vlm_referee.py               # Blinded 3-VLM referee (local servers) + human calibration
│   ├── topk_threat_validation.py    # K0 top-3 vs. blinded VLMs (v1 check)
│   ├── generate_report_figures.py   # Regenerates evaluation/benchmarks/figures/
│   └── benchmarks/
│       ├── gru_encounter_eval/      # GRU runs, one report.md per run
│       ├── ensemble_eval/           # Ensemble reports, paired differences, stacker weights
│       ├── kinetic_score_eval/      # K0 ablation run (2026-08-26)
│       ├── yolo26n_hazards_v1/      # v1 detector evaluation
│       ├── yolo26n_version_comparison/ # v1/v2/v3 accuracy and retention
│       ├── sanpo_edge_realtime/     # Server latency on SANPO (+ retired Jetson proxy)
│       ├── narration_latency/       # Template / Piper latency
│       └── figures/                 # Generated plots + results_summary.md
├── tools/
│   ├── stream_sanpo_perception.py   # Stage 1 over SANPO streamed from the public bucket
│   ├── download_sanpo_valid_streams.py # Bounded downloader + the shared seeded session sampler
│   ├── run_perception.py            # Stage 1 on local frames
│   ├── build_threat_events.py       # Stage-1 CSV → ThreatEvent JSONL
│   ├── run_full_pipeline_demo.py    # Runtime chain on synthetic frames
│   ├── benchmark_edge_realtime.py   # Detector/tracker/depth latency on SANPO
│   ├── benchmark_narration_latency.py # Narration/TTS latency
│   ├── gap_analysis_experiments.py  # Depth-vs-YOLO gap analysis
│   ├── visualize_stage1.py          # Stage-1 CSV visualisation
│   └── run_fact_sheets.py           # v1 SLM fact sheets (to retire)
├── training/
│   ├── scripts/                     # YOLO download, rebalance, train, evaluate, retention compare, export
│   ├── configs/                     # 17-class taxonomy, Roboflow source manifest
│   └── runs/                        # Ultralytics runs and checkpoints (v1, v2, v3)
├── notebooks/                       # sanpo_yolo_gap_analysis.ipynb + gap_analysis_output/
├── simulation/datasets/sanpo/valid_streams.json # The 462 curated SANPO-Real sessions
├── tests/                           # Threat routing, safety envelope, GRU models, ensemble
├── data/                            # Local datasets (gitignored)
└── requirements.txt
```

## Common commands

```bash
# Stage 1 over SANPO, streamed (see docs/ablation_guide.md §2; check --fps against the session's FPS)
python tools/stream_sanpo_perception.py --help

# Build the GRU dataset and train (see models/GRU/README.md)
python -m dataset.GRU.build --csv-dir data/processed/sanpo_real_462 --tracked-only --hazard-m 3 --folds 5
python -m models.GRU.train --arch gru --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_f5_s0 --time-bins 4 --seeds 0 1 2

# Tests
python -m unittest discover tests

# Detector export (Core ML for the iPhone is pending_work.md P3)
python training/scripts/export_yolo26n_edge.py --help
```
