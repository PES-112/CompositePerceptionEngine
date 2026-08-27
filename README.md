# Composite Perception Engine (CPE)

Simulation-based implementation of the Neuro-Symbolic navigation system for blind pedestrians.
**Architecture Type:** Asynchronous Physics-Semantic Dual-Track with Verification Layer.

## Corrected Pipeline

```
Input Layer (Camera + Depth Dataset + Gyro)
    → Sensor Fusion
    → Perception Stack (YOLO26 Nano + ByteTrack + Depth)
    → Threat Prioritizer
         ├── Low Risk  → Ignore
         ├── High Risk → Reflex Layer (TTC, <50ms) ──────────┐
         └── Context   → Cognitive Layer (SLM-1, ~500ms) ───→ Physics Verification (Judge)
                                                              → Narrator SLM-2
                                                              → [Optional] Indic Translation
                                                              → Audio Output (cached clips / local TTS)
System Heartbeat → Audio Output (independent, ambient)
```

Run this chain end-to-end on synthetic data right now with `python tools/run_full_pipeline_demo.py`
(SLM-1 is a rule-based stub — see `docs/pending_work.md` for what's real vs. placeholder in each stage).

## Project Structure

```text
CompositePerceptionEngine/
├── .agents/                         # Agent instructions for required docs/changelog upkeep
├── docs/                            # Canonical project documentation (see docs/README.md for the full map)
│   ├── README.md                    # Documentation ownership map and artifact conventions
│   ├── progress.md                  # Living checklist and immediate next steps
│   ├── progress_presentation.md     # Presentation-ready progress summary and key results
│   ├── pending_work.md              # Prioritized backlog of everything not yet done
│   ├── methodology.md               # Paper-ready methods reference + LLM paper-drafting prompts (dataset, detector, kinetic score, edge latency)
│   ├── architecture.md              # Runtime architecture, contracts, safety behavior, technology audit, and historical implementation plan
│   ├── hardware_targets.md          # GB10 specs, edge target, budgets, and evidence status
│   ├── yolo_training.md             # Current YOLO26n v3 dataset/checkpoint/evaluation guide
│   ├── sanpo_dataset.md             # SANPO layout, intake, edge benchmark evidence, and gap analysis workflow
│   ├── ablation_guide.md            # Kinetic-score ablation + verification run guide and latest results
│   ├── kinetic_score_opinion.md     # Full reasoning trail + decision record for the kinetic-score evaluation strategy
│   ├── related_work.md              # Verified external paper citations, by CPE component
│   └── CHANGELOG.md                 # Date-stamped project changes
├── src/
│   ├── shared/                      # Shared schemas and data contracts
│   ├── perception_stack/            # YOLO26n + ByteTrack + depth post-processing
│   ├── threat_prioritizer/          # ThreatEvent contract and Ignore / Reflex / Cognitive routing
│   ├── reflex_layer/                # Deterministic reflex bridge from ThreatEvent to Physics Verification
│   ├── physics_verification/        # Judge layer for SLM-vs-physics arbitration
│   ├── narration/                   # Template narrator, translation, and TTS adapters (templates.py, translation.py, tts.py, pipeline.py)
│   ├── cognitive_layer/             # stub.py — rule-based SemanticEval placeholder for SLM-1
│   └── pipeline/                    # orchestrator.py — composes prioritizer→cognitive→verification→narration per frame
├── simulation/
│   └── datasets/
│       └── sanpo/valid_streams.json # Curated SANPO stream IDs for the CPE use case
├── training/
│   ├── scripts/                     # YOLO download, training, export, evaluation, comparison scripts
│   └── rl_agent/                    # PPO/LoRA training entry points for SLM-1
├── models/
│   ├── yolo/
│   │   ├── base_yolo26n/            # Local base YOLO26n checkpoint registry entry
│   │   └── cpe_yolo26n_hazards_v3_from_base/ # Preferred detector export registry
│   ├── slm1/                        # Qwen/Phi-style SLM-1 artifacts
│   ├── slm2/                        # Narrator SLM artifacts
│   ├── indic/                       # IndicTrans2 artifacts
│   └── tts/
│       └── piper/                   # Downloaded Piper ONNX voices (e.g. en_US-lessac-low)
├── tools/
│   ├── benchmark_edge_realtime.py   # RGB/depth real-time benchmark with edge profiles
│   ├── benchmark_narration_latency.py # Standalone narration/TTS latency benchmark (no SANPO data needed)
│   ├── build_threat_events.py       # Stage 1 perception CSV → ThreatEvent JSONL bridge
│   ├── download_sanpo_valid_streams.py # Bounded downloader for valid SANPO sessions
│   ├── gap_analysis_experiments.py  # SANPO depth-vs-YOLO perception gap analysis
│   ├── manual_score_inspector.py    # Generates interactive HTML for manual K-score verification
│   ├── run_full_pipeline_demo.py    # Full runtime chain on synthetic data — perception → prioritizer → cognitive → verification → narration
│   ├── run_perception.py            # Stage 1 perception runner
│   ├── stream_sanpo_perception.py   # Stage 1 over a 30% GCS-streamed SANPO sample (no local dataset)
│   └── visualize_stage1.py          # Perception CSV visualization helper
├── evaluation/
│   ├── kinetic_ablation.py          # K0 term ablation + label-free metrics + disagreement export
│   ├── kinetic_ablation_stratified.py # Severity-discriminating-frame-only re-run of the ablation metrics
│   ├── vlm_referee.py               # Blinded 3-VLM referee (local servers) over disagreement frames
│   ├── topk_threat_validation.py    # Per-scene K0 top-3 threats, validated by 3 blinded local VLMs
│   ├── generate_report_figures.py   # Regenerates all plots/tables in benchmarks/figures/ from current results
│   ├── benchmarks/
│   │   ├── sanpo_edge_realtime/     # SANPO latency and edge-simulation metrics
│   │   ├── yolo26n_version_comparison/ # v1/v2/v3 accuracy and retention comparisons
│   │   ├── kinetic_score_eval/      # K0 ablation runs: label-free metrics, referee ballots, reports
│   │   ├── narration_latency/       # Narration/TTS latency benchmark output (report.md, latency.json)
│   │   └── figures/                 # Generated PNG plots + results_summary.md (output only)
│   └── logs/
├── notebooks/                       # EDA and SANPO/YOLO prototyping notebooks
├── tests/                           # Unit/integration tests for threat routing and reflex bridge checks
├── data/                            # Local datasets/downloads (gitignored)
└── README.md
```

## SLM Stack

| Component | Model | Size | Runtime |
|---|---|---|---|
| SLM-1 (Cognitive) | Qwen3-1.7B non-thinking mode | INT4 target | Strict JSON over perception fact sheets |
| SLM-1 fallback | Qwen2.5-1.5B-Instruct | INT4 target | Use if Qwen3 runtime/quantization is blocked |
| Narration | Template-first critical warnings (`src/narration/templates.py`, done); Phi-4-mini optional for non-critical narration (not started) | INT4/ONNX target | Critical alerts stay deterministic |
| Indic Translation | Phrase table + English fallback (done); IndicTrans2 distilled adapter coded, not yet run against real weights | 200M-class target | Phrase table for critical alerts, model for non-critical narration |
| TTS | Cached clips (done) / Piper-ONNX (done, measured 30-68ms/utterance CPU) / FastSpeech2 fallback (not started) | On-device | Must not block reflex alerts |

## Training

```bash
# Phase 1: Supervised warm-up
python training/rl_agent/warmup.py --model qwen2.5-1.5b --steps 500

# Phase 2: PPO from Physics Verification reward
python training/rl_agent/train_ppo.py --lora-rank 16 --steps 10000

# Detector export/runtime baseline
.venv/bin/python training/scripts/export_yolo26n_edge.py \
  --formats onnx \
  --quantize 16 \
  --device 0
```
