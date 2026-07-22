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

## Project Structure

```text
CompositePerceptionEngine/
├── .agents/                         # Agent instructions for required docs/changelog upkeep
├── docs/                            # Canonical project documentation
│   ├── README.md                    # Documentation ownership map and artifact conventions
│   ├── progress.md                  # Living checklist and immediate next steps
│   ├── architecture.md              # Runtime architecture, contracts, and safety behavior
│   ├── hardware_targets.md          # GB10 specs, edge target, budgets, and evidence status
│   ├── roadmap.md                   # Technology decisions and implementation phases
│   ├── sanpo_dataset.md             # SANPO layout, intake, and edge benchmark evidence
│   ├── sanpo_gap_analysis.md        # Depth-versus-YOLO blind-spot workflow
│   ├── yolo_training.md             # Current YOLO26n v3 dataset/checkpoint/evaluation guide
│   ├── research_paper_prompts.md    # Paper prompts aligned with implemented evidence
│   └── CHANGELOG.md                 # Date-stamped project changes
├── src/
│   ├── shared/                      # Shared schemas and data contracts
│   ├── sensor_fusion/               # Camera/depth/gyro fusion components
│   ├── perception_stack/            # YOLO26n + ByteTrack + depth post-processing
│   ├── threat_prioritizer/          # ThreatEvent contract and Ignore / Reflex / Cognitive routing
│   ├── reflex_layer/                # Deterministic reflex bridge from ThreatEvent to Physics Verification
│   ├── cognitive_layer/             # SLM-1 semantic evaluation
│   ├── physics_verification/        # Judge layer for SLM-vs-physics arbitration
│   ├── narrator_slm/                # SLM-2 warning generation
│   ├── indic_translation/           # Optional Indic translation path
│   ├── system_heartbeat/            # Ambient status updates
│   └── audio_output/                # TTS / audio output path
├── simulation/
│   ├── envs/                        # Simulation environments
│   ├── scenarios/                   # Scripted scenarios
│   └── datasets/
│       └── sanpo/valid_streams.json # Curated SANPO stream IDs for the CPE use case
├── training/
│   ├── configs/                     # YOLO/Roboflow/dataset configs
│   ├── scripts/                     # YOLO download, training, export, evaluation, comparison scripts
│   ├── rl_agent/                    # PPO/LoRA training entry points for SLM-1
│   ├── rewards/                     # Reward functions from physics verification logs
│   └── runs/                        # Local Ultralytics outputs and checkpoints (not source-controlled)
├── models/
│   ├── yolo/
│   │   ├── base_yolo26n/            # Local base YOLO26n checkpoint registry entry
│   │   └── cpe_yolo26n_hazards_v3_from_base/ # Preferred detector export registry
│   ├── slm1/                        # Qwen/Phi-style SLM-1 artifacts
│   ├── slm2/                        # Narrator SLM artifacts
│   ├── indic/                       # IndicTrans2 artifacts
│   └── tts/                         # TTS artifacts
├── tools/
│   ├── benchmark_edge_realtime.py   # RGB/depth real-time benchmark with edge profiles
│   ├── build_threat_events.py       # Stage 1 perception CSV → ThreatEvent JSONL bridge
│   ├── download_sanpo_valid_streams.py # Bounded downloader for valid SANPO sessions
│   ├── gap_analysis_experiments.py  # SANPO depth-vs-YOLO perception gap analysis
│   ├── run_perception.py            # Stage 1 perception runner
│   └── visualize_stage1.py          # Perception CSV visualization helper
├── evaluation/
│   ├── benchmarks/
│   │   ├── sanpo_edge_realtime/     # SANPO latency and edge-simulation metrics
│   │   └── yolo26n_version_comparison/ # v1/v2/v3 accuracy and retention comparisons
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
| Narration | Template-first critical warnings; Phi-4-mini optional for non-critical narration | INT4/ONNX target | Critical alerts stay deterministic |
| Indic Translation | IndicTrans2 distilled / phrase table | 200M-class target | Phrase table for critical alerts, model for non-critical narration |
| TTS | Cached clips / Piper-ONNX / FastSpeech2 fallback | On-device | Must not block reflex alerts |

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
