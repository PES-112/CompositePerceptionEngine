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
                                                              → Audio Output (FastSpeech2)
System Heartbeat → Audio Output (independent, ambient)
```

## Project Structure

```text
CompositePerceptionEngine/
├── .agents/                         # Agent instructions for required docs/changelog upkeep
├── docs/                            # Architecture, YOLO training, SANPO bucket, benchmark, and paper notes
│   ├── architecture.md              # Canonical CPE architecture and latency budgets
│   ├── sanpo_bucket_structure.md    # SANPO GCS layout, valid-stream intake, and edge benchmark notes
│   ├── yolo_training.md             # Current YOLO26n v3 dataset/checkpoint/evaluation guide
│   └── CHANGELOG.md                 # Date-stamped project changes
├── src/
│   ├── shared/                      # Shared schemas and data contracts
│   ├── sensor_fusion/               # Camera/depth/gyro fusion components
│   ├── perception_stack/            # YOLO26n + ByteTrack + depth post-processing
│   ├── threat_prioritizer/          # Routes objects into Ignore / Reflex / Cognitive paths
│   ├── reflex_layer/                # Deterministic TTC physics path (<50ms target)
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
│   ├── scripts/                     # YOLO download, training, evaluation, comparison scripts
│   ├── rl_agent/                    # PPO/LoRA training entry points for SLM-1
│   ├── rewards/                     # Reward functions from physics verification logs
│   └── runs/                        # Local Ultralytics outputs and checkpoints (not source-controlled)
├── models/
│   ├── yolo/
│   │   └── base_yolo26n/            # Local base YOLO26n checkpoint registry entry
│   ├── slm1/                        # Qwen/Phi-style SLM-1 artifacts
│   ├── slm2/                        # Narrator SLM artifacts
│   ├── indic/                       # IndicTrans2 artifacts
│   └── tts/                         # TTS artifacts
├── tools/
│   ├── benchmark_edge_realtime.py   # RGB/depth real-time benchmark with edge profiles
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
├── tests/                           # Unit/integration tests
├── data/                            # Local datasets/downloads (gitignored)
└── README.md
```

## SLM Stack

| Component | Model | Size | Runtime |
|---|---|---|---|
| SLM-1 (Cognitive) | Qwen2.5-1.5B-Instruct + LoRA | ~900MB INT4 | llama.cpp + QNN |
| SLM-2 (Narrator) | Phi-3-Mini-4K-Instruct | ~2.2GB INT4 | ONNX Mobile + QNN EP |
| Indic Translation | IndicTrans2 (AI4Bharat) | ~200MB | On-device |
| TTS | FastSpeech2 INT8 | ~150MB | On-device |

## Training

```bash
# Phase 1: Supervised warm-up
python training/rl_agent/warmup.py --model qwen2.5-1.5b --steps 500

# Phase 2: PPO from Physics Verification reward
python training/rl_agent/train_ppo.py --lora-rank 16 --steps 10000

# Phase 3: Export to GGUF
python scripts/export_gguf.py --quant Q4_K_M
```
