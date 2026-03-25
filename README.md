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

```
code/
├── src/
│   ├── shared/                    # Types, constants, NarratorEvent schema
│   ├── sensor_fusion/             # Fuses camera, depth dataset, gyro
│   ├── perception_stack/          # YOLO26 Nano + ByteTrack + depth overlay
│   ├── threat_prioritizer/        # Routes objects: Ignore / Reflex / Cognitive
│   ├── reflex_layer/              # Deterministic TTC physics (<50ms)
│   ├── cognitive_layer/           # SLM-1 semantic evaluation (Qwen2.5-1.5B)
│   ├── physics_verification/      # The Judge: SLM-1 vs kinetic arbiter + RL reward
│   ├── narrator_slm/              # SLM-2 natural language generation (Phi-3-Mini)
│   ├── indic_translation/         # Optional: IndicTrans2, 22 Indian languages
│   ├── system_heartbeat/          # Ambient state updates, bypasses verification
│   └── audio_output/             # FastSpeech2 INT8 + 3D Binaural Pan
│
├── simulation/
│   ├── envs/                      # Custom Gym environments
│   ├── scenarios/                 # Scripted urban traffic scenarios
│   └── datasets/                  # Loaders: Sanpo (depth GT), UASOL, HeadsUp
│
├── training/
│   ├── rl_agent/                  # PPO trainer for SLM-1 (LoRA)
│   ├── rewards/                   # Reward fn consuming Physics Verification logs
│   └── configs/                   # YAML hyperparameters
│
├── models/
│   ├── yolo/                      # YOLO26 Nano weights (PTQAT INT8)
│   ├── slm1/                      # Qwen2.5-1.5B + LoRA adapters (INT4 GGUF)
│   ├── slm2/                      # Phi-3-Mini-4K (INT4 GGUF)
│   ├── indic/                     # IndicTrans2 weights
│   └── tts/                       # FastSpeech2 INT8
│
├── evaluation/
│   ├── benchmarks/                # Latency, reward, narration quality
│   └── logs/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── notebooks/                     # EDA, prototyping
└── scripts/                       # Setup, model download, export to GGUF
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
