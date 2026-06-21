# SANPO Dataset Preprocessing Pipeline

## Overview

This preprocessing pipeline converts SANPO-Real egocentric video data into threat training metrics for the Composite Perception Engine (CPE). The pipeline is designed for **stream processing** — never storing massive intermediate files, only essential training data (~16-20 GB for 100-150K frames).

## Architecture

```
SANPO Raw Data
    ↓
sanpo_loader.py
    (Load video + depth frames)
    ↓
scenario_filter.py
    (Filter outdoor pedestrian scenarios)
    ↓
frame_processor.py
    (YOLO detection + ByteTrack + depth extraction)
    ↓
training_dataset.py
    (Compute threat scores, accumulate metrics)
    ↓
training_cache.pkl (~5-10 GB)
    ↓
threat_prioritizer_finetuner.py
    (Fine-tune threat model on real data)
    ↓
threat_prioritizer_finetuned.pt (~50 MB)
    ↓
demo_inference.py
    (Run CPE on custom video with trained model)
```

## Quick Start

### 1. Prepare SANPO Data

Download SANPO-Real dataset (701 sessions, ~2GB each video + depth pair):
```bash
# Expected structure:
# /path/to/sanpo/
# ├── videos/
# │   ├── 001.mp4
# │   ├── 002.mp4
# │   └── ...
# ├── depth_maps/
# │   ├── 001/
# │   │   ├── depth_0000.npz
# │   │   ├── depth_0001.npz
# │   │   └── ...
# │   └── ...
# └── metadata.json
```

### 2. Run Preprocessing Pipeline

#### Option A: Local (MacBook)
```bash
cd simulation/datasets/

# Process specific sessions (subset for storage constraint)
python sanpo_preprocessing_pipeline.py /path/to/sanpo 001 002 003 004 005

# Outputs:
# - preprocessing_outputs/training_cache.pkl (~50-100 MB for 5 sessions)
# - preprocessing_outputs/preprocessing_stats.json
```

**Estimated time:** ~1-2 hours for 5 sessions on M1/M2 MacBook (CPU)

#### Option B: Google Colab (Faster)
```python
# 1. Mount Google Drive with SANPO subset
# 2. Install dependencies
!pip install ultralytics opencv-python numpy torch

# 3. Upload preprocessing_outputs/preprocessing_pipeline.py
# 4. Run in Colab cell
from sanpo_preprocessing_pipeline import SANPOPreprocessingPipeline

pipeline = SANPOPreprocessingPipeline(
    sanpo_root="/path/to/sanpo",
    session_ids=["001", "002", "003"],  # 10-15 sessions recommended
    device="cuda"  # Free T4 GPU
)
pipeline.run()
```

**Estimated time:** ~30-60 minutes for 15 sessions on Colab T4

### 3. Fine-tune Threat Model

```bash
# Use accumulated training cache to fine-tune threat prioritizer
python threat_prioritizer_finetuner.py \
    preprocessing_outputs/training_cache.pkl \
    threat_prioritizer_finetuned.pt \
    cpu

# Outputs: threat_prioritizer_finetuned.pt (~50 MB)
```

**Estimated time:** ~5-10 minutes on CPU

### 4. Demo on Custom Video

Record a video on your iPhone or MacBook (e.g., walking outside for 30-60 seconds):

```bash
# Run CPE pipeline on custom video
python demo_inference.py \
    /path/to/demo_video.mp4 \
    --model threat_prioritizer_finetuned.pt \
    --output demo_video_annotated.mp4 \
    --device cpu \
    --tts  # Enable audio warnings (optional)

# Real-time output:
# - Console: Threat detections and warnings
# - Optional audio: TTS-generated warnings
# - demo_video_annotated.mp4: Annotated video with bounding boxes + threat scores
```

**Estimated time:** Real-time (FPS depends on device)

---

## Module Documentation

### `sanpo_loader.py`
**Purpose:** Load SANPO video frames and synchronized depth maps

**Key classes:**
- `Frame`: Dataclass containing rgb + depth + metadata
- `SANPOLoader`: Generator-based loader

**Usage:**
```python
from sanpo_loader import SANPOLoader

loader = SANPOLoader("/path/to/sanpo")

# Load specific sessions
for frame in loader.iter_frames("001"):
    # Process frame
    print(f"Frame {frame.frame_id}: RGB {frame.rgb.shape}, Depth {frame.depth.shape}")
    # Memory freed after each iteration
```

---

### `scenario_filter.py`
**Purpose:** Filter SANPO sessions to outdoor pedestrian scenarios

**Filtering stages:**
1. **Metadata filtering** (deterministic): Check scene attributes, environment type
2. **Heuristic filtering** (optional): Brightness check (outdoor > 50%), motion detection

**Key classes:**
- `FilterResult`: Decision + reasoning + confidence
- `ScenarioFilter`: Multi-level filtering logic

**Usage:**
```python
from scenario_filter import ScenarioFilter

filter = ScenarioFilter(min_frames=150, brightness_threshold=50.0)
result = filter.filter_session("001", session_metadata)

if result.is_valid:
    print(f"Session valid! Reason: {result.reason}")
else:
    print(f"Rejected: {result.reason} (confidence: {result.confidence})")
```

---

### `frame_processor.py`
**Purpose:** YOLO detection → Object tracking → Depth extraction

**Processing pipeline per frame:**
1. YOLO inference (ultralytics YOLOv8n)
2. SimpleCentroidTracker (matches detections across frames)
3. Centroid-based depth lookup from depth map
4. Velocity calculation (distance change)
5. Time-to-contact (TTC) estimation

**Key classes:**
- `Detection`: Single object with all metrics (distance, velocity, TTC)
- `SimpleCentroidTracker`: Frame-to-frame object tracking
- `FrameProcessor`: Orchestrator for full processing

**Usage:**
```python
from frame_processor import FrameProcessor

processor = FrameProcessor(model_name="yolov8n", device="cpu")
processor.setup()  # Load YOLO model

detections = processor.process_frame(
    frame_rgb,      # [H, W, 3] uint8
    depth_map,      # [H, W] float32 meters
    frame_id=0,
    timestamp=0.0
)

for det in detections:
    print(f"Track {det.track_id}: {det.class_name} @ {det.depth_m:.1f}m, "
          f"v={det.velocity_mps:.1f} m/s, TTC={det.ttc_s:.2f}s")
```

---

### `training_dataset.py`
**Purpose:** Accumulate threat metrics for model fine-tuning

**Key features:**
- `ThreatCalculator`: Physics-based threat scoring (0-10 scale)
- `TrainingDatasetAccumulator`: Collects frame metrics
- Never stores persistent JSON, only pickle cache

**Threat scoring formula:**
```
kinetic_score = 0.3 * proximity_threat + 0.3 * velocity_threat + 0.4 * time_threat

Where:
- proximity_threat = critical_distance / distance
- velocity_threat = |approaching_velocity| / max_velocity (if approaching)
- time_threat = critical_ttc / ttc (if moving toward)

Score range: [0, 10]
  0-2: Low threat
  2-5: Medium threat
  5-8: High threat
  8-10: Critical
```

**Usage:**
```python
from training_dataset import TrainingDatasetAccumulator, ThreatCalculator

accumulator = TrainingDatasetAccumulator()

for frame_data in preprocessing_stream:
    threat_metrics = accumulator.process_frame(
        detections,
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        session_id="001"
    )

# Save training cache
accumulator.save("training_cache.pkl")

# Get statistics
stats = accumulator.get_summary_stats()
print(f"Threat score mean: {stats['threat_score_mean']:.2f}")
```

---

### `sanpo_preprocessing_pipeline.py`
**Purpose:** End-to-end orchestrator combining all modules

**Key features:**
- Batch processes multiple sessions
- Stream processing (never caches intermediate frames)
- Outputs: `training_cache.pkl` + `preprocessing_stats.json`

**Usage:**
```bash
python sanpo_preprocessing_pipeline.py /path/to/sanpo 001 002 003
```

**Output files:**
```
preprocessing_outputs/
├── training_cache.pkl         # Pickle file with all threat metrics
├── preprocessing_stats.json   # Processing statistics
└── preprocessing_metrics.json # Frame-level details
```

---

### `threat_prioritizer_finetuner.py`
**Purpose:** Fine-tune threat scoring model on SANPO data

**Architecture:**
- **Input:** [distance_m, velocity_mps, ttc_s, class_one_hot] 
- **Hidden:** 64 → 32 → 16 ReLU neurons
- **Output:** [0, 1] threat score (normalized)

**Training details:**
- Loss: MSE between predicted and ground-truth scores
- Optimizer: Adam (lr=1e-3)
- Validation split: 20%
- Early stopping on validation loss

**Usage:**
```bash
python threat_prioritizer_finetuner.py \
    training_cache.pkl \
    threat_prioritizer_finetuned.pt \
    cpu
```

**Output:** `threat_prioritizer_finetuned.pt` (~50 MB model checkpoint)

---

### `demo_inference.py`
**Purpose:** Run full CPE pipeline on custom video (no depth sensor needed)

**Key features:**
- Simulates egocentric perspective
- Generates mock depth maps from brightness heuristics
- Real-time threat detection + TTS warnings
- Exports annotated video

**Usage:**
```bash
python demo_inference.py /path/to/video.mp4 \
    --model threat_prioritizer_finetuned.pt \
    --output annotated.mp4 \
    --device cpu \
    --tts
```

**Real-time warnings:**
```
THREAT WARNING: CRITICAL: vehicle from left, 5.2m away
THREAT WARNING: HIGH: pedestrian from center, 3.1m away
```

---

## Storage Requirements

### Disk Space Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| SANPO raw (10-15 sessions) | ~150 GB | Videos + depth maps |
| training_cache.pkl | ~100 MB | Metrics only, not frames |
| threat_prioritizer_finetuned.pt | ~50 MB | Fine-tuned model |
| Processing working space | ~50 GB | Temporary during pipeline |
| **Total** | **~200 GB** | Fits MacBook |

### Optimization Tips

1. **Reduce sessions:** Process 5-10 instead of 15 (~75-100 GB)
2. **Compress depth:** Store depth as PNG instead of .npz (-50%)
3. **Downsample resolution:** 1280×720 instead of 1920×1080 (-55%)
4. **Stream-only:** Delete raw SANPO after preprocessing (frees 150 GB)

---

## Inferencing Workflow

### Phase 1: Training (Colab, ~2-3 hours)
```
SANPO subset (100-150K frames)
    ↓
sanpo_preprocessing_pipeline.py
    ↓
training_cache.pkl (100 MB)
    ↓
threat_prioritizer_finetuner.py
    ↓
threat_prioritizer_finetuned.pt (50 MB) [DOWNLOAD]
```

### Phase 2: Demo (MacBook, Real-time)
```
Custom video (iPhone recording)
    ↓
demo_inference.py + threat_prioritizer_finetuned.pt
    ↓
Annotated video + Real-time TTS warnings
    ↓
Live demonstration of CPE
```

---

## Troubleshooting

### YOLO not loading?
```
Set environmental:USAGE: python <script> [options]

To disable YOLO (use mock detections):
from frame_processor import FrameProcessor
processor = FrameProcessor()  # No setup() call = mock detections
```

### TTS not working?
```bash
pip install TTS
# If still failing, run --tts flag is optional (falls back to text)
```

### Out of memory?
```
1. Reduce number of sessions
2. Skip depth caching
3. Use Colab instead of local (free GPU)
4. Stream-process instead of batch
```

### Slow processing?
```
1. Use GPU: device="cuda" or "mps" (Mac)
2. Reduce YOLO model size: "yolov8n" → ultra-fast
3. Skip TTS: remove --tts flag
```

---

## Performance Benchmarks

| Configuration | FPS | Memory |
|---|---|---|
| MacBook CPU (M1) | 5-8 FPS | 2-4 GB |
| Colab T4 GPU | 25-35 FPS | 8-12 GB |
| Custom video demo | Real-time | 1-2 GB |

---

## References

- **SANPO Dataset:** https://sites.google.com/view/sanpo-dataset
- **YOLO:** https://docs.ultralytics.com/
- **ByteTrack:** https://github.com/ifzhang/ByteTrack
- **PyTorch:** https://pytorch.org/

---

## Questions?

Refer to individual module docstrings:
```bash
python -c "from sanpo_loader import SANPOLoader; help(SANPOLoader)"
```
