# SANPO from Google Cloud Storage — Ready-to-Run Guide

## Complete Working Example for Colab

```python
# ============================================================================
# SANPO Processing from Google Cloud Storage (Public Bucket)
# ============================================================================
# This notebook processes SANPO data directly from the public GCS bucket:
# gs://gresearch/sanpo_dataset/v0/sanpo-real/
# ============================================================================

# ==== CELL 1: Install Dependencies ====
!pip install -q google-cloud-storage ultralytics opencv-python torch numpy scikit-image

print("✓ Dependencies installed")


# ==== CELL 2: Setup Google Cloud Access ====
from google.colab import auth
auth.authenticate_user()

import logging
logging.basicConfig(level=logging.INFO)

print("✓ Google Cloud authenticated")


# ==== CELL 3: Clone Repository ====
!cd /content && git clone https://github.com/your-user/CompositePerceptionEngine.git 2>/dev/null || echo "Skipped"
%cd /content/CompositePerceptionEngine/simulation/datasets

import sys
sys.path.insert(0, '/content/CompositePerceptionEngine/simulation/datasets')

print("✓ Repository ready")


# ==== CELL 4: Test SANPO Loader ====
from sanpo_loader import SANPOLoader

# Initialize loader for chest camera, left view
loader = SANPOLoader(
    sanpo_root="gs://gresearch/sanpo_dataset/v0/sanpo-real",
    camera="chest",
    view="left"
)

# Get first 5 sessions
sessions = loader.list_sessions()[:5]
print(f"Found {len(sessions)} sessions")
print(f"First 5: {sessions}")

# Load metadata for first session
first_session = sessions[0]
meta = loader.get_session_metadata(first_session)
print(f"\nMetadata for {first_session}:")
print(json.dumps(meta, indent=2)[:500])  # Print first 500 chars


# ==== CELL 5: Stream Frames from First Session ====
from sanpo_loader import Frame
import numpy as np

# Load first 10 frames from first session
frame_count = 0
for frame in loader.iter_frames(first_session, max_frames=10):
    frame_count += 1
    print(f"Frame {frame.frame_id}: "
          f"RGB shape={frame.rgb.shape}, "
          f"Depth range=[{frame.depth.min():.2f}, {frame.depth.max():.2f}]m")

print(f"\n✓ Loaded {frame_count} frames")


# ==== CELL 6: Run Preprocessing Pipeline ====
from sanpo_preprocessing_pipeline import SANPOPreprocessingPipeline

# Create pipeline
pipeline = SANPOPreprocessingPipeline(
    sanpo_root="gs://gresearch/sanpo_dataset/v0/sanpo-real",
    session_ids=sessions,  # Use first 5 sessions
    output_dir="./preprocessing_outputs",
    device="cuda"
)

# Run preprocessing
pipeline.run()

print("✓ Preprocessing complete!")
print(f"Outputs: preprocessing_outputs/")


# ==== CELL 7: Fine-tune Threat Model ====
from threat_prioritizer_finetuner import ThreatPrioritizerFinetuner

finetuner = ThreatPrioritizerFinetuner(
    training_cache_path="./preprocessing_outputs/training_cache.pkl",
    model_output_path="./threat_prioritizer_finetuned.pt",
    device="cuda"
)

finetuner.finetune(
    epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    validation_split=0.2
)

print("✓ Model training complete!")


# ==== CELL 8: Download Results ====
from google.colab import files

files.download("threat_prioritizer_finetuned.pt")
files.download("preprocessing_outputs/training_cache.pkl")
files.download("preprocessing_outputs/preprocessing_stats.json")

print("✓ Files downloaded!")
```

---

## What Changed in the Loader

| Feature | Old | New |
|---------|-----|-----|
| **Data Format** | MP4 videos | PNG frame sequences |
| **Depth Format** | `.npz` files | `.float16.gz` (compressed) |
| **Session IDs** | "001", "002" | Hash strings (hash IDs) |
| **Metadata** | metadata.json in root | description.json per session |
| **Cameras** | Single | Multiple (chest/head, left/right) |
| **Source** | Custom bucket | Public GCS: `gs://gresearch/sanpo_dataset/v0/` |

---

## Key Parameters

```python
loader = SANPOLoader(
    sanpo_root="gs://gresearch/sanpo_dataset/v0/sanpo-real",
    camera="chest",      # or "head"
    view="left"          # or "right"
)
```

---

## Quick Stats

- **Total sessions**: 700+ sessions available
- **Cameras**: chest, head (left and right views each)
- **Depth format**: float16 compressed with gzip
- **Frame format**: PNG sequences
- **Public access**: Yes, no authentication needed

---

## Usage Pattern

```python
# 1. Initialize loader
loader = SANPOLoader("gs://gresearch/sanpo_dataset/v0/sanpo-real")

# 2. List sessions (returns hash IDs)
session_list = loader.list_sessions()

# 3. Iterate through frames
for session_id in session_list[:5]:  # First 5 sessions
    for frame in loader.iter_frames(session_id, max_frames=100):
        # Process frame
        rgb = frame.rgb        # [H, W, 3] uint8
        depth = frame.depth    # [H, W] float32 (meters)
```

---

## Estimated Performance

| Operation | Time | Resource |
|-----------|------|----------|
| Load 100 frames | ~5-10 min | Colab T4 |
| Preprocess 5 sessions | ~1-2 hours | Colab T4 |
| Fine-tune model | ~10 min | Colab T4 |
| **Total** | **~2-3 hours** | **Colab T4** |

---

## Notes

- ✅ No local storage needed (streams from GCS)
- ✅ Public bucket (no authentication required in Colab)
- ✅ Automatic gzip decompression
- ✅ Memory-efficient (generator-based)
- ⚠️ First frame loading may be slow (GCS network latency)
- ⚠️ Depth values in meters (may need normalization for some models)
