# Google Cloud Storage Setup for SANPO Processing in Colab

This guide shows how to run the preprocessing pipeline directly from a Google Cloud Storage (GCS) bucket without needing to download SANPO data locally.

## Prerequisites

1. **SANPO data in GCS bucket** (structure must match):
   ```
   gs://your-bucket/sanpo/
   ├── videos/
   │   ├── 001.mp4
   │   ├── 002.mp4
   │   └── ...
   ├── depth_maps/
   │   ├── 001/
   │   │   ├── depth_0000.npz
   │   │   └── ...
   │   └── ...
   └── metadata.json
   ```

2. **Google Cloud Project** with:
   - Cloud Storage API enabled
   - Service Account with Storage Object Viewer permissions
   - Or use default credentials (if running under your GCP project)

---

## Step 1: Create GCS Bucket and Upload SANPO Data

```bash
# Create bucket
gsutil mb gs://your-sanpo-bucket/

# Upload SANPO data (from local machine)
gsutil -m cp -r /path/to/sanpo/* gs://your-sanpo-bucket/sanpo/

# Verify structure
gsutil ls -r gs://your-sanpo-bucket/sanpo/ | head -20
```

---

## Step 2: Set Up Colab Notebook

### Option A: Using Default GCP Credentials (Recommended)

```python
# Cell 1: Install dependencies and authenticate
!pip install -q google-cloud-storage ultralytics opencv-python torch numpy scikit-image

# Authenticate with Google Cloud
from google.colab import auth
auth.authenticate_user()

print("✓ Authenticated with Google Cloud")
```

### Option B: Using Service Account Key (If needed)

```python
# Cell 1: Install dependencies
!pip install -q google-cloud-storage ultralytics opencv-python torch numpy scikit-image

# Upload your service-account-key.json to Colab
# Then authenticate:
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    '/content/service-account-key.json'
)

print("✓ Using service account authentication")
```

---

## Step 3: Clone Repository and Run Pipeline

```python
# Cell 2: Setup
!cd /content && git clone https://github.com/your-user/CompositePerceptionEngine.git 2>/dev/null || echo "Skipped"
%cd /content/CompositePerceptionEngine/simulation/datasets

import logging
logging.basicConfig(level=logging.INFO)

print("✓ Repository ready")
```

```python
# Cell 3: Run preprocessing from GCS
from sanpo_preprocessing_pipeline import SANPOPreprocessingPipeline

# Replace with your bucket and path
GCS_SANPO_PATH = "gs://your-sanpo-bucket/sanpo"

pipeline = SANPOPreprocessingPipeline(
    sanpo_root=GCS_SANPO_PATH,              # ← GCS path!
    session_ids=["001", "002", "003"],      # Process 3 sessions
    output_dir="./preprocessing_outputs",
    device="cuda"                           # Colab T4 GPU
)

pipeline.run()
print("✓ Preprocessing complete!")
```

---

## Step 4: Fine-tune Model

```python
# Cell 4: Fine-tune threat model
from threat_prioritizer_finetuner import ThreatPrioritizerFinetuner

finetuner = ThreatPrioritizerFinetuner(
    training_cache_path="./preprocessing_outputs/training_cache.pkl",
    model_output_path="./threat_prioritizer_finetuned.pt",
    device="cuda"
)

finetuner.finetune(epochs=10, batch_size=32)
print("✓ Model training complete!")
```

---

## Step 5: Download Results

```python
# Cell 5: Upload results back to GCS
!gsutil cp threat_prioritizer_finetuned.pt gs://your-sanpo-bucket/models/
!gsutil cp preprocessing_outputs/training_cache.pkl gs://your-sanpo-bucket/cache/

# Or download to Colab
from google.colab import files
files.download("threat_prioritizer_finetuned.pt")
files.download("preprocessing_outputs/preprocessing_stats.json")

print("✓ Results saved!")
```

---

## Full Colab Notebook Template

Copy and paste into a new Colab notebook:

```python
# ============================================================================
# SANPO Preprocessing Pipeline — Google Cloud Storage
# ============================================================================

# ==== CELL 1: Install and Authenticate ====
!pip install -q google-cloud-storage ultralytics opencv-python torch numpy scikit-image

from google.colab import auth
auth.authenticate_user()

print("✓ Authenticated with Google Cloud")


# ==== CELL 2: Clone and Setup ====
!cd /content && git clone https://github.com/your-user/CompositePerceptionEngine.git 2>/dev/null || echo "Skipped"
%cd /content/CompositePerceptionEngine/simulation/datasets

import logging
logging.basicConfig(level=logging.INFO)
print("✓ Repository ready")


# ==== CELL 3: Configure GCS Path ====
GCS_SANPO_PATH = "gs://your-bucketname/sanpo"  # ← EDIT THIS
print(f"Using GCS path: {GCS_SANPO_PATH}")


# ==== CELL 4: Run Preprocessing ====
from sanpo_preprocessing_pipeline import SANPOPreprocessingPipeline

pipeline = SANPOPreprocessingPipeline(
    sanpo_root=GCS_SANPO_PATH,
    session_ids=["001", "002", "003", "004", "005"],  # Process 5 sessions
    output_dir="./preprocessing_outputs",
    device="cuda"
)

pipeline.run()
print("✓ Preprocessing complete!")


# ==== CELL 5: Fine-tune Model ====
from threat_prioritizer_finetuner import ThreatPrioritizerFinetuner

finetuner = ThreatPrioritizerFinetuner(
    training_cache_path="./preprocessing_outputs/training_cache.pkl",
    model_output_path="./threat_prioritizer_finetuned.pt",
    device="cuda"
)

finetuner.finetune(epochs=10, batch_size=32, learning_rate=1e-3)
print("✓ Model training complete!")


# ==== CELL 6: Save Results ====
!gsutil cp threat_prioritizer_finetuned.pt gs://your-bucketname/models/
!gsutil cp preprocessing_outputs/training_cache.pkl gs://your-bucketname/cache/

print("✓ Results saved to GCS!")

# Optional: Download to Colab
from google.colab import files
files.download("threat_prioritizer_finetuned.pt")
files.download("preprocessing_outputs/preprocessing_stats.json")
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Permission denied" | Ensure service account has `Storage Object Viewer` role |
| "Bucket not found" | Check GCS path format: `gs://bucket-name/path` |
| "YOLO not available" | Run: `!pip install ultralytics` in Colab |
| "OutOfMemory on GPU" | Reduce `session_ids` list (process 3 instead of 5) |
| "Slow video download" | Normal — videos are streamed from GCS (~50-100 MB each) |

---

## Performance Tips

1. **Use regional bucket**: Create bucket in `us-central1` (same as Colab)
2. **Set session IDs**: Process 3-5 sessions at a time, then re-run
3. **Monitor quota**: Run `!gsutil quota` to check storage quota
4. **Use T4 GPU**: Processing ~10-15 frames/sec on Colab T4

---

## Storage Costs

- **GCS Storage**: $0.020/GB/month
- **Data Transfer**: Free within Google Cloud
- **Example**: 50 GB SANPO data = ~$1/month storage

---

## Next Steps

After fine-tuning:
1. Test model on custom video (use `demo_inference.py`)
2. Iterate with more SANPO sessions (10-20 sessions recommended)
3. Deploy model to mobile device or edge runtime
