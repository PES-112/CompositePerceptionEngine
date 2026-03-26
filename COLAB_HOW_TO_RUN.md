# HOW TO RUN ON COLAB - Step by Step

## 1. Open Google Colab
```
https://colab.research.google.com
```

## 2. Upload the Notebook
- **File** → **Upload notebook**
- Select `COLAB_NOTEBOOK.ipynb` from your computer
- OR **File** → **Open notebook** → **GitHub**
  - Paste: `https://github.com/ksrikrishnareddy/CompositePerceptionEngine`
  - View notebook

## 3. Enable GPU (Critical!)
```
Menu: Runtime → Change runtime type
Hardware accelerator: GPU
Select T4 (free tier)
Click Save
```
⚠️ **Must enable GPU or it will be very slow!**

## 4. Run Cells in Order

Simply click ▶️ play button on each cell:

### Cell 1: Install packages (~1-2 min)
- Installs: google-cloud-storage, ultralytics, torch, opencv
- Wait for: `✓ Dependencies installed`

### Cell 2: Clone repository (~10-15 sec)  
- Clones repo from GitHub
- Sets up Python path
- Wait for: `✓ Repo cloned`

### Cell 3: Run preprocessing pipeline (⏱️ depends on num_sessions)
**Before running:**
- Edit `num_sessions = 10` to control runtime:
  - `5` sessions →  ~15 min
  - `10` sessions → ~1 hour
  - `100` sessions → ~4 hours

- Click ▶️ to start
- Watch progress as it processes each session
- Example output:
  ```
  [1/10] Session: abc123...
    ✓ Processed 240 frames, 156 detections
  [2/10] Session: def456...
    ✓ Processed 210 frames, 189 detections
  ...
  ```
- Wait for: `✓ Preprocessing complete!`

### Cell 4: Show output files (~5 sec)
- Lists the generated files and their sizes
- Example:
  ```
  training_cache.pkl: 45.3 MB
  preprocessing_stats.json: 0.0 MB
  ```

## 5. Download Results

After Cell 4 completes:

**Option A: Colab File Browser (Easy)**
- Left sidebar → **Files** icon
- Navigate to `/content/preprocessing_outputs/`
- Right-click each file → **Download**

**Option B: Drag & Drop**
- Files appear in left sidebar
- Drag to your computer

## Output Files

After preprocessing:
- **`training_cache.pkl`** (10-50 MB)
  - Threat metrics accumulated from all sessions
  - Use this data to train your model
  
- **`preprocessing_stats.json`** (~1 KB)
  - Statistics: sessions processed, frames, detections, time

---

## ⚠️ Common Issues

| Issue | Solution |
|---|---|
| **"CUDA out of memory"** | Reduce `num_sessions` to 5. Click **Runtime → Restart runtime** |
| **"No module google.cloud"** | Re-run Cell 1 (pip install) |
| **"GCS connection failed"** | Rare. Try restarting runtime. |
| **Very slow (~5 min/session instead of 1-2)** | Normal on free T4. Smaller YOLO model taking time. |
| **Colab session disconnected** | Sessions may timeout after 12 hours. Re-run from where it left off. |

---

## Expected Output Structure

After preprocessing 10 sessions, your `training_cache.pkl` will contain:

```
[
  {
    "frame_id": 0,
    "timestamp": 0.0,
    "session_id": "abc123xyz...",
    "detections": [
      {
        "track_id": 1,
        "class_name": "person",
        "distance_m": 3.5,
        "velocity_mps": 1.2,
        "ttc_s": 2.9,
        "kinetic_score": 4.2
      },
      ...
    ]
  },
  ... (more frames)
]
```

This data is ready for training a threat prediction model.

---

## Next Steps (Optional)

After you have `training_cache.pkl`:
1. Download it locally
2. Use `threat_prioritizer_finetuner.py` to train a model
3. Export model weights for deployment

---

**That's it!** 🎉 You've successfully preprocessed SANPO data in Colab.
