# 🚀 COLAB QUICK START — PREPROCESSING ONLY

## What Does This Do?

Preprocesses SANPO egocentric video data:
1. Loads frames + depth from public GCS bucket
2. Detects pedestrians with YOLOv8
3. Tracks objects across frames
4. Calculates threat scores (0-10 based on proximity/velocity/TTC)

**Output:** `training_cache.pkl` (threat metrics data for training)

---

## How to Run

### 1. Open Colab
https://colab.research.google.com → **Upload** → `COLAB_NOTEBOOK.ipynb`

### 2. Enable GPU
**Runtime → Change runtime type → GPU**

### 3. Run 4 Cells

Just click ▶️ on each:
- **Cell 1:** Install packages  
- **Cell 2:** Clone GitHub repo
- **Cell 3:** Run preprocessing
- **Cell 4:** Show output files

### 4. Adjust Runtime (Optional)

Edit `num_sessions` in Cell 3:
```python
num_sessions = 10  # 5, 10, or 100
```

- `5` = ~15 min
- `10` = ~1 hour
- `100` = ~4 hours

---

## Output Files

After Cell 3 completes:
- **`training_cache.pkl`** (~10-50 MB) — Threat metrics for training
- **`preprocessing_stats.json`** (~1 KB) — Processing stats

Download from: `/content/preprocessing_outputs/`

---

**That's it!** You have preprocessed SANPO data. Use `training_cache.pkl` to train your threat model. 🎉
