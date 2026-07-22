# SANPO YOLO Gap Analysis Guide

**Author:** CPE Team  
**Date:** 2026-07-10  
**Status:** Notebook ready (`notebooks/sanpo_yolo_gap_analysis.ipynb`) — this plan documents config decisions and teammate instructions.

---

## 1. Objective

Run the existing gap analysis notebook across the SANPO dataset to **empirically identify** which physical objects are consistently present at hazard range (0.5–6m) but are NOT detected by the current YOLO26n whitelist.

The outputs directly inform which new classes to prioritise in the YOLO training guide (`docs/yolo_training.md`).

---

## 2. What the Gap Analysis Does

For every sampled frame:
1. Run YOLO26n with the current CPE detector configuration.
2. Load the `.float16.gz` SANPO depth map.
3. Find all depth "blobs" in the **0.5–6.0m alert zone** that have **no overlapping YOLO bounding box**.
4. Tag each blob with:
   - **Elevation:** `head_level` (top 45% of frame) | `mid_level` | `foot_level`
   - **Column:** which of 5 vertical navigation columns it occupies
   - **Depth:** nearest and mean distance in metres
5. Generate a 3-panel visualization and save it to the output folder.
6. Export all stats to a CSV for summary analysis.

The frequency of unlabelled blobs across all sessions tells us **what the gap classes are**, and the elevation distribution tells us **how dangerous they are**.

---

## 3. Data Structure

```
data/sanpo/raw/
└── <session_hash>/
    └── camera_head/
        └── left/
            ├── video_frames/        ← RGB PNGs  (000000.png, 000001.png, ...)
            ├── depth_maps/          ← Depth     (000000.float16.gz, ...)
            └── frame_segmentation_annotation_type.json
```

Sessions confirmed locally:
- `01cbb9d502fbee1f...` — 143 frames, SYNTHETIC
- `0393b4b31d79f1d7...` — (check frame count)
- `04bfa5b7f31e6b5c...` — (check frame count)

---

## 4. Key Configuration Parameters

All parameters are in **Cell 1** of `notebooks/sanpo_yolo_gap_analysis.ipynb`.

| Parameter | Default | Description |
|---|---|---|
| `SESSION_FILTER` | `None` | Set to a list of session hashes to restrict analysis. `None` = all sessions. |
| `MAX_FRAMES_PER_SESSION` | `30` | **Critical — controls output volume.** Frames are sampled evenly across the session (not just the first N). |
| `DEPTH_MIN_M` | `0.5` | Minimum depth to count as a hazard candidate. |
| `DEPTH_MAX_M` | `6.0` | Maximum depth. Objects beyond 6m are informational only. |
| `MIN_BLOB_AREA_PX` | `800` | Minimum blob pixel area to report. Filters out noise/reflections. |
| `OBSTACLE_GRID_COLS` | `5` | Number of vertical navigation columns to sweep. |
| `YOLO_MODEL_PATH` | `models/yolo/base_yolo26n/yolo26n.pt` | Path to YOLO model (relative to repo root). |

---

## 5. Output Folder Management

> **This is important to prevent folder bloat.**

The notebook only saves visualizations for frames **that actually have depth gaps** — frames where YOLO detected everything get skipped automatically.

With `MAX_FRAMES_PER_SESSION = 30` and 3 sessions, worst-case output is **90 PNG files**. Typical output will be far fewer since many frames will be fully covered by YOLO.

### Output locations:
```
notebooks/gap_analysis_output/
├── <session_hash>_<frame_id>.png    ← 3-panel visualization (saved only for gap frames)
└── gap_analysis_stats.csv          ← Per-frame statistics table
```

### Recommended settings for different scenarios:

| Scenario | `MAX_FRAMES_PER_SESSION` | Expected output files |
|---|---|---|
| Quick check (1 session) | 10 | ~5–10 PNGs |
| Standard analysis (all 3 sessions) | 30 | ~20–60 PNGs |
| Full SANPO sweep (all sessions) | 20 | ~500–700 PNGs (large run) |

---

## 6. Running the Notebook — Step by Step

### Prerequisites
```bash
# Install dependencies (if not already in .venv)
pip install ultralytics opencv-python matplotlib scipy pandas
```

### Steps

1. **Open the notebook:**
   ```
   notebooks/sanpo_yolo_gap_analysis.ipynb
   ```

2. **Configure Cell 1:**
   - For a quick first run, set `MAX_FRAMES_PER_SESSION = 10` and `SESSION_FILTER = None`.
   - Confirm `YOLO_MODEL_PATH` points to your `models/yolo/base_yolo26n/yolo26n.pt` (default is correct if running from repo root).

3. **Run all cells in order (Cells 1–9).**
   - Cell 7 (the main loop) will print progress per frame and display inline visualizations.

4. **Review Cell 8 (Summary Statistics).**
   This is the most important output — it shows:
   - Which YOLO classes are being detected most (what YOLO IS covering).
   - How many frames have head-level vs foot-level gaps (severity distribution).
   - Top 10 frames by gap count (inspect those frames manually in Cell 10).

5. **Use Cell 10 to manually inspect interesting frames** by changing `FRAME_INDEX`.

---

## 7. How to Read the 3-Panel Visualization

```
[ RGB + YOLO boxes (blue) + gap dots ]  |  [ Depth heatmap 0–10m ]  |  [ Gap-only mask ]
```

| Visual Element | Meaning |
|---|---|
| **Blue rectangle** | YOLO detected this object (labelled, tracked) |
| **Red filled circle** | Depth blob at **head/chest level** — highest priority gap |
| **Orange filled circle** | Depth blob at **mid/waist level** |
| **Green filled circle** | Depth blob at **foot level** — trip hazard |
| **White ring** | Outline of any gap blob |
| Label text | Elevation tag + mean depth in metres |

---

## 8. Interpreting the Gap Statistics CSV

The output CSV (`gap_analysis_stats.csv`) has these columns:

| Column | Description |
|---|---|
| `session` | Session hash (first 20 chars) |
| `frame` | Frame index |
| `yolo_detections` | Number of YOLO detections in this frame |
| `yolo_classes` | Pipe-separated list of detected class names |
| `depth_gaps_total` | Total unlabelled depth blobs found |
| `gaps_head_level` | Count of head-level gaps |
| `gaps_mid_level` | Count of mid-level gaps |
| `gaps_foot_level` | Count of foot-level gaps |
| `nearest_gap_m` | Distance to the nearest unlabelled obstacle (metres) |
| `largest_gap_area` | Pixel area of the largest gap blob |

### Analysis queries to run in pandas after the run:

```python
import pandas as pd
df = pd.read_csv("notebooks/gap_analysis_output/gap_analysis_stats.csv")

# What fraction of frames have significant gaps?
print((df.depth_gaps_total > 0).mean())

# What is the average number of head-level gaps per frame?
print(df.gaps_head_level.mean())

# Frames with a gap closer than 2m (very urgent)
urgent = df[df.nearest_gap_m < 2.0]
print(urgent.shape[0], "urgent frames")
```

---

## 9. What to Do with the Results

Once the analysis is complete:

1. **Export the CSV** and open it with the team. Sort by `depth_gaps_total` to find the most gap-heavy sessions.

2. **Visually inspect** the top 20 gap frames (already saved as PNGs in the output folder). Manually note down what objects you can see in those depth gap regions:
   - Poles / bollards?
   - Benches?
   - Vegetation or other unlabeled overhead/sidewalk clutter?
   - Construction barriers?
   - Stairs / kerbs?

3. **Feed this observation back into the YOLO data plan** (`docs/yolo_training.md`). If 70% of gap frames show a pole in that region, `pole` is your highest-priority new class.

4. **Update CHANGELOG.md** once the analysis run is complete and the gap class list is finalised.

---

## 10. Open Questions for Team

1. **Do we have access to more SANPO sessions?** The local `data/sanpo/raw/` folder has only 3 sessions. The full SANPO dataset is on Google Cloud (GCS bucket). Should we download more sessions for a more statistically robust gap analysis?

2. **Should we also run the gap analysis on real SANPO sessions** (not just synthetic)? Real sessions may show different gap distributions (e.g., more construction hazards).

3. **Segmentation masks are available.** SANPO includes panoptic segmentation masks (`segmentation_masks/` folder). If the gap analysis shows consistent blobs in a region, we can cross-reference the segmentation mask to get the ground-truth class of the unlabelled blob. This would be a very strong paper result.
