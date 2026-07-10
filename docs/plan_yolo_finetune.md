# Implementation Plan: YOLO26n Fine-Tuning for Navigation Hazard Detection

**Author:** CPE Team  
**Date:** 2026-07-10  
**Status:** Ready for Implementation  

---

## 1. Objective

Fine-tune YOLO26n specifically for **egocentric pedestrian navigation** by:
1. Removing redundant COCO classes that have no safety relevance.
2. Adding domain-specific hazard classes absent from COCO.
3. Creating a full training script that auto-processes the SANPO dataset.

The goal is a lighter, more accurate edge model that maximises **hazard recall** for the target use case (assisting visually impaired users) without wasting compute on irrelevant COCO classes.

---

## 2. Class Curation Rationale

### 2.1 Current ALLOWED_CLASSES (from `src/perception_stack/yolo_tracker.py`)

```
person, bicycle, car, motorcycle, bus, truck,
dog, cat, traffic light, stop sign, umbrella,
backpack, suitcase, unlabeled_obstacle
```

### 2.2 Classes to DROP (Redundant / Low Safety Value)

| Class | Reason for Removal |
|---|---|
| `cat` | Rare on public footpaths; low kinetic risk; confusable with debris |
| `umbrella` | Not a navigation hazard; no kinetic risk |
| `backpack` | Almost always worn by `person` (already detected); duplicate spatial footprint |
| `suitcase` | Very low kinetic risk; usually stationary luggage |
| `dog` | Low priority — retain only if dataset has significant dog-trip incidents |

> **Note:** `dog` is a borderline case. It has a real trip/bite risk but is a tiny target for YOLO. Recommend keeping it only if the gap analysis notebook (see Plan 2) shows dogs appear in SANPO depth gaps.

### 2.3 Final Confirmed Class List (14 Classes + depth fallback)

| # | Class | COCO? | Reason to Keep |
|---|---|---|---|
| 1 | `person` | ✅ | Primary dynamic threat; pedestrian intent |
| 2 | `bicycle` | ✅ | Silent, fast, often on footpaths |
| 3 | `car` | ✅ | Most common vehicle threat |
| 4 | `motorcycle` | ✅ | High speed, erratic, no lane discipline |
| 5 | `bus` | ✅ | High mass, wide stop zone |
| 6 | `truck` | ✅ | High mass, wide turning radius, blind spots |
| 7 | `traffic light` | ✅ | Navigation anchor; crossing safety |
| 8 | `stop sign` | ✅ | Navigation anchor |
| 9 | `pole` | ❌ New | Bollards, lamp posts — rigid head/chest impact |
| 10 | `bench` | ❌ New | Fixed obstacle, foot-level trip on overhangs |
| 11 | `fire hydrant` | ✅ COCO | Low ground-level obstacle, often near curbs |
| 12 | `stairs` | ❌ New | Critical level-change alert |
| 13 | `crosswalk` | ❌ New | Navigation anchor for crossing intent inference |
| 14 | `overhanging_hazard` | ❌ New | Branches, scaffolding, open shutters — head level |

> **Depth fallback stays:** The existing `unlabeled_obstacle` depth-grid sweep remains as a catch-all for anything not in this list (e.g. construction debris, unexpected furniture).

---

## 3. Dataset Strategy

### 3.1 Source: SANPO (Google Research)
- **Type:** Egocentric pedestrian navigation — chest and head camera viewpoints. Exact match for our use case.
- **Volume:** 701 real sessions + 1,961 synthetic sessions.
- **Depth:** `.float16.gz` ZED stereo sparse depth + CREStereo dense depth.
- **Existing Labels:** 30-class panoptic segmentation masks (sidewalk, road, pole, fence, pedestrian, rider, tree, etc.).

> **Key insight:** SANPO's panoptic masks can serve as **pseudo-labels** for the new domain classes. Poles, fences, trees, and crosswalks are already labelled in the segmentation masks — we just need to convert them to YOLO bounding boxes.

### 3.2 Label Conversion Pipeline (Panoptic Mask → YOLO Bounding Box)
For each frame with a segmentation mask:
1. Load the PNG segmentation mask.
2. For each class of interest, find connected pixel regions.
3. Compute the bounding box of each connected region.
4. Filter boxes smaller than `MIN_AREA_PX = 500`.
5. Write YOLO-format label file: `<class_id> <cx> <cy> <w> <h>` (normalised).

This avoids manual annotation for the classes already covered by SANPO's taxonomy.

### 3.3 Supplementary Data for New Classes
For classes NOT in SANPO's taxonomy (`overhanging_hazard`, `stairs`, `crosswalk` as a distinct detection target):
- Use **Grounding DINO** (offline) to auto-annotate frames from SANPO real sessions.
- Target: 2,000–3,000 bounding box examples per new class.

### 3.4 Dataset Split
| Split | Fraction | Notes |
|---|---|---|
| Train | 80% | Stratified by session (not by frame) to prevent temporal leakage |
| Val | 10% | Used for mAP monitoring during training |
| Test | 10% | Held out completely; used only for final paper benchmark |

---

## 4. Files to Create

```
training/
├── configs/
│   └── cpe_hazard_classes.yaml       ← Dataset config for YOLO training
├── scripts/
│   ├── convert_sanpo_labels.py       ← Panoptic mask → YOLO bbox labels
│   ├── build_finetune_dataset.py     ← Crawls sessions, splits dataset
│   └── train_yolo_finetune.py        ← Launches YOLO26n fine-tune run
└── README.md                         ← Training instructions for teammates
```

---

## 5. `cpe_hazard_classes.yaml` (Dataset Config)

```yaml
# CPE Navigation Hazard — YOLO26n fine-tune dataset config
path: data/yolo_finetune         # Root of the formatted dataset
train: images/train
val:   images/val
test:  images/test

nc: 14   # Number of classes

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: bus
  5: truck
  6: traffic_light
  7: stop_sign
  8: fire_hydrant
  9: pole
  10: bench
  11: stairs
  12: crosswalk
  13: overhanging_hazard
```

---

## 6. Training Script Outline (`train_yolo_finetune.py`)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")   # Start from pretrained COCO weights

results = model.train(
    data    = "training/configs/cpe_hazard_classes.yaml",
    epochs  = 100,
    imgsz   = 640,           # Standard YOLO training resolution
    batch   = 16,            # Adjust based on GPU VRAM
    lr0     = 0.001,         # Conservative LR for fine-tuning
    warmup_epochs = 3,
    freeze  = 10,            # Freeze first 10 backbone layers
    val     = True,
    project = "training/runs",
    name    = "cpe_yolo26n_v1",
    device  = "0",           # GPU 0
    patience= 20,            # Early stopping
    save    = True,
    plots   = True,
)
```

**Key training considerations:**
- **Freeze backbone layers:** We freeze the first 10 layers since COCO features (edges, shapes, textures) transfer well. Only the neck and head are re-trained heavily.
- **Class imbalance:** Use `cls_pw` (class positive weight) to up-weight underrepresented classes like `overhanging_hazard` and `stairs`.
- **Augmentation:** Enable mosaic augmentation for small objects like `pole` and `fire_hydrant`.

---

## 7. `yolo_tracker.py` Changes Required

After training, update `src/perception_stack/yolo_tracker.py`:

1. **`ALLOWED_CLASSES`:** Replace with the 14-class set above. Remove `cat`, `umbrella`, `backpack`, `suitcase`.
2. **`CLASS_REAL_HEIGHT_M`:** Add height estimates for new classes:
   - `pole`: 3.0m (lamp post); 1.0m (bollard)
   - `stairs`: 0.8m
   - `crosswalk`: 0.1m (flat — ground plane detection)
   - `overhanging_hazard`: 2.5m
3. **`CLASS_SEVERITY`** in `physics.py`: Add severity weights for new classes.
4. **Model path:** Update `DEFAULT_MODEL` to point to the fine-tuned checkpoint.

---

## 8. Verification & Benchmarks

| Metric | Target |
|---|---|
| mAP@50 (all 14 classes) | ≥ 0.65 |
| mAP@50 for `person`, `car`, `bus`, `truck` | ≥ 0.75 |
| mAP@50 for new classes (`pole`, `stairs`, `overhanging_hazard`) | ≥ 0.50 |
| Inference latency on edge GPU (INT8) | ≤ 15ms per frame |
| No regression on original COCO classes kept | mAP within 5% of baseline |

---

## 9. Open Questions for Team

1. **GPU availability:** The fine-tuning run (~100 epochs on 700 sessions) will need a full GPU. Can this be run on Colab Pro or a lab server?
2. **Grounding DINO access:** Do we have a Colab or local setup to run Grounding DINO for auto-labelling `overhanging_hazard` and `stairs`?
3. **`dog` class decision:** After running the gap analysis notebook, check how often dogs appear in depth gaps within 4m. If <5% of frames, drop the class.
4. **Test set identity:** Which SANPO sessions should be held out as the final test set? Should be decided **before** training starts to prevent data leakage.
