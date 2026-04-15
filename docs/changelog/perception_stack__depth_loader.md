# Changelog — `src/perception_stack/depth_loader.py`

> Handles loading SANPO and UASOL depth maps from disk and sampling depth values from bounding-box regions.

---

## [v1.1] — 2026-03-30 | Performance Overhaul

**Session:** *Optimizing CPE Perception Pipeline Performance*

### What Changed

#### `median_depth_in_box()` — Centre-Patch Sampling

**Problem:** The original implementation sampled the *entire* bounding box, averaging edge pixels that bleed from background surfaces (sky, walls, ground). This caused systematic depth overestimation for any object near another surface.

**Fix:** Sample only the **central 20% patch** (controlled by `center_frac` parameter) of the bounding box instead of the full ROI. This targets the object's core pixels, which are the most depth-representative.

```
Before:  roi = depth_map[y1:y2, x1:x2]           (entire bbox)
After:   roi = central 20% patch of the bbox       (centre_frac=0.2)
```

#### `median_depth_in_box()` — Occlusion Exclusion Masking

**Problem:** When object A occludes object B in image space, the depth map at B's bounding box still contains pixels from A's closer surface. This made B appear artificially closer than it actually was in 3D space.

**Fix:** Added `exclude_boxes: list | None` parameter. Before sampling, a boolean mask is built that zeros out all pixels belonging to `exclude_boxes`. The pipeline passes already-processed closer objects as occluders when processing each subsequent (farther) detection.

#### `median_depth_in_box()` — Sparse Depth Fallback

**Problem:** ZED stereo depth and SANPO synthetic depth can have sparse valid-pixel regions, especially for small objects or objects at depth discontinuities.

**Fix:** If no valid pixels are found in the centre patch, automatically expand the sampling region **3×** and retry once before returning `None`. This maintains robustness without polluting normal cases with edge pixels.

### New Function Signatures

```python
# Before
def median_depth_in_box(depth_map, x1, y1, x2, y2) -> float | None:

# After
def median_depth_in_box(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    center_frac: float = 0.2,       # NEW — centre sampling fraction
    exclude_boxes: list | None = None,  # NEW — occlusion mask
) -> float | None:
```

### Backward Compatibility
- ✅ Default `center_frac=0.2` and `exclude_boxes=None` means existing call sites with positional args work unchanged.

---

## [v1.0] — 2026-03-25 | Initial Implementation

**Session:** *Implementing Argus-REI Perception Pipeline*

### What Was Built

- `load_depth_map(depth_path, source)` — auto-detects SANPO `.float16.gz` vs UASOL `.png` depth format:
  - SANPO: gzip-decompresses raw float16 binary, truncates 2 padding values, reshapes to `(1242, 2208)` float32
  - UASOL: reads 16-bit PNG with `cv2.IMREAD_ANYDEPTH`, multiplies by `0.256` to convert to metres
- `median_depth_in_box(depth_map, x1, y1, x2, y2)` — clamps to depth map bounds, filters noise (`0.3m–30m`), returns median of valid pixels
- Constants: `MIN_DEPTH_M = 0.3`, `MAX_DEPTH_M = 30.0`, `DEPTH_SCALES = {sanpo: 1.0, uasol: 0.256}`
- Documented SANPO resolution constants: `SANPO_DEPTH_H = 1242`, `SANPO_DEPTH_W = 2208`

---

## File Reference
- **Source:** [`src/perception_stack/depth_loader.py`](../../src/perception_stack/depth_loader.py)
- **Uses:** `numpy`, `cv2`, `gzip`
- **Used by:** `pipeline.py`, `yolo_tracker.py`
