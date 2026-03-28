"""
depth_loader.py
===============
CPE Perception Stack — Dataset-aware depth map loader.

Handles both SANPO and UASOL depth formats:

  SANPO-Synthetic / SANPO-Real:
      File   : <frame>.float16.gz
      Format : gzip-compressed float16 numpy array, shape (H, W)
      Values : metric depth in metres (no scale needed)
      Res    : 2208×1242 (downsized to match RGB frame by load_depth_map)

  UASOL:
      File   : img_depth.png (16-bit PNG)
      Format : uint16, scale × 0.256 = metres
"""

from __future__ import annotations
import gzip
from pathlib import Path

import cv2
import numpy as np

# Physical depth range filter — anything outside this is sensor noise or irrelevant
MIN_DEPTH_M = 0.3    # below this = too close / noise
MAX_DEPTH_M = 30.0   # beyond this = not actionable

# UASOL-only scale factor (SANPO values are already in metres as float16)
DEPTH_SCALES = {
    "sanpo": 1.0,     # float16 gz — already metres
    "uasol": 0.256,   # uint16 PNG — raw × 0.256 = metres
}

# Known SANPO depth resolution (synthetic + real)
SANPO_DEPTH_H = 1242
SANPO_DEPTH_W = 2208


def load_depth_map(depth_path: Path, source: str = "sanpo") -> np.ndarray | None:
    """
    Load a depth map and return a float32 array in metres, sized to match
    the RGB frame (resized if needed).

    Auto-detects format by file extension:
      *.float16.gz  →  SANPO (decompress → reshape float16 → float32)
      *.png         →  UASOL (cv2 ANYDEPTH → scale)

    Returns None if the file does not exist.
    """
    if not depth_path.exists():
        return None

    suffix = "".join(depth_path.suffixes).lower()   # e.g. ".float16.gz" or ".png"

    if suffix == ".float16.gz":
        # SANPO: gzip-compressed raw float16 binary, row-major (H, W).
        # Files contain 2 extra padding float16 values at the end — truncate to exact size.
        with gzip.open(depth_path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.float16)
        n = SANPO_DEPTH_H * SANPO_DEPTH_W
        depth = raw[:n].reshape(SANPO_DEPTH_H, SANPO_DEPTH_W).astype(np.float32)
        return depth   # already in metres

    elif suffix == ".png":
        # UASOL: 16-bit PNG, scale to metres
        raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if raw is None:
            return None
        scale = DEPTH_SCALES.get(source, 1.0)
        return raw.astype(np.float32) * scale

    else:
        raise ValueError(f"Unknown depth format: {depth_path.name!r}")


def median_depth_in_box(
    depth_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> float | None:
    """
    Return the median metric depth (metres) of valid pixels inside a bounding box.
    Crops ROI, filters noise, returns None if no valid pixels found.
    """
    # Clamp bbox to depth map bounds (SANPO frames may be downscaled by YOLO)
    h, w = depth_map.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x1 >= x2 or y1 >= y2:
        return None

    roi   = depth_map[y1:y2, x1:x2]
    valid = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
