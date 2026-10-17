"""
depth_loader.py
===============
CPE Perception Stack — Dataset-aware depth map loader.

Handles both SANPO and UASOL depth formats:

  SANPO-Synthetic / SANPO-Real:
      File   : <frame>.float16.gz
      Format : gzip-compressed float16: a 2-value header [H, W] followed by
               the H×W depth image, row-major (SANPO-Real depth_maps/: [720, 1280]).
      Values : metric depth in metres (no scale needed)
      Note   : before 2026-10-17 the header was mistaken for padding and the
               921,602 values were reshaped as 960×960 from the start, which
               scrambles the image into three shifted, interleaved copies.
               Every Stage-1 distance built before then is affected.

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

# Fallback shapes for headerless files (none seen in SANPO-Real so far).
SANPO_DEPTH_SHAPES = (
    (720, 1280),
    (1242, 2208),
)


def decode_sanpo_depth(raw: np.ndarray) -> np.ndarray:
    """
    float16 values of one SANPO depth file -> float32 (H, W) metres. The file
    starts with its own shape, [H, W], stored as two float16 values; the image
    follows. A file without that header is reshaped to a known SANPO shape.
    """
    if raw.size > 2:
        h, w = float(raw[0]), float(raw[1])
        if h.is_integer() and w.is_integer() and h > 0 and w > 0 and int(h) * int(w) == raw.size - 2:
            return raw[2:].reshape(int(h), int(w)).astype(np.float32)
    for h, w in SANPO_DEPTH_SHAPES:
        if raw.size == h * w:
            return raw.reshape(h, w).astype(np.float32)
    raise ValueError(f"Cannot decode SANPO depth: {raw.size} float16 values, header {raw[:2].tolist()}")


def scale_box_to_depth(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_w: int,
    frame_h: int,
    depth_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    depth_h, depth_w = depth_shape[:2]
    sx = depth_w / max(frame_w, 1)
    sy = depth_h / max(frame_h, 1)
    dx1 = max(0, min(depth_w, int(round(x1 * sx))))
    dx2 = max(0, min(depth_w, int(round(x2 * sx))))
    dy1 = max(0, min(depth_h, int(round(y1 * sy))))
    dy2 = max(0, min(depth_h, int(round(y2 * sy))))
    return dx1, dy1, dx2, dy2


def scale_point_to_depth(
    x: float,
    y: float,
    frame_w: int,
    frame_h: int,
    depth_shape: tuple[int, int],
) -> tuple[int, int]:
    depth_h, depth_w = depth_shape[:2]
    dx = max(0, min(depth_w - 1, int(round(x * depth_w / max(frame_w, 1)))))
    dy = max(0, min(depth_h - 1, int(round(y * depth_h / max(frame_h, 1)))))
    return dx, dy


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
        # SANPO: gzip-compressed float16, [H, W] header then the image, row-major.
        with gzip.open(depth_path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.float16)
        return decode_sanpo_depth(raw)   # already in metres

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
    center_frac: float = 0.2,
    exclude_boxes: list | None = None,
    frame_shape: tuple[int, int] | None = None,
) -> float | None:
    """
    Sample depth from the **central patch** of a bounding box.

    Uses only the inner `center_frac` fraction of the box — avoids averaging
    depth from edges / background pixels that bleed into the bbox.

    Args:
        depth_map:     2-D float32 depth array (metres).
        x1, y1, x2, y2: Bounding box corners.
        center_frac:   Fraction of bbox width/height to sample (default 0.2).
        exclude_boxes: List of (x1, y1, x2, y2) boxes to mask OUT before
                       sampling.  Used to prevent a background object from
                       reading depth *through* a closer foreground object
                       that overlaps it in image space.

    Returns:
        Median depth in metres, or None if no valid pixels found.
    """
    h, w = depth_map.shape[:2]
    if frame_shape is not None:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = scale_box_to_depth(x1, y1, x2, y2, frame_w, frame_h, depth_map.shape)
        if exclude_boxes:
            exclude_boxes = [
                scale_box_to_depth(ex1, ey1, ex2, ey2, frame_w, frame_h, depth_map.shape)
                for ex1, ey1, ex2, ey2 in exclude_boxes
            ]

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bw = max(1, int((x2 - x1) * center_frac / 2))
    bh = max(1, int((y2 - y1) * center_frac / 2))

    rx1, rx2 = max(0, cx - bw), min(w, cx + bw)
    ry1, ry2 = max(0, cy - bh), min(h, cy + bh)
    if rx1 >= rx2 or ry1 >= ry2:
        return None

    # Build usability mask — True = pixel is valid for sampling
    patch_h = ry2 - ry1
    patch_w = rx2 - rx1
    mask = np.ones((patch_h, patch_w), dtype=bool)

    # Mask out pixels that belong to any foreground (occluding) box
    if exclude_boxes:
        for (ex1, ey1, ex2, ey2) in exclude_boxes:
            mx1 = max(0, ex1 - rx1)
            my1 = max(0, ey1 - ry1)
            mx2 = min(patch_w, ex2 - rx1)
            my2 = min(patch_h, ey2 - ry1)
            if mx1 < mx2 and my1 < my2:
                mask[my1:my2, mx1:mx2] = False

    roi   = depth_map[ry1:ry2, rx1:rx2]
    valid = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M) & mask]

    # ── Sparse depth fallback — expand patch 3× if no readings ────
    if valid.size == 0:
        bw2, bh2 = bw * 3, bh * 3
        rx1, rx2 = max(0, cx - bw2), min(w, cx + bw2)
        ry1, ry2 = max(0, cy - bh2), min(h, cy + bh2)
        patch_h = ry2 - ry1
        patch_w = rx2 - rx1
        mask2 = np.ones((patch_h, patch_w), dtype=bool)
        if exclude_boxes:
            for (ex1, ey1, ex2, ey2) in exclude_boxes:
                mx1 = max(0, ex1 - rx1); my1 = max(0, ey1 - ry1)
                mx2 = min(patch_w, ex2 - rx1); my2 = min(patch_h, ey2 - ry1)
                if mx1 < mx2 and my1 < my2:
                    mask2[my1:my2, mx1:mx2] = False
        roi   = depth_map[ry1:ry2, rx1:rx2]
        valid = roi[(roi > MIN_DEPTH_M) & (roi < MAX_DEPTH_M) & mask2]

    if valid.size == 0:
        return None
    return float(np.median(valid))
