"""
segmentation_metrics.py
=======================
Pixel-level scores for the walkable-surface model (§6.3), accumulated over
frames. Score only SANPO's HUMAN_ANNOTATED frames of the test sessions; the
propagated frames are a training aid, as in the SANPO paper.

  miou, per-class IoU      on the CPE scheme (src/segmentation/labels.py), ignore = 255
  false_safe_rate          unsafe ground (road, curb, stairs, non-walkable) predicted walkable,
                           over all unsafe-ground pixels. The safety-relevant error
                           (SENSATION-DS's road-as-sidewalk rate, generalised)
  road_as_walkable_rate    the road-only part of it, comparable to SENSATION-DS's R->S
  walkable_as_unsafe_rate  the opposite error: walkable ground called unsafe (nuisance, not danger)
  boundary_f[c]            boundary F-score for curb and stairs at a pixel tolerance: thin
                           edges that mIoU barely weighs, and the ones the edge alert needs
"""

from __future__ import annotations

import numpy as np

from src.segmentation.labels import (CPE_CLASSES, EDGE_CLASSES, IGNORE, ROAD, UNSAFE_GROUND,
                                     WALKABLE_CLASSES)

N_CLASSES = len(CPE_CLASSES)
BOUNDARY_TOL_PX = 3


def _boundary(region: np.ndarray) -> np.ndarray:
    import cv2

    r = region.astype(np.uint8)
    return (r - cv2.erode(r, np.ones((3, 3), np.uint8))).astype(bool)


def _near(boundary: np.ndarray, tol: int) -> np.ndarray:
    import cv2

    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    return cv2.dilate(boundary.astype(np.uint8), k).astype(bool)


class SegmentationScore:
    def __init__(self, tol_px: int = BOUNDARY_TOL_PX):
        self.confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)   # [truth, prediction]
        self.tol = tol_px
        self.boundary = {c: np.zeros(4, dtype=np.int64) for c in EDGE_CLASSES}   # pred_hit, pred_n, gt_hit, gt_n
        self.frames = 0

    def add(self, truth: np.ndarray, pred: np.ndarray) -> None:
        if truth.shape != pred.shape:
            raise ValueError(f"shape mismatch {truth.shape} vs {pred.shape}")
        valid = (truth != IGNORE) & (pred < N_CLASSES)
        t, p = truth[valid].astype(np.int64), pred[valid].astype(np.int64)
        self.confusion += np.bincount(t * N_CLASSES + p, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
        for c in EDGE_CLASSES:
            gb = _boundary((truth == c) & valid)
            pb = _boundary((pred == c) & valid)
            self.boundary[c] += [(pb & _near(gb, self.tol)).sum(), pb.sum(), (gb & _near(pb, self.tol)).sum(), gb.sum()]
        self.frames += 1

    def result(self) -> dict:
        cm = self.confusion
        tp = np.diag(cm).astype(float)
        union = cm.sum(0) + cm.sum(1) - tp
        iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
        safe, unsafe = sorted(WALKABLE_CLASSES), sorted(UNSAFE_GROUND)
        unsafe_n = cm[unsafe].sum()
        safe_n = cm[safe].sum()
        bf = {}
        for c, (ph, pn, gh, gn) in self.boundary.items():
            prec = ph / pn if pn else np.nan
            rec = gh / gn if gn else np.nan
            bf[CPE_CLASSES[c]] = float(2 * prec * rec / (prec + rec)) if pn and gn and prec + rec > 0 else float("nan")
        return {
            "frames": self.frames,
            "pixels": int(cm.sum()),
            "miou": float(np.nanmean(iou)),
            "pixel_accuracy": float(tp.sum() / max(1, cm.sum())),
            "iou": {CPE_CLASSES[i]: float(v) for i, v in enumerate(iou)},
            "false_safe_rate": float(cm[np.ix_(unsafe, safe)].sum() / unsafe_n) if unsafe_n else float("nan"),
            "road_as_walkable_rate": (float(cm[ROAD, safe].sum() / cm[ROAD].sum()) if cm[ROAD].sum()
                                      else float("nan")),
            "walkable_as_unsafe_rate": float(cm[np.ix_(safe, unsafe)].sum() / safe_n) if safe_n else float("nan"),
            "boundary_f": bf,
            "boundary_tol_px": self.tol,
        }
