# Changelog — `src/perception_stack/physics.py`

> Physics calculations for perceived objects: bearing, closing velocity, kinetic threat score, and batch tensor operations.

---

## [v1.1] — 2026-03-30 | Batch Tensor Operations Added

**Session:** *Optimizing CPE Perception Pipeline Performance*

### What Changed

Two new vectorised functions were added to enable frame-level batch processing on PyTorch tensors. The original four scalar functions are **unchanged** — this is a pure additive change.

#### `batch_compute_bearing(cx_tensor, frame_width, hfov_deg)`

Vectorised version of `compute_bearing()` for N detections at once. Accepts a list or 1-D tensor of centre-x pixel coordinates, auto-converts to `torch.float32` tensor, returns a tensor of bearing values in degrees.

**Use case:** Computing bearings for an entire frame's detections in one CUDA-parallelisable operation rather than calling the scalar version in a Python loop.

```python
def batch_compute_bearing(cx_tensor, frame_width: int, hfov_deg: float = 70.0):
    # Returns: 1-D tensor of bearing degrees (N,)
```

#### `batch_kinetic_score(distances, velocities, severity_weights)`

Vectorised version of `kinetic_score()`. Formula: `K = severity × v² / clamp(d, min=ε)`. Uses `torch.clamp` for numerically safer division than the scalar `max(d, EPSILON)`.

**Use case:** Computing kinetic scores for all tracked objects in a frame simultaneously, enabling GPU-parallel threat ranking in the Threat Prioritizer.

```python
def batch_kinetic_score(distances, velocities, severity_weights):
    # Returns: 1-D tensor of kinetic scores (N,)
```

### Important Design Note

Both batch functions do a **lazy import** of `torch` inside the function body. This means `physics.py` can still be imported on systems without PyTorch — the import error only surfaces when the batch functions are actually called. This keeps the perception stack lightweight for CPU-only runs.

### Why Not Replace Scalar Functions?

The scalar functions `compute_bearing()` and `kinetic_score()` are called inside tight frame loops where the overhead of tensor creation per-single-object would be wasteful. The batch functions are intended for the Threat Prioritizer and future training-time batch scoring. Both APIs are kept.

---

## [v1.0] — 2026-03-25 | Initial Implementation

**Session:** *Implementing Argus-REI Perception Pipeline*

### What Was Built

- **`CLASS_SEVERITY`** — dict mapping COCO class names to danger weights. Higher = more dangerous at equivalent kinetic energy. Values: `person=1.0`, `bicycle=1.2`, `car=2.0`, `motorcycle=1.8`, `bus=2.5`, `truck=2.5`, `dog=0.8`.

- **`compute_bearing(cx_px, frame_width, hfov_deg=70.0)`** — normalises pixel x-coordinate to `[-1, 1]` then scales by half-HFOV. Returns degrees: negative=left, positive=right, 0=ahead.

- **`compute_velocity(depth_history, fps)`** — takes a list of `(frame_idx, distance_m)` tuples. Computes `Δd / Δt` across the rolling window. Clamps to `≥0` (only reports approaching, not retreating velocity).

- **`kinetic_score(distance_m, velocity_ms, class_name)`** — implements the core threat formula: `K = class_severity × (v²) / max(d, ε)`. `EPSILON = 0.5m` prevents division by zero for very close objects.

- **`bearing_label(deg)`** — maps bearing degrees to human-readable string: `far-left` < -30° < `left` < -10° < `ahead` < 10° < `right` < 30° < `far-right`.

---

## File Reference
- **Source:** [`src/perception_stack/physics.py`](../../src/perception_stack/physics.py)
- **Uses:** `torch` (optional, lazy import for batch functions)
- **Used by:** `pipeline.py`, `fact_sheet_builder.py`, `physics_verification.py`
- **Exports (via `__init__.py`):** `compute_bearing`, `compute_velocity`, `kinetic_score`, `bearing_label`, `batch_compute_bearing`, `batch_kinetic_score`, `CLASS_SEVERITY`
