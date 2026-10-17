"""
priority.py
===========
The fused priority score (docs/architecture.md §6.4, "Level 2" in
models/Ensemble/README.md):

    priority = P(arrives within the horizon) × severity(class)^γ × context
    ties broken by expected arrival time, sooner first
    priority < IGNORE_THRESHOLD → not announced

- WHEN   P(arrive) and the expected arrival time come from an ArrivalPredictor:
         the trained GRU (GRUArrivalPredictor) or, without a checkpoint, the
         physics rule (PhysicsArrivalPredictor: time until the fused relative
         path enters the hazard zone).
- HOW BAD  physics.class_severity (mass law). γ compresses it: γ = 0 ranks by
         arrival alone, γ = 1 uses raw severity (a bus is 13× a person).
         SEVERITY_GAMMA is a declared choice; models/Ensemble/heldout.py-style
         evaluation (tools/replay_threat_engine.py) reports the trade-off.
- WHAT   a context multiplier from walkable-surface segmentation (§6.3). The
         segmentation model doesn't exist yet, so context is 1.0 unless a
         caller supplies SceneContext values; the multiplier table is here so
         the interface is fixed and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.perception_stack.physics import class_severity
from src.threat_engine.envelope import CONE_HALF_ANGLE_DEG, CORRIDOR_WIDTH_M, half_width
from src.threat_engine.fusion import FusedTrack

SEVERITY_GAMMA = 0.25        # bus 13.1 → 1.9×, car 4.6 → 1.5×, stop sign 0.19 → 0.66× a person
IGNORE_THRESHOLD = 0.15      # declared; tune on train sessions only
HAZARD_M = 3.0               # arrival zone radius, as in the arrival dataset
HORIZON_S = 4.0              # 20 steps at SANPO's true 5 Hz processing rate
PHYSICS_SOFTNESS_S = 0.75    # width of the physics predictor's sigmoid around the horizon

# Context multipliers for §6.3's per-track outputs. Declared, not fitted.
# Near a boundary was ×1.25 until 2026-10-17; SANPO's own masks showed it points
# the wrong way (objects at a sidewalk edge are mostly beside the path) and on
# train sessions neither it nor "near a boundary and approaching" raised the
# arrival rate (methodology §4.14), so it is neutral: reported, not weighted.
CONTEXT_IN_PATH = 1.5
CONTEXT_NEAR_BOUNDARY = 1.0
CONTEXT_OTHER_SURFACE = 0.5  # e.g. a car on the road while the walker is on the sidewalk


@dataclass(frozen=True)
class Arrival:
    p_arrive: float              # P(within HAZARD_M inside the cone within the horizon)
    expected_s: float            # expected arrival time; horizon + 1 slice if not expected


@dataclass(frozen=True)
class SceneContext:
    """Per-track output of walkable-surface segmentation (§6.3). All default to 'no information'."""

    in_path: bool = False
    near_boundary: bool = False
    other_surface: bool = False


@dataclass(frozen=True)
class RankedThreat:
    track: FusedTrack
    priority: float
    arrival: Arrival
    severity: float
    context: float


class ArrivalPredictor(Protocol):
    def predict(self, frame_idx: int, tracks: list[FusedTrack]) -> dict[str, Arrival]: ...


def context_multiplier(ctx: SceneContext | None) -> float:
    if ctx is None:
        return 1.0
    m = 1.0
    if ctx.in_path:
        m *= CONTEXT_IN_PATH
    if ctx.near_boundary:
        m *= CONTEXT_NEAR_BOUNDARY
    if ctx.other_surface:
        m *= CONTEXT_OTHER_SURFACE
    return m


def rank(tracks: list[FusedTrack], arrivals: dict[str, Arrival], *,
         gamma: float = SEVERITY_GAMMA, contexts: dict[str, SceneContext] | None = None,
         ignore_threshold: float = IGNORE_THRESHOLD) -> list[RankedThreat]:
    """Tracks above the ignore threshold, highest priority first."""
    out = []
    for t in tracks:
        a = arrivals.get(t.track_id)
        if a is None:
            continue
        sev = class_severity(t.class_name)
        ctx = context_multiplier((contexts or {}).get(t.track_id))
        pr = a.p_arrive * sev ** gamma * ctx
        if pr >= ignore_threshold:
            out.append(RankedThreat(t, pr, a, sev, ctx))
    return sorted(out, key=lambda r: (-r.priority, r.arrival.expected_s, r.track.distance_m))


# ── Arrival predictors ───────────────────────────────────────────────────────

class PhysicsArrivalPredictor:
    """
    No learned model: P(arrive) is a sigmoid of how far inside the horizon the
    fused relative path enters the hazard zone — the walking corridor (default;
    the label since 2026-10-15: 0 < forward <= hazard_m, |lateral| <= corridor
    half-width + object half-width) or the v1 cone. The fallback whenever no
    GRU checkpoint is loaded.
    """

    def __init__(self, hazard_m: float = HAZARD_M, horizon_s: float = HORIZON_S,
                 softness_s: float = PHYSICS_SOFTNESS_S, zone: str = "corridor",
                 width_m: float = CORRIDOR_WIDTH_M, step_s: float = 0.2):
        self.hazard_m, self.horizon_s, self.softness_s = hazard_m, horizon_s, softness_s
        self.zone, self.width_m, self.step_s = zone, width_m, step_s

    def _corridor_entry(self, t: FusedTrack) -> float | None:
        steps = np.arange(0.0, self.horizon_s * 1.25 + 1e-9, self.step_s)
        lat = t.lateral_m + t.rel_velocity[0] * steps
        fwd = t.forward_m + t.rel_velocity[1] * steps
        inside = (fwd > 0) & (fwd <= self.hazard_m) & (np.abs(lat) <= self.width_m / 2 + half_width(t.class_name))
        hit = np.nonzero(inside)[0]
        return float(steps[hit[0]]) if len(hit) else None

    def predict(self, frame_idx: int, tracks: list[FusedTrack]) -> dict[str, Arrival]:
        out = {}
        for t in tracks:
            if self.zone == "corridor":
                self._finish(out, t, self._corridor_entry(t))
                continue
            tz = t.time_to_zone(self.hazard_m)
            if tz is not None:
                # The label counts arrival inside the forward cone only, so the entry point must be in it.
                lat = t.lateral_m + t.rel_velocity[0] * tz
                fwd = t.forward_m + t.rel_velocity[1] * tz
                if fwd <= 0 or abs(math.degrees(math.atan2(lat, fwd))) > CONE_HALF_ANGLE_DEG:
                    tz = None
            self._finish(out, t, tz)
        return out

    def _finish(self, out: dict[str, Arrival], t: FusedTrack, tz: float | None) -> None:
        if tz is None:
            out[t.track_id] = Arrival(0.0, self.horizon_s * 1.25)
            return
        z = min(50.0, (tz - self.horizon_s) / self.softness_s)   # far-future entries: p -> 0 without overflow
        out[t.track_id] = Arrival(1.0 / (1.0 + math.exp(z)), min(tz, self.horizon_s * 1.25))


class GRUArrivalPredictor:
    """
    Runtime wrapper for a models/GRU checkpoint trained with --time-bins.

    It keeps each track's recent Stage-1 rows and encodes them with
    dataset.GRU.samples.history_features — the exact function the training
    data went through. The checkpoint's inputs are Stage-1 row fields
    (camera-frame distance, closing speed and bearing), so it is fed the raw
    rows carried on each FusedTrack, not the fused state; retraining on fused
    features is pending_work.md P1.5.

    Units: checkpoints trained on CSVs built before 2026-10-14 saw speeds 2×
    too high (30 FPS assumed for 15 FPS video). Feed it rows from those same
    CSVs, or set velocity_scale=2.0 when feeding correctly-scaled rows.
    """

    def __init__(self, checkpoint: Path | str, *, step_s: float = 0.2, velocity_scale: float = 1.0,
                 device: str = "cpu"):
        import torch

        from dataset.GRU.samples import CLASS_VOCAB, UNKNOWN_CLASS
        from models.GRU.registry import build_model

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        if not cfg.get("time_bins"):
            raise ValueError(f"{checkpoint} is a yes/no model; the threat engine needs --time-bins")
        self.model = build_model(ckpt["arch"], ckpt["model_kwargs"]).to(device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.torch, self.device = torch, device
        self.time_bins, self.horizon = cfg["time_bins"], cfg["horizon"]
        self.history = cfg["dataset_params"]["history"]
        self.max_objects = cfg["dataset_params"].get("max_objects", 32)
        self.step_s, self.velocity_scale = step_s, velocity_scale
        self._class_index = {c: i for i, c in enumerate(CLASS_VOCAB)}
        self._unknown = UNKNOWN_CLASS
        self._obs: dict[str, dict[int, tuple]] = {}
        self._first_seen: dict[str, int] = {}

    def observe(self, frame_idx: int, tracks: list[FusedTrack]) -> None:
        """Record this frame's measured rows (call once per frame, before predict)."""
        for t in tracks:
            if not t.measured or t.row is None or t.row.get("distance_m") in (None, ""):
                continue
            r = t.row
            # Same as evaluation.kinetic_ablation.prepare(): area from the box when there is one.
            if r.get("bbox_x1") not in (None, ""):
                area = (float(r["bbox_x2"]) - float(r["bbox_x1"])) * (float(r["bbox_y2"]) - float(r["bbox_y1"]))
            else:
                area = r.get("bbox_area_px", 0.0)
            self._obs.setdefault(t.track_id, {})[frame_idx] = (
                float(r.get("confidence", 0.0)), float(r.get("bearing_deg", 0.0)), float(r["distance_m"]),
                float(area), float(r.get("velocity_ms", 0.0)) * self.velocity_scale)
            self._first_seen.setdefault(t.track_id, frame_idx)
        oldest = frame_idx - (self.history - 1)   # processed frames are numbered consecutively
        for tid in list(self._obs):
            self._obs[tid] = {f: v for f, v in self._obs[tid].items() if f >= oldest}
            if not self._obs[tid]:
                del self._obs[tid]
                self._first_seen.pop(tid, None)

    def encode(self, frame_idx: int, tracks: list[FusedTrack]) -> tuple[list[FusedTrack], np.ndarray, np.ndarray]:
        """(objects scored this frame, [n, history, features], [n] class ids), nearest first, as in the dataset."""
        from dataset.GRU.samples import history_features

        live = [t for t in tracks if t.track_id in self._obs and frame_idx in self._obs[t.track_id]]
        live = sorted(live, key=lambda t: float(t.row["distance_m"]))[: self.max_objects]
        if not live:
            return [], np.zeros((0, self.history, 0), dtype=np.float32), np.zeros(0, dtype=np.int64)
        x = np.stack([history_features(self._obs[t.track_id], self._first_seen[t.track_id], frame_idx,
                                       self.history, 1) for t in live])
        cls = np.array([self._class_index.get(t.class_name, self._unknown) for t in live], dtype=np.int64)
        return live, x, cls

    def predict(self, frame_idx: int, tracks: list[FusedTrack]) -> dict[str, Arrival]:
        """observe() then score; safe to call on a frame that was already observed."""
        self.observe(frame_idx, tracks)
        live, x, cls = self.encode(frame_idx, tracks)
        if not live:
            return {}
        torch = self.torch
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x)[None].to(self.device),
                                torch.from_numpy(cls)[None].to(self.device),
                                torch.ones(1, len(live), dtype=torch.bool, device=self.device))[0].cpu().numpy()
        width = -(-self.horizon // self.time_bins)
        lo = np.arange(self.time_bins) * width
        mids = (lo + np.append(lo[1:] - 1, self.horizon)) / 2
        never = self.horizon + width
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p = e / e.sum(axis=-1, keepdims=True)
        expected_steps = p[:, :-1] @ mids + p[:, -1] * never
        return {t.track_id: Arrival(float(1.0 - p[i, -1]), float(expected_steps[i] * self.step_s))
                for i, t in enumerate(live)}
