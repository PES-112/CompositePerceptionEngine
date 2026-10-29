"""
engine.py
=========
The v2 threat engine, one call per processed frame (docs/architecture.md §5,
boxes 2, 4 and 5):

    Stage-1 rows (+ camera pose) ─> GroundTracker (Kalman, ego-motion removed)
        ─> safety envelope (override)        ─┐
        ─> ArrivalPredictor ─> fused priority ─┴─> AnnouncementManager ─> NarratorEvent
    segmentation edge alert + walker surface (optional) ┘

It replaces the v1 chain "K0 routing -> cognitive stub -> PhysicsVerification"
(still available as src.pipeline.orchestrator.process_frame, the baseline the
replay compares against). Untracked depth-sweep pseudo-objects (obs_*) are
skipped: they are re-IDed every frame, so they can't be filtered or ranked —
ground masking by segmentation (§6.3) is their real fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.physics_verification.physics_verification import NarratorEvent
from src.threat_engine import envelope
from src.threat_engine.announcer import Announcement, AnnouncementManager, Edge
from src.threat_engine.fusion import EgoPose, FusedTrack, GroundTracker, Measurement
from src.threat_engine.priority import (IGNORE_THRESHOLD, SEVERITY_GAMMA, ArrivalPredictor,
                                        PhysicsArrivalPredictor, RankedThreat, SceneContext, rank)

UNTRACKED_PREFIX = "obs_"


@dataclass(frozen=True)
class EngineFrame:
    frame_idx: int
    t: float
    tracks: tuple[FusedTrack, ...]
    overrides: tuple[envelope.Override, ...]
    ranked: tuple[RankedThreat, ...]
    announcement: Announcement | None

    @property
    def top(self) -> FusedTrack | None:
        """Most urgent object this frame: the soonest override, else the top-ranked threat."""
        if self.overrides:
            return self.overrides[0].track
        return self.ranked[0].track if self.ranked else None

    def narrator_event(self) -> NarratorEvent | None:
        """The announcement as the narration pipeline's input type, or None if silent."""
        a = self.announcement
        if a is None:
            return None
        if a.kind == "edge":                                  # straight ahead; reason routes it to the edge phrases
            return NarratorEvent(track_id=a.track_id, object_class=a.class_name,
                                 distance_m=a.distance_m, closing_velocity_ms=0.0,
                                 bearing_deg=0.0, reason="edge ahead", is_override=False)
        if a.kind == "surface":                               # underfoot; reason is "now on" or "still on"
            return NarratorEvent(track_id=a.track_id, object_class=a.class_name, distance_m=0.0,
                                 closing_velocity_ms=0.0, bearing_deg=0.0, reason=a.reason, is_override=False)
        track = next(t for t in self.tracks if t.track_id == a.track_id)
        return NarratorEvent(track_id=a.track_id, object_class=a.class_name, distance_m=track.distance_m,
                             closing_velocity_ms=max(0.0, track.closing_speed_ms),
                             bearing_deg=track.bearing_deg, reason=a.reason, is_override=a.kind == "override")


@dataclass
class ThreatEngine:
    predictor: ArrivalPredictor = field(default_factory=PhysicsArrivalPredictor)
    gamma: float = SEVERITY_GAMMA
    ignore_threshold: float = IGNORE_THRESHOLD
    tracker: GroundTracker = field(default_factory=GroundTracker)
    announcer: AnnouncementManager = field(default_factory=AnnouncementManager)

    def step(self, frame_idx: int, t: float, rows: list[dict], pose: EgoPose | None = None,
             contexts: dict[str, SceneContext] | None = None, edge: Edge | None = None,
             surface: str | None = None) -> EngineFrame:
        measurements = [_measurement(r) for r in rows
                        if not str(r.get("track_id", "")).startswith(UNTRACKED_PREFIX)]
        tracks = self.tracker.step(t, measurements, pose)
        overrides = envelope.overrides(tracks)
        arrivals = self.predictor.predict(frame_idx, tracks)
        ranked = rank([tr for tr in tracks if tr.confirmed or tr.measured], arrivals, gamma=self.gamma,
                      contexts=contexts, ignore_threshold=self.ignore_threshold)
        announcement = self.announcer.step(t, overrides, ranked, edge, surface)
        return EngineFrame(frame_idx, t, tuple(tracks), tuple(overrides), tuple(ranked), announcement)


def _measurement(row: dict) -> Measurement:
    depth = row.get("distance_m")
    return Measurement(
        track_id=str(row.get("track_id", "")),
        class_name=str(row.get("class", row.get("class_name", "unknown"))),
        depth_m=None if depth in (None, "") else float(depth),
        bearing_deg=float(row.get("bearing_deg") or 0.0),
        confidence=float(row.get("confidence") or 0.0),
        row=row,
    )
