"""
Threat event contract and routing for the CPE runtime pipeline.

This module bridges Stage 1 perception rows into the deterministic runtime lanes:
ignore, cognitive, and reflex. It intentionally consumes the existing canonical
perception CSV row shape so it can be used both in offline SANPO replay and in a
live frame loop.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

from src.perception_stack.physics import bearing_label, kinetic_score
from src.physics_verification.physics_verification import ReflexResult


Route = Literal["ignore", "cognitive", "reflex"]

DEFAULT_LOW_K_THRESHOLD = 0.5
DEFAULT_HIGH_K_THRESHOLD = 5.0
DEFAULT_REFLEX_TTC_S = 1.0
DEFAULT_NEAR_STATIC_DISTANCE_M = 1.5
STATIC_HAZARD_CLASSES = {"stairs", "pothole", "puddle", "bollard", "pole", "bench", "fire hydrant"}


@dataclass(frozen=True)
class PerceivedObject:
    """One YOLO/ByteTrack/depth object after physics enrichment."""

    frame_idx: int
    source: str
    track_id: str
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    cx_px: float
    bearing_deg: float
    bearing: str
    distance_m: float | None
    velocity_ms: float
    kinetic_score: float
    ttc_s: float | None

    @classmethod
    def from_perception_row(cls, row: dict) -> "PerceivedObject":
        """Build a perceived object from the canonical Stage 1 CSV row dict."""
        class_name = str(row.get("class", row.get("class_name", "unknown")))
        distance_m = _optional_float(row.get("distance_m"))
        velocity_ms = _float(row.get("velocity_ms"), default=0.0)
        k_score = kinetic_score(distance_m, velocity_ms, class_name) if distance_m is not None else 0.0
        ttc_s = compute_ttc(distance_m, velocity_ms)
        bearing_deg = _float(row.get("bearing_deg"), default=0.0)
        return cls(
            frame_idx=int(row.get("frame_idx", 0)),
            source=str(row.get("source", "unknown")),
            track_id=str(row.get("track_id", "")),
            class_name=class_name,
            confidence=_float(row.get("confidence"), default=0.0),
            bbox=(
                _float(row.get("bbox_x1"), default=0.0),
                _float(row.get("bbox_y1"), default=0.0),
                _float(row.get("bbox_x2"), default=0.0),
                _float(row.get("bbox_y2"), default=0.0),
            ),
            cx_px=_float(row.get("cx_px"), default=0.0),
            bearing_deg=bearing_deg,
            bearing=bearing_label(bearing_deg),
            distance_m=distance_m,
            velocity_ms=velocity_ms,
            kinetic_score=k_score,
            ttc_s=ttc_s,
        )

    def to_registry_record(self) -> dict:
        """Return the object shape expected by PhysicsVerification."""
        return {
            "class": self.class_name,
            "distance_m": self.distance_m or 0.0,
            "velocity_ms": self.velocity_ms,
            "bearing_deg": self.bearing_deg,
        }

    def to_reflex_result(self, override_ttc_s: float = DEFAULT_REFLEX_TTC_S) -> ReflexResult:
        """Return the deterministic physics result expected by PhysicsVerification."""
        return ReflexResult(
            track_id=self.track_id,
            kinetic_score=self.kinetic_score,
            ttc_s=self.ttc_s,
            override=self.ttc_s is not None and self.ttc_s <= override_ttc_s,
        )


@dataclass(frozen=True)
class ThreatEvent:
    """Routable event produced by threat prioritization."""

    frame_idx: int
    route: Route
    priority: int
    track_id: str
    class_name: str
    distance_m: float | None
    velocity_ms: float
    ttc_s: float | None
    kinetic_score: float
    bearing_deg: float
    bearing: str
    reason: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


@dataclass(frozen=True)
class ThreatFrame:
    """All routing outputs for one processed frame."""

    frame_idx: int
    objects: tuple[PerceivedObject, ...]
    events: tuple[ThreatEvent, ...]

    @property
    def object_registry(self) -> dict[str, dict]:
        return {obj.track_id: obj.to_registry_record() for obj in self.objects}

    @property
    def reflex_results(self) -> list[ReflexResult]:
        return [obj.to_reflex_result() for obj in self.objects]

    def cognitive_events(self) -> tuple[ThreatEvent, ...]:
        return tuple(event for event in self.events if event.route == "cognitive")

    def reflex_events(self) -> tuple[ThreatEvent, ...]:
        return tuple(event for event in self.events if event.route == "reflex")


def compute_ttc(distance_m: float | None, velocity_ms: float) -> float | None:
    """Return time-to-collision in seconds, or None for static/unknown objects."""
    if distance_m is None or velocity_ms <= 0:
        return None
    return distance_m / velocity_ms


def build_threat_frame(
    rows: Iterable[dict],
    *,
    low_k_threshold: float = DEFAULT_LOW_K_THRESHOLD,
    high_k_threshold: float = DEFAULT_HIGH_K_THRESHOLD,
    reflex_ttc_s: float = DEFAULT_REFLEX_TTC_S,
    near_static_distance_m: float = DEFAULT_NEAR_STATIC_DISTANCE_M,
) -> ThreatFrame:
    """Build route decisions for one processed frame of perception rows."""
    objects = tuple(PerceivedObject.from_perception_row(row) for row in rows)
    frame_idx = objects[0].frame_idx if objects else -1
    events = tuple(
        sorted(
            (
                _route_object(
                    obj,
                    low_k_threshold=low_k_threshold,
                    high_k_threshold=high_k_threshold,
                    reflex_ttc_s=reflex_ttc_s,
                    near_static_distance_m=near_static_distance_m,
                )
                for obj in objects
            ),
            key=lambda event: (event.priority, event.kinetic_score),
            reverse=True,
        )
    )
    return ThreatFrame(frame_idx=frame_idx, objects=objects, events=events)


def load_perception_csv_frames(csv_path: Path) -> dict[int, list[dict]]:
    """Load canonical perception CSV rows grouped by frame index."""
    frames: dict[int, list[dict]] = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            frame_idx = int(row["frame_idx"])
            frames.setdefault(frame_idx, []).append(row)
    return frames


def build_threat_frames_from_csv(csv_path: Path, **kwargs) -> list[ThreatFrame]:
    """Build threat frames from a Stage 1 perception CSV."""
    frames = load_perception_csv_frames(csv_path)
    return [build_threat_frame(frames[idx], **kwargs) for idx in sorted(frames)]


def write_threat_events_jsonl(frames: Iterable[ThreatFrame], out_path: Path) -> int:
    """Write all non-ignore threat events to JSONL and return the count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as file:
        for frame in frames:
            for event in frame.events:
                if event.route == "ignore":
                    continue
                file.write(event.to_json() + "\n")
                count += 1
    return count


def _route_object(
    obj: PerceivedObject,
    *,
    low_k_threshold: float,
    high_k_threshold: float,
    reflex_ttc_s: float,
    near_static_distance_m: float,
) -> ThreatEvent:
    if obj.ttc_s is not None and obj.ttc_s <= reflex_ttc_s:
        return _event(obj, "reflex", 100, f"TTC {obj.ttc_s:.2f}s is inside reflex threshold")

    if obj.kinetic_score >= high_k_threshold:
        return _event(obj, "reflex", 90, f"K-score {obj.kinetic_score:.2f} exceeds high-risk threshold")

    if obj.kinetic_score >= low_k_threshold:
        return _event(obj, "cognitive", 50, f"K-score {obj.kinetic_score:.2f} needs semantic check")

    if (
        obj.distance_m is not None
        and obj.distance_m <= near_static_distance_m
        and obj.class_name in STATIC_HAZARD_CLASSES
    ):
        return _event(obj, "cognitive", 40, f"near static {obj.class_name} at {obj.distance_m:.2f}m")

    return _event(obj, "ignore", 0, "below kinetic and proximity routing thresholds")


def _event(obj: PerceivedObject, route: Route, priority: int, reason: str) -> ThreatEvent:
    return ThreatEvent(
        frame_idx=obj.frame_idx,
        route=route,
        priority=priority,
        track_id=obj.track_id,
        class_name=obj.class_name,
        distance_m=obj.distance_m,
        velocity_ms=obj.velocity_ms,
        ttc_s=obj.ttc_s,
        kinetic_score=obj.kinetic_score,
        bearing_deg=obj.bearing_deg,
        bearing=obj.bearing,
        reason=reason,
    )


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _float(value, *, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)
