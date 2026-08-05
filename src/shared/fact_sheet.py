"""
fact_sheet.py
=============
Defines the Symbolic Fact Sheet — the structured runtime contract between
the Physics Engine / Proof System and SLM-1 (Prioritizer).

Design contract
---------------
Every field in DetectedObject MUST be derivable at phone-camera inference
time. Fields that are only available during offline training (e.g. intent
labels from HEADSUP) must NOT appear here — they belong in the SFT training
data enrichment step only.

Data provenance at inference:
  object_class  : YOLO26n on-device detection
  bearing_deg   : compute_bearing(cx_px, frame_width, hfov_deg) — pixel only
  bearing       : bearing_label(bearing_deg) — derived
  distance_m    : Monocular depth / ARCore / sensor fusion — may be None
  velocity_ms   : Rolling Δd/Δt across depth history frames
  kinetic_score : physics.kinetic_score(d, v, class) — computed locally
  ttc_s         : d / v when v > 0, else None
  route         : Threat prioritizer routing decision (reflex|cognitive|ignore)
  scene_stable  : From gyroscope IMU or optical flow check
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json

# Import canonical severity weights — single source of truth
from src.perception_stack.physics import CLASS_SEVERITY  # noqa: F401 (re-exported)


@dataclass
class DetectedObject:
    """
    A single verified object in the scene — all fields phone-derivable at runtime.

    Removed from prior version:
      - intent_label        : HEADSUP training-data only; not available at inference
      - hallucination_filtered : Internal pipeline flag, not a semantic input
      - threat_score        : Renamed to kinetic_score for precision
      - object_id           : Renamed to track_id (aligns with events.py naming)

    Added:
      - bearing             : Human-readable direction string (from bearing_label())
      - route               : Routing lane assigned by Threat Prioritizer
    """

    # ── Identity ────────────────────────────────────────────────────────────
    track_id: str                   # ByteTrack ID (e.g. "t_042") for temporal reference
    object_class: str               # YOLO26n class label (e.g. "car", "pedestrian")

    # ── Spatial ─────────────────────────────────────────────────────────────
    bearing: str                    # Human-readable: "ahead" | "left" | "right" | "far-left" | "far-right"
    bearing_deg: float              # Numeric bearing from compute_bearing() — negative=left, positive=right

    # ── Physics ─────────────────────────────────────────────────────────────
    distance_m: Optional[float]     # Metric depth in metres (None if depth unavailable)
    velocity_ms: float              # Closing velocity m/s (0.0 if static or retreating)
    ttc_s: Optional[float]          # Time-to-collision in seconds (None if static/unknown)
    kinetic_score: float            # K = f(severity, velocity, distance) — formula TBD post-eval

    # ── Routing ─────────────────────────────────────────────────────────────
    route: str                      # "reflex" | "cognitive" | "ignore" (from Threat Prioritizer)


@dataclass
class FactSheet:
    """
    The Symbolic Fact Sheet — output of the Physics Engine / Proof System.

    Passed as a structured JSON object to SLM-1's prompt.
    Objects are sorted by kinetic_score descending (highest threat first).

    Fields:
      frame_id      : Sequential frame index from the perception loop
      timestamp_ms  : Wall-clock timestamp in milliseconds
      scene_stable  : True if optical flow / IMU gyroscope indicates stable scene
      objects       : List of DetectedObject, highest kinetic_score first
    """

    frame_id: int
    timestamp_ms: float
    scene_stable: bool              # Renamed from is_scene_stable for brevity
    objects: List[DetectedObject] = field(default_factory=list)

    def to_json(self) -> str:
        """Compact JSON — used for logging and debugging."""
        return json.dumps(asdict(self), indent=2)

    def to_prompt_context(self) -> str:
        """
        Formats the fact sheet as a concise LLM-readable prompt context.
        Objects are listed highest-threat-first.
        """
        lines = [
            f"[FACT SHEET | frame={self.frame_id} | stable={self.scene_stable}]",
            f"Objects detected: {len(self.objects)}\n",
        ]
        for obj in self.objects:  # already sorted by kinetic_score desc at build time
            dist  = f"{obj.distance_m:.1f}m" if obj.distance_m is not None else "unknown dist"
            ttc   = f"{obj.ttc_s:.1f}s" if obj.ttc_s is not None else "N/A"
            lines.append(
                f"  [{obj.route.upper()}] {obj.track_id} | {obj.object_class} | "
                f"{dist} | v={obj.velocity_ms:.1f}m/s | TTC={ttc} | "
                f"K={obj.kinetic_score:.2f} | {obj.bearing}"
            )
        return "\n".join(lines)

    def to_slm_user_message(self) -> dict:
        """
        Returns the fact sheet as a structured dict suitable for the SLM-1
        user message slot in the SFT training JSONL.

        This is the canonical input format for SLM-1 training records.
        It is a structured JSON object — NOT a flat string — so the model
        learns to parse structured scene context, not just pattern-match text.
        """
        return {
            "frame_id":     self.frame_id,
            "scene_stable": self.scene_stable,
            "objects": [
                {
                    "track_id":      obj.track_id,
                    "object_class":  obj.object_class,
                    "bearing":       obj.bearing,
                    "bearing_deg":   round(obj.bearing_deg, 1),
                    "distance_m":    round(obj.distance_m, 2) if obj.distance_m is not None else None,
                    "velocity_ms":   round(obj.velocity_ms, 2),
                    "ttc_s":         round(obj.ttc_s, 2) if obj.ttc_s is not None else None,
                    "kinetic_score": round(obj.kinetic_score, 3),
                    "route":         obj.route,
                }
                for obj in self.objects
            ],
        }
