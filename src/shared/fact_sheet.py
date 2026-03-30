"""
fact_sheet.py
Defines the Symbolic Fact Sheet — the structured JSON output of the Proof System.
This is the sole input to SLM-1 (Prioritizer) and the contract between the
deterministic physics layer and the stochastic SLM layer.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json

# ── Import canonical severity weights from physics module ─────────────────────
# (single source of truth — avoids divergent copies)
from src.perception_stack.physics import CLASS_SEVERITY

# Mass weights (used in threat score)
CLASS_MASS_WEIGHT = {
    "car": 1.0,
    "truck": 2.0,
    "bus": 2.0,
    "motorcycle": 0.5,
    "bicycle": 0.2,
    "pedestrian": 0.3,
    "construction": 1.0,
    "unknown": 0.5,
}


@dataclass
class DetectedObject:
    """Represents a single verified object in the scene."""
    object_id: str                    # Unique ID for tracking (e.g., "obj_0042")
    object_class: str                 # YOLO class label
    distance_m: float                 # Metric distance in metres (from depth map)
    velocity_closing_ms: float        # Closing velocity in m/s (+ = approaching)
    bearing_deg: float                # Horizontal angle from user forward vector
    intent_label: Optional[str]       # From HeadsUp dataset ("Looking at Phone", etc.)
    threat_score: float               # S = computed by ProofSystem
    ttc_s: Optional[float]            # Time-to-Collision in seconds (None if static)
    hallucination_filtered: bool = False  # True if object was discarded


@dataclass
class FactSheet:
    """
    The Symbolic Fact Sheet — output of the Physics Engine / Proof System.
    Passed as a JSON string to SLM-1's prompt.
    """
    frame_id: int
    timestamp_ms: float
    is_scene_stable: bool             # From optical flow stability check
    objects: List[DetectedObject] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_prompt_context(self) -> str:
        """Formats the fact sheet as a compact LLM-readable prompt section."""
        lines = [
            f"[SCENE FACT SHEET | frame={self.frame_id} | stable={self.is_scene_stable}]",
            f"Detected {len(self.objects)} verified objects:\n",
        ]
        for obj in sorted(self.objects, key=lambda o: o.threat_score, reverse=True):
            lines.append(
                f"  ID={obj.object_id} | class={obj.object_class} | "
                f"dist={obj.distance_m:.1f}m | vel={obj.velocity_closing_ms:.1f}m/s | "
                f"TTC={obj.ttc_s:.1f}s | threat_score={obj.threat_score:.2f} | "
                f"bearing={obj.bearing_deg:.0f}deg | intent={obj.intent_label}"
            )
        return "\n".join(lines)
