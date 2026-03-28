"""
fact_sheet_builder.py
=====================
CPE Perception Stack — Stage 2: CSV → Kinetic Score + JSONL Fact Sheets.

Reads the flat perception CSV from Stage 1, computes kinetic threat scores
(K₀ present, K₊₂ future look-ahead), and writes train.jsonl for SLM-1 SFT.

Public API:
    load_perception_csv(csv_path)     → dict[frame_idx, list[row]]
    build_fact_sheets(frames, ...)    → (written: int, skipped: int)
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

from src.perception_stack.physics import (
    kinetic_score, bearing_label,
    CLASS_SEVERITY, DEFAULT_SEVERITY,
)

# ── SLM-1 System Prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a pedestrian navigation AI for visually impaired users. "
    "Given a Fact Sheet describing nearby objects with their distance, "
    "velocity, and bearing, identify the single highest-priority threat "
    "and explain your reasoning in one sentence. "
    "Respond ONLY with valid JSON."
)


# ── CSV Loading ───────────────────────────────────────────────────────────────

def load_perception_csv(csv_path: Path) -> dict:
    """
    Load a Stage 1 perception CSV into a frame-indexed dict.

    Returns:
        { frame_idx (int): [ {row_dict}, ... ] }
    Numeric fields are cast to float/int; empty strings become None.
    """
    frames: dict = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["frame_idx"])
            row["distance_m"]  = float(row["distance_m"])  if row["distance_m"]  else None
            row["velocity_ms"] = float(row["velocity_ms"]) if row["velocity_ms"] else 0.0
            row["bearing_deg"] = float(row["bearing_deg"]) if row["bearing_deg"] else 0.0
            frames[idx].append(row)
    return frames


# ── Fact Sheet Rendering ──────────────────────────────────────────────────────

def _render_fact_sheet(objects: list) -> str:
    """Convert a list of scored object dicts into a human-readable Fact Sheet string."""
    if not objects:
        return "No objects detected."
    parts = []
    for i, obj in enumerate(objects, 1):
        dist = f"{obj['distance_m']:.1f}m" if obj["distance_m"] is not None else "unknown dist"
        vel  = f"v={obj['velocity_ms']:.1f}m/s"
        brng = bearing_label(obj["bearing_deg"])
        k    = obj.get("kinetic_score", 0.0)
        parts.append(f"Object_{i:02d}: {obj['class']}, {dist}, {vel}, {brng} [K={k:.3f}]")
    return " | ".join(parts)


def _find_top_threat(objects: list) -> dict | None:
    """Return the object dict with the highest kinetic score, or None."""
    scored = [o for o in objects if o.get("kinetic_score", 0.0) > 0]
    return max(scored, key=lambda o: o["kinetic_score"]) if scored else None


def _reasoning(obj: dict, future_match: bool) -> str:
    dist   = f"{obj['distance_m']:.1f}m" if obj["distance_m"] is not None else "at unknown distance"
    brng   = bearing_label(obj["bearing_deg"])
    future = " Confirmed as highest threat 2 seconds later." if future_match else ""
    return (
        f"{obj['class'].capitalize()} {dist} {brng} approaching at "
        f"{obj['velocity_ms']:.1f} m/s is the highest kinetic threat "
        f"(K={obj['kinetic_score']:.3f}).{future}"
    )


# ── Main Builder ──────────────────────────────────────────────────────────────

def build_fact_sheets(
    frames: dict,
    fps: float,
    lookahead_s: float,
    out_path: Path,
    append: bool = False,
) -> tuple[int, int]:
    """
    Iterate over all frames, score objects with K₀ and K₊₂, write train.jsonl.

    Args:
        frames:       Output of load_perception_csv().
        fps:          Video framerate (used to compute look-ahead window).
        lookahead_s:  Seconds ahead for future threat confirmation (default 2.0).
        out_path:     Destination JSONL file.

    Returns:
        (written, skipped) record counts.
    """
    lookahead_frames = int(fps * lookahead_s)
    frame_indices    = sorted(frames.keys())
    written = skipped = 0

    mode = "a" if append else "w"
    with open(out_path, mode) as f:
        for frame_idx in frame_indices:
            objects = frames[frame_idx]

            # ── Present Kinetic Score (K₀) ──
            for obj in objects:
                if obj["distance_m"] is not None:
                    obj["kinetic_score"] = kinetic_score(
                        obj["distance_m"], obj["velocity_ms"], obj["class"]
                    )
                else:
                    obj["kinetic_score"] = 0.0
            objects.sort(key=lambda o: o["kinetic_score"], reverse=True)

            present_threat = _find_top_threat(objects)
            if present_threat is None:
                skipped += 1
                continue

            # ── Future Kinetic Score (K₊₂) — ground-truth alignment ──
            future_frame = frames.get(frame_idx + lookahead_frames, [])
            future_threat_id = None
            if future_frame:
                for fo in future_frame:
                    fo["kinetic_score"] = (
                        kinetic_score(fo["distance_m"], fo["velocity_ms"], fo["class"])
                        if fo["distance_m"] is not None else 0.0
                    )
                future_best = _find_top_threat(future_frame)
                if future_best:
                    future_threat_id = future_best["track_id"]

            future_match = (
                future_threat_id is not None and
                str(present_threat["track_id"]) == str(future_threat_id)
            )

            # ── Assemble JSONL record ──
            obj_index    = objects.index(present_threat) + 1
            fact_sheet   = _render_fact_sheet(objects)
            assistant    = json.dumps({
                "primary_threat":   f"Object_{obj_index:02d}",
                "track_id":         int(present_threat["track_id"]),
                "class":            present_threat["class"],
                "distance_m":       present_threat["distance_m"],
                "velocity_ms":      round(present_threat["velocity_ms"], 3),
                "kinetic_score":    round(present_threat["kinetic_score"], 4),
                "future_confirmed": future_match,
                "reason":           _reasoning(present_threat, future_match),
            })

            f.write(json.dumps({
                "system":    SYSTEM_PROMPT,
                "user":      f"[SCENARIO FACT SHEET] {fact_sheet}",
                "assistant": assistant,
            }) + "\n")
            written += 1

    return written, skipped
