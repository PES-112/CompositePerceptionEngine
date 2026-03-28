"""
visualize_stage1.py
===================
CPE Phase 1 — Renders an annotated MP4 showing Stage 1 perception output.

Overlays on each frame:
  • Coloured bounding boxes per class
  • Track ID label
  • Distance (m) and velocity (m/s)
  • Bearing arrow (left / ahead / right)
  • Kinetic score bar (filled red = danger)
  • Frame HUD (frame index, object count, top threat)

Usage:
    python tools/visualize_stage1.py ^
        --rgb_dir  data/sanpo/raw/<session>/camera_head/left/video_frames ^
        --csv      data/processed/sanpo_test.csv ^
        --out      data/processed/stage1_preview.mp4 ^
        --fps      33 ^
        --scale    0.5

Output: data/processed/stage1_preview.mp4
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── Colour palette per class (BGR) ───────────────────────────────────────────
CLASS_COLORS = {
    "car":                (0,   200, 255),   # amber
    "person":             (0,   255, 120),   # green
    "bicycle":            (255, 140,   0),   # blue-orange
    "motorcycle":         (255,  60, 200),   # violet
    "bus":                (0,    80, 255),   # orange-red
    "truck":              (50,   50, 255),   # red
    "fire hydrant":       (255, 200,   0),   # cyan-ish
    "unlabeled_obstacle": (255,   0, 255),   # magenta (high visibility)
}
DEFAULT_COLOR   = (180, 180, 180)   # grey for unknown classes
THREAT_COLOR    = (0,   0,   255)   # red for K score bar

# Kinetic score cap for normalising the bar (anything above = full red)
K_MAX_DISPLAY   = 10.0
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.42
FONT_THICK      = 1


# ── CSV Loading ───────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> dict:
    """Load perception CSV into {frame_idx: [row, ...]} dict."""
    frames = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["frame_idx"])
            row["distance_m"]  = float(row["distance_m"])  if row["distance_m"]  else None
            row["velocity_ms"] = float(row["velocity_ms"]) if row["velocity_ms"] else 0.0
            row["bearing_deg"] = float(row["bearing_deg"]) if row["bearing_deg"] else 0.0
            row["kinetic"]     = float(row.get("kinetic", 0.0)) if row.get("kinetic") else 0.0
            frames[idx].append(row)
    return frames


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_box(img, row, scale: float):
    """Draw a bounding box + label + depth + velocity + K-bar on img."""
    x1 = int(int(row["bbox_x1"]) * scale)
    y1 = int(int(row["bbox_y1"]) * scale)
    x2 = int(int(row["bbox_x2"]) * scale)
    y2 = int(int(row["bbox_y2"]) * scale)
    cls  = row["class"]
    tid  = row["track_id"]
    dist = row["distance_m"]
    vel  = row["velocity_ms"]
    brng = row["bearing_deg"]
    color = CLASS_COLORS.get(cls, DEFAULT_COLOR)

    # ── Box ──
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # ── Label background ──
    dist_str = f"{dist:.1f}m" if dist is not None else "?m"
    vel_str  = f"{vel:.1f}m/s" if vel > 0 else "static"
    label    = f"#{tid} {cls}  {dist_str}  {vel_str}"
    (lw, lh), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)
    lx, ly = x1, max(y1 - 4, lh + 4)
    cv2.rectangle(img, (lx, ly - lh - 4), (lx + lw + 6, ly + 2), color, -1)
    cv2.putText(img, label, (lx + 3, ly - 1), FONT, FONT_SCALE, (0, 0, 0), FONT_THICK, cv2.LINE_AA)

    # ── Kinetic score bar (bottom edge of box, filled left→right) ──
    K = row["kinetic"]
    bar_w = x2 - x1
    filled = int(bar_w * min(K, K_MAX_DISPLAY) / K_MAX_DISPLAY)
    cv2.rectangle(img, (x1, y2), (x2, y2 + 4), (50, 50, 50), -1)
    if filled > 0:
        cv2.rectangle(img, (x1, y2), (x1 + filled, y2 + 4), THREAT_COLOR, -1)

    # ── Bearing arrow from box centre ──
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    arrow_len = 25
    angle_rad = np.deg2rad(brng)
    ex = int(cx + arrow_len * np.sin(angle_rad))
    ey = int(cy - arrow_len * np.cos(angle_rad) * 0.3)   # flatten vertically
    cv2.arrowedLine(img, (cx, cy), (ex, ey), color, 2, tipLength=0.4)


def draw_hud(img, frame_idx: int, rows: list, h: int, w: int):
    """Draw a semi-transparent HUD bar at the top of the frame."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    # Find top threat (highest kinetic)
    top = max(rows, key=lambda r: r["kinetic"]) if rows else None

    cv2.putText(img, f"Frame {frame_idx:04d}", (8, 18),
                FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"Objects: {len(rows)}", (8, 38),
                FONT, 0.5, (160, 200, 160), 1, cv2.LINE_AA)

    if top:
        cls   = top["class"]
        dist  = f"{top['distance_m']:.1f}m" if top["distance_m"] else "?m"
        vel   = f"{top['velocity_ms']:.1f}m/s"
        k_val = top["kinetic"]
        threat_label = f"TOP THREAT: #{top['track_id']} {cls}  {dist}  {vel}  K={k_val:.2f}"
        cv2.putText(img, threat_label, (w // 3, 30),
                    FONT, 0.55, (0, 100, 255), 1, cv2.LINE_AA)

    # CPE watermark
    cv2.putText(img, "Argus-REI CPE | Stage 1 Perception", (w - 320, 18),
                FONT, 0.42, (100, 100, 100), 1, cv2.LINE_AA)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CPE Stage 1 — Visualize perception output as MP4")
    parser.add_argument("--rgb_dir", required=True, help="Folder of RGB frames")
    parser.add_argument("--csv",     required=True, help="Stage 1 output CSV")
    parser.add_argument("--out",     default="data/processed/stage1_preview.mp4",
                        help="Output video path")
    parser.add_argument("--fps",     type=float, default=33.0)
    parser.add_argument("--scale",   type=float, default=0.5,
                        help="Resize factor (0.5 = half resolution, faster)")
    args = parser.parse_args()

    rgb_dir  = Path(args.rgb_dir)
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted([p for p in rgb_dir.iterdir()
                          if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not frame_paths:
        print(f"❌  No frames found in {rgb_dir}"); return

    frames_data = load_csv(csv_path)

    # ── Pre-compute kinetic score inline (csv doesn't have it yet) ──
    CLASS_SEVERITY = {"car": 2.0, "bus": 2.5, "truck": 2.5,
                      "motorcycle": 1.8, "bicycle": 1.2, "person": 1.0}
    for rows in frames_data.values():
        for r in rows:
            d = r["distance_m"];  v = r["velocity_ms"]
            if d and d > 0 and v > 0:
                sev = CLASS_SEVERITY.get(r["class"], 1.0)
                r["kinetic"] = sev * v**2 / max(d, 0.5)
        rows.sort(key=lambda x: x["kinetic"], reverse=True)

    # ── Read first frame for dimensions ──
    sample = cv2.imread(str(frame_paths[0]))
    orig_h, orig_w = sample.shape[:2]
    out_h = int(orig_h * args.scale)
    out_w = int(orig_w * args.scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (out_w, out_h))

    print(f"\n=== Rendering Stage 1 visualisation ===")
    print(f"  Frames   : {len(frame_paths)}")
    print(f"  Output   : {out_path}  ({out_w}×{out_h} @ {args.fps}fps)")
    print(f"  Scale    : {args.scale}×  ({orig_w}×{orig_h} → {out_w}×{out_h})\n")

    for frame_idx, rgb_path in enumerate(frame_paths):
        img = cv2.imread(str(rgb_path))
        if img is None: continue

        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rows = frames_data.get(frame_idx, [])

        for row in rows:
            draw_box(img, row, args.scale)

        draw_hud(img, frame_idx, rows, out_h, out_w)
        writer.write(img)

        if frame_idx % 30 == 0:
            print(f"  [{frame_idx:04d}/{len(frame_paths)}] rendered")

    writer.release()
    print(f"\n✅  Video saved → {out_path}")
    print(f"    Open in VLC, Windows Media Player, or any video player.")


if __name__ == "__main__":
    main()
