"""
relabel.py
==========
Replace the arrival label of a built GRU dataset with the "in my path"
corridor label, keeping the inputs (x) unchanged.

Why. The original label — within hazard_m inside a ±30° bearing cone — was
written for a 70° camera. SANPO's head camera (ZED-M) is 70-78°, so ±30° is
most of the image, and the cone is ~3.5 m wide at 3 m whatever the camera:
"arrives" mostly meant "is near and in view", which is why the nearest-object
rule ties every learned model.

The corridor label, per object and frame T:

    the object is in the walker's path at some step t in [T, T + horizon]:
        0 < forward <= depth_m  and  |lateral| <= width_m / 2 + half_width(class)

where forward/lateral are measured along the walker's direction of travel at
step t (from SANPO's fixed_camera_poses.csv: the 1 s ground displacement
ahead of t, expressed in the camera frame; the camera heading when standing
still), and the object's position comes from its depth and its pinhole bearing
(the session's real fx). tte = steps until it first is.

    python -m dataset.GRU.relabel --dataset dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official
    python -m dataset.GRU.relabel --dataset ... --width 1.0 --horizon 25   # sensitivity grid

Width 0.75 m follows Fruin's comfortable lateral spacing for a walking person
(0.71-0.76 m); class half-widths are declared values (CLASS_HALF_WIDTH_M).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from pathlib import Path

import numpy as np

from dataset.GRU.samples import SPLITS, load_split, save_split
from src.threat_engine.envelope import CLASS_HALF_WIDTH_M, DEFAULT_HALF_WIDTH_M
from src.threat_engine.fusion import EgoPose, sanpo_session_poses

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT_ROOT / "simulation/datasets/sanpo/session_audit.json"
POSE_CACHE = PROJECT_ROOT / "data/sanpo_poses_fixed"
STRIDE = 3                 # processed frame k = source frame 3k
SOURCE_FPS = 15
LEGACY_HALF_FOV = 35.0     # Stage-1 CSVs before 2026-10-14 mapped the image linearly onto ±35°
MIN_WALK_M = 0.3           # 1 s displacement below this = standing; use the camera heading

# The class half-widths (declared, not measured) live with the runtime envelope so the
# label and the runtime share one table.


def load_session_poses(sid: str) -> list[EgoPose]:
    path = POSE_CACHE / f"{sid}.csv"
    if not path.exists():
        from tools.audit_sanpo_sessions import fetch, object_url
        from tools.download_sanpo_valid_streams import BASE_PREFIX
        POSE_CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(object_url(f"{BASE_PREFIX}/{sid}/camera_head/fixed_camera_poses.csv")))
    return sanpo_session_poses(list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))))


def walking_angle(poses: list[EgoPose], k: int) -> float:
    """Direction of travel at processed frame k, as an angle from the camera's forward axis (+ = right)."""
    i = min(STRIDE * k, len(poses) - 1)
    j = min(i + SOURCE_FPS, len(poses) - 1)
    d = np.subtract(poses[j].position, poses[i].position)
    if np.linalg.norm(d) < MIN_WALK_M:
        return 0.0
    return math.atan2(float(d @ np.asarray(poses[i].right)), float(d @ np.asarray(poses[i].forward)))


def pinhole_bearing(legacy_bearing_deg: np.ndarray, fx: float, width: float) -> np.ndarray:
    dx = legacy_bearing_deg / LEGACY_HALF_FOV * width / 2
    return np.arctan2(dx, fx)


def in_path(distance: np.ndarray, bearing_rad: np.ndarray, phi: float, classes: list[str],
            width_m: float, depth_m: float) -> np.ndarray:
    x, z = distance * np.tan(bearing_rad), distance
    forward = x * math.sin(phi) + z * math.cos(phi)
    lateral = x * math.cos(phi) - z * math.sin(phi)
    half = np.array([CLASS_HALF_WIDTH_M.get(c, DEFAULT_HALF_WIDTH_M) for c in classes])
    return (forward > 0) & (forward <= depth_m) & (np.abs(lateral) <= width_m / 2 + half)


def relabel_session(samples: list[dict], poses: list[EgoPose], fx: float, width: float,
                    width_m: float, depth_m: float, horizon: int) -> None:
    """Overwrite y, tte and frame_has_encounter in place."""
    samples.sort(key=lambda s: s["frame_idx"])
    hits: dict[int, set[str]] = {}
    for s in samples:
        phi = walking_angle(poses, s["frame_idx"])
        mask = in_path(np.asarray(s["distance"], float), pinhole_bearing(np.asarray(s["bearing"], float), fx, width),
                       phi, s["classes"], width_m, depth_m)
        hits[s["frame_idx"]] = {t for t, m in zip(s["track_ids"], mask) if m}
    first: dict[str, list[int]] = {}
    for f in sorted(hits):
        for t in hits[f]:
            first.setdefault(t, []).append(f)
    for s in samples:
        f0 = s["frame_idx"]
        tte = []
        for t in s["track_ids"]:
            fs = first.get(t, [])
            i = np.searchsorted(fs, f0)
            tte.append(fs[i] - f0 if i < len(fs) and fs[i] <= f0 + horizon else -1)
        s["tte"] = np.array(tte, dtype=np.int64)
        s["y"] = (s["tte"] >= 0).astype(np.float32)
        s["frame_has_encounter"] = any(any(f0 <= f <= f0 + horizon for f in fs) for fs in first.values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--width", type=float, default=0.75, help="Corridor width, m (plus each object's half-width).")
    p.add_argument("--depth", type=float, default=3.0, help="How far ahead the corridor reaches, m.")
    p.add_argument("--horizon", type=int, default=20, help="Steps (0.2 s each): 13 = 2.6 s, 20 = 4 s, 25 = 5 s.")
    p.add_argument("--name", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    name = args.name or f"{src['name']}_corridor_w{args.width:g}_d{args.depth:g}_h{args.horizon}"
    out = args.dataset.parent / name
    if (out / "manifest.json").exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists (pass --overwrite)")
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}

    parts = ["all"] if "folds" in src else list(SPLITS)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for part in parts:
        samples = load_split(args.dataset / f"{part}.npz")
        by: dict[str, list[dict]] = {}
        for s in samples:
            by.setdefault(s["session"], []).append(s)
        for sid, ss in by.items():
            a = audit[sid]
            relabel_session(ss, load_session_poses(sid), a["fx"], a["image_width"], args.width, args.depth,
                            args.horizon)
        save_split(samples, out / f"{part}.npz")
        y = np.concatenate([s["y"] for s in samples])
        stats[part] = {"sessions": len(by), "frames": len(samples), "objects": int(len(y)),
                       "positive_rate": round(float(y.mean()), 4),
                       "sessions_with_encounter": len({s["session"] for s in samples if s["y"].any()}),
                       "frames_with_encounter": int(sum(s["frame_has_encounter"] for s in samples)),
                       "positive_tracks": len({(s["session"], t) for s in samples
                                               for t, yy in zip(s["track_ids"], s["y"]) if yy})}
    manifest = {**src, "name": name, "created": time.strftime("%Y-%m-%d"), "relabelled_from": src["name"],
                "stats": stats if "folds" not in src else {"all": stats["all"]},
                "label": {"definition": f"in the walking corridor (width {args.width:g} m + class half-width, "
                                        f"0 < forward <= {args.depth:g} m along the direction of travel from "
                                        f"fixed_camera_poses.csv) within {args.horizon} steps "
                                        f"({args.horizon * 0.2:g} s); tracked objects only",
                          "kind": "corridor", "width_m": args.width, "depth_m": args.depth,
                          "HORIZON_FRAMES": args.horizon, "THETA_DEG": src["label"].get("THETA_DEG", 30.0),
                          "hazard_m": args.depth, "class_half_width_m": CLASS_HALF_WIDTH_M}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for part, s in stats.items():
        print(f"{part:5s}: {s['sessions']} sessions ({s['sessions_with_encounter']} with an arrival), "
              f"{s['positive_tracks']} arriving tracks, positive rate {s['positive_rate']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
