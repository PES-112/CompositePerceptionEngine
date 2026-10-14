"""
validate_sanpo_poses.py
=======================
Pass/fail gates for using SANPO camera poses to remove the walker's own motion
(src/threat_engine/fusion.py). Run before claiming ego-motion removal works.

  Gate 1  heading   median angle between the walker's 1 s ground displacement
                    and the camera's forward axis < 15 deg (walking sessions)
  Gate 2  static    near (< 8 m) static-class tracks move less per step in the
                    world frame than in the camera frame (median, all sessions
                    with tracked static objects)
  Gate 3  smooth    path length / net displacement over 1 s windows < 1.3
                    (per session; failing sessions are listed, not dropped)

Pose files are fetched from the public bucket and cached under data/. Gate 2
uses the Stage-1 rows stored in the GRU dataset (all 462 sessions' tracked
objects), converted to correct units as in tools/replay_threat_engine.py.

    python tools/validate_sanpo_poses.py                       # fixed_camera_poses.csv, opencv axes
    python tools/validate_sanpo_poses.py --file camera_poses.csv --convention zed
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.GRU.samples import load_split
from src.threat_engine.fusion import EgoPose, sanpo_session_poses
from tools.audit_sanpo_sessions import fetch, object_url
from tools.download_sanpo_valid_streams import BASE_PREFIX, DEFAULT_VALID_STREAMS, load_manifest
from tools.replay_threat_engine import AUDIT, STRIDE, session_rows

ALL_SESSIONS = PROJECT_ROOT / "dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_f5_s0/all.npz"
STATIC = {"pole", "bollard", "bench", "fire hydrant", "traffic light", "stop sign"}
SOURCE_FPS = 15
HEADING_MAX_DEG = 15.0
SMOOTH_MAX = 1.3
NEAR_M = 8.0


def load_poses(sid: str, file: str, convention: str, cache: Path) -> list[EgoPose]:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{sid}.csv"
    if not path.exists():
        path.write_bytes(fetch(object_url(f"{BASE_PREFIX}/{sid}/camera_head/{file}")))
    return sanpo_session_poses(list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))), convention)


def heading_and_smoothness(poses: list[EgoPose]) -> tuple[list[float], float | None]:
    """(angles between 1 s displacement and camera forward, deg; median 1 s path/net ratio)."""
    P = np.array([p.position for p in poses])
    angles, ratios = [], []
    for i in range(0, len(P) - SOURCE_FPS, SOURCE_FPS):
        d = P[i + SOURCE_FPS] - P[i]
        net = float(np.linalg.norm(d))
        if net < 0.5:                       # standing still: heading undefined
            continue
        path = float(np.sum(np.linalg.norm(np.diff(P[i:i + SOURCE_FPS + 1], axis=0), axis=1)))
        ratios.append(path / net)
        f = np.asarray(poses[i].forward)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, float(d @ f) / net)))))
    return angles, (float(np.median(ratios)) if ratios else None)


def static_steps(frames: dict[int, list[dict]], poses: list[EgoPose]) -> tuple[list[float], list[float]]:
    """Per consecutive-frame step of a near static track: (world-frame, camera-frame) displacement, m."""
    world, cam, last = [], [], {}
    for k in sorted(frames):
        pose = poses[min(STRIDE * k, len(poses) - 1)]
        for r in frames[k]:
            if r["class"] not in STATIC or r["distance_m"] > NEAR_M:
                continue
            x = r["distance_m"] * math.tan(math.radians(r["bearing_deg"]))
            c = np.array([x, r["distance_m"]])
            w = pose.to_world(*c)
            prev = last.get(r["track_id"])
            if prev is not None and prev[0] == k - 1:
                world.append(float(np.linalg.norm(w - prev[1])))
                cam.append(float(np.linalg.norm(c - prev[2])))
            last[r["track_id"]] = (k, w, c)
    return world, cam


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default="fixed_camera_poses.csv", choices=["fixed_camera_poses.csv", "camera_poses.csv"])
    p.add_argument("--convention", default="opencv", choices=["opencv", "zed"])
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "evaluation/benchmarks/pose_validation")
    args = p.parse_args()
    cache = PROJECT_ROOT / ("data/sanpo_poses_fixed" if args.file.startswith("fixed") else "data/sanpo_poses")

    ids = [e["session_id"] for e in load_manifest(DEFAULT_VALID_STREAMS)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        poses = dict(zip(ids, pool.map(lambda s: load_poses(s, args.file, args.convention, cache), ids)))

    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}
    by_session = defaultdict(list)
    for s in load_split(ALL_SESSIONS):
        by_session[s["session"]].append(s)

    angles_all, per_session, world_all, cam_all = [], [], [], []
    for sid in ids:
        angles, ratio = heading_and_smoothness(poses[sid])
        angles_all += angles
        w = c = []
        if sid in by_session:
            a = audit[sid]
            w, c = static_steps(session_rows(by_session[sid], a["fx"], a["image_width"]), poses[sid])
            world_all += w
            cam_all += c
        per_session.append({"session_id": sid, "walking_windows": len(angles),
                            "heading_median_deg": float(np.median(angles)) if angles else None,
                            "path_net_ratio": ratio, "static_steps": len(w),
                            "static_world_median_m": float(np.median(w)) if w else None,
                            "static_camera_median_m": float(np.median(c)) if c else None})

    heading = float(np.median(angles_all))
    world, cam = float(np.median(world_all)), float(np.median(cam_all))
    ratios = [r["path_net_ratio"] for r in per_session if r["path_net_ratio"] is not None]
    rough = [r["session_id"] for r in per_session if r["path_net_ratio"] is not None and r["path_net_ratio"] >= SMOOTH_MAX]
    gates = {
        "heading": {"median_deg": heading, "p75_deg": float(np.percentile(angles_all, 75)),
                    "windows": len(angles_all), "pass": heading < HEADING_MAX_DEG},
        "static": {"world_median_m": world, "camera_median_m": cam, "steps": len(world_all),
                   "world_p75_m": float(np.percentile(world_all, 75)), "camera_p75_m": float(np.percentile(cam_all, 75)),
                   "pass": world < cam},
        "smooth": {"median_ratio": float(np.median(ratios)), "sessions_failing": len(rough),
                   "sessions_with_walking": len(ratios), "failing_ids": rough,
                   "pass": float(np.median(ratios)) < SMOOTH_MAX},
    }
    out = args.out / f"{Path(args.file).stem}_{args.convention}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gates.json").write_text(json.dumps({"file": args.file, "convention": args.convention,
                                                "generated": time.strftime("%Y-%m-%d"), "gates": gates,
                                                "sessions": per_session}, indent=2) + "\n", encoding="utf-8")
    g = gates
    print(f"{args.file} ({args.convention}), {len(ids)} sessions")
    print(f"  gate 1 heading : median {g['heading']['median_deg']:.1f} deg over {g['heading']['windows']} walking "
          f"windows -> {'PASS' if g['heading']['pass'] else 'FAIL'}")
    print(f"  gate 2 static  : world {world:.3f} m vs camera {cam:.3f} m per step over {len(world_all)} steps "
          f"-> {'PASS' if g['static']['pass'] else 'FAIL'}")
    print(f"  gate 3 smooth  : median path/net {g['smooth']['median_ratio']:.2f}, {len(rough)}/{len(ratios)} sessions "
          f">= {SMOOTH_MAX} -> {'PASS' if g['smooth']['pass'] else 'FAIL'}")
    print(f"-> {out / 'gates.json'}")


if __name__ == "__main__":
    main()
