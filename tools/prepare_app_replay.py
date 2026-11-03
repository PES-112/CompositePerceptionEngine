"""
prepare_app_replay.py
=====================
SANPO test walks laid out for the CPEWalk app's own Swift pipeline (ios/CPEWalk/Tests/SanpoReplayTests.swift),
so the finished app can be run on recorded walks in the iOS Simulator.

Per session, under --out-dir/<session_id>/:
    frames/0000.jpg ...   RGB at half resolution (1104 x 621), every 3rd source frame = 5 Hz, as the replay
    depth/0000.f32 ...    SANPO stereo depth, metres, float32 row-major, resized to 256 wide (LiDAR-like)
    meta.json             intrinsics; per frame t, EgoPose (sanpo_session_poses, as the Python replay),
                          camera position (Z up, for StairMotion) and the DepthGeometry the app's LiDAR
                          obstacle scan needs

The depth geometry maps the app's portrait-sensor convention onto an upright landscape image: the scan reads
sensor x = row, sensor y = W - 1 - column, so the "camera" axes are +x = image down, +y = image right,
-z = forward, and the world is the replay's ground basis with up as y: (p·e1, p·up, -(p·e2)).

    python tools/prepare_app_replay.py --sessions 20
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception_stack.depth_loader import decode_sanpo_depth
from src.threat_engine.fusion import CAMERA_AXES, _ground_basis, _quat_to_matrix, sanpo_session_poses
from tools.audit_sanpo_sessions import camera_details, fetch, object_url
from tools.download_sanpo_valid_streams import BASE_PREFIX, download_object, gcs_list, stem_map

POSES = ROOT / "data/sanpo_poses_fixed"
STRIDE, MAX_FRAMES, SCALE, DEPTH_W = 3, 300, 0.5, 256
# app "camera" axes (portrait-sensor convention) in OpenCV camera axes: image down, image right, backward
FAKE_AXES = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float).T


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data.nosync/app_replay")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def pick_sessions(n: int, seed: int) -> list[str]:
    split = json.loads((ROOT / "simulation/datasets/sanpo/official_split.json").read_text())
    valid = {e["session_id"] for e in json.loads((ROOT / "simulation/datasets/sanpo/valid_streams.json").read_text())
             if e.get("camera", "head") == "head" and e.get("view", "left") == "left"}
    pool = sorted(s for s in split["test"] if s in valid and (POSES / f"{s}.csv").exists())
    return sorted(random.Random(seed).sample(pool, n))


def fetch_retry(name: str, path: Path, size: int, tries: int = 6) -> None:
    for i in range(tries):
        try:
            download_object(name, path, size)
            return
        except Exception:
            path.unlink(missing_ok=True)
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def geometry(rows: list[dict]) -> list[dict]:
    """Per pose row: the app's DepthGeometry rotation / position and the stairs camera position."""
    keys = ("pos_x", "pos_y", "pos_z", "q_x", "q_y", "q_z", "q_w")
    vals = np.array([[float(r[k]) for k in keys] for r in rows], dtype=float)
    ready = np.array([str(r.get("tracking_state", "")).endswith("READY") and "NOT" not in str(r.get("tracking_state", ""))
                      for r in rows])
    cam_up = np.asarray(CAMERA_AXES["opencv"]["up"], dtype=float)
    up = np.mean([_quat_to_matrix(*v[3:]) @ cam_up for v in vals[ready]], axis=0)
    up /= np.linalg.norm(up)
    e1, e2 = _ground_basis(up)
    B = np.stack([e1, up, -e2])
    idx = np.arange(len(rows))
    good = idx[ready]
    pos = np.stack([np.interp(idx, good, vals[ready, k]) for k in range(3)], axis=1)
    nearest = good[np.abs(idx[:, None] - good[None, :]).argmin(axis=1)]
    out = []
    for i in idx:
        R = B @ _quat_to_matrix(*vals[nearest[i], 3:]) @ FAKE_AXES
        out.append({"rotation_cols": R.T.round(6).tolist(), "position": (B @ pos[i]).round(4).tolist(),
                    "camera": [round(float(pos[i][0]), 4), round(float(pos[i][1]), 4), round(float(pos[i][2]), 4)]})
    return out


def prepare(sid: str, out: Path, workers: int) -> None:
    dest = out / sid
    if (dest / "meta.json").exists():
        print(f"{sid}: already prepared", flush=True)
        return
    desc = json.loads(fetch(object_url(f"{BASE_PREFIX}/{sid}/description.json")))
    cam = camera_details(desc, "head")["left_camera_params"]
    branch = f"{BASE_PREFIX}/{sid}/camera_head/left/"
    rgb = stem_map(gcs_list(branch + "video_frames/"), branch + "video_frames/")
    dep = stem_map(gcs_list(branch + "depth_maps/"), branch + "depth_maps/")
    stems = sorted(set(rgb) & set(dep))[::STRIDE][:MAX_FRAMES]
    raw = dest / "raw"
    jobs = [(it, raw / Path(it["name"]).name) for s in stems for it in (rgb[s], dep[s])]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda j: fetch_retry(j[0]["name"], j[1], int(j[0].get("size", 0))), jobs))

    rows = list(csv.DictReader(io.StringIO((POSES / f"{sid}.csv").read_text())))
    poses, geo = sanpo_session_poses(rows), geometry(rows)
    all_stems = sorted(set(rgb) & set(dep))
    (dest / "frames").mkdir(parents=True, exist_ok=True)
    (dest / "depth").mkdir(exist_ok=True)
    frames, dshape = [], None
    for k, s in enumerate(stems):
        img = cv2.imread(str(raw / Path(rgb[s]["name"]).name))
        img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest / f"frames/{k:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with gzip.open(raw / Path(dep[s]["name"]).name, "rb") as f:
            d = decode_sanpo_depth(np.frombuffer(f.read(), dtype=np.float16))
        dh = round(d.shape[0] * DEPTH_W / d.shape[1])
        d = cv2.resize(np.nan_to_num(d, nan=0, posinf=0, neginf=0), (DEPTH_W, dh), interpolation=cv2.INTER_NEAREST)
        d.astype("<f4").tofile(dest / f"depth/{k:04d}.f32")
        dshape = (DEPTH_W, dh)
        src = all_stems.index(s)                       # source frame index = pose row (15 fps, as the replay)
        pi = min(src, len(poses) - 1)
        p = poses[pi]
        frames.append({"t": round(k * 0.2, 3), "pose": {"position": list(p.position), "right": list(p.right),
                                                       "forward": list(p.forward)}, **geo[pi]})
    W, H = cam["image_width"] * SCALE, cam["image_height"] * SCALE
    s = dshape[0] / cam["image_width"]                 # depth px per source px
    dw = dshape[0]
    meta = {
        "session": sid, "image_size": [round(W), round(H)], "depth_size": list(dshape),
        "focal_px": cam["fx"] * SCALE, "principal_x": cam["cx"] * SCALE,
        # app convention: x = row, y = dw - 1 - column
        "depth_geometry": {"fx": cam["fy"] * s, "fy": cam["fx"] * s, "cx": cam["cy"] * s,
                           "cy": dw - 1 - cam["cx"] * s, "sensor_height": dw},
        "frames": frames,
    }
    (dest / "meta.json").write_text(json.dumps(meta))
    shutil.rmtree(raw)
    print(f"{sid}: {len(frames)} frames, image {meta['image_size']}, depth {dshape}", flush=True)


def main() -> None:
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    sessions = pick_sessions(a.sessions, a.seed)
    (a.out_dir / "sessions.json").write_text(json.dumps(sessions, indent=1))
    for i, sid in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}]", end=" ", flush=True)
        try:
            prepare(sid, a.out_dir, a.workers)
        except Exception as e:                       # one bad session must not stop the rest
            print(f"{sid}: FAILED {e!r}", flush=True)


if __name__ == "__main__":
    main()
