"""
build_session_viewer.py
=======================
A local, self-contained viewer for checking the v2 engine by eye on one SANPO
walk: the video with every tracked box, what the engine thought of each object
(distance, closing speed, P(arrive), priority), when it spoke and what it said,
and where objects actually went (the corridor-arrival ground truth).

Pipeline, as the runtime runs it:
  SANPO RGB + SANPO depth (as is)  -> Stage 1 (YOLO26n v3 + ByteTrack + box depth, pinhole bearing)
  + fixed_camera_poses.csv          -> ThreatEngine (Kalman fusion, safety envelope, GRU arrival,
                                       fused priority, announcement manager) -> narration phrase
Ground truth: an object "arrives" at the first frame it is inside the walking
corridor (0 < forward <= 3 m, |lateral| <= 0.375 m + half its width, along the
direction of travel), the same rule as the arrival label (dataset/GRU/relabel.py).
Every box shows how many seconds until it arrives within the 4 s horizon.

Output: <out>/index.html + data.js + frames/*.jpg. Open index.html in a browser
(works from file://; nothing is uploaded).

    python tools/build_session_viewer.py                       # a test walk with arrivals, 150 frames (30 s)
    python tools/build_session_viewer.py --session <id> --frames 300 --threshold 0.4 --no-pose
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.GRU.relabel import walking_angle  # noqa: E402
from dataset.GRU.samples import load_dataset  # noqa: E402
from src.narration.templates import narrate  # noqa: E402
from src.threat_engine.announcer import AnnouncementManager  # noqa: E402
from src.threat_engine.engine import ThreatEngine  # noqa: E402
from src.threat_engine.envelope import CORRIDOR_WIDTH_M, half_width  # noqa: E402
from src.threat_engine.fusion import GroundTracker  # noqa: E402
from src.threat_engine.priority import GRUArrivalPredictor  # noqa: E402
from tools.replay_threat_engine import DEFAULT_DATASET, session_poses  # noqa: E402
from tools.stream_sanpo_perception import DEFAULT_MODEL, fetch_session_frames, session_camera  # noqa: E402

GRU_RUN = PROJECT_ROOT / "models/GRU/GRU/runs/corridor_h8_seed0"
STRIDE = 3
STEP_S = 0.2
HORIZON = 20          # steps = 4 s
DEPTH_M = 3.0
VIEW_W = 960


def default_session() -> str:
    """The official-test walk with the most arriving tracks in the arrival dataset."""
    _, splits = load_dataset(DEFAULT_DATASET, splits=("test",))
    count = defaultdict(set)
    for s in splits["test"]:
        for t, y in zip(s["track_ids"], s["y"]):
            if y:
                count[s["session"]].add(t)
    return max(count, key=lambda sid: len(count[sid]))


def ground_truth(frames: list[list[dict]], poses) -> dict[tuple[int, str], int]:
    """{(frame, track): steps until the track is first in the corridor within the horizon, 0 = in it now}."""
    inside: dict[str, list[int]] = defaultdict(list)
    for k, rows in enumerate(frames):
        phi = walking_angle(poses, k) if poses else 0.0
        for r in rows:
            d = float(r["distance_m"] or 0)
            if d <= 0 or str(r["track_id"]).startswith("obs_"):
                continue
            th = math.radians(float(r["bearing_deg"]))
            x, z = d * math.tan(th), d
            fwd = x * math.sin(phi) + z * math.cos(phi)
            lat = x * math.cos(phi) - z * math.sin(phi)
            if 0 < fwd <= DEPTH_M and abs(lat) <= CORRIDOR_WIDTH_M / 2 + half_width(r["class"]):
                inside[str(r["track_id"])].append(k)
    out = {}
    for k, rows in enumerate(frames):
        for r in rows:
            ks = [f for f in inside.get(str(r["track_id"]), []) if k <= f <= k + HORIZON]
            if ks:
                out[(k, str(r["track_id"]))] = ks[0] - k
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", default=None, help="SANPO session id (default: the test walk with most arrivals).")
    p.add_argument("--frames", type=int, default=150, help="Processed frames (5 per second); 150 = 30 s.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Ignore threshold (0.5 = v2_gru's validation-chosen early_budget_5 point, corridor_tuned).")
    p.add_argument("--no-pose", action="store_true", help="Track in the camera frame (no ego-motion removal).")
    p.add_argument("--out", type=Path, default=None, help="Default: data/viewer/<session>/ (git-ignored).")
    args = p.parse_args()
    started = time.time()

    sid = args.session or default_session()
    out = args.out or PROJECT_ROOT / "data/viewer" / sid
    scratch = PROJECT_ROOT / "data/viewer/_scratch" / sid
    print(f"session {sid}")
    camera = session_camera({"session_id": sid})
    for attempt in range(1, 6):   # flaky links: finished files are skipped, so a retry resumes
        try:
            rgb_dir, depth_dir, rgb_objects = fetch_session_frames(
                {"session_id": sid}, scratch,
                SimpleNamespace(depth_kind="depth_maps", frame_stride=STRIDE, max_frames=args.frames,
                                download_workers=4))
            break
        except OSError as exc:   # URLError, timeouts and connection resets are all OSErrors
            if attempt == 5:
                raise
            print(f"download attempt {attempt} failed ({type(exc).__name__}: {exc}); retrying", flush=True)
            time.sleep(5 * attempt)
    print(f"{len(rgb_objects)} frames downloaded ({time.time() - started:.0f} s); running Stage 1 on CPU...")

    from src.perception_stack import run_perception_stream
    frames = [rows for rows in run_perception_stream(rgb_dir, depth_dir, camera["fps"] / STRIDE, source="sanpo",
                                                     frame_step=1, model_path=str(DEFAULT_MODEL),
                                                     focal_px=camera["fx"])]
    poses_all = session_poses(sid)
    poses = None if args.no_pose else poses_all
    truth = ground_truth(frames, poses_all)
    print(f"Stage 1 done ({time.time() - started:.0f} s); running the v2 engine...")

    images = sorted(rgb_dir.iterdir())
    from PIL import Image
    w0, h0 = Image.open(images[0]).size
    engine = ThreatEngine(predictor=GRUArrivalPredictor(GRU_RUN / "model.pt"), ignore_threshold=args.threshold,
                          tracker=GroundTracker(),
                          announcer=AnnouncementManager())
    (out / "frames").mkdir(parents=True, exist_ok=True)
    scale = VIEW_W / w0
    data = {"session": sid, "step_s": STEP_S, "threshold": args.threshold, "pose": not args.no_pose,
            "image_w": VIEW_W, "image_h": round(h0 * scale), "frames": [], "announcements": [], "arrivals": []}
    arrived, first_seen = set(), {}
    for k, rows in enumerate(frames):
        for r in rows:
            first_seen.setdefault(str(r["track_id"]), k)
        pose = poses[min(STRIDE * k, len(poses) - 1)] if poses else None
        f = engine.step(k, k * STEP_S, rows, pose)
        ranked = {r.track.track_id: r for r in f.ranked}
        fused = {t.track_id: t for t in f.tracks}
        overrides = {o.track.track_id for o in f.overrides}
        boxes = []
        for r in rows:
            tid = str(r["track_id"])
            if tid.startswith("obs_") or r.get("bbox_x1") in (None, ""):
                continue
            t, rk = fused.get(tid), ranked.get(tid)
            tte = truth.get((k, tid))
            if tte == 0 and tid not in arrived:
                arrived.add(tid)
                if first_seen[tid] < k:   # as in the replay: only arrivals that could have been warned
                    data["arrivals"].append({"frame": k, "track": tid, "class": r["class"],
                                             "seen": first_seen[tid]})
            boxes.append({
                "id": tid, "cls": r["class"], "conf": round(float(r["confidence"]), 2),
                "box": [round(float(r[c]) * scale) for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")],
                "d": round(float(r["distance_m"]), 2) if r["distance_m"] not in (None, "") else None,
                "closing": round(t.closing_speed_ms, 2) if t else None,
                "lat": round(t.lateral_m, 2) if t else None, "fwd": round(t.forward_m, 2) if t else None,
                "p": round(rk.arrival.p_arrive, 3) if rk else None,
                "prio": round(rk.priority, 3) if rk else None,
                "override": tid in overrides, "tte": tte,
            })
        a = f.announcement
        if a:
            ev = f.narrator_event()
            tte = truth.get((k, a.track_id), -1)
            data["announcements"].append({"frame": k, "track": a.track_id, "kind": a.kind, "class": a.class_name,
                                          "text": narrate(ev) if ev else a.reason, "reason": a.reason,
                                          "truth": "true" if tte >= 1 else "late" if tte == 0 else "false"})
        Image.open(images[k]).convert("RGB").resize((VIEW_W, data["image_h"])).save(out / "frames" / f"{k:04d}.jpg",
                                                                                        quality=80)
        data["frames"].append({"boxes": boxes, "said": a.track_id if a else None})
    for arr in data["arrivals"]:
        heard = [x for x in data["announcements"] if x["track"] == arr["track"]
                 and arr["frame"] - HORIZON <= x["frame"] <= arr["frame"]]
        arr["warned_at"] = heard[0]["frame"] if heard else None
    (out / "data.js").write_text("window.DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("session_viewer.html"), out / "index.html")
    shutil.rmtree(scratch, ignore_errors=True)
    n_arr, n_ann = len(data["arrivals"]), len(data["announcements"])
    print(f"{len(frames)} frames, {n_ann} announcements, {n_arr} arrivals ({time.time() - started:.0f} s)")
    print(f"open: {(out / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
