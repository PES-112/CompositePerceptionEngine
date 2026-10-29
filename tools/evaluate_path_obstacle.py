"""
evaluate_path_obstacle.py
=========================
Does "Obstacle ahead." (src/perception_stack/path_obstacle.py) warn before the walker gets close to
something in the way, and how often does it speak?

Every 6th source frame (2.5 Hz, the HUMAN_ANNOTATED frames, as in evaluate_edge_alerts.py) of the 53
sessions with head-camera masks: SANPO's raw class ids + depth -> corridor histogram, cached in
data/path_obstacle_track/gt/<sid>.npz. This is the oracle: SANPO's masks stand in for a perfect
segmentation model, so it scores the rule, not a model.

Obstacles: SANPO_OBSTACLE_IDS (vertical classes the detector doesn't track: obstacle, pole, sign,
light, bike rack, bus stop, tree, building, wall/fence, rails, doors, gates). Pedestrians, riders,
animals and vehicles are left to the object engine.

Reference (declared): an *encounter* starts on a frame with >= ENCOUNTER_PIXELS obstacle pixels in
the corridor within ENCOUNTER_M, after >= 2 frames without. SANPO has no "bumped into" event, and a
walker usually steps around things, so this is "came close", the analogue of the curb alert's
step-off. Scores (session-bootstrap 95% CIs):
  alerts / min
  encounters warned        an alert in the LOOKBACK_S before (or at) the encounter; warned >= 1 s
  alerts followed          share of alerts followed by an encounter within LOOKBACK_S (a lower bound
                           on precision: an obstacle the walker steered around isn't followed)
  by class                 encounters warned, by the dominant SANPO class of the close pixels

--sweep chooses alert_m x min_pixels x confirm_frames on official-train sessions only (most
encounters warned >= 1 s ahead with <= SWEEP_MAX_ALERTS_PER_MIN, ties: fewer alerts) and scores
test once with the choice.

    python tools/evaluate_path_obstacle.py --sweep
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from dataset.GRU.relabel import load_session_poses
from src.perception_stack.depth_loader import decode_sanpo_depth
from src.perception_stack.path_obstacle import (CLEAR_S, SANPO_OBSTACLE_IDS, PathObstacleAlerter,
                                                corridor_histogram, pixels_within)
from tools.audit_sanpo_sessions import fetch, object_url
from tools.download_sanpo_valid_streams import BASE_PREFIX

AUDIT = PROJECT_ROOT / "simulation/datasets/sanpo/session_audit.json"
SOURCE_FPS, FRAME_STEP, DOWNSAMPLE = 15, 6, 4
N_CLASSES = 31
ENCOUNTER_M, ENCOUNTER_PIXELS, ENCOUNTER_GAP = 1.0, 300, 2
LOOKBACK_S = 10.0
SWEEP_ALERT_M = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
SWEEP_MIN_PIXELS = (100, 300, 1000, 3000)
SWEEP_CONFIRM = (1, 2, 3)
SWEEP_MAX_ALERTS_PER_MIN = 1.0      # as the curb alert's sweep
CACHE = PROJECT_ROOT / "data/path_obstacle_track/gt"
OUT = PROJECT_ROOT / "evaluation/benchmarks/path_obstacle"
NAMES = {4: "guard rail", 7: "building", 8: "wall/fence", 9: "hand rail", 10: "opening-door", 11: "opening-gate",
         20: "obstacle", 22: "traffic sign", 23: "traffic light", 24: "pole", 25: "bus stop", 26: "bike rack",
         28: "tree"}
N_BOOT = 1000


def frame_hist(sid: str, f: int, fx_full: float, width_full: int) -> np.ndarray | None:
    base = f"{BASE_PREFIX}/{sid}/camera_head/left"
    try:
        m = cv2.imdecode(np.frombuffer(fetch(object_url(f"{base}/segmentation_masks/{f:06d}.png")), np.uint8),
                         cv2.IMREAD_UNCHANGED)
        depth = decode_sanpo_depth(np.frombuffer(gzip.decompress(fetch(object_url(
            f"{base}/depth_maps/{f:06d}.float16.gz"))), np.float16))
    except Exception:
        return None
    ids = (m if m.ndim == 2 else m[..., 2])[::DOWNSAMPLE, ::DOWNSAMPLE]
    depth = cv2.resize(depth.astype(np.float32), (ids.shape[1], ids.shape[0]), interpolation=cv2.INTER_NEAREST)
    fx = fx_full * ids.shape[1] / width_full
    return corridor_histogram(ids, depth, fx, N_CLASSES)


def session_track(sid: str, rec: dict) -> tuple[np.ndarray, np.ndarray]:
    path = CACHE / f"{sid}.npz"
    if path.exists():
        z = np.load(path)
        return z["frames"], z["hist"]
    frames, hists = [], []
    for f in range(0, len(load_session_poses(sid)), FRAME_STEP):
        for attempt in range(3):
            try:
                h = frame_hist(sid, f, rec["fx"], rec["image_width"])
                break
            except (ConnectionError, TimeoutError, OSError):
                time.sleep(2 ** attempt)
                h = None
        if h is not None:
            frames.append(f)
            hists.append(h)
    frames, hists = np.array(frames, np.int32), np.array(hists, np.int32).reshape(-1, N_CLASSES, 8)
    np.savez_compressed(path, frames=frames, hist=hists)
    return frames, hists


def score_session(frames, hists, alert_m, min_pixels, confirm) -> dict:
    t = frames / SOURCE_FPS
    alerter = PathObstacleAlerter(alert_m, min_pixels, confirm, CLEAR_S)
    alerts = [a.t for ti, h in zip(t, hists) if (a := alerter.step(ti, pixels_within(h, SANPO_OBSTACLE_IDS, alert_m)))]
    enc, gap = [], ENCOUNTER_GAP
    for ti, h in zip(t, hists):
        if pixels_within(h, SANPO_OBSTACLE_IDS, ENCOUNTER_M) >= ENCOUNTER_PIXELS:
            if gap >= ENCOUNTER_GAP:
                close = h[list(SANPO_OBSTACLE_IDS), :2].sum(axis=1)       # bins below 1.0 m
                enc.append((ti, NAMES[SANPO_OBSTACLE_IDS[int(np.argmax(close))]]))
            gap = 0
        else:
            gap += 1
    leads = []
    for te, _ in enc:
        before = [a for a in alerts if 0.0 <= te - a <= LOOKBACK_S]
        leads.append(te - max(before) if before else None)
    followed = [any(0.0 <= te - a <= LOOKBACK_S for te, _ in enc) for a in alerts]
    minutes = (t[-1] - t[0]) / 60 if len(t) > 1 else 0.0
    return {"minutes": minutes, "alerts": alerts, "encounters": enc, "leads": leads, "followed": followed}


def summary(sc: dict[str, dict]) -> dict:
    s = list(sc.values())
    minutes = sum(x["minutes"] for x in s)
    leads = [l for x in s for l in x["leads"]]
    fol = [f for x in s for f in x["followed"]]
    n = len(leads)
    return {"sessions": len(s), "minutes": round(minutes, 1), "alerts": len(fol),
            "alerts_per_min": len(fol) / minutes if minutes else float("nan"), "encounters": n,
            "warned": sum(l is not None for l in leads) / n if n else float("nan"),
            "warned_1s": sum(l is not None and l >= 1.0 for l in leads) / n if n else float("nan"),
            "followed": sum(fol) / len(fol) if fol else float("nan"),
            "median_lead_s": float(np.median([l for l in leads if l is not None])) if any(l is not None for l in leads) else float("nan")}


def with_ci(sc: dict[str, dict]) -> dict:
    point = summary(sc)
    rng = np.random.default_rng(0)
    keys = sorted(sc)
    boots = [summary({f"{i}_{k}": sc[k] for i, k in enumerate(rng.choice(keys, len(keys)))}) for _ in range(N_BOOT)]
    for m in ("alerts_per_min", "warned", "warned_1s", "followed"):
        v = np.array([b[m] for b in boots], float)
        v = v[np.isfinite(v)]
        point[m + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else None
    by = {}
    for x in sc.values():
        for (_, cls), l in zip(x["encounters"], x["leads"]):
            b = by.setdefault(cls, [0, 0])
            b[0] += 1
            b[1] += l is not None
    point["by_class"] = {k: {"encounters": v[0], "warned": v[1]} for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])}
    return point


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}
    ids = sorted(s for s, r in audit.items() if r["segmentation_masks"])
    CACHE.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tracks = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (sid, tr) in enumerate(zip(ids, pool.map(lambda s: session_track(s, audit[s]), ids)), 1):
            tracks[sid] = tr
            print(f"  {i}/{len(ids)} {sid} {len(tr[0])} frames {time.time() - started:.0f} s", flush=True)
    usable = [s for s in ids if len(tracks[s][0]) > 1]
    train = [s for s in usable if audit[s]["split"] == "train"]
    test = [s for s in usable if audit[s]["split"] == "test"]

    def score(group, a, n, c):
        return {s: score_session(*tracks[s], a, n, c) for s in group}

    from src.perception_stack import path_obstacle as po
    choice = {"alert_m": po.ALERT_M, "min_pixels": po.MIN_PIXELS, "confirm_frames": po.CONFIRM_FRAMES}
    grid = []
    if args.sweep:
        for a in SWEEP_ALERT_M:
            for n in SWEEP_MIN_PIXELS:
                for c in SWEEP_CONFIRM:
                    r = summary(score(train, a, n, c))
                    grid.append({"alert_m": a, "min_pixels": n, "confirm_frames": c, **r})
        ok = [r for r in grid if r["alerts_per_min"] <= SWEEP_MAX_ALERTS_PER_MIN]
        best = max(ok, key=lambda r: (r["warned_1s"], r["warned"], -r["alerts_per_min"]))
        choice = {k: best[k] for k in ("alert_m", "min_pixels", "confirm_frames")}
        print("chosen on train:", best, flush=True)
    res = {"choice": choice, "declared_default": {"alert_m": po.ALERT_M, "min_pixels": po.MIN_PIXELS,
                                                  "confirm_frames": po.CONFIRM_FRAMES},
           "train": with_ci(score(train, *choice.values())), "test": with_ci(score(test, *choice.values())),
           "sweep_train": grid}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "oracle_gt.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for split in ("train", "test"):
        print(split, json.dumps(res[split]), flush=True)


if __name__ == "__main__":
    main()
