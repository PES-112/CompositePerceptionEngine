"""
build_scene_context.py
======================
Per-frame scene context (src/segmentation/context.py) for replay sessions, so
tools/replay_threat_engine.py --context-dir can run its context-on arms.

For each processed frame k of a session (source frame 3k on camera_head/left):
the class map comes from SANPO's own masks (--source gt: an oracle, the upper
bound on what segmentation can add) or from a trained model run on the video
frame (--source <weights>.pt; needs ultralytics, i.e. the lab server). The
ground under each tracked object is found from the row's bearing and distance
plus SANPO's depth map, so the GRU dataset is enough; no Stage-1 CSV is needed.

Only sessions whose head camera has masks can use --source gt: 53 of the 462
curated (32 train / 5 val / 10 test in the GRU split). The chest camera has
masks for 130 more, but the tracks are on the head camera and the two cameras'
poses are in separate world frames, so chest masks can't be used without
re-running Stage 1 on the chest stream.

    python tools/build_scene_context.py --source gt
    python tools/build_scene_context.py --source training/runs/cpe_yolo26s_sem_walkable/weights/best.pt --all-sessions

Writes data/sanpo_scene_context/<source name>/<session>.json:
    {"frames": {k: {"walker": cls|null, "edge": [cls, m]|null,
                    "tracks": {track_id: [surface|null, in_path, near_boundary, other_surface]}}}}
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

import numpy as np

from dataset.GRU.samples import load_dataset
from src.perception_stack.depth_loader import _infer_sanpo_shape
from src.segmentation.context import ObjectRef, scene_frame
from src.segmentation.labels import read_sanpo_mask
from tools.audit_sanpo_sessions import fetch, object_url
from tools.download_sanpo_valid_streams import BASE_PREFIX

DEFAULT_DATASET = PROJECT_ROOT / "dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official_corridor_w0.75_d3_h20"
AUDIT = PROJECT_ROOT / "simulation/datasets/sanpo/session_audit.json"
OUT_ROOT = PROJECT_ROOT / "data/sanpo_scene_context"
STRIDE = 3
LEGACY_HALF_FOV = 35.0      # Stage-1 CSVs before 2026-10-14 mapped the image linearly onto ±35°: exact pixel column
DOWNSAMPLE = 4              # class maps are scored at 1/4 resolution (552 x 310 for 2208 x 1242)
MIN_HALF_WIDTH = 0.005


def depth_map(sid: str, frame: int) -> np.ndarray:
    raw = np.frombuffer(gzip.decompress(fetch(object_url(
        f"{BASE_PREFIX}/{sid}/camera_head/left/depth_maps/{frame:06d}.float16.gz"))), dtype=np.float16)
    h, w = _infer_sanpo_shape(raw)
    return raw[: h * w].reshape(h, w).astype(np.float32)


def class_map_source(source: str):
    if source == "gt":
        def gt(sid: str, frame: int) -> np.ndarray | None:
            try:
                return read_sanpo_mask(fetch(object_url(
                    f"{BASE_PREFIX}/{sid}/camera_head/left/segmentation_masks/{frame:06d}.png")))
            except Exception:
                return None
        return gt
    import cv2
    from ultralytics import YOLO

    model = YOLO(source)

    def predicted(sid: str, frame: int) -> np.ndarray | None:
        img = cv2.imdecode(np.frombuffer(fetch(object_url(
            f"{BASE_PREFIX}/{sid}/camera_head/left/video_frames/{frame:06d}.png")), np.uint8), cv2.IMREAD_COLOR)
        r = model.predict(img, imgsz=1024, verbose=False)[0]
        m = r.semantic_mask.data.cpu().numpy().astype(np.uint8)
        return cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return predicted


def objects(sample: dict, width: float) -> list[ObjectRef]:
    out = []
    for j, tid in enumerate(sample["track_ids"]):
        area = float(np.expm1(sample["x"][j, -1][7] * 15))         # log_area feature -> box area, px²
        out.append(ObjectRef(tid, 0.5 + float(sample["bearing"][j]) / LEGACY_HALF_FOV * 0.5,
                             float(sample["distance"][j]), max(MIN_HALF_WIDTH, 0.5 * area ** 0.5 / width)))
    return out


def session_context(sid: str, samples: list[dict], width: float, get_class_map) -> dict:
    frames = {}
    for s in sorted(samples, key=lambda s: s["frame_idx"]):
        k = s["frame_idx"]
        cls = get_class_map(sid, STRIDE * k)
        if cls is None:
            continue
        f = scene_frame(cls[::DOWNSAMPLE, ::DOWNSAMPLE], depth_map(sid, STRIDE * k), objects(s, width))
        frames[str(k)] = {
            "walker": f.walker_surface,
            "edge": [f.edge.kind, round(f.edge.distance_m, 2)] if f.edge else None,
            "tracks": {tid: [f.surfaces[tid], c.in_path, c.near_boundary, c.other_surface]
                       for tid, c in f.contexts.items()},
        }
    return {"session": sid, "frames": frames}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--source", default="gt", help="'gt' (SANPO masks) or a YOLO26-sem checkpoint.")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--all-sessions", action="store_true",
                   help="Every session in --splits, not only those with head-camera masks (model sources only).")
    p.add_argument("--max-sessions", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    if args.source == "gt" and args.all_sessions:
        raise SystemExit("--source gt needs head-camera masks; drop --all-sessions")
    audit = {r["session_id"]: r for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]}
    _, splits = load_dataset(args.dataset, splits=tuple(args.splits))
    by: dict[str, list[dict]] = {}
    for part in splits.values():
        for s in part:
            by.setdefault(s["session"], []).append(s)
    ids = sorted(sid for sid in by if args.all_sessions or audit[sid]["segmentation_masks"])
    ids = ids[: args.max_sessions or None]
    name = "gt" if args.source == "gt" else Path(args.source).parents[1].name
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    get_class_map = class_map_source(args.source)
    print(f"{len(ids)} sessions -> {out}", flush=True)
    started = time.time()

    def one(sid: str) -> str:
        path = out / f"{sid}.json"
        if not path.exists():
            ctx = session_context(sid, by[sid], audit[sid]["image_width"], get_class_map)
            path.write_text(json.dumps(ctx), encoding="utf-8")
        return sid

    with ThreadPoolExecutor(max_workers=args.workers if args.source == "gt" else 1) as pool:
        for i, sid in enumerate(pool.map(one, ids), 1):
            print(f"  {i}/{len(ids)} {sid}  {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
