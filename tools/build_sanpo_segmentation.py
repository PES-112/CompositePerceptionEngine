"""
build_sanpo_segmentation.py
===========================
Builds the walkable-surface training set (docs/architecture.md §6.3) from
SANPO-Real's panoptic masks, in the Ultralytics semantic-segmentation layout:

    <out>/images/{train,val,test}/<session>_<camera>_<frame>.jpg
    <out>/masks/{train,val,test}/<session>_<camera>_<frame>.png   raw SANPO class ids (R channel)
    <out>/cpe_walkable.yaml        label_mapping SANPO -> CPE (src/segmentation/labels.py)
    <out>/manifest.json            sessions, frames and human/propagated counts per split

Masks are stored as SANPO ids, not CPE ids, so the taxonomy can change by
rewriting the YAML alone.

Which data:
  sessions   every SANPO-Real session with masks on either camera (--scope all, default; the
             paper reports 237) or only the 462 curated replay sessions (--scope curated).
             Masks are on the chest camera (ZED 2i) for most sessions and the head camera
             (ZED-M) for some; both are used, and the camera is in each file name.
  split      SANPO's official train/test. Validation = the curated sessions in the GRU
             dataset's val split, plus a seeded --val-frac of the other train sessions. So no
             replay val/test session is ever trained on.
  frames     val/test: HUMAN_ANNOTATED frames only (every 6th frame), as the paper scores.
             train: human frames plus every --machine-stride-th propagated frame (0 = none).

Downloads are ~3.5 MB per source frame; images are resized to --width (default 1024, aspect
kept) and written as JPEG, masks with nearest-neighbour as PNG. Run on the lab server:

    python tools/build_sanpo_segmentation.py --out data/sanpo_walkable_seg --workers 32
    python tools/build_sanpo_segmentation.py --out /tmp/seg_smoke --max-sessions 2 --max-frames 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.segmentation.labels import CPE_CLASSES, read_sanpo_ids, ultralytics_label_mapping
from tools.audit_sanpo_sessions import ANNOTATION_TYPES, BUCKET, fetch, list_dirs, load_official_split, object_url
from tools.download_sanpo_valid_streams import BASE_PREFIX

AUDIT = PROJECT_ROOT / "simulation/datasets/sanpo/session_audit.json"
GRU_SPLIT_MANIFEST = PROJECT_ROOT / "dataset/GRU/sanpo_real_462_tracked_d3_h8_current_m32_official/manifest.json"
CAMERAS = ("chest", "head")
VIEW = "left"


def all_sessions() -> list[str]:
    out, token = [], None
    while True:
        params = {"prefix": f"{BASE_PREFIX}/", "delimiter": "/", "fields": "prefixes,nextPageToken"}
        if token:
            params["pageToken"] = token
        data = json.loads(fetch(f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(params)))
        out += [p[len(BASE_PREFIX) + 1:].rstrip("/") for p in data.get("prefixes", [])]
        token = data.get("nextPageToken")
        if not token:
            return sorted(s for s in out if s != "splits")


def masked_streams(sid: str) -> list[dict]:
    """[{session, camera, human: [...], machine: [...]}] for each camera of the session that has masks."""
    out = []
    for cam in CAMERAS:
        dirs, files = list_dirs(f"{BASE_PREFIX}/{sid}/camera_{cam}/{VIEW}/")
        if "segmentation_masks" not in dirs or ANNOTATION_TYPES not in files:
            continue
        kinds = json.loads(fetch(object_url(f"{BASE_PREFIX}/{sid}/camera_{cam}/{VIEW}/{ANNOTATION_TYPES}")))
        out.append({"session": sid, "camera": cam,
                    "human": sorted(int(k) for k, v in kinds.items() if v == "HUMAN_ANNOTATED"),
                    "machine": sorted(int(k) for k, v in kinds.items() if v == "MACHINE_ANNOTATED")})
    return out


def assign_splits(sessions: list[str], official: dict[str, set[str]], val_frac: float, seed: int) -> dict[str, str]:
    gru = json.loads(GRU_SPLIT_MANIFEST.read_text(encoding="utf-8"))["split"]
    replay_val, replay_test = set(gru["val"]), set(gru["test"])
    rng = random.Random(seed)
    out = {}
    for sid in sessions:
        if sid in official["test"] or sid in replay_test:
            out[sid] = "test"
        elif sid in replay_val:
            out[sid] = "val"
        elif sid in official["train"]:
            out[sid] = "val" if rng.random() < val_frac else "train"
    return out


def pick_frames(stream: dict, split: str, machine_stride: int, max_frames: int) -> list[tuple[int, str]]:
    frames = [(f, "human") for f in stream["human"]]
    if split == "train" and machine_stride > 0:
        frames += [(f, "machine") for f in stream["machine"][machine_stride // 2::machine_stride]]
    frames.sort()
    if max_frames:
        frames = frames[:: max(1, len(frames) // max_frames)][:max_frames]
    return frames


def write_frame(stream: dict, frame: int, split: str, out: Path, width: int) -> bool:
    import cv2

    stem = f"{stream['session']}_{stream['camera']}_{frame:06d}"
    img_path, mask_path = out / "images" / split / f"{stem}.jpg", out / "masks" / split / f"{stem}.png"
    if img_path.exists() and mask_path.exists():
        return False
    base = f"{BASE_PREFIX}/{stream['session']}/camera_{stream['camera']}/{VIEW}"
    img = cv2.imdecode(np.frombuffer(fetch(object_url(f"{base}/video_frames/{frame:06d}.png")), np.uint8), cv2.IMREAD_COLOR)
    ids = read_sanpo_ids(fetch(object_url(f"{base}/segmentation_masks/{frame:06d}.png")))
    h = round(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)
    ids = cv2.resize(ids, (width, h), interpolation=cv2.INTER_NEAREST)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(mask_path), ids)
    return True


def write_yaml(out: Path) -> Path:
    lines = [
        "# CPE walkable-surface segmentation (tools/build_sanpo_segmentation.py). Masks hold raw SANPO",
        "# class ids; label_mapping remaps them to the CPE scheme in src/segmentation/labels.py.",
        f"path: {out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "masks_dir: masks",
        "names:",
        *[f"  {i}: {name}" for i, name in CPE_CLASSES.items()],
        "label_mapping:",
        *[f"  {k}: {v}" for k, v in ultralytics_label_mapping().items()],
    ]
    path = out / "cpe_walkable.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/sanpo_walkable_seg")
    p.add_argument("--scope", choices=["all", "curated"], default="all")
    p.add_argument("--width", type=int, default=1024, help="Stored image width, px (aspect kept).")
    p.add_argument("--machine-stride", type=int, default=6,
                   help="Train only: also take every Nth propagated frame (0 = human frames only).")
    p.add_argument("--val-frac", type=float, default=0.15, help="Share of non-curated train sessions held out for val.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-sessions", type=int, default=0, help="0 = all (smoke testing).")
    p.add_argument("--max-frames", type=int, default=0, help="Per stream; 0 = all.")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--list-only", action="store_true", help="Write the manifest without downloading frames.")
    args = p.parse_args()

    started = time.time()
    official = load_official_split()
    if args.scope == "curated":
        sessions = [r["session_id"] for r in json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]]
    else:
        sessions = all_sessions()
    splits = assign_splits(sessions, official, args.val_frac, args.seed)
    sessions = [s for s in sessions if s in splits]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        streams = [st for found in pool.map(masked_streams, sessions) for st in found]
    if args.max_sessions:
        keep = sorted({st["session"] for st in streams})[: args.max_sessions]
        streams = [st for st in streams if st["session"] in keep]
    print(f"{len(sessions)} sessions listed, {len(streams)} masked streams "
          f"({len({st['session'] for st in streams})} sessions)  {time.time() - started:.0f} s", flush=True)

    jobs, stats = [], {}
    for st in streams:
        split = splits[st["session"]]
        frames = pick_frames(st, split, args.machine_stride, args.max_frames)
        jobs += [(st, f, split) for f, _ in frames]
        s = stats.setdefault(split, {"sessions": set(), "streams": Counter(), "frames": Counter()})
        s["sessions"].add(st["session"])
        s["streams"][st["camera"]] += 1
        s["frames"].update(kind for _, kind in frames)

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.list_only:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, _ in enumerate(pool.map(lambda j: write_frame(j[0], j[1], j[2], args.out, args.width), jobs), 1):
                if i % 500 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} frames  {time.time() - started:.0f} s", flush=True)
    yaml_path = write_yaml(args.out)
    manifest = {
        "created": time.strftime("%Y-%m-%d"), "scope": args.scope, "width": args.width,
        "machine_stride": args.machine_stride, "val_frac": args.val_frac, "seed": args.seed,
        "label_scheme": CPE_CLASSES, "yaml": yaml_path.name,
        "splits": {k: {"sessions": sorted(v["sessions"]), "streams_by_camera": dict(v["streams"]),
                       "frames": dict(v["frames"])} for k, v in sorted(stats.items())},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for k, v in sorted(stats.items()):
        print(f"{k:5s}: {len(v['sessions'])} sessions, streams {dict(v['streams'])}, frames {dict(v['frames'])}")
    print(f"-> {args.out}  ({yaml_path.name})")


if __name__ == "__main__":
    main()
