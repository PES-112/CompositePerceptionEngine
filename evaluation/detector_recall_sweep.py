"""Can poles be recovered without retraining? Detector recall on SANPO vs inference size and pole confidence.

Same objects and matching rule as evaluation/detector_recall_sanpo.py (on v2-segmentation, methodology
§4.21): an object is a connected component of a SANPO class covering >= 0.3% of the image with median
depth <= 10 m; it is found if a v3 box of a matching class covers >= 50% of its bounding box.

Two inference-time knobs, no retraining:
  A. imgsz: the detector's input size (Stage 1 uses Ultralytics' default 640 on 2208x1242 frames).
  B. pole_conf: a lower confidence threshold for pole-type boxes (pole, bollard) only; others keep 0.30.

Selection set: HUMAN_ANNOTATED head-camera frames of official-train sessions (the detector never saw
SANPO). Rule, fixed before looking: the highest pole recall with person recall >= baseline - 0.05 and
pole-box support >= 0.5; ties go to the smaller imgsz, then the higher conf. Test (10 sessions, the
§4.21 frames) is scored once for the baseline, A alone, B alone and A+B.

Pole-box support (a precision proxy; SANPO has no box labels): the share of predicted pole/bollard
boxes in which >= 20% of the pixels are a thin vertical SANPO class (pole, traffic sign, traffic
light, bike rack, obstacle, bus stop, rails).

    python evaluation/detector_recall_sweep.py --split train   # download + select (cached)
    python evaluation/detector_recall_sweep.py --split test    # score once with the chosen setting
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.perception_stack.depth_loader import decode_sanpo_depth  # noqa: E402
from tools.audit_sanpo_sessions import ANNOTATION_TYPES, fetch, object_url  # noqa: E402
from tools.download_sanpo_valid_streams import BASE_PREFIX  # noqa: E402

SANPO = {12: "pedestrian", 21: "vehicle", 24: "pole", 23: "traffic light", 26: "bike rack", 20: "obstacle"}
MATCH = {"pedestrian": {"person"}, "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle"},
         "pole": {"pole", "bollard", "traffic light", "stop sign"}, "traffic light": {"traffic light", "pole"},
         "bike rack": {"bicycle", "bollard", "pole", "bench"}, "obstacle": set()}
MIN_AREA, MAX_DEPTH_M, BASE_CONF = 0.003, 10.0, 0.30
POLE_TYPE = {"pole", "bollard"}
THIN_VERTICAL = np.zeros(256, bool)
THIN_VERTICAL[[4, 9, 20, 22, 23, 24, 25, 26]] = True
IMGSZ = (640, 960, 1280)
POLE_CONF = (0.30, 0.20, 0.15, 0.10)
FLOOR_CONF = 0.05  # predict once at this floor, then apply each threshold offline
MODEL = ROOT / "training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt"
CACHE = ROOT / "data/detector_recall"
OUT = ROOT / "evaluation/benchmarks/detector_recall_sanpo"


def sessions(split: str) -> list[str]:
    audit = json.loads((ROOT / "simulation/datasets/sanpo/session_audit.json").read_text(encoding="utf-8"))["sessions"]
    return sorted(r["session_id"] for r in audit if r["split"] == split and r["segmentation_masks"])


def frame_list(sid: str, per_session: int) -> list[int]:
    base = f"{BASE_PREFIX}/{sid}/camera_head/left"
    kinds = json.loads(fetch(object_url(f"{base}/{ANNOTATION_TYPES}")))
    human = sorted(int(k) for k, v in kinds.items() if v == "HUMAN_ANNOTATED")
    return human[:: max(1, len(human) // per_session)][:per_session]


def download(sid: str, f: int, split: str, tries: int = 4) -> Path:
    for attempt in range(tries):  # fetch() retries HTTP/URL errors but not a reset connection
        try:
            return _download(sid, f, split)
        except (ConnectionError, TimeoutError, OSError):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def _download(sid: str, f: int, split: str) -> Path:
    d = CACHE / split / f"{sid}_{f:06d}"
    if (d / "done").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    base = f"{BASE_PREFIX}/{sid}/camera_head/left"
    (d / "frame.png").write_bytes(fetch(object_url(f"{base}/video_frames/{f:06d}.png")))
    (d / "mask.png").write_bytes(fetch(object_url(f"{base}/segmentation_masks/{f:06d}.png")))
    (d / "depth.float16.gz").write_bytes(fetch(object_url(f"{base}/depth_maps/{f:06d}.float16.gz")))
    (d / "done").touch()
    return d


def predict_all(dirs: list[Path]) -> None:
    """Boxes at FLOOR_CONF for every imgsz, cached per frame as JSON."""
    from ultralytics import YOLO

    model = YOLO(str(MODEL))
    for i, d in enumerate(dirs):
        out = d / "boxes.json"
        have = json.loads(out.read_text()) if out.exists() else {}
        todo = [s for s in IMGSZ if str(s) not in have]
        if not todo:
            continue
        img = cv2.imread(str(d / "frame.png"), cv2.IMREAD_COLOR)
        for s in todo:
            r = model.predict(img, conf=FLOOR_CONF, imgsz=s, verbose=False)[0]
            have[str(s)] = [[r.names[int(c)], float(p), *map(float, b)]
                            for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
        out.write_text(json.dumps(have))
        if i % 20 == 0:
            print(f"  predicted {i + 1}/{len(dirs)}", flush=True)


def load_truth(d: Path):
    img_hw = cv2.imread(str(d / "frame.png"), cv2.IMREAD_REDUCED_COLOR_2).shape[:2]
    h, w = img_hw[0] * 2, img_hw[1] * 2
    m = cv2.imread(str(d / "mask.png"), cv2.IMREAD_UNCHANGED)
    ids = m if m.ndim == 2 else m[..., 2]
    depth = decode_sanpo_depth(np.frombuffer(gzip.decompress((d / "depth.float16.gz").read_bytes()), np.float16))
    ids = cv2.resize(ids, (w, h), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    objs = []
    for sid_cls, name in SANPO.items():
        n, lab, stats, _ = cv2.connectedComponentsWithStats((ids == sid_cls).astype(np.uint8), 8)
        for k in range(1, n):
            x, y, bw, bh, area = stats[k]
            if area < MIN_AREA * h * w:
                continue
            dd = float(np.median(depth[lab == k]))
            if 0.3 < dd <= MAX_DEPTH_M:
                objs.append((name, int(x), int(y), int(bw), int(bh)))
    return ids, objs


def keep(boxes, pole_conf):
    return [b for b in boxes if b[1] >= (pole_conf if b[0] in POLE_TYPE else BASE_CONF)]


def score(frames, imgsz, pole_conf):
    found, total = defaultdict(int), defaultdict(int)
    n_pole_boxes = supported = 0
    for ids, objs, boxes_by_size in frames:
        boxes = keep(boxes_by_size[str(imgsz)], pole_conf)
        for name, x, y, bw, bh in objs:
            total[name] += 1
            for cname, _, x1, y1, x2, y2 in boxes:
                ix = max(0, min(x + bw, x2) - max(x, x1)) * max(0, min(y + bh, y2) - max(y, y1))
                if cname in MATCH[name] and ix / (bw * bh) >= 0.5:
                    found[name] += 1
                    break
        for cname, _, x1, y1, x2, y2 in boxes:
            if cname not in POLE_TYPE:
                continue
            n_pole_boxes += 1
            patch = ids[int(y1):int(math.ceil(y2)), int(x1):int(math.ceil(x2))]
            supported += patch.size > 0 and THIN_VERTICAL[patch].mean() >= 0.2
    return found, total, n_pole_boxes, supported


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def row(label, frames, imgsz, pole_conf):
    f, t, nb, sup = score(frames, imgsz, pole_conf)
    r = {"setting": label, "imgsz": imgsz, "pole_conf": pole_conf, "frames": len(frames),
         "pole_boxes_per_frame": nb / len(frames), "pole_box_support": sup / nb if nb else float("nan")}
    for name in SANPO.values():
        if t[name]:
            r[name] = {"found": f[name], "total": t[name], "recall": f[name] / t[name],
                       "ci95": wilson(f[name], t[name])}
    return r


def fmt(r):
    def cell(name):
        c = r.get(name)
        return f"{c['found']}/{c['total']} = {c['recall']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]" if c else "-"
    return (f"| {r['setting']} | {r['imgsz']} | {r['pole_conf']:.2f} | {cell('pole')} | {cell('pedestrian')} | "
            f"{cell('vehicle')} | {r['pole_boxes_per_frame']:.2f} | {r['pole_box_support']:.2f} |")


HEADER = ("| setting | imgsz | pole conf | poles | pedestrians | vehicles | pole boxes/frame | pole-box support |\n"
          "|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--per-session", type=int, default=None, help="default: 4 (train), 8 (test, as in §4.21)")
    args = ap.parse_args()
    per = args.per_session or (8 if args.split == "test" else 4)
    sids = sessions(args.split)
    print(f"{args.split}: {len(sids)} sessions with head masks, {per} frames each", flush=True)

    jobs = [(sid, f) for sid in sids for f in frame_list(sid, per)]
    with ThreadPoolExecutor(8) as ex:
        dirs = list(ex.map(lambda j: download(j[0], j[1], args.split), jobs))
    print(f"  {len(dirs)} frames cached", flush=True)
    predict_all(dirs)

    frames = []
    for d in dirs:
        ids, objs = load_truth(d)
        frames.append((ids, objs, json.loads((d / "boxes.json").read_text())))

    OUT.mkdir(parents=True, exist_ok=True)
    if args.split == "train":
        grid = [row(f"{s}/{c:.2f}", frames, s, c) for s in IMGSZ for c in POLE_CONF]
        base = grid[0]
        ok = [r for r in grid if r["pedestrian"]["recall"] >= base["pedestrian"]["recall"] - 0.05
              and r["pole_box_support"] >= 0.5]
        best = max(ok, key=lambda r: (r["pole"]["recall"], -r["imgsz"], r["pole_conf"]))
        choice = {"imgsz": best["imgsz"], "pole_conf": best["pole_conf"]}
        (OUT / "selection_train.json").write_text(json.dumps({"grid": grid, "choice": choice}, indent=1))
        print(HEADER)
        for r in grid:
            print(fmt(r))
        print("chosen:", choice)
    else:
        sel = OUT / "selection_train.json"
        if not sel.exists():
            sys.exit("Run --split train first: the setting is chosen there.")
        c = json.loads(sel.read_text())["choice"]
        rows = [row("before (Stage 1 today)", frames, 640, BASE_CONF),
                row("A: imgsz only", frames, c["imgsz"], BASE_CONF),
                row("B: pole conf only", frames, 640, c["pole_conf"]),
                row("A+B (chosen on train)", frames, c["imgsz"], c["pole_conf"])]
        (OUT / "test.json").write_text(json.dumps({"choice": c, "rows": rows}, indent=1))
        print(HEADER)
        for r in rows:
            print(fmt(r))


if __name__ == "__main__":
    main()
