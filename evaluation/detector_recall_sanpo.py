"""Detector recall on SANPO itself (run with .venv-seg/Scripts/python; ~5 min on CPU): SANPO-mask objects (connected components of a class) vs v3 boxes.

Frames: HUMAN_ANNOTATED frames of the 10 head-masked official-test sessions (head camera, the
stream Stage 1 runs on). An object counts if its component is >= MIN_AREA of the image and within
MAX_DEPTH_M (median SANPO depth). It is found if a v3 box of a matching class (conf >= 0.30, the
pipeline's threshold) covers >= 50% of the component's bounding box.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO  # noqa: E402

from src.perception_stack.depth_loader import decode_sanpo_depth  # noqa: E402
from src.segmentation.labels import read_sanpo_ids  # noqa: E402
from tools.audit_sanpo_sessions import ANNOTATION_TYPES, fetch, object_url  # noqa: E402
from tools.download_sanpo_valid_streams import BASE_PREFIX  # noqa: E402

SANPO = {12: "pedestrian", 21: "vehicle", 24: "pole", 23: "traffic light", 26: "bike rack", 20: "obstacle"}
MATCH = {"pedestrian": {"person"}, "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle"},
         "pole": {"pole", "bollard", "traffic light", "stop sign"}, "traffic light": {"traffic light", "pole"},
         "bike rack": {"bicycle", "bollard", "pole", "bench"}, "obstacle": set()}
MIN_AREA, MAX_DEPTH_M, PER_SESSION = 0.003, 10.0, 8

audit = json.loads((ROOT / "simulation/datasets/sanpo/session_audit.json").read_text(encoding="utf-8"))["sessions"]
sessions = sorted(r["session_id"] for r in audit if r["split"] == "test" and r["segmentation_masks"])
model = YOLO(str(ROOT / "training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt"))
found, total = defaultdict(int), defaultdict(int)
anything = defaultdict(int)
frames = 0
for sid in sessions:
    base = f"{BASE_PREFIX}/{sid}/camera_head/left"
    kinds = json.loads(fetch(object_url(f"{base}/{ANNOTATION_TYPES}")))
    human = sorted(int(k) for k, v in kinds.items() if v == "HUMAN_ANNOTATED")
    for f in human[:: max(1, len(human) // PER_SESSION)][:PER_SESSION]:
        img = cv2.imdecode(np.frombuffer(fetch(object_url(f"{base}/video_frames/{f:06d}.png")), np.uint8), cv2.IMREAD_COLOR)
        ids = read_sanpo_ids(fetch(object_url(f"{base}/segmentation_masks/{f:06d}.png")))
        depth = decode_sanpo_depth(np.frombuffer(gzip.decompress(fetch(object_url(f"{base}/depth_maps/{f:06d}.float16.gz"))), np.float16))
        h, w = img.shape[:2]
        ids = cv2.resize(ids, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        r = model.predict(img, conf=0.30, verbose=False)[0]
        boxes = [(r.names[int(c)], *map(float, b)) for c, b in zip(r.boxes.cls, r.boxes.xyxy)]
        frames += 1
        for sid_cls, name in SANPO.items():
            n, lab, stats, _ = cv2.connectedComponentsWithStats((ids == sid_cls).astype(np.uint8), 8)
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area < MIN_AREA * h * w:
                    continue
                d = float(np.median(depth[lab == i]))
                if not (0.3 < d <= MAX_DEPTH_M):
                    continue
                total[name] += 1
                cover_any = 0.0
                hit = False
                for cname, x1, y1, x2, y2 in boxes:
                    ix = max(0, min(x + bw, x2) - max(x, x1)) * max(0, min(y + bh, y2) - max(y, y1))
                    cov = ix / (bw * bh)
                    cover_any = max(cover_any, cov)
                    if cname in MATCH[name] and cov >= 0.5:
                        hit = True
                found[name] += hit
                anything[name] += cover_any >= 0.5
    print(sid, frames, dict(total), flush=True)

print(f"\n{frames} frames, {len(sessions)} sessions (head camera, official test)")
for name in SANPO.values():
    if total[name]:
        print(f"{name:14s} objects {total[name]:4d}  found (matching class) {found[name] / total[name]:.2f}"
              f"  covered by any box {anything[name] / total[name]:.2f}")
