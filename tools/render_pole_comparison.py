"""Before/after videos for the pole fix: the v3 detector at Stage 1's setting vs the setting chosen in
evaluation/detector_recall_sweep.py, side by side on a SANPO head-camera clip.

Raw detector boxes per frame (no tracker, no size check), as in the recall measurement. Orange = pole or
bollard, green = person, grey = other classes. Cyan outline = SANPO's pole mask, on the frames SANPO
annotated (human or machine). The clips are picked for having poles, so they illustrate; the recall
table in evaluation/benchmarks/detector_recall_sanpo/ is the measurement.

    python tools/render_pole_comparison.py --pick 3              # the 3 test frames with the most poles
    python tools/render_pole_comparison.py --session SID --center 1234
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.detector_recall_sweep import (  # noqa: E402
    BASE_CONF, CACHE, MODEL, OUT, POLE_TYPE, load_truth)
from tools.audit_sanpo_sessions import ANNOTATION_TYPES, fetch, object_url  # noqa: E402
from tools.download_sanpo_valid_streams import BASE_PREFIX  # noqa: E402

FPS, PANEL_W = 15, 960
ORANGE, GREEN, GREY, CYAN = (0, 140, 255), (60, 200, 60), (170, 170, 170), (255, 255, 0)


def pick(n: int) -> list[tuple[str, int]]:
    """The test frames with the most SANPO poles within 10 m, one per session."""
    best = {}
    for d in sorted((CACHE / "test").iterdir()):
        if not (d / "done").exists():
            continue
        sid, f = d.name.rsplit("_", 1)
        k = sum(o[0] == "pole" for o in load_truth(d)[1])
        if k > best.get(sid, (0, 0))[0]:
            best[sid] = (k, int(f))
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:n]
    for sid, (k, f) in ranked:
        print(f"  {sid} frame {f}: {k} poles")
    return [(sid, f) for sid, (k, f) in ranked]


def clip_frames(sid: str, center: int, seconds: float) -> tuple[list[Path], dict]:
    base = f"{BASE_PREFIX}/{sid}/camera_head/left"
    kinds = {int(k): v for k, v in json.loads(fetch(object_url(f"{base}/{ANNOTATION_TYPES}"))).items()}
    half = int(seconds * FPS / 2)
    out = CACHE / "clips" / sid
    out.mkdir(parents=True, exist_ok=True)

    def get(f):
        p = out / f"{f:06d}.png"
        if not p.exists():
            try:
                p.write_bytes(fetch(object_url(f"{base}/video_frames/{f:06d}.png")))
            except Exception:
                return None
        if f in kinds and not (out / f"{f:06d}_mask.png").exists():
            try:
                (out / f"{f:06d}_mask.png").write_bytes(fetch(object_url(f"{base}/segmentation_masks/{f:06d}.png")))
            except Exception:
                pass
        return p

    with ThreadPoolExecutor(8) as ex:
        paths = [p for p in ex.map(get, range(max(0, center - half), center + half)) if p]
    return paths, kinds


def draw(img, boxes, pole_conf, mask_path, title):
    h, w = img.shape[:2]
    s = PANEL_W / w
    im = cv2.resize(img, (PANEL_W, int(h * s)), interpolation=cv2.INTER_AREA)
    if mask_path is not None and mask_path.exists():
        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        ids = m if m.ndim == 2 else m[..., 2]
        pole = cv2.resize((ids == 24).astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(pole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, CYAN, 1)
    n_pole = 0
    for name, conf, x1, y1, x2, y2 in boxes:
        if conf < (pole_conf if name in POLE_TYPE else BASE_CONF):
            continue
        col, t = (ORANGE, 3) if name in POLE_TYPE else (GREEN, 2) if name == "person" else (GREY, 1)
        n_pole += name in POLE_TYPE
        p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
        cv2.rectangle(im, p1, p2, col, t)
        cv2.putText(im, f"{name} {conf:.2f}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    bar = np.full((44, PANEL_W, 3), 25, np.uint8)
    cv2.putText(bar, f"{title}   pole/bollard boxes: {n_pole}", (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def render(sid, center, seconds, after, model, out_dir) -> Path:
    paths, kinds = clip_frames(sid, center, seconds)
    out = out_dir / f"poles_{sid[:8]}_{center:06d}.mp4"
    ff = None
    from imageio_ffmpeg import get_ffmpeg_exe
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        f = int(p.stem)
        mask = p.with_name(f"{f:06d}_mask.png") if f in kinds else None
        panels = []
        for (sz, pc), title in (((640, BASE_CONF), "BEFORE  640 px, conf 0.30"),
                                (after, f"AFTER  {after[0]} px, pole conf {after[1]:.2f}")):
            r = model.predict(img, conf=min(pc, BASE_CONF), imgsz=sz, verbose=False)[0]
            boxes = [(r.names[int(c)], float(q), *map(float, b)) for c, q, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
            panels.append(draw(img, boxes, pc, mask, title))
        frame = np.hstack(panels)
        if ff is None:
            H, W = frame.shape[:2]
            ff = subprocess.Popen([get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                   "-crf", "23", str(out)], stdin=subprocess.PIPE)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"  wrote {out} ({len(paths)} frames)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--session")
    ap.add_argument("--center", type=int)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=ROOT / "data/demo/poles")
    args = ap.parse_args()
    c = json.loads((OUT / "selection_train.json").read_text())["choice"]
    after = (c["imgsz"], c["pole_conf"])
    clips = pick(args.pick) if args.pick else [(args.session, args.center)]
    args.out.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(MODEL))
    for sid, center in clips:
        render(sid, center, args.seconds, after, model, args.out)


if __name__ == "__main__":
    main()
