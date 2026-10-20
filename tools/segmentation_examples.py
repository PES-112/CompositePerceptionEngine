"""
segmentation_examples.py
========================
Paper figure F6: qualitative segmentation examples, image | SANPO mask | model mask,
for the walkable-surface model (YOLO26n-sem @640, paper §6.5).

Frame selection, declared before looking at any output:
  candidates  SANPO official-test sessions among the 462 curated ones that have masks on the
              chest or head camera (simulation/datasets/sanpo/session_audit.json: split "test"),
              sorted by session id; chest camera before head; within a camera, the first 10
              HUMAN_ANNOTATED frames in frame order (the frames the paper scores).
  targets     curb, stairs, road, crosswalk, in that order. A frame qualifies for a target if, in
              SANPO's mask, the target class covers >= 3% of pixels and walkable >= 3%.
  choice      for each target, the first qualifying candidate in that order; a frame is used
              for one target only. Every chosen frame is shown, whatever the model does on it.

    python tools/segmentation_examples.py                  # CPU
    python tools/segmentation_examples.py --device mps     # Apple silicon

Needs ultralytics and matplotlib (the .venv-seg environment). Writes
evaluation/benchmarks/figures/f6_segmentation_examples.png and a .json with the chosen frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.segmentation.labels import CPE_CLASSES, CROSSWALK, CURB, IGNORE, ROAD, STAIRS, WALKABLE, read_sanpo_mask
from tools.audit_sanpo_sessions import ANNOTATION_TYPES, fetch, object_url
from tools.download_sanpo_valid_streams import BASE_PREFIX

AUDIT = PROJECT_ROOT / "simulation/datasets/sanpo/session_audit.json"
WEIGHTS = PROJECT_ROOT / "training/runs/cpe_yolo26n_sem_walkable/weights/best.pt"
OUT = PROJECT_ROOT / "evaluation/benchmarks/figures/f6_segmentation_examples"
TARGETS = (CURB, STAIRS, ROAD, CROSSWALK)
MIN_SHARE = 0.03
FRAMES_PER_STREAM = 10
# One colour per CPE class (src/segmentation/labels.py); ignore (unlabeled) is black.
COLOURS = {0: (0.30, 0.75, 0.35), 1: (1.00, 0.85, 0.20), 2: (0.45, 0.45, 0.50), 3: (0.95, 0.35, 0.20),
           4: (0.70, 0.30, 0.85), 5: (0.55, 0.40, 0.20), 6: (0.20, 0.45, 0.90), 7: (0.65, 0.85, 1.00)}


def streams() -> list[tuple[str, str]]:
    rows = json.loads(AUDIT.read_text(encoding="utf-8"))["sessions"]
    out = []
    for r in sorted(rows, key=lambda r: r["session_id"]):
        if r["split"] != "test":
            continue
        if r["chest_segmentation_masks"]:
            out.append((r["session_id"], "chest"))
        if r["segmentation_masks"]:
            out.append((r["session_id"], "head"))
    return out


def base(sid: str, cam: str) -> str:
    return f"{BASE_PREFIX}/{sid}/camera_{cam}/left"


def human_frames(sid: str, cam: str) -> list[int]:
    kinds = json.loads(fetch(object_url(f"{base(sid, cam)}/{ANNOTATION_TYPES}")))
    return sorted(int(k) for k, v in kinds.items() if v == "HUMAN_ANNOTATED")[:FRAMES_PER_STREAM]


def select() -> list[dict]:
    chosen, used = {}, set()
    for sid, cam in streams():
        for f in human_frames(sid, cam):
            mask = read_sanpo_mask(fetch(object_url(f"{base(sid, cam)}/segmentation_masks/{f:06d}.png")))
            share = {c: float(np.mean(mask == c)) for c in (*TARGETS, WALKABLE)}
            for t in TARGETS:
                if t not in chosen and (sid, cam, f) not in used \
                        and share[t] >= MIN_SHARE and share[WALKABLE] >= MIN_SHARE:
                    chosen[t] = {"target": CPE_CLASSES[t], "session": sid, "camera": cam, "frame": f,
                                 "share_target": round(share[t], 3), "share_walkable": round(share[WALKABLE], 3)}
                    used.add((sid, cam, f))
                    print(f"  {CPE_CLASSES[t]}: {sid} {cam} {f}", flush=True)
            if len(chosen) == len(TARGETS):
                return [chosen[t] for t in TARGETS]
    print(f"not found: {[CPE_CLASSES[t] for t in TARGETS if t not in chosen]}")
    return [chosen[t] for t in TARGETS if t in chosen]


def colourise(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3))
    for c, col in COLOURS.items():
        rgb[mask == c] = col
    return rgb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from ultralytics import YOLO

    print("selecting frames (declared rule, see docstring)...")
    picks = select()
    model = YOLO(str(args.weights))
    fig, axes = plt.subplots(len(picks), 3, figsize=(12, 2.6 * len(picks)), squeeze=False)
    for row, pick in zip(axes, picks):
        b = base(pick["session"], pick["camera"])
        img = cv2.imdecode(np.frombuffer(fetch(object_url(f"{b}/video_frames/{pick['frame']:06d}.png")), np.uint8),
                           cv2.IMREAD_COLOR)
        gt = read_sanpo_mask(fetch(object_url(f"{b}/segmentation_masks/{pick['frame']:06d}.png")))
        r = model.predict(img, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        pred = cv2.resize(r.semantic_mask.data.cpu().numpy().astype(np.uint8), (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        labelled = gt != IGNORE
        pick["pixel_agreement"] = round(float(np.mean(pred[labelled] == gt[labelled])), 3)
        for ax, panel, title in zip(row, (img[..., ::-1] / 255.0, colourise(gt), colourise(pred)),
                                    (f"{pick['target']}: {pick['session'][:8]} {pick['camera']} #{pick['frame']}",
                                     "SANPO mask", f"model (agrees on {pick['pixel_agreement']:.0%} of labelled px)")):
            ax.imshow(panel)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.legend(handles=[Patch(color=COLOURS[c], label=n) for c, n in CPE_CLASSES.items()]
               + [Patch(color="black", label="unlabeled")], loc="lower center", ncol=9, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=130)
    OUT.with_suffix(".json").write_text(json.dumps({"rule": "see tools/segmentation_examples.py docstring",
                                                    "weights": str(args.weights.relative_to(PROJECT_ROOT)),
                                                    "imgsz": args.imgsz, "device": args.device,
                                                    "frames": picks}, indent=2), encoding="utf-8")
    print(f"wrote {OUT.with_suffix('.png').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
