"""
Two fixes for "it calls the road the sidewalk / the crosswalk", tuned on SANPO validation and scored
once on test, at the level the narration uses: the walker's surface (context.walker_surface, the
majority ground class in the bottom-centre patch).

1. Road bias. The segmenter's logits (before its in-graph argmax) get +b on the road class. Calling
   road "walkable" is the dangerous error; calling walkable "road" is an extra "Careful."
2. Crosswalk stripe check. A crosswalk is painted: its pixels hold bright stripes on dark asphalt.
   If the walker patch is called crosswalk but its luminance spread (p90 - p10 over the crosswalk
   pixels, 0-255) is below s, it is plain road.

Declared selection rules (written before the validation run):
- b in {0, 0.25, ..., 4}: minimise walker-level road-called-safe (ground-truth road patches
  predicted walkable or crosswalk), subject to walkable-called-road <= 5 % of ground-truth walkable
  patches. Ties -> smaller b.
- s in {0, 5, ..., 120}, applied after the chosen b: the largest s that keeps >= 90 % of
  ground-truth crosswalk patches that were predicted crosswalk.

    .venv-seg.nosync/bin/python tools/tune_surface_rules.py --split val
    .venv-seg.nosync/bin/python tools/tune_surface_rules.py --split test --bias B --stripe S

Writes evaluation/benchmarks/segmentation/surface_rules_<split>/{metrics.json,report.md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.context import walker_surface
from src.segmentation.labels import read_sanpo_mask

WEIGHTS = PROJECT_ROOT / "training/runs/cpe_yolo26n_sem_walkable/weights/best.pt"
DATA = PROJECT_ROOT / "data/sanpo_walkable_seg"
OUT = PROJECT_ROOT / "evaluation/benchmarks/segmentation"
WALKABLE, CROSSWALK, ROAD = 0, 1, 2
SAFE = (WALKABLE, CROSSWALK)
BIASES = [i * 0.25 for i in range(17)]
STRIPES = list(range(0, 125, 5))
MAX_FALSE_ROAD = 0.05
KEEP_CROSSWALK = 0.90
SIZE = 640


def letterbox(img: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Square 640 letterbox, grey 114 padding, as the phone (Detector.letterbox)."""
    import cv2

    h, w = img.shape[:2]
    r = min(SIZE / w, SIZE / h)
    nw, nh = round(w * r), round(h * r)
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    out = np.full((SIZE, SIZE, 3), 114, np.uint8)
    out[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (px, py)


def logits_fn(device: str, weights: Path = WEIGHTS):
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    net = YOLO(str(weights)).model.to(device).eval()

    def run(img_bgr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Logits (8, H, W) at the mask's resolution."""
        lb, r, (px, py) = letterbox(img_bgr)
        x = torch.from_numpy(lb[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(device)
        with torch.no_grad():
            y = net(x)
            y = y[0] if isinstance(y, (list, tuple)) else y
            y = F.interpolate(y.float(), size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0]
        h, w = img_bgr.shape[:2]
        y = y[:, py:py + round(h * r), px:px + round(w * r)][None]
        return F.interpolate(y, size=shape, mode="bilinear", align_corners=False)[0].cpu().numpy()

    return run


def stripe_spread(gray: np.ndarray, class_map: np.ndarray) -> float | None:
    """p90 - p10 luminance over the crosswalk pixels of the walker patch (context.walker_surface's patch)."""
    h, w = class_map.shape
    sl = (slice(int(0.8 * h), h), slice(int(0.3 * w), int(0.7 * w)))
    px = gray[sl][class_map[sl] == CROSSWALK]
    return float(np.percentile(px, 90) - np.percentile(px, 10)) if px.size >= 5 else None


def main() -> None:
    import cv2

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--bias", type=float, default=None, help="test: the bias chosen on val")
    p.add_argument("--stripe", type=float, default=None, help="test: the stripe threshold chosen on val")
    p.add_argument("--device", default="mps")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--name", default="surface_rules", help="output folder prefix")
    args = p.parse_args()
    biases = BIASES if args.split == "val" else [0.0, args.bias]

    run = logits_fn(args.device, args.weights)
    images = sorted((args.data / "images" / args.split).glob("*.jpg"))
    if args.limit:
        images = images[:: max(1, len(images) // args.limit)]
    # walker-level confusion per bias: (truth, pred) counts, overall and per camera
    conf: dict[float, Counter] = {b: Counter() for b in biases}
    by_cam: dict[float, dict[str, Counter]] = {b: defaultdict(Counter) for b in biases}
    # crosswalk-predicted patches (at each bias): (truth surface, stripe spread)
    cross: dict[float, list[tuple[int, float | None]]] = {b: [] for b in biases}
    for i, path in enumerate(images, 1):
        _, camera, _ = path.stem.rsplit("_", 2)
        truth_map = read_sanpo_mask(args.data / "masks" / args.split / f"{path.stem}.png")
        truth = walker_surface(truth_map)
        if truth is None:
            continue
        img = cv2.imread(str(path))
        if img.shape[:2] != truth_map.shape:
            img = cv2.resize(img, truth_map.shape[::-1], interpolation=cv2.INTER_AREA)
        logits = run(img, truth_map.shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for b in biases:
            lg = logits.copy()
            lg[ROAD] += b
            pred_map = lg.argmax(0).astype(np.uint8)
            pred = walker_surface(pred_map)
            pred = -1 if pred is None else pred
            conf[b][(truth, pred)] += 1
            by_cam[b][camera][(truth, pred)] += 1
            if pred == CROSSWALK:
                cross[b].append((truth, stripe_spread(gray, pred_map)))
        if i % 500 == 0:
            print(f"  {i}/{len(images)}", flush=True)

    def rates(c: Counter) -> dict:
        road = sum(v for (t, _), v in c.items() if t == ROAD)
        walk = sum(v for (t, _), v in c.items() if t == WALKABLE)
        cw = sum(v for (t, _), v in c.items() if t == CROSSWALK)
        return {
            "road_patches": road, "walkable_patches": walk, "crosswalk_patches": cw,
            "road_called_safe": sum(v for (t, p), v in c.items() if t == ROAD and p in SAFE) / max(road, 1),
            "road_called_crosswalk": c[(ROAD, CROSSWALK)] / max(road, 1),
            "walkable_called_road": c[(WALKABLE, ROAD)] / max(walk, 1),
            "crosswalk_called_road": c[(CROSSWALK, ROAD)] / max(cw, 1),
            "agreement": sum(v for (t, p), v in c.items() if t == p) / max(sum(c.values()), 1),
        }

    def with_stripe(b: float, s: float) -> dict:
        """Rates after relabelling low-spread crosswalk patches as road."""
        c = Counter(conf[b])
        for truth, spread in cross[b]:
            if spread is not None and spread < s:
                c[(truth, CROSSWALK)] -= 1
                c[(truth, ROAD)] += 1
        return rates(c)

    table = {str(b): rates(conf[b]) for b in biases}
    cams = {str(b): {cam: rates(c) for cam, c in sorted(by_cam[b].items())} for b in biases}
    if args.split == "val":
        ok = [b for b in biases if table[str(b)]["walkable_called_road"] <= MAX_FALSE_ROAD]
        bias = min(ok, key=lambda b: (table[str(b)]["road_called_safe"], b))
        true_cw = [sp for t, sp in cross[bias] if t == CROSSWALK and sp is not None]
        keep = [s for s in STRIPES if np.mean([sp >= s for sp in true_cw]) >= KEEP_CROSSWALK] if true_cw else [0]
        stripe = float(max(keep)) if keep else 0.0
    else:
        bias, stripe = args.bias, args.stripe
    final = with_stripe(bias, stripe)
    stripes = {str(s): with_stripe(bias, s) for s in STRIPES} if args.split == "val" else {}
    result = {"split": args.split, "frames": len(images), "weights": str(args.weights),
              "data": str(args.data),
              "chosen_bias": bias, "chosen_stripe": stripe, "by_bias": table, "by_bias_camera": cams,
              "baseline": rates(conf[0.0]), "final": final, "by_stripe": stripes,
              "crosswalk_spreads": {"truth_crosswalk": [sp for t, sp in cross[bias] if t == CROSSWALK],
                                    "truth_road": [sp for t, sp in cross[bias] if t == ROAD]}}
    out = OUT / f"{args.name}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    keys = ["road_called_safe", "road_called_crosswalk", "walkable_called_road", "crosswalk_called_road", "agreement"]
    lines = [f"# Walker-surface rules, SANPO {args.split}", "",
             f"{len(images)} frames; walker patch = context.walker_surface. Chosen: road bias **{bias}**, "
             f"stripe threshold **{stripe}**.", "",
             "| | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for name, r in [("model as shipped", rates(conf[0.0])), (f"road bias {bias}", table[str(bias)]),
                    (f"+ stripe check {stripe}", final)]:
        lines.append(f"| {name} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    if args.split == "val":
        lines += ["", "Bias sweep:", "", "| b | " + " | ".join(keys) + " |", "|---:|" + "---:|" * len(keys)]
        for b in biases:
            lines.append(f"| {b} | " + " | ".join(f"{table[str(b)][k]:.3f}" for k in keys) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
