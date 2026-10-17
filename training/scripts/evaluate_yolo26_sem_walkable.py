"""
Score a walkable-surface checkpoint on the SANPO segmentation split built by
tools/build_sanpo_segmentation.py (val/test hold HUMAN_ANNOTATED frames only).

Reports evaluation/segmentation_metrics.SegmentationScore (mIoU, per-class IoU,
false-safe rate, road-as-walkable, curb/stairs boundary F), overall and per
camera (head ZED-M vs chest ZED 2i), with a session-bootstrap 95% CI on mIoU
and the false-safe rate.

    python training/scripts/evaluate_yolo26_sem_walkable.py \
        --weights training/runs/cpe_yolo26s_sem_walkable/weights/best.pt \
        --data data/sanpo_walkable_seg --split test

Writes evaluation/benchmarks/segmentation/<run>_<split>/{metrics.json,report.md}.
Pass --weights gt to check the pipeline end to end (scores the labels against themselves).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.segmentation_metrics import SegmentationScore
from src.segmentation.labels import read_sanpo_mask

N_BOOT = 1000


def predictor(weights: str, imgsz: int, device: str):
    if weights == "gt":
        return None
    from ultralytics import YOLO

    model = YOLO(weights)

    def predict(path: Path) -> np.ndarray:
        r = model.predict(str(path), imgsz=imgsz, device=device, verbose=False)[0]
        return r.semantic_mask.data.cpu().numpy().astype(np.uint8)

    return predict


def main() -> None:
    import cv2

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="Checkpoint, or 'gt' for a self-check.")
    p.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/sanpo_walkable_seg")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    predict = predictor(args.weights, args.imgsz, args.device)
    per_session: dict[str, SegmentationScore] = defaultdict(SegmentationScore)
    by_camera: dict[str, SegmentationScore] = defaultdict(SegmentationScore)
    overall = SegmentationScore()
    images = sorted((args.data / "images" / args.split).glob("*.jpg"))
    for i, img in enumerate(images, 1):
        session, camera, _ = img.stem.rsplit("_", 2)
        truth = read_sanpo_mask(args.data / "masks" / args.split / f"{img.stem}.png")
        pred = truth if predict is None else predict(img)
        if pred.shape != truth.shape:
            pred = cv2.resize(pred, truth.shape[::-1], interpolation=cv2.INTER_NEAREST)
        for s in (overall, per_session[session], by_camera[camera]):
            s.add(truth, pred)
        if i % 500 == 0:
            print(f"  {i}/{len(images)}", flush=True)

    sessions = sorted(per_session)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(N_BOOT):
        agg = SegmentationScore()
        for j in rng.integers(0, len(sessions), len(sessions)):
            agg.confusion += per_session[sessions[j]].confusion
        r = agg.result()
        draws.append((r["miou"], r["false_safe_rate"]))
    draws = np.array(draws)
    result = {
        "weights": args.weights, "split": args.split, "imgsz": args.imgsz, "sessions": len(sessions),
        "overall": overall.result(), "by_camera": {c: s.result() for c, s in sorted(by_camera.items())},
        "ci95": {"miou": [float(np.nanpercentile(draws[:, 0], q)) for q in (2.5, 97.5)],
                 "false_safe_rate": [float(np.nanpercentile(draws[:, 1], q)) for q in (2.5, 97.5)]},
    }
    run = args.name or (Path(args.weights).parents[1].name if args.weights != "gt" else "gt")
    out = PROJECT_ROOT / "evaluation/benchmarks/segmentation" / f"{run}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    o = result["overall"]
    lines = [f"# Walkable-surface segmentation - {run}, {args.split}", "",
             f"{len(images)} human-annotated frames, {len(sessions)} sessions. 95% CIs: session bootstrap.", "",
             f"- mIoU **{o['miou']:.3f}** [{result['ci95']['miou'][0]:.3f}, {result['ci95']['miou'][1]:.3f}]",
             f"- false-safe rate **{o['false_safe_rate']:.3f}** [{result['ci95']['false_safe_rate'][0]:.3f}, "
             f"{result['ci95']['false_safe_rate'][1]:.3f}] (unsafe ground called walkable)",
             f"- road as walkable {o['road_as_walkable_rate']:.3f}; walkable as unsafe {o['walkable_as_unsafe_rate']:.3f}",
             f"- boundary F (±{o['boundary_tol_px']} px): " + ", ".join(f"{k} {v:.3f}" for k, v in o["boundary_f"].items()),
             "", "| Class | IoU | " + " | ".join(result["by_camera"]) + " |",
             "|---|---:|" + "---:|" * len(result["by_camera"])]
    for c, v in o["iou"].items():
        lines.append(f"| {c} | {v:.3f} | " + " | ".join(f"{r['iou'][c]:.3f}" for r in result["by_camera"].values()) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
