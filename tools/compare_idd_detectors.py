"""
Does the IDD fine-tune forget? Decides which classes the original detector keeps on the phone
(ios/CPEWalk/prepare_app.py LEGACY_CLASSES, methodology §4.26).

The original test set (Roboflow) is on the lab server, so forgetting is measured as retention against
the old model itself, on SANPO frames neither model trained on (data/sanpo_walkable_seg images/val):
    retention(c) = share of the old model's class-c boxes (conf >= CONF) that the new model also finds
                   (same class, IoU >= 0.5, conf >= CONF)
plus IDD test mAP50 per class for the classes IDD labels.

Declared rule (written before the comparison was run):
- a class IDD labels goes to the new model if its IDD test mAP50 is higher, else stays legacy;
- a class IDD does not label goes to the new model only if retention >= 0.8 with at least 20 old
  boxes; otherwise it stays with the original detector.

    .venv-seg.nosync/bin/python tools/compare_idd_detectors.py --new training/runs/cpe_yolo26n_hazards_v4_idd/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_idd_detection import IDD_TO_CPE, NAMES, iou

OLD = PROJECT_ROOT / "training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt"
IDD_YAML = PROJECT_ROOT / "data/cpe_hazards_idd/cpe_hazards_idd.yaml"
SANPO_IMAGES = PROJECT_ROOT / "data/sanpo_walkable_seg/images/val"
OUT = PROJECT_ROOT / "evaluation/benchmarks/detector_idd"
CONF, IOU_MATCH, MIN_RETENTION, MIN_BOXES = 0.3, 0.5, 0.8, 20
IDD_CLASSES = sorted({c for c in IDD_TO_CPE.values() if c})


def main() -> None:
    from ultralytics import YOLO

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    old, new = YOLO(str(OLD)), YOLO(str(args.new))
    ap = {}
    for tag, m in (("old", old), ("new", new)):
        r = m.val(data=str(IDD_YAML), split="test", imgsz=640, device=args.device, batch=16, plots=False,
                  verbose=False, project=str(OUT / "val"), name=tag, exist_ok=True)
        ap[tag] = {NAMES[int(c)]: float(r.box.ap50[i]) for i, c in enumerate(r.box.ap_class_index)}

    frames = sorted(SANPO_IMAGES.glob("*.jpg"))
    frames = frames[:: max(1, len(frames) // args.frames)][: args.frames]
    old_n, kept, new_n = Counter(), Counter(), Counter()
    for i in range(0, len(frames), 32):
        batch = [str(f) for f in frames[i:i + 32]]
        ro = old.predict(batch, conf=CONF, imgsz=640, device=args.device, verbose=False)
        rn = new.predict(batch, conf=CONF, imgsz=640, device=args.device, verbose=False)
        for a, b in zip(ro, rn):
            nb = list(zip(b.boxes.xyxy.tolist(), b.boxes.cls.tolist()))
            for c in b.boxes.cls.tolist():
                new_n[NAMES[int(c)]] += 1
            for box, c in zip(a.boxes.xyxy.tolist(), a.boxes.cls.tolist()):
                name = NAMES[int(c)]
                old_n[name] += 1
                if any(int(nc) == int(c) and iou(box, nbox) >= IOU_MATCH for nbox, nc in nb):
                    kept[name] += 1

    rows, legacy = [], []
    for name in NAMES:
        ret = kept[name] / old_n[name] if old_n[name] else None
        if name in IDD_CLASSES:
            to_new = ap["new"].get(name, 0.0) > ap["old"].get(name, 0.0)
            why = f"IDD mAP50 {ap['old'].get(name, 0):.3f} -> {ap['new'].get(name, 0):.3f}"
        else:
            to_new = ret is not None and ret >= MIN_RETENTION and old_n[name] >= MIN_BOXES
            why = f"retention {ret:.2f} of {old_n[name]}" if ret is not None else "no old boxes on SANPO"
        if not to_new:
            legacy.append(name)
        rows.append({"class": name, "idd_ap50_old": ap["old"].get(name), "idd_ap50_new": ap["new"].get(name),
                     "sanpo_old_boxes": old_n[name], "sanpo_new_boxes": new_n[name], "retention": ret,
                     "model": "new" if to_new else "legacy", "why": why})

    OUT.mkdir(parents=True, exist_ok=True)
    result = {"old": str(OLD), "new": str(args.new), "sanpo_frames": len(frames), "conf": CONF,
              "legacy_classes": legacy, "classes": rows}
    (OUT / "metrics.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    lines = ["# IDD fine-tune vs original detector", "",
             f"IDD test mAP50 (IDD-labelled classes); retention on {len(frames)} SANPO val frames at conf {CONF}.", "",
             "| class | IDD AP50 old | new | SANPO boxes old | new | retention | phone uses |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        f = lambda v: "—" if v is None else f"{v:.3f}"
        lines.append(f"| {r['class']} | {f(r['idd_ap50_old'])} | {f(r['idd_ap50_new'])} | {r['sanpo_old_boxes']} | "
                     f"{r['sanpo_new_boxes']} | {f(r['retention'])} | {r['model']} |")
    lines += ["", f"LEGACY_CLASSES = {legacy}"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
