"""Does an open-vocabulary detector see SANPO poles out of the box? YOLOE-11s prompt-free (4,585 built-in
classes, no training, no text encoder) on the frames cached by detector_recall_sweep.py, same objects
and matching rule (methodology §4.21-4.22).

A pole counts as found if a YOLOE box of a pole-like class covers >= 50% of its bounding box. The
confidence threshold is chosen on the train frames (highest pole recall with pole-box support >= 0.5;
ties to the higher threshold); test is scored once.

    python evaluation/open_vocab_recall.py     # needs data/weights/yoloe-11s-seg-pf.pt (auto-downloaded)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluation.detector_recall_sweep as sweep  # noqa: E402

WEIGHTS = ROOT / "data/weights/yoloe-11s-seg-pf.pt"
POLE_LIKE = {"pole", "flag pole", "lamp post", "street light", "telegraph pole", "post", "pillar",
             "street sign", "traffic sign", "road sign", "stop sign", "parking sign", "warning sign",
             "speed limit sign", "parking meter", "signal"}
CONFS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
KEY = "yoloe11s_pf"


def predict(dirs):
    from ultralytics import YOLOE

    model = YOLOE(str(WEIGHTS))
    for i, d in enumerate(dirs):
        boxes = json.loads((d / "boxes.json").read_text())
        if KEY in boxes:
            continue
        r = model.predict(cv2.imread(str(d / "frame.png")), conf=min(CONFS), imgsz=640, verbose=False)[0]
        boxes[KEY] = [[r.names[int(c)], float(p), *map(float, b)] for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
        (d / "boxes.json").write_text(json.dumps(boxes))
        if i % 40 == 0:
            print(f"  {i + 1}/{len(dirs)}", flush=True)


def frames(split):
    out = []
    for d in sorted((sweep.CACHE / split).iterdir()):
        if (d / "boxes.json").exists():
            ids, objs = sweep.load_truth(d)
            b = json.loads((d / "boxes.json").read_text())
            # Pole-like YOLOE classes become "pole" so the shared matcher and support score apply.
            b[KEY] = [["pole" if c in POLE_LIKE else c, *rest] for c, *rest in b[KEY]]
            out.append((ids, objs, b))
    return out


def main():
    for split in ("train", "test"):
        predict([d for d in sorted((sweep.CACHE / split).iterdir()) if (d / "boxes.json").exists()])
    sweep.POLE_TYPE = {"pole"}
    tr = frames("train")
    grid = []
    for c in CONFS:
        sweep.BASE_CONF = c
        r = sweep.row(f"YOLOE conf {c:.2f}", tr, KEY, c)
        grid.append(r)
        print(f"train conf {c:.2f}: poles {r['pole']['found']}/{r['pole']['total']}, "
              f"pole boxes/frame {r['pole_boxes_per_frame']:.2f}, support {r['pole_box_support']:.2f}")
    ok = [r for r in grid if r["pole_box_support"] >= 0.5]
    best = max(ok, key=lambda r: (r["pole"]["recall"], r["pole_conf"]))
    c = best["pole_conf"]
    te = frames("test")
    sweep.BASE_CONF = c
    t = sweep.row(f"YOLOE-11s prompt-free, conf {c:.2f} (chosen on train)", te, KEY, c)
    print(f"chosen conf {c:.2f}; TEST poles {t['pole']['found']}/{t['pole']['total']} = {t['pole']['recall']:.2f} "
          f"[{t['pole']['ci95'][0]:.2f}, {t['pole']['ci95'][1]:.2f}], pole boxes/frame {t['pole_boxes_per_frame']:.2f}, "
          f"support {t['pole_box_support']:.2f}")
    (sweep.OUT / "open_vocab_yoloe.json").write_text(json.dumps({"pole_like": sorted(POLE_LIKE), "train_grid": grid,
                                                                 "chosen_conf": c, "test": t}, indent=1))


if __name__ == "__main__":
    main()
