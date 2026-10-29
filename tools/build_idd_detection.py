"""
build_idd_detection.py
======================
Indian street boxes for the 17-class hazard detector: IDD Detection (IIIT Hyderabad, Varma et al.
WACV 2019; Pascal VOC XML) in YOLO layout, with the classes IDD does not label filled in by the
current detector (pseudo-labels), so fine-tuning on IDD does not teach it that poles, bollards,
potholes ... are background.

    IDD class                                    -> CPE class (training/configs/cpe_hazard_classes.yaml)
    person, rider                                -> person
    bicycle, car, motorcycle, bus, truck         -> same
    autorickshaw                                 -> car   (no autorickshaw class; similar size and speed)
    vehicle fallback, caravan, trailer, train    -> truck
    animal                                       -> dog   (CPE's only animal class; cows are common)
    traffic light                                -> traffic light
    traffic sign                                 -> dropped (CPE has stop sign only; the teacher labels those)

Pseudo-labelled from the v3 detector (conf >= --pseudo-conf), boxes not overlapping an IDD box
(IoU < 0.5): stop sign, fire hydrant, pole, bollard, stairs, crosswalk, pothole, puddle, bench.

The original Roboflow training set is not on this machine, so the teacher stands in for it
(methodology §4.26). Splits:
    train  --train IDD train images from sequences not held out
    val    --val images from held-out IDD train sequences (checkpoint choice)
    test   --test images from IDD's official val split (scored once)

    .venv-seg.nosync/bin/python tools/build_idd_detection.py --idd data/idd/IDD_Detection --out data/cpe_hazards_idd
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEACHER = PROJECT_ROOT / "training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt"
NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign", "fire hydrant",
         "pole", "bollard", "stairs", "crosswalk", "pothole", "puddle", "dog", "bench"]
CPE = {n: i for i, n in enumerate(NAMES)}
IDD_TO_CPE = {"person": "person", "rider": "person", "bicycle": "bicycle", "car": "car",
              "motorcycle": "motorcycle", "bus": "bus", "truck": "truck", "autorickshaw": "car",
              "vehicle fallback": "truck", "caravan": "truck", "trailer": "truck", "train": "truck",
              "animal": "dog", "traffic light": "traffic light", "traffic sign": None}
PSEUDO = ["stop sign", "fire hydrant", "pole", "bollard", "stairs", "crosswalk", "pothole", "puddle", "bench"]
MAX_W = 1280


def parse(xml: Path) -> tuple[int, int, list[tuple[int, float, float, float, float]], Counter]:
    """VOC XML -> (width, height, [(cpe class, x1, y1, x2, y2)], unknown names)."""
    r = ET.parse(xml).getroot()
    w, h = int(float(r.findtext("size/width"))), int(float(r.findtext("size/height")))
    boxes, unknown = [], Counter()
    for o in r.iter("object"):
        name = (o.findtext("name") or "").strip().lower()
        if name not in IDD_TO_CPE:
            unknown[name] += 1
            continue
        if IDD_TO_CPE[name] is None:
            continue
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append((CPE[IDD_TO_CPE[name]], x1, y1, x2, y2))
    return w, h, boxes, unknown


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def items(root: Path) -> dict[str, list[tuple[str, Path, Path]]]:
    """{'train'|'val': [(sequence, xml, image)]}, from IDD's train.txt / val.txt."""
    out = {}
    for split in ("train", "val"):
        lst = root / f"{split}.txt"
        rows = []
        for line in lst.read_text(encoding="utf-8").split():
            xml = root / "Annotations" / f"{line}.xml"
            img = next((p for p in (root / "JPEGImages" / f"{line}.jpg", root / "JPEGImages" / f"{line}.png")
                        if p.exists()), None)
            if xml.exists() and img is not None:
                rows.append((str(Path(line).parent), xml, img))
        out[split] = rows
    return out


def main() -> None:
    import cv2

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--idd", type=Path, required=True, help="Extracted IDD Detection root (has train.txt)")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/cpe_hazards_idd")
    p.add_argument("--train", type=int, default=10000)
    p.add_argument("--val", type=int, default=1500)
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--pseudo-conf", type=float, default=0.5)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    src = items(args.idd)
    print(f"IDD: {len(src['train'])} train, {len(src['val'])} val annotated images")
    rng = random.Random(args.seed)
    seqs = sorted({s for s, _, _ in src["train"]})
    held = set(rng.sample(seqs, max(1, round(0.1 * len(seqs)))))
    pool = {"train": [r for r in src["train"] if r[0] not in held],
            "val": [r for r in src["train"] if r[0] in held], "test": src["val"]}
    chosen = {k: rng.sample(v, min(n, len(v))) for (k, v), n in
              zip(pool.items(), (args.train, args.val, args.test))}

    from ultralytics import YOLO

    teacher = YOLO(str(TEACHER))
    pseudo_ids = [CPE[n] for n in PSEUDO]
    counts, unknown, pseudo = defaultdict(Counter), Counter(), Counter()
    for split, rows in chosen.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

        def load(row):
            seq, xml, img = row
            stem = "idd_" + "_".join(Path(os.path.relpath(xml, args.idd / "Annotations")).with_suffix("").parts)
            out_img = args.out / "images" / split / f"{stem}.jpg"
            w, h, boxes, unk = parse(xml)
            im = cv2.imread(str(img))
            s = min(1.0, MAX_W / im.shape[1])
            if s < 1:
                im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_img), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return stem, out_img, im.shape[1] / w, boxes, unk

        with ThreadPoolExecutor(8) as ex:
            loaded = list(ex.map(load, rows))
        for i in range(0, len(loaded), 32):
            batch = loaded[i:i + 32]
            results = teacher.predict([str(b[1]) for b in batch], conf=args.pseudo_conf, classes=pseudo_ids,
                                      imgsz=640, device=args.device, verbose=False)
            for (stem, out_img, s, boxes, unk), r in zip(batch, results):
                unknown.update(unk)
                H, W = r.orig_shape
                gt = [(c, x1 * s, y1 * s, x2 * s, y2 * s) for c, x1, y1, x2, y2 in boxes]
                extra = []
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
                    if all(iou((x1, y1, x2, y2), g[1:]) < 0.5 for g in gt):
                        extra.append((int(c), x1, y1, x2, y2))
                        pseudo[NAMES[int(c)]] += 1
                lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                         for c, x1, y1, x2, y2 in gt + extra]
                (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                for c, *_ in gt:
                    counts[split][NAMES[c]] += 1
            print(f"  {split} {min(i + 32, len(loaded))}/{len(loaded)}", flush=True)

    yaml = args.out / "cpe_hazards_idd.yaml"
    yaml.write_text("\n".join([
        "# 17-class hazards: IDD Detection + v3 pseudo-labels for classes IDD lacks (tools/build_idd_detection.py)",
        f"path: {args.out.resolve()}", "train: images/train", "val: images/val", "test: images/test",
        f"nc: {len(NAMES)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]) + "\n", encoding="utf-8")
    manifest = {"idd": str(args.idd), "held_out_sequences": sorted(held), "images": {k: len(v) for k, v in chosen.items()},
                "idd_boxes": {k: dict(v) for k, v in counts.items()}, "pseudo_boxes": dict(pseudo),
                "pseudo_conf": args.pseudo_conf, "unknown_labels": dict(unknown), "teacher": str(TEACHER)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
