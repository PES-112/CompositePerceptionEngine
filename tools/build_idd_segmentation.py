"""
build_idd_segmentation.py
=========================
Indian street data for the walkable-surface model: IDD Segmentation (IDD 20k Part I, IIIT Hyderabad,
Varma et al. WACV 2019) mixed with SANPO, in the layout of tools/build_sanpo_segmentation.py.

IDD's polygons are rasterised straight into SANPO class ids (IDD_TO_SANPO), so the combined set reuses
SANPO's label_mapping (src/segmentation/labels.py) unchanged and SANPO's masks are linked, not copied.

    <out>/images/{train,val,test}/idd_<seq>_<frame>.jpg | <sanpo stem>.jpg (symlink)
    <out>/masks/{train,val,test}/...png
    <out>/cpe_walkable_idd.yaml, manifest.json

Splits (IDD's own test labels are not public):
    train  IDD train minus held-out sequences + --sanpo-train SANPO train frames (seeded)
    val    IDD train sequences held out (--val-frac, by sequence) + --sanpo-val SANPO val frames:
           chooses the checkpoint
    test   IDD's official val split: scored once (tools/tune_surface_rules.py --data ...)
SANPO's own val/test stay where they are for the SANPO comparison.

IDD is filmed from a car's dashboard, not by a pedestrian: sidewalks sit at the image sides and
there are no crosswalk or stairs labels (painted crossings are "road"). Mixing with SANPO keeps those.

    .venv-seg.nosync/bin/python tools/build_idd_segmentation.py --idd data/idd/IDD_Segmentation \\
        --out data/cpe_walkable_idd
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.labels import CPE_CLASSES, ultralytics_label_mapping

SANPO = PROJECT_ROOT / "data/sanpo_walkable_seg"
WIDTH = 1024

# IDD label name -> SANPO class id (labels.SANPO_CLASSES), so labels.SANPO_TO_CPE applies.
# 0 = unlabeled (ignored in training).
IDD_TO_SANPO = {
    "road": 1, "parking": 1, "drivable fallback": 1,          # -> road
    "sidewalk": 3,                                            # -> walkable
    "curb": 2,                                                # -> curb
    "rail track": 19,                                         # -> road (railway)
    "non-drivable fallback": 30,                              # mud, grass shoulders -> non-walkable ground
    "sky": 27,
    "vegetation": 28,                                         # mostly trees and bushes -> obstacle
    **{k: 20 for k in ("person", "rider", "animal", "motorcycle", "bicycle", "autorickshaw", "car", "truck",
                       "bus", "caravan", "trailer", "train", "vehicle fallback", "wall", "fence", "guard rail",
                       "billboard", "traffic sign", "traffic light", "pole", "polegroup",
                       "obs-str-bar-fallback", "building", "bridge", "tunnel", "license plate")},
    **{k: 0 for k in ("fallback background", "unlabeled", "ego vehicle", "rectification border",
                      "out of roi", "ground")},
}


def rasterise(job: tuple[str, str, str, str]) -> tuple[str, Counter]:
    """(polygons json, image, out image, out mask) -> write both at WIDTH; returns unknown label counts."""
    import cv2

    js, img_path, out_img, out_mask = job
    d = json.loads(Path(js).read_text(encoding="utf-8"))
    h, w = d["imgHeight"], d["imgWidth"]
    s = WIDTH / w
    H = round(h * s)
    mask = np.zeros((H, WIDTH), np.uint8)
    unknown = Counter()
    for obj in d["objects"]:
        if obj.get("deleted"):
            continue
        name = obj["label"].strip().lower()
        if name.endswith("group") and name != "polegroup":
            name = name[:-5]
        if name not in IDD_TO_SANPO:
            unknown[name] += 1
            continue
        pts = np.round(np.asarray(obj["polygon"], np.float64) * s).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], int(IDD_TO_SANPO[name]))
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return "unreadable " + img_path, unknown
    cv2.imwrite(out_img, cv2.resize(img, (WIDTH, H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(out_mask, mask)
    return "", unknown


def idd_pairs(root: Path) -> dict[str, list[tuple[str, Path, Path]]]:
    """{'train'|'val': [(sequence, polygons json, image)]} from a Cityscapes-style IDD tree."""
    out: dict[str, list] = {"train": [], "val": []}
    for js in sorted(root.rglob("*_gtFine_polygons.json")):
        parts = js.parts
        split = next((p for p in ("train", "val") if p in parts), None)
        if split is None:
            continue
        stem = js.name.replace("_gtFine_polygons.json", "")
        img_dir = Path(*[("leftImg8bit" if p == "gtFine" else p) for p in js.parent.parts])
        cands = [img_dir / f"{stem}_leftImg8bit.png", img_dir / f"{stem}_leftImg8bit.jpg"]
        img = next((c for c in cands if c.exists()), None)
        if img is not None:
            out[split].append((js.parent.name, js, img))
    return out


def link(src: Path, dst: Path) -> None:
    if not dst.exists():
        os.symlink(os.path.relpath(src, dst.parent), dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--idd", type=Path, required=True, help="Extracted IDD Segmentation root")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/cpe_walkable_idd")
    p.add_argument("--val-frac", type=float, default=0.1, help="IDD train sequences held out for val")
    p.add_argument("--sanpo-train", type=int, default=7000)
    p.add_argument("--sanpo-val", type=int, default=1000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pairs = idd_pairs(args.idd)
    print(f"IDD: {len(pairs['train'])} train, {len(pairs['val'])} val frames with polygons")
    if not pairs["train"]:
        sys.exit("no *_gtFine_polygons.json under --idd")
    rng = random.Random(args.seed)
    seqs = sorted({s for s, _, _ in pairs["train"]})
    held = set(rng.sample(seqs, max(1, round(args.val_frac * len(seqs)))))
    split_of = {"train": lambda s: "val" if s in held else "train", "val": lambda s: "test"}

    jobs, counts = [], Counter()
    for split in ("train", "val", "test"):
        for kind in ("images", "masks"):
            (args.out / kind / split).mkdir(parents=True, exist_ok=True)
    for src_split, items in pairs.items():
        for seq, js, img in items:
            split = split_of[src_split](seq)
            stem = f"idd_{seq}_{js.name.replace('_gtFine_polygons.json', '')}"
            oi, om = args.out / "images" / split / f"{stem}.jpg", args.out / "masks" / split / f"{stem}.png"
            counts[f"idd_{split}"] += 1
            if not (oi.exists() and om.exists()):
                jobs.append((str(js), str(img), str(oi), str(om)))

    unknown = Counter()
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (err, unk) in enumerate(ex.map(rasterise, jobs, chunksize=16), 1):
            unknown.update(unk)
            if err:
                print(" ", err)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    for split, n in (("train", args.sanpo_train), ("val", args.sanpo_val)):
        stems = sorted(q.stem for q in (SANPO / "images" / split).glob("*.jpg"))
        for stem in rng.sample(stems, min(n, len(stems))):
            link(SANPO / "images" / split / f"{stem}.jpg", args.out / "images" / split / f"{stem}.jpg")
            link(SANPO / "masks" / split / f"{stem}.png", args.out / "masks" / split / f"{stem}.png")
            counts[f"sanpo_{split}"] += 1

    yaml = args.out / "cpe_walkable_idd.yaml"
    lines = ["# CPE walkable surface: IDD Segmentation + SANPO (tools/build_idd_segmentation.py). Masks hold SANPO",
             "# class ids (IDD rasterised through IDD_TO_SANPO); label_mapping remaps to CPE.",
             f"path: {args.out.resolve()}", "train: images/train", "val: images/val", "test: images/test",
             "masks_dir: masks", "names:"]
    lines += [f"  {k}: {v}" for k, v in CPE_CLASSES.items()]
    lines += ["label_mapping:"] + [f"  {k}: {v}" for k, v in ultralytics_label_mapping().items()]
    yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"idd": str(args.idd), "held_out_sequences": sorted(held), "counts": dict(counts),
                "unknown_labels": dict(unknown), "seed": args.seed, "width": WIDTH}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "unknown_labels": unknown}, indent=1))


if __name__ == "__main__":
    main()
