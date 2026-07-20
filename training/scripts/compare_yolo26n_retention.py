"""
Compare a fine-tuned CPE YOLO26n checkpoint against base YOLO26n.

The candidate checkpoint is evaluated on all 17 compact CPE classes. The base
checkpoint is evaluated only on retained COCO classes by mapping prediction class
names (person, bicycle, car, etc.) into the compact CPE class IDs. This avoids the
COCO-ID vs CPE-ID mismatch for classes such as traffic light and fire hydrant.

This is a thresholded detection check, not COCO mAP. Use
`evaluate_yolo26n_hazards.py` for Ultralytics mAP curves and plots.

Example:
    .venv/bin/python training/scripts/compare_yolo26n_retention.py --split test
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE = PROJECT_ROOT / "training" / "runs" / "cpe_yolo26n_hazards_v2" / "weights" / "best.pt"
FALLBACK_CANDIDATE = PROJECT_ROOT / "training" / "runs" / "cpe_yolo26n_hazards" / "weights" / "best.pt"
DEFAULT_BASE = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_DATA = PROJECT_ROOT / "training" / "configs" / "cpe_hazard_classes.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "training" / "runs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CPE_CLASS_NAMES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'bus',
    'truck',
    'traffic light',
    'stop sign',
    'fire hydrant',
    'pole',
    'bollard',
    'stairs',
    'crosswalk',
    'pothole',
    'puddle',
    'dog',
    'bench',
]
CPE_NAME_TO_ID = {name: idx for idx, name in enumerate(CPE_CLASS_NAMES)}
RETAINED_CLASS_IDS = [*range(9), 15, 16]


@dataclass(frozen=True)
class Box:
    cls_id: int
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0


def parse_args() -> argparse.Namespace:
    default_candidate = DEFAULT_CANDIDATE if DEFAULT_CANDIDATE.exists() else FALLBACK_CANDIDATE
    parser = argparse.ArgumentParser(description="Compare fine-tuned YOLO26n against base YOLO26n.")
    parser.add_argument("--candidate", type=Path, default=default_candidate, help="Fine-tuned checkpoint to validate.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Base YOLO26n checkpoint for retained-class comparison.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CPE dataset YAML.")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Split to compare.")
    parser.add_argument("--imgsz", type=int, default=640, help="Prediction image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for true-positive matching.")
    parser.add_argument("--device", default="0", help="CUDA device or cpu.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for quick checks.")
    parser.add_argument("--name", default=None, help="Output folder name under training/runs.")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_device(value: str):
    if value == "cpu":
        return value
    if "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    try:
        return int(value)
    except ValueError:
        return value


def read_dataset_root(data_yaml: Path) -> Path:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        return project_path(Path(data["path"]))
    except Exception:
        for raw in data_yaml.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("path:"):
                return project_path(Path(line.split(":", 1)[1].strip()))
    raise ValueError(f"Could not read dataset path from {data_yaml}")


def image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = [path for path in images_dir.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTS]
    return matches[0] if matches else None


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    x1 = (cx - w / 2.0) * image_w
    y1 = (cy - h / 2.0) * image_h
    x2 = (cx + w / 2.0) * image_w
    y2 = (cy + h / 2.0) * image_h
    return x1, y1, x2, y2


def load_ground_truth(label_path: Path, image_path: Path) -> list[Box]:
    with Image.open(image_path) as image:
        image_w, image_h = image.size
    boxes: list[Box] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        if 0 <= cls_id < len(CPE_CLASS_NAMES):
            boxes.append(Box(cls_id=cls_id, xyxy=yolo_to_xyxy(cx, cy, w, h, image_w, image_h)))
    return boxes


def collect_samples(dataset_root: Path, split: str, max_images: int | None) -> list[tuple[Path, Path, list[Box]]]:
    labels_dir = dataset_root / "labels" / split
    images_dir = dataset_root / "images" / split
    samples = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = image_for_stem(images_dir, label_path.stem)
        if image_path is None:
            continue
        gt = load_ground_truth(label_path, image_path)
        samples.append((image_path, label_path, gt))
        if max_images is not None and len(samples) >= max_images:
            break
    return samples


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def predict_boxes(model: YOLO, image_path: Path, args: argparse.Namespace, allowed_ids: set[int] | None) -> list[Box]:
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.7,
        device=parse_device(str(args.device)),
        verbose=False,
    )
    if not results or results[0].boxes is None:
        return []
    names = results[0].names
    boxes = []
    for box in results[0].boxes:
        pred_id = int(box.cls.item())
        pred_name = str(names.get(pred_id, pred_id))
        cpe_id = CPE_NAME_TO_ID.get(pred_name)
        if cpe_id is None:
            continue
        if allowed_ids is not None and cpe_id not in allowed_ids:
            continue
        xyxy = tuple(float(v) for v in box.xyxy[0].cpu().tolist())
        conf = float(box.conf.item()) if box.conf is not None else 1.0
        boxes.append(Box(cls_id=cpe_id, xyxy=xyxy, conf=conf))
    return boxes


def update_stats(stats: dict[int, dict[str, int]], gt: list[Box], preds: list[Box], class_ids: list[int], iou_thresh: float) -> None:
    for cls_id in class_ids:
        cls_gt = [box for box in gt if box.cls_id == cls_id]
        cls_preds = sorted([box for box in preds if box.cls_id == cls_id], key=lambda box: box.conf, reverse=True)
        matched_gt: set[int] = set()
        tp = 0
        fp = 0
        for pred in cls_preds:
            best_idx = None
            best_iou = 0.0
            for idx, target in enumerate(cls_gt):
                if idx in matched_gt:
                    continue
                score = iou(pred.xyxy, target.xyxy)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx
            if best_idx is not None and best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn = len(cls_gt) - len(matched_gt)
        stats[cls_id]["gt"] += len(cls_gt)
        stats[cls_id]["pred"] += len(cls_preds)
        stats[cls_id]["tp"] += tp
        stats[cls_id]["fp"] += fp
        stats[cls_id]["fn"] += fn


def empty_stats(class_ids: list[int]) -> dict[int, dict[str, int]]:
    return {cls_id: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0} for cls_id in class_ids}


def finalize_rows(model_name: str, stats: dict[int, dict[str, int]], class_ids: list[int]) -> list[dict]:
    rows = []
    for cls_id in class_ids:
        item = stats[cls_id]
        tp, fp, fn = item["tp"], item["fp"], item["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall and precision + recall else None
        rows.append(
            {
                "model": model_name,
                "class_id": cls_id,
                "class_name": CPE_CLASS_NAMES[cls_id],
                "gt": item["gt"],
                "pred": item["pred"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def evaluate_model(model_name: str, model: YOLO, samples, args: argparse.Namespace, class_ids: list[int], allowed_ids: set[int] | None) -> list[dict]:
    stats = empty_stats(class_ids)
    for image_path, _label_path, gt in samples:
        preds = predict_boxes(model, image_path, args, allowed_ids)
        update_stats(stats, gt, preds, class_ids, args.iou)
    return finalize_rows(model_name, stats, class_ids)


def clean_value(value):
    return "" if value is None else value


def write_csv(rows: list[dict], path: Path) -> None:
    keys = ["model", "class_id", "class_name", "gt", "pred", "tp", "fp", "fn", "precision", "recall", "f1"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_value(row.get(key, "")) for key in keys})


def write_report(rows: list[dict], output_dir: Path, args: argparse.Namespace) -> None:
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    lines = [
        "# YOLO26n Retention Comparison",
        "",
        f"Split: `{args.split}`",
        f"Confidence: `{args.conf}`",
        f"IoU match threshold: `{args.iou}`",
        "",
    ]
    for model_name, model_rows in by_model.items():
        lines.extend([
            f"## {model_name}",
            "",
            "| Class | GT | Pred | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in model_rows:
            printable = {key: clean_value(value) for key, value in row.items()}
            lines.append(
                "| {class_name} | {gt} | {pred} | {tp} | {fp} | {fn} | {precision} | {recall} | {f1} |".format(**printable)
            )
        lines.append("")
    lines.append("Note: base YOLO is compared only for retained COCO classes by class name. If GT is zero, add retained-class validation data before judging retention.")
    (output_dir / "retention_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    candidate = project_path(args.candidate)
    base = project_path(args.base)
    data = project_path(args.data)
    for path in (candidate, base, data):
        if not path.exists():
            raise FileNotFoundError(path)

    output_name = args.name or f"compare_{candidate.parent.parent.name}_{args.split}"
    output_dir = DEFAULT_PROJECT / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = read_dataset_root(data)
    samples = collect_samples(dataset_root, args.split, args.max_images)
    if not samples:
        raise RuntimeError(f"No labelled samples found in {dataset_root}/labels/{args.split}")

    candidate_model = YOLO(str(candidate))
    base_model = YOLO(str(base))

    rows = []
    rows.extend(evaluate_model("candidate", candidate_model, samples, args, list(range(len(CPE_CLASS_NAMES))), None))
    rows.extend(evaluate_model("base_yolo26n", base_model, samples, args, RETAINED_CLASS_IDS, set(RETAINED_CLASS_IDS)))

    write_csv(rows, output_dir / "retention_comparison.csv")
    (output_dir / "retention_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(rows, output_dir, args)

    retained_gt = sum(row["gt"] for row in rows if row["model"] == "base_yolo26n")
    print(f"Compared {len(samples)} images from split={args.split}")
    if retained_gt == 0:
        print("WARNING: no retained COCO-class ground truth exists in this split; base-vs-candidate retention cannot be judged yet.")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
