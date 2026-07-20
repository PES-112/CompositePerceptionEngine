"""
Evaluate a fine-tuned YOLO26n CPE hazard checkpoint.

This script reports validation/test metrics for the 17-class CPE hazard taxonomy
and saves qualitative prediction images for the new navigation-hazard classes.

Examples:
    .venv/bin/python training/scripts/evaluate_yolo26n_hazards.py --split test
    .venv/bin/python training/scripts/evaluate_yolo26n_hazards.py --split val --samples-per-class 12
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "training" / "runs" / "cpe_yolo26n_hazards" / "weights" / "best.pt"
DEFAULT_DATA = PROJECT_ROOT / "training" / "configs" / "cpe_hazard_classes.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "training" / "runs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "fire hydrant",
    "dog",
    "bench",
    "pole",
    "bollard",
    "stairs",
    "crosswalk",
    "pothole",
    "puddle",
]
NEW_CLASS_IDS = [11, 12, 13, 14, 15, 16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned YOLO26n hazards.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt or another checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Ultralytics dataset YAML.")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device, device list, or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence for qualitative previews.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for validation/prediction NMS.")
    parser.add_argument("--samples-per-class", type=int, default=8, help="Preview images to save per new class.")
    parser.add_argument("--name", default=None, help="Run folder name under training/runs.")
    parser.add_argument("--skip-previews", action="store_true", help="Only run metrics; do not save prediction previews.")
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


def labels_by_class(dataset_root: Path, split: str) -> dict[int, list[Path]]:
    labels_dir = dataset_root / "labels" / split
    images_dir = dataset_root / "images" / split
    by_class: dict[int, list[Path]] = defaultdict(list)
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = image_for_stem(images_dir, label_path.stem)
        if image_path is None:
            continue
        present: set[int] = set()
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            if not parts:
                continue
            try:
                present.add(int(float(parts[0])))
            except ValueError:
                continue
        for cls_id in present:
            by_class[cls_id].append(image_path)
    return by_class


def metric_array(metrics, names: Iterable[str]):
    box = getattr(metrics, "box", None)
    if box is None:
        return None
    for name in names:
        value = getattr(box, name, None)
        if value is not None:
            return value
    return None


def to_list(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def summarize_metrics(metrics) -> dict:
    box = getattr(metrics, "box", None)
    summary = {
        "map50_95": getattr(box, "map", None),
        "map50": getattr(box, "map50", None),
        "map75": getattr(box, "map75", None),
    }
    per_class = [
        {"class_id": cls_id, "class_name": name}
        for cls_id, name in enumerate(CLASS_NAMES)
    ]
    if box is not None:
        for metric_row, cls_id in enumerate(getattr(box, "ap_class_index", [])):
            cls_id = int(cls_id)
            if not 0 <= cls_id < len(per_class):
                continue
            try:
                precision, recall, map50, map50_95 = box.class_result(metric_row)
            except Exception:
                precision = recall = map50 = None
                maps = to_list(getattr(box, "maps", None))
                map50_95 = maps[cls_id] if isinstance(maps, list) and cls_id < len(maps) else None
            per_class[cls_id].update(
                {
                    "precision": precision,
                    "recall": recall,
                    "map50": map50,
                    "map50_95": map50_95,
                }
            )
    summary["per_class"] = per_class
    return summary


def save_csv(rows: list[dict], path: Path) -> None:
    keys = ["class_id", "class_name", "precision", "recall", "map50", "map50_95"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def save_report(summary: dict, class_image_counts: dict[int, int], output_dir: Path) -> None:
    report = output_dir / "hazard_eval_report.md"
    lines = [
        "# YOLO26n Hazard Evaluation",
        "",
        f"mAP50-95: {summary.get('map50_95')}",
        f"mAP50: {summary.get('map50')}",
        f"mAP75: {summary.get('map75')}",
        "",
        "## New Hazard Classes",
        "",
        "| Class | Images | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = {row["class_id"]: row for row in summary["per_class"]}
    for cls_id in NEW_CLASS_IDS:
        row = rows.get(cls_id, {})
        if class_image_counts.get(cls_id, 0) == 0:
            precision = recall = map50 = map50_95 = ""
        else:
            precision = row.get("precision", "")
            recall = row.get("recall", "")
            map50 = row.get("map50", "")
            map50_95 = row.get("map50_95", "")
        lines.append(
            "| {name} | {images} | {precision} | {recall} | {map50} | {map50_95} |".format(
                name=CLASS_NAMES[cls_id],
                images=class_image_counts.get(cls_id, 0),
                precision=precision,
                recall=recall,
                map50=map50,
                map50_95=map50_95,
            )
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_prediction_previews(
    model: YOLO,
    by_class: dict[int, list[Path]],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[int, int]:
    preview_root = output_dir / "prediction_previews"
    counts: dict[int, int] = {}
    for cls_id in NEW_CLASS_IDS:
        images = by_class.get(cls_id, [])[: args.samples_per_class]
        counts[cls_id] = len(images)
        if not images:
            continue
        model.predict(
            source=[str(path) for path in images],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=parse_device(str(args.device)),
            project=str(preview_root),
            name=CLASS_NAMES[cls_id].replace(" ", "_"),
            exist_ok=True,
            save=True,
            save_txt=True,
            verbose=False,
        )
    return counts


def main() -> None:
    args = parse_args()
    weights = project_path(args.weights)
    data = project_path(args.data)
    if not weights.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Missing dataset config: {data}")

    run_name = args.name or f"eval_{weights.parent.parent.name}_{args.split}"
    output_dir = DEFAULT_PROJECT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = read_dataset_root(data)
    by_class = labels_by_class(dataset_root, args.split)
    class_image_counts = {cls_id: len(paths) for cls_id, paths in by_class.items()}

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=parse_device(str(args.device)),
        workers=args.workers,
        iou=args.iou,
        project=str(DEFAULT_PROJECT),
        name=run_name,
        exist_ok=True,
        plots=True,
        save_json=False,
    )

    summary = summarize_metrics(metrics)
    (output_dir / "hazard_eval_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_csv(summary["per_class"], output_dir / "hazard_eval_per_class.csv")
    save_report(summary, class_image_counts, output_dir)

    if not args.skip_previews:
        preview_counts = save_prediction_previews(model, by_class, output_dir, args)
        print("Saved prediction previews:")
        for cls_id in NEW_CLASS_IDS:
            print(f"  {CLASS_NAMES[cls_id]}: {preview_counts.get(cls_id, 0)} images")

    print(f"Evaluation artifacts: {output_dir}")
    print(f"Report: {output_dir / 'hazard_eval_report.md'}")


if __name__ == "__main__":
    main()
