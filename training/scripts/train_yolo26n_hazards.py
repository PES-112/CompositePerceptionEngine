"""
Fine-tune YOLO26n for CPE navigation hazard detection.

This script keeps the edge architecture small by starting from the existing
YOLO26n nano checkpoint and changing only the dataset/classes used for
fine-tuning. It does not switch to a larger detector.

Dataset layout expected by Ultralytics:
    data/yolo_finetune/
      images/train, images/val, images/test
      labels/train, labels/val, labels/test

Example:
    python training/scripts/train_yolo26n_hazards.py --device 0 --batch -1

For a longer lab run:
    python training/scripts/train_yolo26n_hazards.py ^
        --epochs 120 --device 0 --batch -1 --export-formats onnx engine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "training" / "configs" / "cpe_hazard_classes.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolo26n.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "training" / "runs"

EDGE_HAZARD_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "fire hydrant",
    "pole",
    "bollard",
    "stairs",
    "crosswalk",
    "pothole",
    "puddle",
    "overhanging_hazard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO26n on CPE navigation hazard classes."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Pretrained YOLO26n checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Ultralytics dataset YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs. Early stopping is enabled.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size. Keep 640 for edge-latency parity.",
    )
    parser.add_argument(
        "--batch",
        type=float,
        default=-1,
        help="Batch size. Use -1 for Ultralytics auto-batch on the lab GPU.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device, device list, -1 for idle GPU, or cpu.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Data loader workers.",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Freeze the first N model layers for conservative fine-tuning.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate for transfer learning.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--cls-pw",
        type=float,
        default=0.5,
        help="Partial inverse-frequency class weighting for rare hazards.",
    )
    parser.add_argument(
        "--name",
        default="cpe_yolo26n_hazards",
        help="Run name under training/runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an interrupted Ultralytics training run.",
    )
    parser.add_argument(
        "--distill-model",
        type=Path,
        default=None,
        help="Optional teacher checkpoint such as YOLO26s/x. No inference cost.",
    )
    parser.add_argument(
        "--export-formats",
        nargs="*",
        default=["onnx"],
        choices=["onnx", "engine", "openvino", "torchscript"],
        help="Export formats after training. Use no values to skip export.",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip training and export the checkpoint passed with --weights.",
    )
    parser.add_argument(
        "--no-export-int8",
        action="store_true",
        help="Disable INT8 export calibration.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help="Fraction of training data used for INT8 calibration export.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reusing an existing run directory.",
    )
    return parser.parse_args()


def normalized_path(path: Path) -> Path:
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


def validate_inputs(weights: Path, data: Path) -> None:
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights file: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {data}")

    dataset_root = None
    for raw_line in data.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("path:"):
            dataset_root = line.split(":", 1)[1].strip()
            break

    if not dataset_root:
        return

    root_path = normalized_path(Path(dataset_root))
    expected = [
        root_path / "images" / "train",
        root_path / "images" / "val",
        root_path / "labels" / "train",
        root_path / "labels" / "val",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Dataset folders are missing. Create YOLO-format images/labels first:\n"
            f"  {formatted}"
        )


def print_class_plan(classes: Iterable[str]) -> None:
    print("CPE hazard classes:")
    for idx, name in enumerate(classes):
        source = "COCO" if idx <= 8 else "custom"
        print(f"  {idx:2d}: {name} ({source})")
    print(
        "\nEdge constraint: this run keeps the YOLO26n architecture and reduces "
        "the 80-class COCO head to a 16-class CPE head."
    )


def best_checkpoint_from(results, model: YOLO) -> Path | None:
    trainer = getattr(model, "trainer", None)
    trainer_best = getattr(trainer, "best", None)
    if trainer_best is not None and Path(trainer_best).exists():
        return Path(trainer_best)

    save_dir = getattr(results, "save_dir", None) or getattr(trainer, "save_dir", None)
    if save_dir is None:
        return None

    best = Path(save_dir) / "weights" / "best.pt"
    last = Path(save_dir) / "weights" / "last.pt"
    if best.exists():
        return best
    return last if last.exists() else None


def export_model(best_pt: Path, data: Path, args: argparse.Namespace) -> None:
    if not args.export_formats:
        print("Skipping export.")
        return

    model = YOLO(str(best_pt))
    for export_format in args.export_formats:
        kwargs = {
            "format": export_format,
            "imgsz": args.imgsz,
            "batch": 1,
            "device": parse_device(str(args.device)),
        }
        if not args.no_export_int8:
            kwargs.update(
                {
                    "int8": True,
                    "data": str(data),
                    "fraction": args.calibration_fraction,
                }
            )

        try:
            print(f"Exporting {best_pt.name} to {export_format}...")
            exported = model.export(**kwargs)
            print(f"Export complete: {exported}")
        except Exception as exc:  # Ultralytics may need extra export packages.
            print(
                f"Export to {export_format} failed: {exc}\n"
                "The trained PyTorch checkpoint is still usable. Install the "
                "required Ultralytics export dependencies on the lab system and "
                "rerun with --export-only --weights path/to/best.pt if needed.",
                file=sys.stderr,
            )


def main() -> None:
    args = parse_args()
    weights = normalized_path(args.weights)
    data = normalized_path(args.data)
    validate_inputs(weights, data)
    print_class_plan(EDGE_HAZARD_CLASSES)

    if args.export_only:
        export_model(weights, data, args)
        return

    model = YOLO(str(weights))

    train_kwargs = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": parse_device(str(args.device)),
        "workers": args.workers,
        "freeze": args.freeze,
        "lr0": args.lr0,
        "warmup_epochs": 3,
        "patience": args.patience,
        "optimizer": "auto",
        "cls_pw": args.cls_pw,
        "close_mosaic": 10,
        "amp": True,
        "val": True,
        "plots": True,
        "save": True,
        "project": str(DEFAULT_PROJECT),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "resume": args.resume,
    }
    if args.distill_model is not None:
        train_kwargs["distill_model"] = str(normalized_path(args.distill_model))

    results = model.train(**train_kwargs)
    best_pt = best_checkpoint_from(results, model)
    if best_pt is None:
        print("Training finished, but best.pt was not found in the run directory.")
        return

    print(f"Best checkpoint: {best_pt}")
    export_model(best_pt, data, args)


if __name__ == "__main__":
    main()
