"""
Fine-tune YOLO26n for CPE navigation hazard detection.

This script keeps the edge architecture small by starting from the existing
YOLO26n nano checkpoint and changing only the dataset/classes used for
fine-tuning. It does not switch to a larger detector.

Dataset layout expected by Ultralytics:
    data/yolo_finetune/
      images/train, images/val, images/test
      labels/train, labels/val, labels/test

Example GB10 high-throughput run:
    python training/scripts/train_yolo26n_hazards.py \
        --weights training/runs/cpe_yolo26n_hazards/weights/best.pt \
        --device 0 \
        --epochs 50 \
        --name cpe_yolo26n_hazards_v2 \
        --export-formats

Defaults are tuned for the ARM Grace-Blackwell GB10 lab server: RAM cache,
AutoBatch, high worker count, no epoch-time validation, and no plots.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "training" / "configs" / "cpe_hazard_classes.yaml"
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo" / "base_yolo26n" / "yolo26n.pt"
DEFAULT_PROJECT = PROJECT_ROOT / "training" / "runs"
DEFAULT_RESERVED_CPU_CORES = 4

EDGE_HAZARD_CLASSES = [
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
        type=int,
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
        default=-1,
        help=(
            "Data loader workers. Use -1 to saturate CPU minus reserved OS cores "
            "(16 workers on the 20-core GB10 with the default reserve)."
        ),
    )
    parser.add_argument(
        "--reserve-cpu-cores",
        type=int,
        default=DEFAULT_RESERVED_CPU_CORES,
        help="CPU cores left free when --workers -1 is used.",
    )
    parser.add_argument(
        "--cache",
        choices=["false", "ram", "disk"],
        default="ram",
        help="Ultralytics dataset cache mode. Default 'ram' is intended for the 120GB GB10 memory pool.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile through Ultralytics when supported by the local stack.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision on CUDA. Pass --no-amp to disable.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic training. Slower, but useful for exact reproducibility.",
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
        help="Early stopping patience. Only applies when --val is enabled.",
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
        "--val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Ultralytics validation during training. Disabled by default for maximum throughput.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate Ultralytics training plots/previews. Disabled by default for maximum throughput.",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=-1,
        help="Checkpoint save interval in epochs. -1 saves only normal best/last checkpoints.",
    )
    parser.add_argument(
        "--export-formats",
        nargs="*",
        default=[],
        choices=["onnx", "engine", "openvino", "torchscript"],
        help="Export formats after training. Default skips export to avoid dependency overhead during fast lab runs.",
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


def configure_cuda_runtime(device, *, autobatch: bool = False) -> None:
    """Enable safe CUDA speed knobs and print the selected GPU plan."""
    try:
        import torch
    except Exception as exc:
        print(f"Torch import failed during GPU setup: {exc}", file=sys.stderr)
        return

    if device == "cpu" or not torch.cuda.is_available():
        print("Training runtime: CPU")
        return

    torch.backends.cudnn.benchmark = not autobatch
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if isinstance(device, list):
        device_ids = device
    elif isinstance(device, int):
        device_ids = [device]
    else:
        device_ids = list(range(torch.cuda.device_count()))

    print("Training runtime: CUDA")
    for idx in device_ids:
        if idx >= torch.cuda.device_count():
            print(f"  cuda:{idx}: unavailable", file=sys.stderr)
            continue
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        print(f"  cuda:{idx}: {props.name}, total={total_gb:.1f}GB, reserved={reserved_gb:.1f}GB")
    print(f"  cudnn_benchmark={torch.backends.cudnn.benchmark} ({'AutoBatch search' if autobatch else 'static batch speed path'})")


def cache_arg(value: str):
    return False if value == "false" else value


def worker_count(value: int, reserve_cpu_cores: int) -> int:
    if value >= 0:
        return value
    cpu_count = os.cpu_count() or 2
    reserve = max(0, reserve_cpu_cores)
    return max(2, cpu_count - reserve)


def dataset_root_from_yaml(data: Path) -> Path | None:
    for raw_line in data.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("path:"):
            return normalized_path(Path(line.split(":", 1)[1].strip()))
    return None


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def print_throughput_plan(args: argparse.Namespace, data: Path) -> None:
    root = dataset_root_from_yaml(data)
    images_size = directory_size(root / "images") if root else 0
    labels_size = directory_size(root / "labels") if root else 0
    print("GB10 throughput profile:")
    print(f"  cache={cache_arg(args.cache)!r} (images={format_bytes(images_size)}, labels={format_bytes(labels_size)})")
    print(f"  batch={args.batch} ({'AutoBatch maximum-fit search' if args.batch == -1 else 'static batch'})")
    print(f"  workers={worker_count(args.workers, args.reserve_cpu_cores)} (reserve_cpu_cores={args.reserve_cpu_cores})")
    print(f"  val_during_train={args.val}, plots={args.plots}, save_period={args.save_period}")


def validate_inputs(weights: Path, data: Path) -> None:
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights file: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"Missing dataset YAML: {data}")

    root_path = dataset_root_from_yaml(data)
    if root_path is None:
        return
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
        source = "COCO" if idx <= 8 or idx in (15, 16) else "custom"
        print(f"  {idx:2d}: {name} ({source})")
    print(
        "\nEdge constraint: this run keeps the YOLO26n architecture and reduces "
        "the 80-class COCO head to a 17-class CPE head."
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
    device = parse_device(str(args.device))
    if not args.export_formats:
        print("Skipping export.")
        return

    model = YOLO(str(best_pt))
    for export_format in args.export_formats:
        kwargs = {
            "format": export_format,
            "imgsz": args.imgsz,
            "batch": 1,
            "device": device,
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
    device = parse_device(str(args.device))
    configure_cuda_runtime(device, autobatch=args.batch == -1)
    print_throughput_plan(args, data)

    if args.export_only:
        export_model(weights, data, args)
        return

    model = YOLO(str(weights))

    train_kwargs = {
        "data": str(data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "workers": worker_count(args.workers, args.reserve_cpu_cores),
        "freeze": args.freeze,
        "lr0": args.lr0,
        "warmup_epochs": 3,
        "patience": args.patience,
        "optimizer": "auto",
        "cls_pw": args.cls_pw,
        "close_mosaic": 10,
        "amp": args.amp,
        "cache": cache_arg(args.cache),
        "compile": args.compile,
        "deterministic": args.deterministic,
        "val": args.val,
        "plots": args.plots,
        "save_period": args.save_period,
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

    checkpoint_label = "Best checkpoint" if best_pt.name == "best.pt" else "Last checkpoint"
    print(f"{checkpoint_label}: {best_pt}")
    export_model(best_pt, data, args)


if __name__ == "__main__":
    main()
