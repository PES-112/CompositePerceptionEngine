"""
Export the preferred CPE YOLO26n checkpoint for edge-runtime benchmarking.

This script is intentionally separate from training so a completed checkpoint
can be exported and benchmarked without rerunning epochs. It uses the modern
Ultralytics `quantize` argument instead of deprecated `half` / `int8` aliases.

Examples:
    .venv/bin/python training/scripts/export_yolo26n_edge.py \
        --formats onnx \
        --quantize 16 \
        --device 0

    .venv/bin/python training/scripts/export_yolo26n_edge.py \
        --formats engine \
        --quantize 8 \
        --device 0 \
        --calibration-fraction 0.25
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "training"
    / "runs"
    / "cpe_yolo26n_hazards_v3_from_base"
    / "weights"
    / "best.pt"
)
DEFAULT_DATA = PROJECT_ROOT / "training" / "configs" / "cpe_hazard_classes.yaml"
DEFAULT_EXPORT_ROOT = (
    PROJECT_ROOT / "models" / "yolo" / "cpe_yolo26n_hazards_v3_from_base" / "exports"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CPE YOLO26n v3 to edge runtime formats."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Checkpoint to export.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Dataset YAML used for INT8 calibration.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx"],
        choices=["onnx", "engine", "openvino", "torchscript"],
        help="One or more Ultralytics export formats.",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "16", "8"],
        default="16",
        help="Precision request. Use 16 for FP16, 8 for INT8 calibration, none for FP32/default.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size.")
    parser.add_argument("--device", default="0", help="CUDA device or cpu.")
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export with dynamic input shapes where supported.",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simplify ONNX graphs where supported.",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help="Fraction of training images used for INT8 calibration.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=None,
        help="TensorRT workspace size in GiB. Leave unset for Ultralytics default.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Local model-registry folder where exported artifacts and manifest are copied.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print export arguments without running Ultralytics export.",
    )
    return parser.parse_args()


def normalized_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_device(value: str) -> Any:
    if value == "cpu":
        return value
    if "," in value:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    try:
        return int(value)
    except ValueError:
        return value


def quantize_value(value: str) -> int | None:
    return None if value == "none" else int(value)


def build_export_kwargs(
    args: argparse.Namespace,
    export_format: str,
    weights: Path,
    data: Path,
) -> dict[str, Any]:
    quantize = quantize_value(args.quantize)
    kwargs: dict[str, Any] = {
        "format": export_format,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": parse_device(str(args.device)),
        "dynamic": args.dynamic,
        "simplify": args.simplify,
    }
    if quantize is not None:
        kwargs["quantize"] = quantize
    if quantize == 8:
        kwargs["data"] = str(data)
        kwargs["fraction"] = args.calibration_fraction
    if export_format == "engine" and args.workspace is not None:
        kwargs["workspace"] = args.workspace

    print(f"Export plan for {weights.name} -> {export_format}:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")
    return kwargs


def copy_exported_artifact(exported: Any, export_root: Path) -> list[str]:
    copied: list[str] = []
    paths = exported if isinstance(exported, (list, tuple)) else [exported]
    export_root.mkdir(parents=True, exist_ok=True)

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        destination = export_root / path.name
        if path.resolve() != destination.resolve():
            shutil.copy2(path, destination)
        copied.append(str(destination))
    return copied


def main() -> None:
    args = parse_args()
    weights = normalized_path(args.weights)
    data = normalized_path(args.data)
    export_root = normalized_path(args.export_root)

    if not weights.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights}")
    if quantize_value(args.quantize) == 8 and not data.exists():
        raise FileNotFoundError(f"Missing calibration data YAML: {data}")

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "data": str(data),
        "exports": [],
    }

    model = YOLO(str(weights))
    for export_format in args.formats:
        kwargs = build_export_kwargs(args, export_format, weights, data)
        if args.dry_run:
            manifest["exports"].append(
                {"format": export_format, "kwargs": kwargs, "status": "dry_run"}
            )
            continue

        try:
            exported = model.export(**kwargs)
            copied = copy_exported_artifact(exported, export_root)
            manifest["exports"].append(
                {
                    "format": export_format,
                    "kwargs": kwargs,
                    "status": "ok",
                    "ultralytics_output": str(exported),
                    "registry_paths": copied,
                }
            )
            print(f"Export complete: {exported}")
            for path in copied:
                print(f"Copied registry artifact: {path}")
        except Exception as exc:
            manifest["exports"].append(
                {
                    "format": export_format,
                    "kwargs": kwargs,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            print(
                f"Export to {export_format} failed: {exc}\n"
                "The PyTorch checkpoint is still usable. Install the export "
                "dependencies required by this format, then rerun this script.",
                file=sys.stderr,
            )

    export_root.mkdir(parents=True, exist_ok=True)
    manifest_path = export_root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Export manifest: {manifest_path}")


if __name__ == "__main__":
    main()
