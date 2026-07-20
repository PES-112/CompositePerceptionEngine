"""
Download Roboflow Universe datasets and merge selected labels into CPE YOLO data.

The manifest maps source labels from each Universe dataset into the 17-class CPE
hazard taxonomy in training/configs/cpe_hazard_classes.yaml.

Example:
    export ROBOFLOW_API_KEY="..."
    cp training/configs/roboflow_universe_sources.example.json \
        training/configs/roboflow_universe_sources.json
    python training/scripts/download_roboflow_universe.py \
        --manifest training/configs/roboflow_universe_sources.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_CLASSES = [
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
TARGET_CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_IMAGE_LIMITS = {
    "person": 500,
    "bicycle": 400,
    "car": 500,
    "motorcycle": 400,
    "bus": 300,
    "truck": 300,
    "traffic light": 300,
    "stop sign": 250,
    "fire hydrant": 250,
    "dog": 400,
    "bench": 400,
    "stairs": 1500,
    "crosswalk": 600,
    "pothole": 1500,
    "puddle": 1000,
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Roboflow Universe datasets and remap labels to CPE hazards."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON manifest listing Universe datasets and source-to-CPE class maps.",
    )
    parser.add_argument(
        "--api-key-env",
        default="ROBOFLOW_API_KEY",
        help="Environment variable containing the Roboflow private API key.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifest and print the download plan without copying data.",
    )
    parser.add_argument(
        "--no-class-limits",
        action="store_true",
        help="Copy every matching image instead of enforcing class_image_limits.",
    )
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_dotenv_files() -> None:
    """Load ROBOFLOW_API_KEY from local .env files without requiring python-dotenv."""
    candidates = [
        Path.cwd() / ".env",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip(chr(39) + chr(34))
            if key and key not in os.environ:
                os.environ[key] = value


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("datasets"):
        raise ValueError("Manifest must contain a non-empty 'datasets' list.")
    return manifest


def class_image_limits(manifest: dict[str, Any], disabled: bool) -> dict[str, int | None]:
    if disabled:
        return {name: None for name in TARGET_CLASSES}
    raw_limits = manifest.get("class_image_limits") or DEFAULT_CLASS_IMAGE_LIMITS
    limits: dict[str, int | None] = {}
    for class_name in TARGET_CLASSES:
        value = raw_limits.get(class_name)
        limits[class_name] = None if value is None else int(value)
    return limits


def under_limit(class_name: str, limits: dict[str, int | None], counts: Counter[str]) -> bool:
    limit = limits.get(class_name)
    return limit is None or counts[class_name] < limit


def read_dataset_names(data_yaml: Path) -> dict[int, str]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        raw_names = data.get("names", {})
        if isinstance(raw_names, list):
            return {idx: str(name) for idx, name in enumerate(raw_names)}
        return {int(idx): str(name) for idx, name in raw_names.items()}
    except Exception:
        names: dict[int, str] = {}
        in_names = False
        for raw in data_yaml.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if stripped.startswith("names:"):
                in_names = True
                remainder = stripped.split(":", 1)[1].strip()
                if remainder.startswith("[") and remainder.endswith("]"):
                    items = [item.strip().strip(chr(39) + chr(34)) for item in remainder[1:-1].split(",")]
                    return {idx: item for idx, item in enumerate(items) if item}
                continue
            if in_names and ":" in stripped:
                idx, name = stripped.split(":", 1)
                if idx.strip().isdigit():
                    names[int(idx.strip())] = name.strip().strip(chr(39) + chr(34))
        if not names:
            raise ValueError(f"Could not parse class names from {data_yaml}")
        return names


def find_download_root(dataset_location: Path) -> Path:
    if (dataset_location / "data.yaml").exists():
        return dataset_location
    candidates = list(dataset_location.rglob("data.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No data.yaml found under {dataset_location}")
    return candidates[0].parent


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = [path for path in images_dir.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTS]
    return matches[0] if matches else None


def ensure_yolo_dirs(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def source_dirs(source_root: Path, split: str) -> tuple[Path, Path] | None:
    labels_dir = source_root / split / "labels"
    images_dir = source_root / split / "images"
    if labels_dir.exists() and images_dir.exists():
        return images_dir, labels_dir
    labels_dir = source_root / "labels" / split
    images_dir = source_root / "images" / split
    if labels_dir.exists() and images_dir.exists():
        return images_dir, labels_dir
    return None


def remap_split(
    source_root: Path,
    output_root: Path,
    source_split: str,
    target_split: str,
    source_names: dict[int, str],
    class_map: dict[str, str],
    prefix: str,
    limits: dict[str, int | None],
    running_images: Counter[str],
) -> tuple[Counter[str], Counter[str]]:
    dirs = source_dirs(source_root, source_split)
    if dirs is None:
        return Counter(), Counter()
    images_dir, labels_dir = dirs
    image_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    normalized_map = {key.lower(): value for key, value in class_map.items()}

    for label_path in sorted(labels_dir.glob("*.txt")):
        remapped_lines: list[str] = []
        present_classes: set[str] = set()
        present_instances: Counter[str] = Counter()
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            try:
                source_id = int(float(parts[0]))
            except ValueError:
                continue
            source_class = source_names.get(source_id, "").lower()
            target_class = normalized_map.get(source_class)
            if target_class not in TARGET_CLASS_TO_ID:
                continue
            remapped_lines.append(
                " ".join([str(TARGET_CLASS_TO_ID[target_class]), *parts[1:5]])
            )
            present_classes.add(target_class)
            present_instances[target_class] += 1

        if not remapped_lines:
            continue

        if present_classes and not any(
            under_limit(class_name, limits, running_images) for class_name in present_classes
        ):
            continue

        image_path = find_image(images_dir, label_path.stem)
        if image_path is None:
            continue

        out_stem = f"{prefix}_{source_split}_{label_path.stem}"
        out_image = output_root / "images" / target_split / f"{out_stem}{image_path.suffix.lower()}"
        out_label = output_root / "labels" / target_split / f"{out_stem}.txt"
        shutil.copy2(image_path, out_image)
        out_label.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
        for class_name in present_classes:
            image_counts[class_name] += 1
            running_images[class_name] += 1
        instance_counts.update(present_instances)

    return image_counts, instance_counts


def safe_prefix(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(project_path(args.manifest))
    output_root = project_path(manifest.get("output_root", "data/yolo_finetune"))
    download_root = project_path(manifest.get("download_root", "data/roboflow_universe"))
    model_format = manifest.get("format", "yolov8")

    load_dotenv_files()
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        raise RuntimeError(f"Set {args.api_key_env} before downloading from Roboflow.")

    if not args.dry_run:
        ensure_yolo_dirs(output_root)
        download_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        rf = None
    else:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)
    total_images: Counter[str] = Counter()
    total_instances: Counter[str] = Counter()
    limits = class_image_limits(manifest, args.no_class_limits)
    if not args.no_class_limits:
        print("Class image caps:")
        for class_name in TARGET_CLASSES:
            if limits[class_name] is not None:
                print(f"  {class_name:20s} max_images={limits[class_name]}")

    for item in manifest["datasets"]:
        name = item.get("name") or f"{item['workspace']}_{item['project']}_v{item['version']}"
        prefix = safe_prefix(name)
        print(f"\nDataset: {name}")
        print(f"  {item['workspace']}/{item['project']}/{item['version']} -> {model_format}")
        print(f"  class_map: {item['class_map']}")
        if args.dry_run:
            continue

        project = rf.workspace(item["workspace"]).project(item["project"])  # type: ignore[union-attr]
        version = project.version(int(item["version"]))
        downloaded = version.download(model_format=model_format, location=str(download_root / prefix))
        source_root = find_download_root(Path(downloaded.location))
        source_names = read_dataset_names(source_root / "data.yaml")

        for source_split, target_split in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
            image_counts, instance_counts = remap_split(
                source_root=source_root,
                output_root=output_root,
                source_split=source_split,
                target_split=target_split,
                source_names=source_names,
                class_map=item["class_map"],
                prefix=prefix,
                limits=limits,
                running_images=total_images,
            )
            if image_counts:
                print(f"  merged {source_split} -> {target_split}: {dict(image_counts)}")
            total_images.update(image_counts)
            total_instances.update(instance_counts)

    if args.dry_run:
        print("\nDry run complete. Replace manifest placeholders with real Universe URLs.")
        return

    print("\nMerged Roboflow image counts by CPE class:")
    for class_name in TARGET_CLASSES:
        if total_images[class_name] or total_instances[class_name]:
            print(
                f"  {class_name:20s} images={total_images[class_name]:5d} "
                f"instances={total_instances[class_name]:5d}"
            )


if __name__ == "__main__":
    main()
