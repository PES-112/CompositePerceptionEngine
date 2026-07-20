"""
Rebalance a YOLO detection dataset so train-only Roboflow sources contribute val/test examples.

The script moves image/label pairs from train into val/test using a deterministic
hash order. It prefers images whose labels include classes below their requested
validation/test target, which keeps all-class evaluation usable without creating
train/validation leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move YOLO train images into val/test to satisfy split targets.")
    parser.add_argument("--data-root", type=Path, required=True, help="YOLO dataset root with images/ and labels/ folders.")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Target validation image fraction per class.")
    parser.add_argument("--test-frac", type=float, default=0.10, help="Target test image fraction per class.")
    parser.add_argument("--min-val", type=int, default=30, help="Minimum validation images per class when enough train images exist.")
    parser.add_argument("--min-test", type=int, default=30, help="Minimum test images per class when enough train images exist.")
    parser.add_argument("--seed", default="cpe-yolo-v2", help="Stable hash seed for deterministic moves.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without changing files.")
    return parser.parse_args()


def image_for_label(data_root: Path, split: str, stem: str) -> Path | None:
    image_dir = data_root / "images" / split
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    for candidate in image_dir.glob(f"{stem}.*"):
        if candidate.suffix.lower() in IMAGE_EXTS:
            return candidate
    return None


def label_classes(label_path: Path) -> set[int]:
    classes: set[int] = set()
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if not parts:
            continue
        try:
            classes.add(int(float(parts[0])))
        except ValueError:
            continue
    return classes


def split_counts(data_root: Path) -> dict[str, Counter[int]]:
    counts: dict[str, Counter[int]] = {split: Counter() for split in ("train", "val", "test")}
    for split in counts:
        for label_path in sorted((data_root / "labels" / split).glob("*.txt")):
            for class_id in label_classes(label_path):
                counts[split][class_id] += 1
    return counts


def target_count(total: int, frac: float, minimum: int) -> int:
    if total <= 0:
        return 0
    desired = max(round(total * frac), min(minimum, total // 5))
    return min(desired, max(1, total - 1))


def stable_key(seed: str, label_path: Path) -> str:
    return hashlib.sha256(f"{seed}:{label_path.name}".encode("utf-8")).hexdigest()


def move_pair(data_root: Path, label_path: Path, image_path: Path, target_split: str, dry_run: bool) -> None:
    target_label = data_root / "labels" / target_split / label_path.name
    target_image = data_root / "images" / target_split / image_path.name
    target_label.parent.mkdir(parents=True, exist_ok=True)
    target_image.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    shutil.move(str(label_path), str(target_label))
    shutil.move(str(image_path), str(target_image))


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    counts = split_counts(data_root)
    all_classes = sorted(set(counts["train"]) | set(counts["val"]) | set(counts["test"]))
    targets: dict[str, Counter[int]] = {"val": Counter(), "test": Counter()}
    for class_id in all_classes:
        total = counts["train"][class_id] + counts["val"][class_id] + counts["test"][class_id]
        targets["val"][class_id] = target_count(total, args.val_frac, args.min_val)
        targets["test"][class_id] = target_count(total, args.test_frac, args.min_test)

    train_labels = sorted((data_root / "labels" / "train").glob("*.txt"), key=lambda p: stable_key(args.seed, p))
    moved: dict[str, Counter[int]] = {"val": Counter(), "test": Counter()}
    moved_files: Counter[str] = Counter()

    for target_split in ("val", "test"):
        for label_path in list(train_labels):
            if not label_path.exists():
                continue
            classes = label_classes(label_path)
            if not classes:
                continue
            needs = [cid for cid in classes if counts[target_split][cid] < targets[target_split][cid]]
            if not needs:
                continue
            image_path = image_for_label(data_root, "train", label_path.stem)
            if image_path is None:
                continue
            move_pair(data_root, label_path, image_path, target_split, args.dry_run)
            moved_files[target_split] += 1
            for cid in classes:
                counts["train"][cid] -= 1
                counts[target_split][cid] += 1
                moved[target_split][cid] += 1

    print("Moved image counts by target split:")
    for split in ("val", "test"):
        print(f"  {split}: files={moved_files[split]}")
        for cid, count in sorted(moved[split].items()):
            print(f"    class_id={cid:2d} moved={count:4d} final={counts[split][cid]:4d} target={targets[split][cid]:4d}")


if __name__ == "__main__":
    main()
