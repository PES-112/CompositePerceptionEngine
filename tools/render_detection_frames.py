"""
render_detection_frames.py
===========================
Generates annotated still images of the CPE YOLO26n v3 hazard detector running
on real frames, for the Phase-3 review slide "Demo: Perception Stack — YOLO
Hazard Detection." Produces PNGs you can screenshot/paste directly into the
slide — this script does not touch the .pptx itself.

Two source modes:

  1. --source <path>   Run on a local image, a folder of images, or a video
                       file (e.g. a phone recording of a street/hallway).

  2. (no --source)     Auto-downloads a handful of real RGB frames from the
                       public SANPO-Real GCS bucket (no auth/API key/local
                       dataset needed — same bucket `tools/download_sanpo_valid_streams.py`
                       uses) across a few sessions, runs detection on all of
                       them, then keeps the frames with the most distinct
                       classes detected. This is biased toward showing one of
                       the CPE-added hazard classes (pole, bollard, stairs,
                       crosswalk, pothole, puddle) that base COCO YOLO does not
                       have at all, alongside ordinary retained classes
                       (person, car, bicycle, ...), so the saved frames make
                       the "new classes" story visible rather than just
                       another car/person detection screenshot.

Boxes for CPE-added (non-COCO) hazard classes are drawn in magenta with a
"NEW CLASS" tag so they stand out against the ordinary retained-COCO boxes.

Usage:
    # Auto-download real SANPO frames + detect (default, no local dataset needed):
    .venv/bin/python tools/render_detection_frames.py

    # Run on your own footage instead:
    .venv/bin/python tools/render_detection_frames.py --source path/to/clip.mp4
    .venv/bin/python tools/render_detection_frames.py --source path/to/folder_of_images/

Output: data/demo_frames/annotated/*.png (gitignored — copy/paste manually
into the slide; nothing here is committed automatically).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.download_sanpo_valid_streams import (  # noqa: E402
    DEFAULT_VALID_STREAMS,
    download_object,
    gcs_list,
    load_manifest,
    stem_map,
    BASE_PREFIX,
)

DEFAULT_MODEL = str(
    PROJECT_ROOT / "training/runs/cpe_yolo26n_hazards_v3_from_base/weights/best.pt"
)
BASE_MODEL = str(PROJECT_ROOT / "models/yolo/base_yolo26n/yolo26n.pt")
DEFAULT_CONF = 0.30

# Classes CPE added on top of base COCO — the ones worth calling out on the slide.
NEW_HAZARD_CLASSES = {"pole", "bollard", "stairs", "crosswalk", "pothole", "puddle"}

RETAINED_COLOR = (0, 200, 255)   # amber (BGR) — ordinary retained COCO classes
NEW_COLOR = (255, 0, 255)        # magenta — CPE-added hazard classes
FONT = cv2.FONT_HERSHEY_SIMPLEX


def download_demo_frames(out_dir: Path, num_sessions: int, frames_per_session: int,
                          seed: int | None = None) -> list[Path]:
    """Pull a small set of real SANPO-Real RGB frames from the public GCS bucket.

    seed=None (the default) picks a different random set of sessions on every run —
    this is a demo/visualisation script, not the frozen 30% ablation subset, so there
    is no reason to pin it to a fixed sample. Pass --seed to reproduce a specific pick.
    """
    manifest = load_manifest(DEFAULT_VALID_STREAMS)
    rng = random.Random(seed)
    sessions = rng.sample(manifest, min(num_sessions, len(manifest)))

    saved: list[Path] = []
    for entry in sessions:
        sid = entry["session_id"]
        camera = entry.get("camera", "head")
        view = entry.get("view", "left")
        rgb_prefix = f"{BASE_PREFIX}/{sid}/camera_{camera}/{view}/video_frames/"
        rgb_items = stem_map(gcs_list(rgb_prefix), rgb_prefix)
        stems = sorted(rgb_items)
        if not stems:
            print(f"[WARN] no frames for session {sid}, skipping")
            continue
        # Spread picks across the session instead of only its first few frames.
        idxs = np.linspace(0, len(stems) - 1, num=frames_per_session, dtype=int)
        picks = sorted({stems[i] for i in idxs})

        for stem in picks:
            item = rgb_items[stem]
            dest = out_dir / f"{sid}_{stem}.jpg"
            download_object(item["name"], dest, int(item.get("size", 0)))
            saved.append(dest)
        print(f"  {sid}: downloaded {len(picks)} frames")

    return saved


def class_color(class_name: str) -> tuple[int, int, int]:
    return NEW_COLOR if class_name in NEW_HAZARD_CLASSES else RETAINED_COLOR


def draw_detections(img: np.ndarray, boxes, class_names: dict) -> set[str]:
    seen_classes: set[str] = set()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = class_names[cls_idx]
        seen_classes.add(cls_name)
        color = class_color(cls_name)
        is_new = cls_name in NEW_HAZARD_CLASSES

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if is_new else 2)
        tag = f"NEW CLASS: {cls_name} {conf:.2f}" if is_new else f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.55, 2)
        ly = max(y1 - 8, th + 8)
        cv2.rectangle(img, (x1, ly - th - 8), (x1 + tw + 8, ly + 4), color, -1)
        text_color = (255, 255, 255) if is_new else (0, 0, 0)
        cv2.putText(img, tag, (x1 + 4, ly - 2), FONT, 0.55, text_color, 2, cv2.LINE_AA)
    return seen_classes


def draw_hud(img: np.ndarray, seen_classes: set[str], model_name: str) -> None:
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    new_seen = sorted(seen_classes & NEW_HAZARD_CLASSES)
    label = f"CPE {model_name} | {len(seen_classes)} classes"
    if new_seen:
        label += f" | new hazard classes: {', '.join(new_seen)}"
    cv2.putText(img, label, (10, 26), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def run_detection(model, img: np.ndarray, conf: float) -> tuple[np.ndarray, set[str]]:
    """Run one model on one frame, return an annotated copy + the classes it found."""
    results = model.predict(img, conf=conf, verbose=False)
    boxes = results[0].boxes
    annotated = img.copy()
    seen_classes: set[str] = set()
    if boxes is not None and len(boxes) > 0:
        seen_classes = draw_detections(annotated, boxes, results[0].names)
    return annotated, seen_classes


def stitch_before_after(before_img: np.ndarray, after_img: np.ndarray,
                         before_label: str, after_label: str) -> np.ndarray:
    """Side-by-side panel with a labelled banner over each half."""
    h = max(before_img.shape[0], after_img.shape[0])

    def pad(img):
        if img.shape[0] == h:
            return img
        canvas = np.zeros((h, img.shape[1], 3), dtype=img.dtype)
        canvas[: img.shape[0]] = img
        return canvas

    before_img, after_img = pad(before_img), pad(after_img)
    gutter = np.full((h, 6, 3), 255, dtype=np.uint8)
    panel = np.hstack([before_img, gutter, after_img])

    for x_off, label, bg in (
        (0, before_label, (60, 60, 180)),          # "before" banner — dull red
        (before_img.shape[1] + 6, after_label, (60, 140, 30)),  # "after" banner — green
    ):
        w = before_img.shape[1]
        cv2.rectangle(panel, (x_off, 0), (x_off + w, 46), bg, -1)
        cv2.putText(panel, label, (x_off + 12, 31), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CPE YOLO26n v3 detections on real frames")
    parser.add_argument("--source", type=str, default=None,
                         help="Image, folder of images, or video file. Omit to auto-download SANPO frames.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/demo_frames/annotated")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/demo_frames/raw")
    parser.add_argument("--num-sessions", type=int, default=3,
                         help="SANPO sessions to sample from when --source is omitted.")
    parser.add_argument("--frames-per-session", type=int, default=10,
                         help="Frames per session when --source is omitted.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Session-selection seed. Omit for a different random set of "
                              "SANPO sessions every run; pass a fixed value to reproduce one.")
    parser.add_argument("--top-k", type=int, default=8,
                         help="How many best frames to keep (ranked by distinct classes, "
                              "prioritising frames with a new hazard class).")
    parser.add_argument("--compare-base", action="store_true",
                         help="Also run base (stock COCO) YOLO26n on each kept frame and save "
                              "a side-by-side 'before/after' panel showing what the CPE "
                              "hazard fine-tune adds. Saved to --before-after-out.")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--before-after-out", type=Path,
                         default=PROJECT_ROOT / "data/demo_frames/before_after")
    args = parser.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)

    if args.source:
        source_path = Path(args.source)
        if source_path.is_dir():
            image_paths = sorted(
                p for p in source_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
        elif source_path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
            args.raw_dir.mkdir(parents=True, exist_ok=True)
            cap = cv2.VideoCapture(str(source_path))
            image_paths = []
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % 5 == 0:
                    p = args.raw_dir / f"video_frame_{idx:05d}.jpg"
                    cv2.imwrite(str(p), frame)
                    image_paths.append(p)
                idx += 1
            cap.release()
        else:
            image_paths = [source_path]
    else:
        args.raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"No --source given: downloading real SANPO frames into {args.raw_dir} ...")
        image_paths = download_demo_frames(
            args.raw_dir, args.num_sessions, args.frames_per_session, seed=args.seed
        )

    if not image_paths:
        print("No frames to process.")
        return

    print(f"\nRunning {Path(args.model).name} on {len(image_paths)} frames (conf={args.conf}) ...")
    model = YOLO(args.model)

    scored: list[tuple[int, bool, Path, np.ndarray, set[str]]] = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        results = model.predict(img, conf=args.conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        annotated = img.copy()
        seen_classes = draw_detections(annotated, boxes, results[0].names)
        draw_hud(annotated, seen_classes, Path(args.model).parent.parent.name)
        has_new = bool(seen_classes & NEW_HAZARD_CLASSES)
        scored.append((len(seen_classes), has_new, img_path, annotated, seen_classes))

    if not scored:
        print("No detections found in any frame — try a lower --conf or different --source.")
        return

    # Prefer frames with a new hazard class, then frames with more distinct classes overall.
    scored.sort(key=lambda t: (t[1], t[0]), reverse=True)
    keep = scored[: args.top_k]

    print(f"\nSaving top {len(keep)}/{len(scored)} frames to {args.out} :")
    for _, has_new, img_path, annotated, seen_classes in keep:
        out_path = args.out / f"detect_{img_path.stem}.png"
        cv2.imwrite(str(out_path), annotated)
        flag = " [has new hazard class]" if has_new else ""
        print(f"  {out_path.name} — classes: {sorted(seen_classes)}{flag}")

    print("\nDone. Open the PNGs above and paste your pick(s) into the "
          "'Demo: Perception Stack — YOLO Hazard Detection' slide.")

    if args.compare_base:
        print(f"\n--compare-base set: running {Path(args.base_model).name} (stock COCO) on the "
              f"same {len(keep)} frames for a before/after panel ...")
        base_model = YOLO(args.base_model)
        args.before_after_out.mkdir(parents=True, exist_ok=True)
        for _, has_new, img_path, after_annotated, after_classes in keep:
            raw = cv2.imread(str(img_path))
            before_annotated, before_classes = run_detection(base_model, raw, args.conf)
            # No draw_hud() here: stitch_before_after()'s banner is taller (46px vs
            # draw_hud's 40px) and fully overpaints this region anyway — drawing a
            # HUD label here would be immediately covered, so skip the wasted work.
            panel = stitch_before_after(
                before_annotated, after_annotated,
                "BEFORE: Base YOLO26n (COCO)", "AFTER: CPE Fine-Tuned v3 (Hazard-Aware)",
            )
            out_path = args.before_after_out / f"compare_{img_path.stem}.png"
            cv2.imwrite(str(out_path), panel)
            new_only = sorted(after_classes - before_classes)
            note = f" — v3 adds: {new_only}" if new_only else ""
            print(f"  {out_path.name}{note}")
        print(f"\nBefore/after panels saved to {args.before_after_out} — paste one into the "
              "'Before vs. After: Base YOLO vs. CPE Fine-Tuned Hazard Detector' slide.")


if __name__ == "__main__":
    main()
