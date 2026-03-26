"""
prepare_sample_data.py
======================
One-time helper to turn a raw SANPO/UASOL video file into a folder of
RGB frames (and optionally depth PNGs) so that extract_perception.py can
process them.

Usage
-----
# Extract every 3rd frame from a SANPO video + its depth map zip
python tools/prepare_sample_data.py \\
    --video    data/sanpo/sample/raw/clip_01.mp4 \\
    --depth    data/sanpo/sample/raw/clip_01_depth.zip \\
    --out_rgb  data/sanpo/sample/rgb \\
    --out_dep  data/sanpo/sample/depth \\
    --stride   3

# UASOL (no depth)
python tools/prepare_sample_data.py \\
    --video    data/uasol/sample/raw/scene_42.mp4 \\
    --out_rgb  data/uasol/sample/rgb \\
    --stride   3
"""

import argparse
import zipfile
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path, stride: int = 1):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1
    cap.release()
    print(f"  Saved {saved} RGB frames → {out_dir}")


def extract_depth_zip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print(f"  Extracted depth maps → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",    required=True, help="Input video file (.mp4)")
    p.add_argument("--depth",    default=None,  help="Zip of 16-bit depth PNGs (SANPO only)")
    p.add_argument("--out_rgb",  required=True, help="Output folder for RGB frames")
    p.add_argument("--out_dep",  default=None,  help="Output folder for depth maps")
    p.add_argument("--stride",   type=int, default=3, help="Save every N-th frame (default: 3)")
    args = p.parse_args()

    print(f"\n=== Prepare Sample Data ===")
    extract_frames(Path(args.video), Path(args.out_rgb), args.stride)
    if args.depth and args.out_dep:
        extract_depth_zip(Path(args.depth), Path(args.out_dep))
    print("Done.\n")


if __name__ == "__main__":
    main()
