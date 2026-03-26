"""
prepare_sample_data.py
======================
One-time helper to turn a raw SANPO or UASOL video/image sequence into
a folder of RGB frames plus co-registered depth PNGs so that
extract_perception.py can process them.

Dataset Breakdown (Phase 1 target)
------------------------------------
  SANPO  ~80% of curated frames  — lidar-fused ground-truth depth
  UASOL  ~20% of curated frames  — stereo-rectified ground-truth depth

Both datasets supply 16-bit depth PNGs; only the physical scale differs:
  SANPO  → depth_px × 0.001  = metres   (values stored in millimetres)
  UASOL  → depth_px × 0.256  = metres   (stereo disparity, per dataset spec)

Usage
-----
# SANPO — extract every 3rd frame + accompanying depth zip
python tools/prepare_sample_data.py \\
    --video    data/sanpo/sample/raw/clip_01.mp4 \\
    --depth    data/sanpo/sample/raw/clip_01_depth.zip \\
    --out_rgb  data/sanpo/sample/rgb \\
    --out_dep  data/sanpo/sample/depth \\
    --source   sanpo \\
    --stride   3

# UASOL — extract frames + stereo depth zip
python tools/prepare_sample_data.py \\
    --video    data/uasol/sample/raw/scene_42.mp4 \\
    --depth    data/uasol/sample/raw/scene_42_depth.zip \\
    --out_rgb  data/uasol/sample/rgb \\
    --out_dep  data/uasol/sample/depth \\
    --source   uasol \\
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
    p = argparse.ArgumentParser(description="CPE Phase 1 — Dataset Frame Extractor")
    p.add_argument("--video",    required=True, help="Input video file (.mp4)")
    p.add_argument("--depth",    default=None,
                   help="Zip of 16-bit depth PNGs — supported for both SANPO and UASOL")
    p.add_argument("--out_rgb",  required=True, help="Output folder for RGB frames")
    p.add_argument("--out_dep",  default=None,  help="Output folder for depth maps")
    p.add_argument("--source",   choices=["sanpo", "uasol"], default="sanpo",
                   help="Dataset source: sanpo (lidar depth) or uasol (stereo depth)")
    p.add_argument("--stride",   type=int, default=3, help="Save every N-th frame (default: 3)")
    args = p.parse_args()

    depth_note = {
        "sanpo": "lidar ground-truth depth  (scale=0.001, mm→m)",
        "uasol": "stereo ground-truth depth (scale=0.256)",
    }[args.source]

    print(f"\n=== Prepare Sample Data ===")
    print(f"  Source : {args.source.upper()} — {depth_note}")
    extract_frames(Path(args.video), Path(args.out_rgb), args.stride)
    if args.depth and args.out_dep:
        extract_depth_zip(Path(args.depth), Path(args.out_dep))
        print(f"  Depth type: {depth_note}")
    elif args.source == "uasol" and not args.depth:
        print("  [WARN] UASOL has stereo depth — pass --depth <zip> to include it!")
    print("Done.\n")


if __name__ == "__main__":
    main()
