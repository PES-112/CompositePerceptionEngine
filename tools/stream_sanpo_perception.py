"""
stream_sanpo_perception.py
==========================
Stage 1 over a 30% sample of the SANPO valid streams, streamed straight from the
public GCS bucket — no local copy of SANPO required.

Written for a shared college SSH box: frames for one session are pulled to a
scratch directory, turned into a per-session CSV, then deleted before the next
session starts, so peak disk stays at one session (~tens of MB) regardless of
how many sessions you run. Every session is independent and already-written
CSVs are skipped, so a dropped SSH connection costs at most one session.

Usage
-----
    # 30% of the 462 valid streams, 300 frames each, every 3rd source frame
    nohup python tools/stream_sanpo_perception.py \\
        --out-dir data/processed/ablation_30pct > logs/stage1.log 2>&1 &

    # smoke test first — 2 sessions, 30 frames each
    python tools/stream_sanpo_perception.py --max-sessions 2 --max-frames 30 \\
        --out-dir data/processed/smoke

Output: one CSV per session, named `<session_id>.csv`, in --out-dir. That is the
input to `evaluation/kinetic_ablation.py`, which bootstraps at the session level.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception_stack import run_perception_stream, StreamingCSVWriter
from tools.download_sanpo_valid_streams import (
    BASE_PREFIX,
    DEFAULT_VALID_STREAMS,
    download_object,
    gcs_list,
    load_manifest,
    stem_map,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream a SANPO sample from GCS through Stage 1.")
    p.add_argument("--valid-streams", type=Path, default=DEFAULT_VALID_STREAMS)
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/processed/ablation_30pct",
                   help="Destination for per-session CSVs.")
    p.add_argument("--scratch-dir", type=Path, default=PROJECT_ROOT / "data/.sanpo_scratch",
                   help="Working directory for the frames of the session being processed.")
    p.add_argument("--session-fraction", type=float, default=0.30,
                   help="Fraction of valid streams to process (default 0.30).")
    p.add_argument("--seed", type=int, default=20260819,
                   help="Seed for the session sample — keep fixed so the sample is reproducible.")
    p.add_argument("--max-sessions", type=int, default=None, help="Hard cap, applied after sampling.")
    p.add_argument("--max-frames", type=int, default=300,
                   help="Frames kept per session AFTER striding (300 @ stride 3 ≈ 30 s of video).")
    p.add_argument("--frame-stride", type=int, default=3,
                   help="Keep every Nth source frame. 3 = 10 Hz from 30 fps source.")
    p.add_argument("--fps", type=float, default=30.0, help="Source framerate.")
    p.add_argument("--download-workers", type=int, default=8,
                   help="Parallel frame downloads. Raise on a fast link, lower if GCS throttles.")
    p.add_argument("--depth-kind", default="depth_maps", choices=["depth_maps", "zed_depth_maps"])
    p.add_argument("--keep-frames", action="store_true", help="Do not delete scratch frames (debugging).")
    p.add_argument("--overwrite", action="store_true", help="Reprocess sessions that already have a CSV.")
    p.add_argument("--dry-run", action="store_true", help="Print the sampled session list and exit.")
    return p.parse_args()


def sample_sessions(manifest: list[dict], fraction: float, seed: int, cap: int | None) -> list[dict]:
    """
    Deterministic sample of the manifest.

    Sorted first so the sample depends only on the seed, never on manifest order
    — a reordered JSON must not silently change which 30% you evaluated.
    """
    ordered = sorted(manifest, key=lambda e: e["session_id"])
    n = max(1, round(len(ordered) * fraction))
    chosen = random.Random(seed).sample(ordered, n)
    chosen.sort(key=lambda e: e["session_id"])
    return chosen[:cap] if cap else chosen


def fetch_session_frames(entry: dict, scratch: Path, args: argparse.Namespace) -> tuple[Path, Path, list[str]]:
    """
    Download one session's RGB + depth frames to scratch.

    Returns (rgb_dir, depth_dir, rgb_object_names). The object-name list is the
    map from the pipeline's local frame_idx back to the GCS object, which is
    what lets the VLM referee re-fetch a handful of frames later instead of us
    keeping every image on disk.
    """
    sid = entry["session_id"]
    branch = f"camera_{entry.get('camera', 'head')}/{entry.get('view', 'left')}"
    root_prefix = f"{BASE_PREFIX}/{sid}/{branch}/"
    rgb_prefix = root_prefix + "video_frames/"
    depth_prefix = root_prefix + f"{args.depth_kind}/"

    rgb_items = stem_map(gcs_list(rgb_prefix), rgb_prefix)
    depth_items = stem_map(gcs_list(depth_prefix), depth_prefix)
    stems = sorted(set(rgb_items) & set(depth_items))
    selected = stems[:: max(args.frame_stride, 1)][: args.max_frames]

    rgb_dir, depth_dir = scratch / "rgb", scratch / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # One HTTPS GET per frame, so the run is latency-bound, not bandwidth-bound;
    # a small thread pool cuts session fetch time several-fold on a shared box.
    jobs = [(item, dest_dir) for stem in selected
            for item, dest_dir in ((rgb_items[stem], rgb_dir), (depth_items[stem], depth_dir))]
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        for future in as_completed([
            pool.submit(download_object, item["name"], dest_dir / Path(item["name"]).name,
                        int(item.get("size", 0)))
            for item, dest_dir in jobs
        ]):
            future.result()      # re-raise here: a half-downloaded session must fail loudly

    return rgb_dir, depth_dir, [rgb_items[s]["name"] for s in selected]


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.valid_streams)
    sessions = sample_sessions(manifest, args.session_fraction, args.seed, args.max_sessions)

    # Frames are strided at download time, so the pipeline sees consecutive
    # indices for what were every-Nth source frames. Velocity divides a frame
    # delta by fps, so the effective rate must be scaled down to match or every
    # closing speed comes out `stride` times too fast.
    effective_fps = args.fps / max(args.frame_stride, 1)

    print(f"Sampled {len(sessions)}/{len(manifest)} sessions "
          f"(fraction={args.session_fraction}, seed={args.seed})")
    print(f"Stride {args.frame_stride} -> effective {effective_fps:.2f} fps, "
          f"≤{args.max_frames} frames/session")
    if args.dry_run:
        print(json.dumps([e["session_id"] for e in sessions], indent=2))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    done = failed = skipped = 0

    for i, entry in enumerate(sessions, 1):
        sid = entry["session_id"]
        out_csv = args.out_dir / f"{sid}.csv"
        if out_csv.exists() and not args.overwrite:
            skipped += 1
            continue

        scratch = args.scratch_dir / sid
        started = time.time()
        print(f"\n[{i}/{len(sessions)}] {sid}", flush=True)
        try:
            rgb_dir, depth_dir, rgb_objects = fetch_session_frames(entry, scratch, args)
            if not rgb_objects:
                print("  no matched RGB/depth frames — skipping", flush=True)
                failed += 1
                continue

            tmp_csv = out_csv.with_suffix(".csv.partial")
            with StreamingCSVWriter(tmp_csv) as writer:
                for frame_rows in run_perception_stream(
                    rgb_dir, depth_dir, effective_fps, source="sanpo", frame_step=1
                ):
                    writer.write_rows(frame_rows)
            # Rename only on success: a partial file must never look like a
            # finished session to the resume check or to the ablation script.
            tmp_csv.replace(out_csv)
            (args.out_dir / f"{sid}.frames.json").write_text(json.dumps({
                "session_id": sid,
                "camera": entry.get("camera", "head"),
                "view": entry.get("view", "left"),
                "frame_stride": args.frame_stride,
                "effective_fps": effective_fps,
                # index in this list == frame_idx in the CSV
                "rgb_objects": rgb_objects,
            }, indent=2) + "\n")
            done += 1
            print(f"  {len(rgb_objects)} frames -> {writer.rows_written} rows "
                  f"in {time.time() - started:.0f}s", flush=True)
        except Exception as exc:                      # keep going: one bad session must not end the run
            failed += 1
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if not args.keep_frames:
                shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nDone. written={done} skipped_existing={skipped} failed={failed} -> {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
