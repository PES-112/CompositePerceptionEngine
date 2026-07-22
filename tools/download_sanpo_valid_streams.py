"""Download bounded SANPO-Real valid-stream RGB/depth frames from public GCS."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "gresearch"
BASE_PREFIX = "sanpo_dataset/v0/sanpo-real"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALID_STREAMS = PROJECT_ROOT / "simulation/datasets/sanpo/valid_streams.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/sanpo/valid_streams"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected SANPO valid stream frames.")
    parser.add_argument("--valid-streams", type=Path, default=DEFAULT_VALID_STREAMS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--session-id", action="append", default=None, help="Specific session ID to download. May be repeated.")
    parser.add_argument("--max-streams", type=int, default=1, help="Number of valid streams to download when --session-id is omitted.")
    parser.add_argument("--camera", default=None, choices=["head", "chest"], help="Override manifest camera.")
    parser.add_argument("--view", default=None, choices=["left", "right"], help="Override manifest view.")
    parser.add_argument("--depth-kind", default="depth_maps", choices=["depth_maps", "zed_depth_maps"], help="Depth branch to download.")
    parser.add_argument("--frame-step", type=int, default=1, help="Download every Nth matched frame.")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum matched frames per stream.")
    parser.add_argument("--dry-run", action="store_true", help="List planned downloads without downloading files.")
    return parser.parse_args()


def gcs_list(prefix: str, max_pages: int = 100) -> list[dict]:
    items: list[dict] = []
    token = None
    pages = 0
    while True:
        params = {"prefix": prefix, "maxResults": "1000"}
        if token:
            params["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.load(response)
        items.extend(data.get("items", []))
        token = data.get("nextPageToken")
        pages += 1
        if not token or pages >= max_pages:
            return items


def download_object(object_name: str, destination: Path, expected_size: int) -> bool:
    if destination.exists() and destination.stat().st_size == expected_size:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/gresearch/" + urllib.parse.quote(object_name)
    with urllib.request.urlopen(url, timeout=120) as response:
        destination.write_bytes(response.read())
    return True


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of valid stream entries")
    return data


def select_entries(manifest: list[dict], session_ids: list[str] | None, max_streams: int) -> list[dict]:
    if session_ids:
        by_id = {entry["session_id"]: entry for entry in manifest}
        missing = [sid for sid in session_ids if sid not in by_id]
        if missing:
            raise ValueError(f"Session IDs not found in manifest: {missing}")
        return [by_id[sid] for sid in session_ids]
    return manifest[:max_streams]


def frame_stem(rel_path: str) -> str:
    name = Path(rel_path).name
    if name.endswith(".float16.gz"):
        return name.removesuffix(".float16.gz")
    return Path(name).stem


def stem_map(items: list[dict], prefix: str) -> dict[str, dict]:
    mapped = {}
    for item in items:
        name = item["name"]
        if name.endswith("_$folder$"):
            continue
        rel = name[len(prefix):]
        mapped[frame_stem(rel)] = item
    return mapped


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.valid_streams)
    entries = select_entries(manifest, args.session_id, args.max_streams)
    total_downloaded = 0
    total_skipped = 0

    for entry in entries:
        sid = entry["session_id"]
        camera = args.camera or entry.get("camera", "head")
        view = args.view or entry.get("view", "left")
        branch = f"camera_{camera}/{view}"
        root_prefix = f"{BASE_PREFIX}/{sid}/{branch}/"
        rgb_prefix = root_prefix + "video_frames/"
        depth_prefix = root_prefix + f"{args.depth_kind}/"

        rgb_items = stem_map(gcs_list(rgb_prefix), rgb_prefix)
        depth_items = stem_map(gcs_list(depth_prefix), depth_prefix)
        stems = sorted(set(rgb_items) & set(depth_items))
        selected = stems[:: max(args.frame_step, 1)][: args.max_frames]
        if not selected:
            print(f"[WARN] No matched RGB/depth frames for {sid} {branch}")
            continue

        local_root = args.output_root / sid / f"camera_{camera}" / view
        meta_items = gcs_list(f"{BASE_PREFIX}/{sid}/", max_pages=1)
        camera_items = gcs_list(f"{BASE_PREFIX}/{sid}/camera_{camera}/", max_pages=1)
        metadata = {
            "session_id": sid,
            "camera": camera,
            "view": view,
            "depth_kind": args.depth_kind,
            "gcs_prefix": root_prefix,
            "selected_frame_count": len(selected),
            "selected_stems": selected,
        }
        print(f"\n{sid} {branch}: {len(stems)} matched frames, downloading {len(selected)}")

        planned_bytes = 0
        downloads: list[tuple[str, Path, int]] = []
        for stem in selected:
            for kind, item in (("video_frames", rgb_items[stem]), (args.depth_kind, depth_items[stem])):
                object_name = item["name"]
                rel = object_name[len(root_prefix):]
                dest = local_root / rel
                size = int(item.get("size", 0))
                planned_bytes += size
                downloads.append((object_name, dest, size))

        for item in meta_items + camera_items:
            rel_name = item["name"].split(f"{sid}/", 1)[-1]
            if rel_name in {"description.json", f"camera_{camera}/camera_poses.csv", f"camera_{camera}/fixed_camera_poses.csv"}:
                downloads.append((item["name"], args.output_root / sid / rel_name, int(item.get("size", 0))))

        metadata["planned_download_mb"] = round(planned_bytes / 1_048_576, 2)
        if args.dry_run:
            print(json.dumps(metadata, indent=2))
            continue

        local_root.mkdir(parents=True, exist_ok=True)
        for object_name, dest, size in downloads:
            changed = download_object(object_name, dest, size)
            if changed:
                total_downloaded += 1
                print(f"  downloaded {dest.relative_to(args.output_root)}")
            else:
                total_skipped += 1
        (local_root / "download_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"\nDone. downloaded={total_downloaded} skipped_cached={total_skipped}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
