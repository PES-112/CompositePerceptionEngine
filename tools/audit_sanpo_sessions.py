"""
audit_sanpo_sessions.py
=======================
Metadata-only audit of the curated SANPO-Real sessions, read straight from the
public GCS bucket (no frames are downloaded). It answers the P0 data questions
in docs/pending_work.md:

  - What frame rate and field of view does each session really have?
    (Stage 1 used to assume 30 FPS and a 70° HFOV; SANPO-Real reports 15 FPS
    and ~58.4°.)
  - Which sessions are in SANPO's official train / test split?
    (sanpo-real/splits/{train,test}_session_ids.txt)
  - Which sessions have segmentation masks and camera poses? Masks exist on
    the chest camera only, so both cameras are checked; mask_frames counts the
    HUMAN_ANNOTATED vs propagated frames per session.

Writes two files next to valid_streams.json:

  session_audit.json   one record per session + a summary
  official_split.json  {"train": [...], "test": [...]} restricted to the
                       curated sessions — the split every reported number uses

    python tools/audit_sanpo_sessions.py
    python tools/audit_sanpo_sessions.py --max-sessions 5   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.download_sanpo_valid_streams import BASE_PREFIX, BUCKET, DEFAULT_VALID_STREAMS, load_manifest

OUT_DIR = DEFAULT_VALID_STREAMS.parent
MASK_CAMERA = "chest"
ANNOTATION_TYPES = "frame_segmentation_annotation_type.json"
SPLIT_FILES = {"train": "splits/train_session_ids.txt", "test": "splits/test_session_ids.txt"}


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise
            if attempt == retries - 1:
                raise
        except urllib.error.URLError:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def object_url(name: str) -> str:
    return f"https://storage.googleapis.com/{BUCKET}/" + urllib.parse.quote(name)


def list_dirs(prefix: str) -> tuple[set[str], set[str]]:
    """(sub-folder names, file names) directly under prefix."""
    params = {"prefix": prefix, "delimiter": "/", "fields": "prefixes,items(name)"}
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?" + urllib.parse.urlencode(params)
    data = json.loads(fetch(url))
    dirs = {p[len(prefix):].rstrip("/") for p in data.get("prefixes", [])}
    files = {i["name"][len(prefix):] for i in data.get("items", []) if not i["name"].endswith("_$folder$")}
    return dirs, files


def camera_details(desc: dict, camera: str) -> dict:
    """
    The description.json entry for one camera. session_camera_details is parallel to
    session_camera_location (["camera_chest", "camera_head"] in every session checked):
    the chest camera is a ZED 2i (fx ~1973, HFOV 51-60 deg), the head camera a ZED-M
    (fx ~1368, HFOV 70-78 deg). Taking element 0 silently reads the chest camera.
    """
    details = desc.get("session_camera_details") or [{}]
    locations = desc.get("session_camera_location") or []
    name = f"camera_{camera}"
    if name not in locations:
        raise KeyError(f"{name} not in session_camera_location {locations}")
    return details[locations.index(name)]


def load_official_split() -> dict[str, set[str]]:
    return {part: set(fetch(object_url(f"{BASE_PREFIX}/{rel}")).decode().split())
            for part, rel in SPLIT_FILES.items()}


def audit_session(entry: dict, split: dict[str, set[str]]) -> dict:
    sid = entry["session_id"]
    camera, view = entry.get("camera", "head"), entry.get("view", "left")
    record: dict = {"session_id": sid, "camera": camera, "view": view,
                    "split": next((p for p, ids in split.items() if sid in ids), None)}
    try:
        desc = json.loads(fetch(object_url(f"{BASE_PREFIX}/{sid}/description.json")))
        cam = camera_details(desc, camera)
        params = cam.get(f"{view}_camera_params", {})
        record.update(
            camera_model=cam.get("model"),
            fps=cam.get("fps"),
            h_fov=params.get("h_fov"),
            v_fov=params.get("v_fov"),
            fx=params.get("fx"),
            image_width=params.get("image_width"),
            image_height=params.get("image_height"),
            ego_motion=desc.get("session_video_metadata", {}).get("ego_motion"),
        )
        chest = camera_details(desc, MASK_CAMERA).get(f"{view}_camera_params", {})
        record.update(chest_fx=chest.get("fx"), chest_h_fov=chest.get("h_fov"))
    except urllib.error.HTTPError as err:
        record["description_error"] = f"HTTP {err.code}"
    _, cam_files = list_dirs(f"{BASE_PREFIX}/{sid}/camera_{camera}/")
    view_dirs, _ = list_dirs(f"{BASE_PREFIX}/{sid}/camera_{camera}/{view}/")
    record.update(
        camera_poses="camera_poses.csv" in cam_files,
        streams=sorted(view_dirs),
        segmentation_masks=any(d.startswith("segmentation_mask") for d in view_dirs),
    )
    # SANPO's panoptic masks are on the chest camera only; count them for the segmentation work.
    chest_dirs, chest_files = list_dirs(f"{BASE_PREFIX}/{sid}/camera_{MASK_CAMERA}/{view}/")
    record["chest_segmentation_masks"] = "segmentation_masks" in chest_dirs
    if record["chest_segmentation_masks"] and ANNOTATION_TYPES in chest_files:
        kinds = json.loads(fetch(object_url(f"{BASE_PREFIX}/{sid}/camera_{MASK_CAMERA}/{view}/{ANNOTATION_TYPES}")))
        record["mask_frames"] = {"human": sum(v == "HUMAN_ANNOTATED" for v in kinds.values()),
                                 "machine": sum(v == "MACHINE_ANNOTATED" for v in kinds.values())}
    return record


def summarize(records: list[dict]) -> dict:
    def counts(key):
        return {str(k): v for k, v in Counter(r.get(key) for r in records).most_common()}
    return {
        "sessions": len(records),
        "fps": counts("fps"),
        "h_fov": value_range(records, "h_fov"),
        "fx": value_range(records, "fx"),
        "image_size": {f"{w}x{h}": n for (w, h), n in
                       Counter((r.get("image_width"), r.get("image_height")) for r in records).most_common()},
        "camera_model": counts("camera_model"),
        "ego_motion": counts("ego_motion"),
        "split": counts("split"),
        "with_segmentation_masks": sum(r["segmentation_masks"] for r in records),
        "with_segmentation_masks_by_split": counts_by(records, "segmentation_masks"),
        "with_chest_segmentation_masks": sum(r.get("chest_segmentation_masks", False) for r in records),
        "with_chest_segmentation_masks_by_split": counts_by(records, "chest_segmentation_masks"),
        "chest_mask_frames_by_split": mask_frames_by_split(records),
        "chest_fx": value_range(records, "chest_fx"),
        "chest_h_fov": value_range(records, "chest_h_fov"),
        "with_camera_poses": sum(r["camera_poses"] for r in records),
        "description_errors": sum("description_error" in r for r in records),
    }


def value_range(records: list[dict], key: str) -> dict[str, float]:
    values = sorted(r[key] for r in records if r.get(key) is not None)
    if not values:
        return {}
    return {"min": values[0], "median": values[len(values) // 2], "max": values[-1]}


def counts_by(records: list[dict], flag: str) -> dict[str, int]:
    return {str(k): v for k, v in Counter(r["split"] for r in records if r.get(flag)).most_common()}


def mask_frames_by_split(records: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = {}
    for r in records:
        out.setdefault(str(r["split"]), Counter()).update(r.get("mask_frames", {}))
    return {k: dict(v) for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--valid-streams", type=Path, default=DEFAULT_VALID_STREAMS)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--max-sessions", type=int, default=0, help="0 = all.")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    manifest = load_manifest(args.valid_streams)
    if args.max_sessions:
        manifest = manifest[:args.max_sessions]
    split = load_official_split()
    print(f"Official split: {len(split['train'])} train / {len(split['test'])} test sessions "
          f"(overlap {len(split['train'] & split['test'])})")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda e: audit_session(e, split), manifest))
    records.sort(key=lambda r: r["session_id"])
    summary = summarize(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "session_audit.json").write_text(
        json.dumps({"summary": summary, "sessions": records}, indent=2) + "\n", encoding="utf-8")
    curated = {part: sorted(r["session_id"] for r in records if r["split"] == part) for part in SPLIT_FILES}
    unassigned = sorted(r["session_id"] for r in records if r["split"] is None)
    (args.out_dir / "official_split.json").write_text(json.dumps({
        "source": f"gs://{BUCKET}/{BASE_PREFIX}/splits/",
        "note": "SANPO's official session-level split, restricted to the curated valid_streams.json sessions.",
        **curated,
        "unassigned": unassigned,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
