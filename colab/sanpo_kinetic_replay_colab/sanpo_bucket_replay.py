from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from google.api_core.exceptions import NotFound
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage


DEFAULT_BUCKET = "gresearch"
DEFAULT_BASE_PREFIX = "sanpo_dataset/v0/sanpo-synthetic"
DEFAULT_CAMERA = "camera_head"
DEFAULT_SIDE = "left"


DEFAULT_FORMULA = {
    "eps_distance": 0.5,
    "velocity_exp": 2.0,
    "distance_exp": 1.0,
    "unknown_distance_score": 0.0,
    "default_severity": 1.0,
    "ahead_weight": 1.00,
    "lateral_weight": 0.85,
    "far_lateral_weight": 0.70,
    "rear_weight": 0.60,
    "unknown_direction_weight": 0.75,
    "severity_by_class": {
        "person": 1.2,
        "bicycle": 1.4,
        "motorcycle": 1.8,
        "car": 2.0,
        "bus": 2.5,
        "truck": 2.5,
        "unlabeled_obstacle": 1.3,
        "fire hydrant": 0.8,
        "shadow": 0.2,
        "boat": 0.5,
    },
}


@dataclass
class ObjectFact:
    object_id: str
    obj_class: str
    distance_m: Optional[float]
    velocity_ms: float
    position: str


OBJECT_PREFIX = re.compile(r"^(Object_\d+):\s*(.+)$")


def get_anonymous_client() -> storage.Client:
    """Create an anonymous GCS client for public SANPO bucket access."""
    return storage.Client(credentials=AnonymousCredentials(), project="none")


def _session_prefix(
    session_id: str,
    base_prefix: str = DEFAULT_BASE_PREFIX,
    camera: str = DEFAULT_CAMERA,
    side: str = DEFAULT_SIDE,
) -> str:
    return f"{base_prefix.strip('/')}/{session_id}/{camera}/{side}/video_frames"


def list_sessions(
    client: storage.Client,
    bucket_name: str = DEFAULT_BUCKET,
    base_prefix: str = DEFAULT_BASE_PREFIX,
    limit: Optional[int] = None,
) -> list[str]:
    """List SANPO session ids under the base prefix."""
    blobs = client.list_blobs(bucket_name, prefix=f"{base_prefix.strip('/')}/", delimiter="/")
    _ = list(blobs)
    sessions = sorted(p.rstrip("/").split("/")[-1] for p in blobs.prefixes)
    if limit is not None:
        return sessions[:limit]
    return sessions


def fetch_session_fps(
    client: storage.Client,
    session_id: str,
    bucket_name: str = DEFAULT_BUCKET,
    base_prefix: str = DEFAULT_BASE_PREFIX,
) -> Optional[float]:
    """Read session FPS from description.json in bucket if available."""
    blob_name = f"{base_prefix.strip('/')}/{session_id}/description.json"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        text = blob.download_as_text(encoding="utf-8")
    except NotFound:
        return None

    data = json.loads(text)
    details = data.get("session_camera_details", [])
    if not details:
        return None
    fps = details[0].get("fps")
    return float(fps) if fps else None


def parse_distance(token: str) -> Optional[float]:
    token = token.strip().lower()
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)m", token)
    return float(m.group(1)) if m else None


def parse_object_segment(segment: str) -> Optional[ObjectFact]:
    m = OBJECT_PREFIX.match(segment.strip())
    if not m:
        return None

    object_id = m.group(1)
    payload = m.group(2)
    parts = [p.strip() for p in payload.split(",")]
    if len(parts) < 2:
        return None

    obj_class = parts[0]
    distance_m = parse_distance(parts[1])

    vel_match = re.search(r"v=([-+]?\d*\.?\d+)m/s", payload)
    velocity_ms = float(vel_match.group(1)) if vel_match else 0.0

    pos_match = re.search(r"\),\s*([^\[]+)\s*\[", payload)
    position = pos_match.group(1).strip() if pos_match else "unknown"

    return ObjectFact(
        object_id=object_id,
        obj_class=obj_class,
        distance_m=distance_m,
        velocity_ms=velocity_ms,
        position=position,
    )


def parse_jsonl_record(raw_line: str):
    row = json.loads(raw_line)
    assistant = json.loads(row.get("assistant", "{}"))
    if "frame_id" not in assistant:
        return None

    frame_id = int(assistant["frame_id"])
    user_text = row.get("user", "")
    fact_text = user_text.split("] ", 1)[1] if "] " in user_text else user_text

    object_segments = [
        seg.strip()
        for seg in fact_text.split("|")
        if seg.strip().startswith("Object_")
    ]

    objects = []
    for seg in object_segments:
        obj = parse_object_segment(seg)
        if obj is not None:
            objects.append(obj)

    return frame_id, objects


def parse_jsonl_session(jsonl_path: Path | str) -> dict[int, list[ObjectFact]]:
    """Load one processed JSONL and return frame_id -> list[ObjectFact]."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL not found: {path}")

    frame_objects: dict[int, list[ObjectFact]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parsed = parse_jsonl_record(line)
            if parsed is None:
                continue
            frame_id, objects = parsed
            frame_objects[frame_id] = objects

    if not frame_objects:
        raise RuntimeError("No frame records found in JSONL.")

    return frame_objects


def infer_stride(frame_ids: list[int]) -> int:
    if len(frame_ids) < 2:
        return 1
    diffs = [b - a for a, b in zip(frame_ids[:-1], frame_ids[1:]) if b > a]
    if not diffs:
        return 1
    return max(1, int(round(float(np.median(np.array(diffs))))))


def _direction_weight(position: str, cfg: dict) -> float:
    p = position.lower()
    if "ahead" in p:
        return float(cfg["ahead_weight"])
    if "far-left" in p or "far-right" in p:
        return float(cfg["far_lateral_weight"])
    if "left" in p or "right" in p:
        return float(cfg["lateral_weight"])
    if "behind" in p:
        return float(cfg["rear_weight"])
    return float(cfg["unknown_direction_weight"])


def kinetic_score_custom(obj: ObjectFact, formula_cfg: dict) -> float:
    if obj.distance_m is None:
        return float(formula_cfg["unknown_distance_score"])

    severity_map = formula_cfg.get("severity_by_class", {})
    default_sev = float(formula_cfg.get("default_severity", 1.0))
    severity = float(severity_map.get(obj.obj_class, default_sev))

    v = abs(float(obj.velocity_ms))
    d = max(float(obj.distance_m), float(formula_cfg["eps_distance"]))
    v_exp = float(formula_cfg["velocity_exp"])
    d_exp = float(formula_cfg["distance_exp"])

    return float(severity * _direction_weight(obj.position, formula_cfg) * (v ** v_exp) / (d ** d_exp))


def _try_download_blob_bytes(client: storage.Client, bucket_name: str, blob_name: str) -> Optional[bytes]:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    try:
        return blob.download_as_bytes()
    except NotFound:
        return None


def fetch_frame_image(
    client: storage.Client,
    session_id: str,
    frame_id: int,
    cache_dir: Optional[Path] = None,
    bucket_name: str = DEFAULT_BUCKET,
    base_prefix: str = DEFAULT_BASE_PREFIX,
    camera: str = DEFAULT_CAMERA,
    side: str = DEFAULT_SIDE,
) -> Optional[np.ndarray]:
    """
    Fetch one frame directly from SANPO bucket (with optional local cache).
    Returns BGR image or None if frame is unavailable.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    prefix = _session_prefix(session_id, base_prefix=base_prefix, camera=camera, side=side)

    for ext in ("png", "jpg", "jpeg"):
        file_name = f"{frame_id:06d}.{ext}"

        if cache_dir is not None:
            cached = cache_dir / file_name
            if cached.exists():
                img = cv2.imread(str(cached))
                if img is not None:
                    return img

        blob_name = f"{prefix}/{file_name}"
        data = _try_download_blob_bytes(client, bucket_name, blob_name)
        if data is None:
            continue

        if cache_dir is not None:
            (cache_dir / file_name).write_bytes(data)

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img

    return None


def _score_color(value: float, max_value: float) -> tuple[int, int, int]:
    if max_value <= 0:
        return (180, 180, 180)
    ratio = max(0.0, min(1.0, value / max_value))
    b = int(40 * (1 - ratio))
    g = int(220 * (1 - ratio))
    r = int(60 + 195 * ratio)
    return (b, g, r)


def render_replay_from_bucket(
    client: storage.Client,
    session_id: str,
    frame_objects: dict[int, list[ObjectFact]],
    output_mp4: Path | str,
    formula_cfg: Optional[dict] = None,
    cache_dir: Optional[Path | str] = None,
    draw_top_n: int = 8,
    scale: float = 0.60,
    output_fps: Optional[float] = None,
    bucket_name: str = DEFAULT_BUCKET,
    base_prefix: str = DEFAULT_BASE_PREFIX,
    camera: str = DEFAULT_CAMERA,
    side: str = DEFAULT_SIDE,
) -> dict:
    """Render MP4 by streaming only required frame_ids from SANPO bucket."""
    cfg = dict(DEFAULT_FORMULA)
    if formula_cfg:
        cfg.update(formula_cfg)
        if "severity_by_class" in formula_cfg:
            merged = dict(DEFAULT_FORMULA["severity_by_class"])
            merged.update(formula_cfg["severity_by_class"])
            cfg["severity_by_class"] = merged

    frame_ids = sorted(frame_objects.keys())
    if not frame_ids:
        raise RuntimeError("frame_objects is empty.")

    cache_path = Path(cache_dir) if cache_dir else None

    sample = None
    for fid in frame_ids:
        sample = fetch_frame_image(
            client,
            session_id,
            fid,
            cache_dir=cache_path,
            bucket_name=bucket_name,
            base_prefix=base_prefix,
            camera=camera,
            side=side,
        )
        if sample is not None:
            break
    if sample is None:
        raise RuntimeError("Could not fetch any frame from bucket for the given session_id/frame_ids.")

    h0, w0 = sample.shape[:2]
    out_w = int(w0 * scale)
    out_h = int(h0 * scale)

    session_fps = fetch_session_fps(client, session_id, bucket_name=bucket_name, base_prefix=base_prefix)
    stride = infer_stride(frame_ids)
    fps = float(output_fps) if output_fps is not None else (session_fps / stride if session_fps else 10.0)
    fps = max(1.0, float(fps))

    output_path = Path(output_mp4)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    rendered = 0
    missing = 0

    for i, fid in enumerate(frame_ids):
        img = fetch_frame_image(
            client,
            session_id,
            fid,
            cache_dir=cache_path,
            bucket_name=bucket_name,
            base_prefix=base_prefix,
            camera=camera,
            side=side,
        )
        if img is None:
            missing += 1
            continue

        if scale != 1.0:
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

        scored = [(obj, kinetic_score_custom(obj, cfg)) for obj in frame_objects[fid]]
        scored.sort(key=lambda x: x[1], reverse=True)
        max_k = max((k for _, k in scored), default=1.0)

        overlay = img.copy()
        panel_h = min(out_h - 20, 95 + draw_top_n * 28)
        cv2.rectangle(overlay, (10, 10), (min(out_w - 10, 1150), panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.58, img, 0.42, 0, img)

        header = (
            f"Session: {session_id[:12]}... Frame: {fid} Objects: {len(scored)} "
            f"formula: sev*dir*|v|^{cfg['velocity_exp']}/d^{cfg['distance_exp']}"
        )
        cv2.putText(img, header, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        y = 64
        for rank, (obj, score) in enumerate(scored[:draw_top_n], start=1):
            dist_txt = f"{obj.distance_m:.2f}m" if obj.distance_m is not None else "unknown"
            line = (
                f"{rank:02d}. {obj.object_id} | {obj.obj_class:<18} "
                f"d={dist_txt:<8} v={obj.velocity_ms:>5.2f} "
                f"pos={obj.position:<9} K*={score:>8.4f}"
            )

            color = _score_color(score, max_k)
            if rank == 1:
                cv2.rectangle(img, (18, y - 16), (min(out_w - 14, 1040), y + 8), (35, 35, 95), 1)

            cv2.putText(img, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

            bar_x = min(out_w - 360, 760)
            bar_w = min(280, out_w - bar_x - 20)
            if bar_w > 20:
                cv2.rectangle(img, (bar_x, y - 11), (bar_x + bar_w, y + 5), (60, 60, 60), -1)
                fill = int(bar_w * (score / max_k)) if max_k > 0 else 0
                if fill > 0:
                    cv2.rectangle(img, (bar_x, y - 11), (bar_x + fill, y + 5), color, -1)

            y += 28

        writer.write(img)
        rendered += 1

        if i % 30 == 0:
            print(f"Rendered {i + 1}/{len(frame_ids)} frames")

    writer.release()

    return {
        "output_mp4": str(output_path),
        "rendered_frames": rendered,
        "missing_frames": missing,
        "total_frame_ids": len(frame_ids),
        "fps": fps,
        "stride": stride,
        "session_fps": session_fps,
    }
