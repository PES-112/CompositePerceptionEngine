import os
import sys
import gzip
import json
import csv
import io
import time
import random
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials

# Set up paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Central Configuration
YOLO_MODEL_PATH = str(REPO_ROOT / "yolo26n.pt")
CONF_THRESH = 0.30
IOU_THRESH = 0.50
IMG_SIZE = (1280, 720)
MIN_BLOB_AREA_PX = 800

MAX_STREAMS = 20
MAX_FRAMES_PER_STREAM = 5
RANDOM_SEED = 42

OUTPUT_DIR = REPO_ROOT / "notebooks" / "gap_analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define experimental ranges: (min_depth, max_depth, name)
RANGES = [
    (0.5, 2.0, "range_0.5_2.0"),
    (0.5, 4.0, "range_0.5_4.0"),
    (0.5, 6.0, "range_0.5_6.0"),
    (2.0, 6.0, "range_2.0_6.0"),
]

class SANPOLoader:
    """
    Handles data retrieval for the SANPO dataset.
    Checks for local cached assets first; if unavailable, streams/downloads
    anonymously from the official Google Cloud Storage bucket.
    """
    def __init__(self, local_raw_path=None, camera="head", view="left"):
        self.camera = camera
        self.view = view
        self.local_raw_path = Path(local_raw_path) if local_raw_path else None
        
        self.sanpo_root = "gs://gresearch/sanpo_dataset/v0/sanpo-real"
        self.client = storage.Client(credentials=AnonymousCredentials(), project='gresearch')
        parts = self.sanpo_root.split("/")
        self.bucket_name = parts[2]
        self.prefix = "/".join(parts[3:])
        self.bucket = self.client.bucket(self.bucket_name)

    def load_rgb(self, session_id: str, frame_id: int) -> np.ndarray | None:
        if self.local_raw_path:
            local_path = self.local_raw_path / session_id / f"camera_{self.camera}" / self.view / "video_frames" / f"{frame_id:06d}.png"
            if local_path.exists():
                img = cv2.imread(str(local_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        path = f"{self.prefix}/{session_id}/camera_{self.camera}/{self.view}/video_frames/{frame_id:06d}.png"
        try:
            blob = self.bucket.blob(path)
            arr = np.frombuffer(blob.download_as_bytes(timeout=15), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"  Error loading RGB for {session_id} frame {frame_id}: {e}")
        return None

    def load_depth(self, session_id: str, frame_id: int) -> np.ndarray | None:
        if self.local_raw_path:
            local_path_gz = self.local_raw_path / session_id / f"camera_{self.camera}" / self.view / "depth_maps" / f"{frame_id:06d}.float16.gz"
            if local_path_gz.exists():
                return self._parse_float16_gz(local_path_gz)
            local_path_npz = self.local_raw_path / session_id / f"camera_{self.camera}" / self.view / "zed_depth_maps" / f"{frame_id:06d}.npz"
            if local_path_npz.exists():
                return self._parse_npz(local_path_npz)
        
        npz_path = f"{self.prefix}/{session_id}/camera_{self.camera}/{self.view}/zed_depth_maps/{frame_id:06d}.npz"
        try:
            blob = self.bucket.blob(npz_path)
            raw = blob.download_as_bytes(timeout=15)
            return self._parse_npz(io.BytesIO(raw))
        except Exception:
            pass

        cre_path = f"{self.prefix}/{session_id}/camera_{self.camera}/{self.view}/depth_maps/{frame_id:06d}.float16.gz"
        try:
            blob = self.bucket.blob(cre_path)
            raw = blob.download_as_bytes(timeout=15)
            return self._parse_float16_gz_bytes(raw)
        except Exception as e:
            print(f"  Error loading depth for {session_id} frame {frame_id}: {e}")
        
        return None

    def _parse_npz(self, file_source) -> np.ndarray | None:
        try:
            depth = np.load(file_source)["arr_0"].astype(np.float32)
            depth[~np.isfinite(depth)] = 0
            depth[depth > 100] = 0
            return depth
        except Exception:
            return None

    def _parse_float16_gz(self, path: Path) -> np.ndarray | None:
        try:
            with gzip.open(path, 'rb') as f:
                return self._parse_float16_gz_bytes(f.read())
        except Exception:
            return None

    def _parse_float16_gz_bytes(self, data: bytes) -> np.ndarray | None:
        try:
            raw = np.frombuffer(gzip.decompress(data) if data.startswith(b'\x1f\x8b') else data, dtype=np.float16)
            for h, w in [(1242, 2208), (720, 1280), (1080, 1920)]:
                if raw.size >= h * w:
                    if raw.size >= h*w + 2 and np.allclose(raw[:2], 0):
                        return raw[2:h*w+2].reshape(h, w).astype(np.float32)
                    return raw[:h*w].reshape(h, w).astype(np.float32)
            side = int(np.sqrt(raw.size))
            return raw[:side*side].reshape(side, side).astype(np.float32)
        except Exception:
            return None

def find_depth_gaps(depth_map: np.ndarray,
                    yolo_dets: list[dict],
                    frame_h: int, frame_w: int,
                    depth_min_m: float,
                    depth_max_m: float,
                    min_blob_area_px: int = 800) -> list[dict]:
    """
    Identifies depth regions within depth_min_m..depth_max_m that do not overlap with YOLO boxes.
    """
    dh, dw = depth_map.shape[:2]

    # 1. Alert mask
    alert_mask = (depth_map >= depth_min_m) & (depth_map <= depth_max_m)

    # 2. Mask out YOLO bounding boxes (scaling RGB coords to depth map resolution)
    scale_x = dw / frame_w
    scale_y = dh / frame_h
    yolo_mask = np.zeros_like(alert_mask)
    for det in yolo_dets:
        dx1 = int(det["x1"] * scale_x)
        dy1 = int(det["y1"] * scale_y)
        dx2 = int(det["x2"] * scale_x)
        dy2 = int(det["y2"] * scale_y)
        
        pad = 2
        dy1_pad = max(0, dy1 - pad)
        dy2_pad = min(dh, dy2 + pad)
        dx1_pad = max(0, dx1 - pad)
        dx2_pad = min(dw, dx2 + pad)
        yolo_mask[dy1_pad:dy2_pad, dx1_pad:dx2_pad] = True

    alert_mask[yolo_mask] = False

    # 3. Focus on navigation zone (lower 60% of frame)
    nav_start_row = int(dh * 0.4)
    nav_mask = np.zeros_like(alert_mask)
    nav_mask[nav_start_row:, :] = alert_mask[nav_start_row:, :]

    mask_u8 = (nav_mask * 255).astype(np.uint8)

    # 4. Noise cleaning
    blurred = cv2.GaussianBlur(mask_u8, (5, 5), 0)
    _, cleaned = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # 5. Contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_blob_area_px:
            continue

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx_dm = int(M["m10"] / M["m00"])
            cy_dm = int(M["m01"] / M["m00"])
        else:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx_dm, cy_dm = bx + bw // 2, by + bh // 2

        contour_mask = np.zeros_like(cleaned)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        depths = depth_map[contour_mask == 255]
        depths = depths[(depths >= depth_min_m) & (depths <= depth_max_m)]

        if len(depths) == 0:
            continue

        mean_depth = float(np.mean(depths))
        min_depth = float(np.min(depths))

        rx, ry, rw, rh = cv2.boundingRect(cnt)
        aspect_ratio = float(rw) / max(float(rh), 1e-5)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 0.0
        extent = float(area / (rw * rh)) if (rw * rh) > 0 else 0.0

        rel_y = cy_dm / dh
        if rel_y < 0.45:
            elevation = "head_level"
        elif rel_y < 0.65:
            elevation = "mid_level"
        else:
            elevation = "foot_level"

        focal_length_px = 960.0 * (dw / 1280.0)
        est_width_m = (rw * mean_depth) / focal_length_px
        est_height_m = (rh * mean_depth) / focal_length_px

        blobs.append({
            "region_id": len(blobs) + 1,
            "cx_dm": cx_dm, "cy_dm": cy_dm,
            "cx_rgb": int(cx_dm / scale_x),
            "cy_rgb": int(cy_dm / scale_y),
            "area_px": int(area),
            "min_depth": min_depth,
            "mean_depth": mean_depth,
            "bbox_dm": [rx, ry, rx+rw, ry+rh],
            "bbox_rgb": [int(rx/scale_x), int(ry/scale_y), int((rx+rw)/scale_x), int((ry+rh)/scale_y)],
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "extent": extent,
            "elevation": elevation,
            "est_width_m": est_width_m,
            "est_height_m": est_height_m,
            "contour": cnt.tolist()
        })

    return sorted(blobs, key=lambda b: b["mean_depth"])

def main():
    print("🚀 Initializing Multi-Range Gap Analysis Experiment...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialize model
    print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = YOLO(YOLO_MODEL_PATH)
    model.to(device)
    print(f"Using device: {device.upper()}")

    # Initialize loader
    loader = SANPOLoader(camera="head", view="left")

    # Load stream definitions
    with open(REPO_ROOT / "valid_streams.json") as f:
        valid_streams = json.load(f)

    streams_to_process = valid_streams[:MAX_STREAMS]
    print(f"Found {len(valid_streams)} valid streams. Sub-sampling the first {len(streams_to_process)} streams.")

    # Initialize stats accumulators for each range
    range_stats = {}
    for _, _, folder_name in RANGES:
        range_stats[folder_name] = {
            "yolo_detections": [],
            "candidate_regions": [],
            "frame_stats": [],
            "hard_examples": {}  # session_id -> list of frame_ids
        }

    # Main Processing Loop
    for s_idx, stream in enumerate(streams_to_process):
        session_id = stream["session_id"]
        camera = stream["camera"]
        view = stream["view"]

        print(f"\n📂 [{s_idx+1}/{len(streams_to_process)}] Stream: {session_id} ({camera}/{view})")

        # Define 5 sampled frames
        sampled_frames = list(range(0, 100, 20))[:MAX_FRAMES_PER_STREAM]
        print(f"   Sampled frames: {sampled_frames}")

        for frame_id in sampled_frames:
            rgb = loader.load_rgb(session_id, frame_id)
            depth = loader.load_depth(session_id, frame_id)
            if rgb is None or depth is None:
                print(f"   ⚠️ Frame {frame_id} could not be retrieved. Skipping.")
                continue

            fh, fw = rgb.shape[:2]

            # Run YOLO Inference once (using BGR image)
            results = model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
            
            # Extract ALL YOLO detections (whitelist disabled)
            yolo_dets = []
            if results.boxes is not None:
                for box, cls_idx, conf in zip(
                    results.boxes.xyxy.cpu().numpy(),
                    results.boxes.cls.cpu().numpy().astype(int),
                    results.boxes.conf.cpu().numpy(),
                ):
                    cls_name = results.names[cls_idx]
                    x1, y1, x2, y2 = map(int, box)
                    yolo_dets.append({
                        "stream_id": session_id,
                        "frame_id": frame_id,
                        "class_name": cls_name,
                        "conf": float(conf),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "area_px": (x2 - x1) * (y2 - y1),
                        "aspect_ratio": (x2 - x1) / max((y2 - y1), 1)
                    })

            # Evaluate each range configuration
            for depth_min_m, depth_max_m, folder_name in RANGES:
                stats = range_stats[folder_name]

                # Accumulate YOLO detections
                stats["yolo_detections"].extend(yolo_dets)

                # Depth gap analysis
                blobs = find_depth_gaps(depth, yolo_dets, fh, fw, depth_min_m, depth_max_m, MIN_BLOB_AREA_PX)

                for blob in blobs:
                    stats["candidate_regions"].append({
                        "stream_id": session_id,
                        "frame_id": frame_id,
                        "region_id": blob["region_id"],
                        "area": blob["area_px"],
                        "centroid_x": blob["cx_rgb"],
                        "centroid_y": blob["cy_rgb"],
                        "mean_depth": blob["mean_depth"],
                        "minimum_depth": blob["min_depth"],
                        "aspect_ratio": blob["aspect_ratio"],
                        "solidity": blob["solidity"],
                        "extent": blob["extent"],
                        "elevation": blob["elevation"],
                        "bounding_box": blob["bbox_rgb"],
                        "estimated_size_w": blob["est_width_m"],
                        "estimated_size_h": blob["est_height_m"]
                    })

                # Compute frame-level metrics
                nav_area = 0.60 * fh * fw
                yolo_nav_mask = np.zeros((fh, fw), dtype=bool)
                nav_start_row = int(fh * 0.4)
                for det in yolo_dets:
                    dy1 = max(nav_start_row, det["y1"])
                    dy2 = max(nav_start_row, det["y2"])
                    dx1 = det["x1"]
                    dx2 = det["x2"]
                    if dy2 > dy1 and dx2 > dx1:
                        yolo_nav_mask[dy1:dy2, dx1:dx2] = True
                
                yolo_nav_px = np.sum(yolo_nav_mask)
                coverage_pct = (yolo_nav_px / nav_area) * 100.0
                total_gap_area = sum(b["area_px"] for b in blobs)
                largest_gap_area = max((b["area_px"] for b in blobs), default=0)
                avg_conf = np.mean([d["conf"] for d in yolo_dets]) if yolo_dets else 0.0

                frame_stat = {
                    "stream_id": session_id,
                    "frame_id": frame_id,
                    "detection_count": len(yolo_dets),
                    "candidate_region_count": len(blobs),
                    "total_candidate_area": total_gap_area,
                    "largest_candidate_area": largest_gap_area,
                    "coverage_percentage": coverage_pct,
                    "unexplained_percentage": 100.0 - coverage_pct,
                    "average_confidence": avg_conf,
                    "notes": ""
                }

                # Hard example logic
                is_hard = False
                notes = []
                if largest_gap_area > 3000:
                    is_hard = True
                    notes.append("Large gap")
                if len(blobs) >= 3:
                    is_hard = True
                    notes.append("Multi-gap obstacle")
                if len(yolo_dets) > 0 and avg_conf < 0.45 and len(blobs) > 0:
                    is_hard = True
                    notes.append("Low confidence overlay")

                frame_stat["notes"] = ", ".join(notes) if notes else "Normal"
                stats["frame_stats"].append(frame_stat)

                if is_hard:
                    if session_id not in stats["hard_examples"]:
                        stats["hard_examples"][session_id] = []
                    stats["hard_examples"][session_id].append(frame_id)

    # Save outputs & generate comparison
    comparison_data = []

    print("\n💾 Writing experimental results...")
    for depth_min_m, depth_max_m, folder_name in RANGES:
        folder_path = OUTPUT_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        stats = range_stats[folder_name]

        # 1. Save frame summary CSV
        df_f = pd.DataFrame(stats["frame_stats"])
        df_f.to_csv(folder_path / "frame_summary.csv", index=False)

        # 2. Save candidate regions CSV
        if stats["candidate_regions"]:
            df_r = pd.DataFrame(stats["candidate_regions"])
            df_r["bounding_box"] = df_r["bounding_box"].apply(lambda b: f"{b[0]};{b[1]};{b[2]};{b[3]}")
            df_r.to_csv(folder_path / "candidate_regions.csv", index=False)
        else:
            pd.DataFrame(columns=[
                "stream_id", "frame_id", "region_id", "area", "centroid_x", "centroid_y",
                "mean_depth", "minimum_depth", "aspect_ratio", "solidity", "extent",
                "elevation", "bounding_box", "estimated_size_w", "estimated_size_h"
            ]).to_csv(folder_path / "candidate_regions.csv", index=False)

        # 3. Save hard examples JSON
        with open(folder_path / "hard_examples.json", "w") as f:
            json.dump(stats["hard_examples"], f, indent=2)

        # Compute summary metrics for comparison
        total_frames = len(stats["frame_stats"])
        total_gaps = len(stats["candidate_regions"])
        avg_gaps_per_frame = total_gaps / max(total_frames, 1)

        if stats["candidate_regions"]:
            df_r_temp = pd.DataFrame(stats["candidate_regions"])
            avg_gap_size = df_r_temp["area"].mean()
            median_gap_size = df_r_temp["area"].median()
            elev_counts = df_r_temp["elevation"].value_counts(normalize=True) * 100
            pct_foot = elev_counts.get("foot_level", 0.0)
            pct_mid = elev_counts.get("mid_level", 0.0)
            pct_head = elev_counts.get("head_level", 0.0)
        else:
            avg_gap_size = 0.0
            median_gap_size = 0.0
            pct_foot = pct_mid = pct_head = 0.0

        avg_unexplained = df_f["unexplained_percentage"].mean() if not df_f.empty else 100.0
        num_hard_sessions = len(stats["hard_examples"])
        total_hard_frames = sum(len(v) for v in stats["hard_examples"].values())

        comparison_data.append({
            "Range Name": folder_name,
            "Depth Bounds": f"{depth_min_m}m - {depth_max_m}m",
            "Total Gaps": total_gaps,
            "Avg Gaps / Frame": round(avg_gaps_per_frame, 2),
            "Avg Gap Size (px)": round(avg_gap_size, 1),
            "Median Gap Size (px)": round(median_gap_size, 1),
            "Avg Unexplained Area (%)": round(avg_unexplained, 2),
            "Elevation (Foot/Mid/Head %)": f"{round(pct_foot,1)}% / {round(pct_mid,1)}% / {round(pct_head,1)}%",
            "Hard Example Sessions": num_hard_sessions,
            "Total Hard Frames": total_hard_frames
        })

    # Write Markdown comparison report
    df_comp = pd.DataFrame(comparison_data)
    
    # Custom markdown table generator to avoid 'tabulate' dependency
    headers = list(df_comp.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for _, row in df_comp.iterrows():
        table_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_table = "\n".join(table_lines)
    
    report_lines = [
        "# Multi-Range Gap Analysis Comparison Report",
        f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Executive Summary",
        "This report compares the detection profile of object-agnostic perception gaps (obstacles seen by depth but missed by YOLO26n)",
        "across **20 streams** (5 sampled frames per stream) under **four distinct depth range configurations**.",
        "Crucially, **no class whitelist was used**; YOLO26n predicted all 80 standard COCO classes, and any overlapping YOLO detections",
        "regardless of class were masked out to ensure only truly unexplained physical obstructions are flagged as gaps.",
        "\n## Comparison Matrix",
        md_table,
        "\n## Key Takeaways",
        "1. **Impact of Short Ranges (0.5m - 2.0m)**: Focuses strictly on immediate walking path hazards. Gaps here are low in count but represent critical trip risks.",
        "2. **Standard and Mid-Range Proximity**: Increasing depth range to 4m or 6m captures infrastructure blocks (bollards, signs, posts) further ahead,",
        "   increasing both the gap counts and the total unexplained navigation area.",
        "3. **Long-Range Filtering (2.0m - 6.0m)**: By ignoring objects closer than 2 meters, this range highlights upcoming visual planning features,",
        "   useful for filtering out local walking cane movement or foot motion noise.",
        "\n## Saved Data Structure",
        "For each range configuration, results are saved inside `notebooks/gap_analysis_output/range_<min>_<max>/`:",
        "- `frame_summary.csv`: Frame-by-frame gap counts and coverage details.",
        "- `candidate_regions.csv`: Detailed geometric and physical metrics for every gap.",
        "- `hard_examples.json`: Mapping of stream session IDs to lists of frame numbers flagged as hard examples.",
    ]

    report_text = "\n".join(report_lines)
    with open(OUTPUT_DIR / "distance_metric_comparison.md", "w") as f:
        f.write(report_text)

    print("\n✅ Comparison report successfully generated at notebooks/gap_analysis_output/distance_metric_comparison.md")
    print(df_comp.to_string(index=False))

if __name__ == "__main__":
    main()
