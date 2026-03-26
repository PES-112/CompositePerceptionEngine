"""
frame_processor.py — Process frames for YOLO detection, tracking, and depth extraction.

This module:
1. Runs YOLO26 Nano inference on frames (from ultralytics)
2. Tracks objects across frames (simple centroid-based, or ByteTrack if available)
3. Extracts depth values at detection centroids
4. Computes velocity from depth changes
5. Calculates Time-To-Contact (TTC)

Design Philosophy - EFFICIENT & MEMORY-FRIENDLY:
- Load YOLO model once, reuse across frames
- Process frames one-at-a-time (generator pattern)
- Never cache full detections, summarize per-frame
- Depth extraction uses centroid-based lookup (fast)

Output Format (per-frame):
    {
        "frame_id": 1200,
        "timestamp": 40.0,
        "detections": [
            {
                "track_id": 1,
                "class": "vehicle",
                "bbox": [x1, y1, x2, y2],  # pixel coordinates
                "confidence": 0.95,
                "center": [x_c, y_c],
                "depth_m": 25.5,           # from depth map at centroid
                "velocity_mps": 8.3,       # change in distance/time
                "ttc_s": 3.07              # time-to-contact, if moving toward
            }
        ]
    }
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not installed; detection will be mocked")

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection in a frame."""
    track_id: int
    class_name: str
    class_id: int
    bbox: Tuple[int, int, int, int]  # [x1, y1, x2, y2]
    confidence: float
    centroid: Tuple[float, float]  # [x_c, y_c]
    depth_m: Optional[float] = None
    velocity_mps: Optional[float] = None
    ttc_s: Optional[float] = None


class SimpleCentroidTracker:
    """
    Simple object tracker using centroid distances.
    
    For production, replace with ByteTrack for better multi-person scenarios.
    This is sufficient for proof-of-concept.
    
    Algorithm:
    1. Compute centroids of current detections
    2. Find nearest centroids from previous frame using distance
    3. Assign track IDs, creating new tracks for unmatched detections
    4. Remove tracks without recent detections (>5 frames old)
    """
    
    def __init__(self, max_distance: float = 50.0, max_age: int = 10):
        """
        Args:
            max_distance: Max pixel distance to consider same object
            max_age: Max frames to keep track alive without detections
        """
        self.centroids = {}  # {track_id: (x, y)}
        self.track_ages = {}  # {track_id: frames_since_seen}
        self.next_track_id = 0
        self.max_distance = max_distance
        self.max_age = max_age
    
    def update(self, detections: List[Dict]) -> List[int]:
        """
        Update tracker with new detections; assign track IDs.
        
        Args:
            detections: List of {bbox, confidence, class} (no track_id yet)
        
        Returns:
            List of track IDs corresponding to input detections
        """
        if len(detections) == 0:
            # No detections: age all tracks
            self.track_ages = {tid: age + 1 for tid, age in self.track_ages.items()}
            # Remove old tracks
            old_ids = [tid for tid, age in self.track_ages.items() if age > self.max_age]
            for tid in old_ids:
                del self.centroids[tid]
                del self.track_ages[tid]
            return []
        
        # Current centroids
        current_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            current_centroids.append((cx, cy))
        
        # Match to previous centroids
        track_ids = []
        matched = set()
        
        for i, curr_centroid in enumerate(current_centroids):
            if not self.centroids:
                # No previous tracks: create new
                track_id = self.next_track_id
                self.next_track_id += 1
                track_ids.append(track_id)
            else:
                # Find nearest previous centroid
                distances = []
                for track_id, prev_centroid in self.centroids.items():
                    dist = np.sqrt((curr_centroid[0] - prev_centroid[0])**2 +
                                  (curr_centroid[1] - prev_centroid[1])**2)
                    distances.append((dist, track_id))
                
                nearest_dist, nearest_id = min(distances, key=lambda x: x[0])
                
                if nearest_dist < self.max_distance and nearest_id not in matched:
                    # Match to existing track
                    track_ids.append(nearest_id)
                    matched.add(nearest_id)
                else:
                    # Create new track
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    track_ids.append(track_id)
        
        # Update centroids and ages
        self.centroids = {}
        for track_id, centroid in zip(track_ids, current_centroids):
            self.centroids[track_id] = centroid
            self.track_ages[track_id] = 0
        
        # Age unmatched tracks
        for track_id in list(self.track_ages.keys()):
            if track_id not in track_ids:
                self.track_ages[track_id] += 1
        
        # Remove old tracks
        old_ids = [tid for tid, age in self.track_ages.items() if age > self.max_age]
        for tid in old_ids:
            del self.centroids[tid]
            del self.track_ages[tid]
        
        return track_ids


class FrameProcessor:
    """
    Process frames: YOLO detection → Tracking → Depth extraction → Threat metrics.
    
    Usage:
        processor = FrameProcessor(model_name="yolov8n", device="cpu")
        processor.setup()
        
        for frame_data in frame_stream:
            detections = processor.process_frame(
                frame_data.rgb,
                frame_data.depth,
                frame_id=frame_data.frame_id
            )
            # Use detections
    """
    
    def __init__(self,
                 model_name: str = "yolov8n",  # nano = lightweight
                 device: str = "cpu",
                 conf_threshold: float = 0.5):
        """
        Args:
            model_name: YOLO model ("yolov8n", "yolov8s", etc.)
            device: Device ("cpu", "cuda", "mps" for Mac)
            conf_threshold: Minimum detection confidence to keep
        """
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = None
        self.tracker = SimpleCentroidTracker()
        self.prev_detections = {}  # {track_id: Detection} for velocity calc
        self.prev_timestamp = None
    
    def setup(self):
        """Load YOLO model (call once before processing)."""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available; using mock detections")
            return
        
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info(f"Loaded {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            self.model = None
    
    def extract_detections(self, frame_rgb: np.ndarray) -> List[Dict]:
        """
        Run YOLO inference on frame.
        
        Args:
            frame_rgb: [H, W, 3] uint8 image
        
        Returns:
            List of {bbox, confidence, class_id, class_name}
        """
        if self.model is None:
            logger.warning("YOLO model not loaded; returning mock detections")
            return self._mock_detections(frame_rgb)
        
        try:
            # YOLO inference
            results = self.model(frame_rgb, conf=self.conf_threshold, verbose=False)
            detections = []
            
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf)
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    })
            
            return detections
        
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return []
    
    def _mock_detections(self, frame_rgb: np.ndarray) -> List[Dict]:
        """Generate mock detections for testing (when YOLO unavailable)."""
        h, w = frame_rgb.shape[:2]
        mock_detections = [
            {
                "bbox": (int(w*0.3), int(h*0.4), int(w*0.5), int(h*0.7)),
                "confidence": 0.95,
                "class_id": 2,
                "class_name": "vehicle"
            },
            {
                "bbox": (int(w*0.6), int(h*0.3), int(w*0.85), int(h*0.8)),
                "confidence": 0.88,
                "class_id": 0,
                "class_name": "person"
            }
        ]
        return mock_detections
    
    def extract_depth_at_centroid(self,
                                  depth_map: np.ndarray,
                                  centroid: Tuple[float, float]) -> Optional[float]:
        """
        Extract depth value at detection centroid from depth map.
        
        Args:
            depth_map: [H, W] float32 depth in meters
            centroid: (x, y) pixel coordinates
        
        Returns:
            Depth in meters at centroid, or None if invalid
        """
        x, y = int(centroid[0]), int(centroid[1])
        h, w = depth_map.shape[:2]
        
        if 0 <= x < w and 0 <= y < h:
            depth_val = float(depth_map[y, x])
            # Valid if positive (0 typically means invalid/no reading)
            if depth_val > 0:
                return depth_val
        
        return None
    
    def compute_velocity_and_ttc(self,
                                 track_id: int,
                                 current_depth: float,
                                 current_timestamp: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute velocity and time-to-contact from depth changes.
        
        Args:
            track_id: Object track ID
            current_depth: Current depth in meters
            current_timestamp: Current frame timestamp
        
        Returns:
            (velocity_mps, ttc_s) or (None, None) if can't compute
        """
        if track_id not in self.prev_detections:
            return None, None
        
        prev_det = self.prev_detections[track_id]
        if prev_det.depth_m is None or self.prev_timestamp is None:
            return None, None
        
        dt = current_timestamp - self.prev_timestamp
        if dt <= 0:
            return None, None
        
        # Velocity: change in distance per time
        # Positive = moving away, Negative = moving toward
        depth_change = current_depth - prev_det.depth_m
        velocity = depth_change / dt  # meters per second
        
        # TTC (time-to-contact): if approaching (velocity < 0)
        ttc = None
        if velocity < 0:  # Moving toward
            ttc = abs(current_depth / velocity)  # seconds until contact
        
        return velocity, ttc
    
    def process_frame(self,
                     frame_rgb: np.ndarray,
                     depth_map: np.ndarray,
                     frame_id: int,
                     timestamp: float) -> List[Detection]:
        """
        Full pipeline: YOLO → Track → Depth → Velocity → TTC.
        
        Args:
            frame_rgb: [H, W, 3] uint8 RGB image
            depth_map: [H, W] float32 depth in meters
            frame_id: Frame index
            timestamp: Time in seconds from video start
        
        Returns:
            List of Detection objects with all metrics computed
        """
        # Step 1: YOLO detection
        raw_detections = self.extract_detections(frame_rgb)
        
        # Step 2: Tracking (assign IDs)
        track_ids = self.tracker.update(raw_detections)
        
        # Step 3: Enrich with depth, velocity, TTC
        detections = []
        for track_id, raw_det in zip(track_ids, raw_detections):
            x1, y1, x2, y2 = raw_det["bbox"]
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
            # Extract depth at centroid
            depth_m = self.extract_depth_at_centroid(depth_map, centroid)
            
            # Compute velocity and TTC
            velocity_mps, ttc_s = None, None
            if depth_m is not None:
                velocity_mps, ttc_s = self.compute_velocity_and_ttc(
                    track_id, depth_m, timestamp
                )
            
            detection = Detection(
                track_id=track_id,
                class_name=raw_det["class_name"],
                class_id=raw_det["class_id"],
                bbox=raw_det["bbox"],
                confidence=raw_det["confidence"],
                centroid=centroid,
                depth_m=depth_m,
                velocity_mps=velocity_mps,
                ttc_s=ttc_s
            )
            detections.append(detection)
            
            # Cache for next frame
            self.prev_detections[track_id] = detection
        
        self.prev_timestamp = timestamp
        
        return detections
