"""
sanpo_loader.py — Load SANPO-Real dataset from Google Cloud Storage.

This module handles:
1. Reading video frames from SANPO PNG frame sequences (GCS)
2. Loading aligned depth maps (.float16.gz format, meters)
3. Parsing metadata from description.json
4. Efficiently iterating through session data with memory management

SANPO-Real GCS Structure (gs://gresearch/sanpo_dataset/v0/):
    sanpo-real/
    ├── {session_id_hash}/
    │   ├── description.json (metadata)
    │   ├── camera_chest/
    │   │   ├── left/
    │   │   │   ├── video_frames_$folder$ (PNG images)
    │   │   │   │   ├── 000000.png
    │   │   │   │   ├── 000001.png
    │   │   │   │   └── ...
    │   │   │   └── depth_maps/ (.float16.gz files)
    │   │   │       ├── 000000.float16.gz
    │   │   │       ├── 000001.float16.gz
    │   │   │       └── ...
    │   │   └── right/
    │   └── camera_head/
    └── ...

Usage:
    loader = SANPOLoader("gs://gresearch/sanpo_dataset/v0/sanpo-real", camera="chest", view="left")
    
    for session_id in loader.list_sessions()[:5]:
        for frame in loader.iter_frames(session_id):
            process_frame(frame)

Key Design:
- Streams PNG frames from GCS (no local storage)
- Decompresses .float16.gz depth on-the-fly
- Generator-based iteration (memory efficient)
- Supports multiple cameras (chest/head) and views (left/right)
"""

import os
import cv2
import numpy as np
import json
import logging
import tempfile
import gzip
import io
from typing import Generator, Dict, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """A single frame with synchronized depth data."""
    frame_id: int
    timestamp: float  # seconds from video start
    rgb: np.ndarray  # [H, W, 3] uint8
    depth: np.ndarray  # [H, W] float32, in meters
    session_id: str
    camera: str  # "chest" or "head"
    view: str  # "left" or "right"
    fps: int
    resolution: Tuple[int, int]


class SANPOLoader:
    """
    Loader for SANPO-Real dataset from Google Cloud Storage.
    
    Handles PNG frame sequences + .float16.gz depth maps.
    Supports multiple camera views (chest/head, left/right).
    
    Usage:
        loader = SANPOLoader(
            "gs://gresearch/sanpo_dataset/v0/sanpo-real",
            camera="chest",
            view="left"
        )
        
        for session_id in loader.list_sessions()[:5]:
            for frame in loader.iter_frames(session_id):
                process_frame(frame)
    """
    
    def __init__(self, 
                 sanpo_root: str = "gs://gresearch/sanpo_dataset/v0/sanpo-real",
                 camera: str = "chest",
                 view: str = "left"):
        """
        Args:
            sanpo_root: Root path (must be GCS gs:// path)
            camera: "chest" or "head"
            view: "left" or "right"
        """
        if not sanpo_root.startswith("gs://"):
            raise ValueError("sanpo_root must be a GCS path starting with gs://")
        
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage not installed. Run: pip install google-cloud-storage")
        
        self.sanpo_root = sanpo_root.rstrip("/")
        self.camera = camera
        self.view = view
        self.gcs_client = storage.Client()
        
        # Parse bucket and prefix
        parts = self.sanpo_root.split("/")
        self.bucket_name = parts[2]
        self.gcs_prefix = "/".join(parts[3:])
        self.bucket = self.gcs_client.bucket(self.bucket_name)
        
        logger.info(f"Initialized SANPO loader: bucket={self.bucket_name}, prefix={self.gcs_prefix}")
        logger.info(f"Camera: {camera}/{view}")
        
        self.session_cache = None
    
    def list_sessions(self) -> List[str]:
        """
        List all available session IDs in the bucket.
        
        Returns:
            List of session hash IDs
        """
        if self.session_cache is not None:
            return self.session_cache
        
        try:
            # List all blobs with prefix
            prefix = f"{self.gcs_prefix}/"
            blobs = self.bucket.list_blobs(prefix=prefix, delimiter="/")
            
            sessions = []
            for blob in blobs:
                # Each subfolder is a session
                session_id = blob.name.split("/")[-2]
                if session_id and not session_id.endswith("_"):
                    sessions.append(session_id)
            
            self.session_cache = sorted(sessions)
            logger.info(f"Found {len(self.session_cache)} sessions")
            return self.session_cache
        
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    def get_session_metadata(self, session_id: str) -> Dict:
        """
        Load description.json for a session.
        
        Args:
            session_id: Session hash ID
        
        Returns:
            Metadata dict from description.json
        """
        try:
            metadata_path = f"{self.gcs_prefix}/{session_id}/description.json"
            blob = self.bucket.blob(metadata_path)
            metadata_content = blob.download_as_string().decode('utf-8')
            return json.loads(metadata_content)
        except Exception as e:
            logger.warning(f"Could not load metadata for {session_id}: {e}")
            return {}
    
    def _get_png_frame(self, session_id: str, frame_id: int) -> Optional[np.ndarray]:
        """
        Load a single PNG frame from GCS.
        
        Returns:
            [H, W, 3] uint8 RGB array
        """
        try:
            frame_path = (
                f"{self.gcs_prefix}/{session_id}/camera_{self.camera}/{self.view}/"
                f"video_frames_{self.view}/{frame_id:06d}.png"
            )
            blob = self.bucket.blob(frame_path)
            frame_bytes = blob.download_as_bytes()
            
            # Decode PNG
            frame_array = cv2.imdecode(
                np.frombuffer(frame_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            
            # Convert BGR to RGB
            if frame_array is not None:
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            
            return frame_array
        
        except Exception as e:
            logger.warning(f"Could not load frame {frame_id}: {e}")
            return None
    
    def _get_depth_frame(self, session_id: str, frame_id: int) -> Optional[np.ndarray]:
        """
        Load and decompress .float16.gz depth map from GCS.
        
        Returns:
            [H, W] float32 depth array (in meters)
        """
        try:
            depth_path = (
                f"{self.gcs_prefix}/{session_id}/camera_{self.camera}/{self.view}/"
                f"depth_maps/{frame_id:06d}.float16.gz"
            )
            blob = self.bucket.blob(depth_path)
            compressed_bytes = blob.download_as_bytes()
            
            # Decompress gzip
            depth_bytes = gzip.decompress(compressed_bytes)
            
            # Load float16 array
            depth_float16 = np.frombuffer(depth_bytes, dtype=np.float16)
            
            # Reshape (assuming square or standard resolution)
            # SANPO default is typically 720x1280 or similar
            # We'll infer from the byte count
            num_pixels = len(depth_float16)
            h = int(np.sqrt(num_pixels * 720 / 1280))  # Maintain aspect ratio
            w = int(h * 1280 / 720)
            
            # If that doesn't work, try common resolutions
            if h * w != num_pixels:
                common_resolutions = [(720, 1280), (1080, 1920), (480, 640)]
                for try_h, try_w in common_resolutions:
                    if try_h * try_w == num_pixels:
                        h, w = try_h, try_w
                        break
            
            depth_float16 = depth_float16.reshape((h, w))
            
            # Convert to float32
            depth_float32 = depth_float16.astype(np.float32)
            
            return depth_float32
        
        except Exception as e:
            logger.warning(f"Could not load depth frame {frame_id}: {e}")
            return None
    
    def iter_frames(self, session_id: str, max_frames: Optional[int] = None) -> Generator[Frame, None, None]:
        """
        Generator: Iterate through frames of a session with synchronized depth.
        
        Args:
            session_id: Session hash ID
            max_frames: Limit number of frames (for testing)
        
        Yields:
            Frame objects with rgb + depth + metadata
        """
        # Get metadata
        metadata = self.get_session_metadata(session_id)
        fps = metadata.get("fps", 15)  # Default to 15 fps
        
        # Detect resolution from first frame
        first_frame = self._get_png_frame(session_id, 0)
        if first_frame is None:
            logger.error(f"Could not load first frame for {session_id}")
            return
        
        resolution = (first_frame.shape[1], first_frame.shape[0])  # (W, H)
        logger.info(f"Session {session_id}: Resolution {resolution}, FPS {fps}")
        
        frame_id = 0
        while True:
            if max_frames and frame_id >= max_frames:
                break
            
            # Load PNG frame
            rgb_frame = self._get_png_frame(session_id, frame_id)
            if rgb_frame is None:
                break  # No more frames
            
            # Load depth frame
            depth_frame = self._get_depth_frame(session_id, frame_id)
            if depth_frame is None:
                logger.warning(f"Skipping frame {frame_id} (no depth data)")
                frame_id += 1
                continue
            
            timestamp = frame_id / fps
            
            yield Frame(
                frame_id=frame_id,
                timestamp=timestamp,
                rgb=rgb_frame,
                depth=depth_frame,
                session_id=session_id,
                camera=self.camera,
                view=self.view,
                fps=fps,
                resolution=resolution
            )
            
            frame_id += 1


# ============================================================================
# Utility: Quick validation script
# ============================================================================

if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python sanpo_loader.py <sanpo_root_dir>")
        sys.exit(1)
    
    sanpo_root = sys.argv[1]
    loader = SANPOLoader(sanpo_root)
    
    # Test: Load first 10 frames from first session
    first_session = loader.list_sessions()[0]
    print(f"\nTesting session: {first_session}")
    print(f"Metadata: {loader.get_session_info(first_session)}")
    
    frame_count = 0
    for frame in loader.iter_frames(first_session):
        print(f"  Frame {frame.frame_id}: RGB {frame.rgb.shape}, "
              f"Depth {frame.depth.shape}, "
              f"Depth range: [{frame.depth.min():.2f}, {frame.depth.max():.2f}] m")
        frame_count += 1
        if frame_count >= 10:  # Just test 10 frames
            break
