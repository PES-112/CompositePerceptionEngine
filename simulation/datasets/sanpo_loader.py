"""
sanpo_loader.py — Load SANPO-Real dataset frames and depth maps with validation.

This module handles:
1. Reading video frames from SANPO MP4 files (local or GCS)
2. Loading aligned depth maps (.npz format, meters)
3. Validating frame-depth synchronization
4. Efficiently iterating through session data with memory management

SANPO-Real Structure (works with both local and gs:// GCS paths):
    sanpo_data/
    ├── videos/
    │   ├── 001.mp4
    │   ├── 002.mp4
    │   └── ...
    ├── depth_maps/
    │   ├── 001/
    │   │   ├── depth_0000.npz (contains array of shape [H, W])
    │   │   ├── depth_0001.npz
    │   │   └── ...
    │   └── ...
    └── metadata.json

Usage:
    # Local path
    loader = SANPOLoader("/path/to/sanpo")
    
    # Google Cloud Storage
    loader = SANPOLoader("gs://my-bucket/sanpo")

Key Design:
- Generator-based frame iteration (never loads full video into memory)
- On-demand depth loading synchronized with frames
- GCS streaming via google-cloud-storage library
- Early validation of frame-depth alignment
- Logging for debugging data issues
"""

import os
import cv2
import numpy as np
import json
import logging
import tempfile
import io
from typing import Generator, Dict, Optional, Tuple
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
    fps: int
    resolution: Tuple[int, int]


class SANPOLoader:
    """
    Generator-based loader for SANPO-Real dataset (local or GCS).
    
    Supports both local paths and Google Cloud Storage gs:// URIs.
    
    Usage:
        # Local
        loader = SANPOLoader("/path/to/sanpo")
        
        # GCS
        loader = SANPOLoader("gs://bucket-name/sanpo")
        
        for session_id in ["001", "002", "003"]:
            for frame in loader.iter_frames(session_id):
                process_frame(frame)
    """
    
    def __init__(self, sanpo_root: str, metadata_file: str = "metadata.json"):
        """
        Args:
            sanpo_root: Path to root SANPO directory (local or gs://bucket/path)
            metadata_file: Name of metadata JSON file
        """
        self.sanpo_root = sanpo_root
        self.is_gcs = sanpo_root.startswith("gs://")
        self.gcs_client = None
        self.gcs_bucket = None
        
        if self.is_gcs:
            if not GCS_AVAILABLE:
                raise ImportError("google-cloud-storage not installed. Run: pip install google-cloud-storage")
            self._init_gcs()
        
        # Load metadata
        metadata_path = f"{sanpo_root.rstrip('/')}/{metadata_file}"
        metadata_content = self._read_file(metadata_path)
        self.metadata = json.loads(metadata_content)
        
        self.sessions = {s["session_id"]: s for s in self.metadata.get("sessions", [])}
        logger.info(f"Loaded SANPO metadata: {len(self.sessions)} sessions found (GCS: {self.is_gcs})")
    
    def _init_gcs(self):
        """Initialize Google Cloud Storage client."""
        self.gcs_client = storage.Client()
        # Extract bucket name from gs://bucket-name/path
        parts = self.sanpo_root.split("/")
        bucket_name = parts[2]
        self.gcs_bucket = self.gcs_client.bucket(bucket_name)
        self.gcs_prefix = "/".join(parts[3:])  # Path within bucket
        logger.info(f"Initialized GCS: bucket={bucket_name}, prefix={self.gcs_prefix}")
    
    def _read_file(self, file_path: str) -> str:
        """
        Read file content as string (works with local and GCS).
        
        Args:
            file_path: Path to file (local or gs://bucket/path)
        
        Returns:
            File content as string
        """
        if self.is_gcs:
            # Convert gs:// path to blob path within bucket
            blob_path = file_path.replace(f"gs://{self.gcs_bucket.name}/", "")
            blob = self.gcs_bucket.blob(blob_path)
            return blob.download_as_string().decode('utf-8')
        else:
            with open(file_path, 'r') as f:
                return f.read()
    
    def _read_bytes(self, file_path: str) -> bytes:
        """Read file content as bytes (works with local and GCS)."""
        if self.is_gcs:
            blob_path = file_path.replace(f"gs://{self.gcs_bucket.name}/", "")
            blob = self.gcs_bucket.blob(blob_path)
            return blob.download_as_bytes()
        else:
            with open(file_path, 'rb') as f:
                return f.read()
    
    def _get_video_file(self, session_id: str) -> Optional[str]:
        """
        Get path to video file (download from GCS if needed).
        
        Returns:
            Path to video file (local for both local and GCS)
        """
        session_meta = self.sessions.get(session_id)
        if not session_meta:
            return None
        
        video_filename = session_meta["video_file"].split("/")[-1]
        
        if self.is_gcs:
            # Download to temp file
            gcs_path = f"{self.gcs_prefix}/videos/{video_filename}"
            blob = self.gcs_bucket.blob(gcs_path)
            
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            blob.download_to_filename(temp_file.name)
            logger.info(f"Downloaded {gcs_path} to {temp_file.name}")
            return temp_file.name
        else:
            return str(Path(self.sanpo_root) / session_meta["video_file"])
    
    def _get_depth_file(self, session_id: str, frame_id: int) -> Optional[np.ndarray]:
        """
        Get depth map for frame (works with both local and GCS).
        
        Returns:
            Depth array [H, W] float32
        """
        depth_filename = f"depth_{frame_id:04d}.npz"
        
        if self.is_gcs:
            gcs_path = f"{self.gcs_prefix}/depth_maps/{session_id}/{depth_filename}"
            try:
                blob = self.gcs_bucket.blob(gcs_path)
                depth_bytes = blob.download_as_bytes()
                depth_file = io.BytesIO(depth_bytes)
                depth_data = np.load(depth_file)
                return depth_data['arr_0'].astype(np.float32)
            except:
                logger.warning(f"Could not load depth from GCS: {gcs_path}")
                return None
        else:
            depth_path = Path(self.sanpo_root) / "depth_maps" / session_id / depth_filename
            try:
                depth_data = np.load(depth_path)
                return depth_data['arr_0'].astype(np.float32)
            except:
                logger.warning(f"Could not load depth file: {depth_path}")
                return None
    
    def validate_session(self, session_id: str) -> bool:
        """
        Validate that video files exist for a session.
        
        For GCS, we skip validation and rely on download errors.
        
        Args:
            session_id: Session identifier (e.g., "001")
        
        Returns:
            True if session exists in metadata
        """
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not in metadata")
            return False
        
        if not self.is_gcs:
            # Local validation
            session_meta = self.sessions[session_id]
            video_path = Path(self.sanpo_root) / session_meta["video_file"]
            if not video_path.exists():
                logger.error(f"Video missing: {video_path}")
                return False
        
        return True
    
    def iter_frames(self, session_id: str) -> Generator[Frame, None, None]:
        """
        Generator: Iterate through frames of a session with synchronized depth.
        
        Memory efficient: Each frame is yielded and can be processed/discarded
        immediately without keeping history in memory.
        Works with both local and GCS paths.
        
        Args:
            session_id: Session to load (e.g., "001")
        
        Yields:
            Frame objects with rgb + depth synchronized
        
        Raises:
            FileNotFoundError: If session files don't exist
            ValueError: If frame-depth synchronization fails
        """
        if not self.validate_session(session_id):
            raise FileNotFoundError(f"Session {session_id} not found")
        
        session_meta = self.sessions[session_id]
        fps = session_meta["fps"]
        resolution = tuple(session_meta["resolution"])
        
        # Get video file (downloads if GCS)
        video_path = self._get_video_file(session_id)
        if not video_path:
            raise FileNotFoundError(f"Could not get video for session {session_id}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        frame_id = 0
        try:
            while True:
                ret, frame_rgb = cap.read()
                if not ret:
                    break
                
                # Ensure correct shape
                if frame_rgb.shape[:2] != resolution:
                    frame_rgb = cv2.resize(frame_rgb, (resolution[1], resolution[0]))
                
                # Get synchronized depth
                depth_m = self._get_depth_file(session_id, frame_id)
                if depth_m is None:
                    logger.warning(f"Depth missing for frame {frame_id}, skipping")
                    frame_id += 1
                    continue
                
                timestamp = frame_id / fps
                
                yield Frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    rgb=frame_rgb,
                    depth=depth_m,
                    session_id=session_id,
                    fps=fps,
                    resolution=resolution
                )
                
                frame_id += 1
        
        finally:
            cap.release()
            # Clean up temp file if GCS
            if self.is_gcs:
                try:
                    os.remove(video_path)
                    logger.info(f"Cleaned up temp video file: {video_path}")
                except:
                    pass
        """
        if not self.validate_session(session_id):
            raise FileNotFoundError(f"Invalid session: {session_id}")
        
        session_meta = self.sessions[session_id]
        video_file = self.video_dir / session_meta["video_file"].replace("videos/", "")
        depth_folder = self.depth_dir / session_id
        fps = session_meta["fps"]
        resolution = tuple(session_meta["resolution"])
        
        # Open video
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_file}")
        
        frame_id = 0
        try:
            while True:
                ret, frame_rgb = cap.read()
                if not ret:
                    break  # End of video
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                
                # Load corresponding depth map
                depth_file = depth_folder / f"depth_{frame_id:04d}.npz"
                if not depth_file.exists():
                    logger.warning(f"Depth file missing: {depth_file}, skipping frame")
                    frame_id += 1
                    continue
                
                try:
                    depth_data = np.load(depth_file)
                    # Extract the depth array (npz may have multiple arrays)
                    # Typical key: 'arr_0' or 'depth'
                    key = list(depth_data.files)[0]
                    depth = depth_data[key].astype(np.float32)
                except Exception as e:
                    logger.error(f"Error loading depth {depth_file}: {e}")
                    frame_id += 1
                    continue
                
                # Validate alignment
                if frame_rgb.shape[:2] != depth.shape[:2]:
                    raise ValueError(
                        f"Frame {frame_id}: RGB shape {frame_rgb.shape} != "
                        f"Depth shape {depth.shape}"
                    )
                
                timestamp = frame_id / fps
                yield Frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    rgb=frame_rgb,
                    depth=depth,
                    session_id=session_id,
                    fps=fps,
                    resolution=resolution
                )
                
                frame_id += 1
        
        finally:
            cap.release()
            logger.info(f"Session {session_id}: Processed {frame_id} frames")
    
    def get_session_info(self, session_id: str) -> Dict:
        """Get metadata for a specific session."""
        return self.sessions.get(session_id, {})
    
    def list_sessions(self) -> list:
        """List all available session IDs."""
        return list(self.sessions.keys())


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
