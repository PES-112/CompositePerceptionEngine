"""
sanpo_loader.py — Load SANPO-Real dataset frames and depth maps with validation.

This module handles:
1. Reading video frames from SANPO MP4 files
2. Loading aligned depth maps (.npz format, meters)
3. Validating frame-depth synchronization
4. Efficiently iterating through session data with memory management

SANPO-Real Structure:
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
        {
          "sessions": [
            {
              "session_id": "001",
              "video_file": "videos/001.mp4",
              "depth_dir": "depth_maps/001/",
              "num_frames": 450,
              "fps": 15,
              "resolution": [1920, 1080],
              "scene_attributes": {...}
            }
          ]
        }

Key Design:
- Generator-based frame iteration (never loads full video into memory)
- On-demand depth loading synchronized with frames
- Early validation of frame-depth alignment
- Logging for debugging data issues
"""

import os
import cv2
import numpy as np
import json
import logging
from typing import Generator, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

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
    Generator-based loader for SANPO-Real dataset.
    
    Usage:
        loader = SANPOLoader(sanpo_root_dir)
        for session_id in ["001", "002", "003"]:  # Load only specific sessions
            for frame in loader.iter_frames(session_id):
                process_frame(frame)
                # Memory is freed after each frame (generator pattern)
    """
    
    def __init__(self, sanpo_root: str, metadata_file: str = "metadata.json"):
        """
        Args:
            sanpo_root: Path to root SANPO-Real directory
            metadata_file: Name of metadata JSON file
        """
        self.sanpo_root = Path(sanpo_root)
        self.video_dir = self.sanpo_root / "videos"
        self.depth_dir = self.sanpo_root / "depth_maps"
        
        # Load metadata
        metadata_path = self.sanpo_root / metadata_file
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.sessions = {s["session_id"]: s for s in self.metadata.get("sessions", [])}
        logger.info(f"Loaded SANPO metadata: {len(self.sessions)} sessions found")
    
    def validate_session(self, session_id: str) -> bool:
        """
        Validate that video and depth files exist for a session.
        
        Args:
            session_id: Session identifier (e.g., "001")
        
        Returns:
            True if all required files exist, False otherwise
        """
        if session_id not in self.sessions:
            logger.error(f"Session {session_id} not in metadata")
            return False
        
        session_meta = self.sessions[session_id]
        video_file = self.video_dir / session_meta["video_file"].replace("videos/", "")
        depth_folder = self.depth_dir / session_id
        
        video_exists = video_file.exists()
        depth_exists = depth_folder.exists()
        
        if not video_exists:
            logger.error(f"Video missing: {video_file}")
        if not depth_exists:
            logger.error(f"Depth folder missing: {depth_folder}")
        
        return video_exists and depth_exists
    
    def iter_frames(self, session_id: str) -> Generator[Frame, None, None]:
        """
        Generator: Iterate through frames of a session with synchronized depth.
        
        Memory efficient: Each frame is yielded and can be processed/discarded
        immediately without keeping history in memory.
        
        Args:
            session_id: Session to load (e.g., "001")
        
        Yields:
            Frame objects with rgb + depth synchronized
        
        Raises:
            FileNotFoundError: If session files don't exist
            ValueError: If frame-depth synchronization fails
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
