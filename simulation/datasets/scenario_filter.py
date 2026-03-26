"""
scenario_filter.py — Filter SANPO sessions to outdoor pedestrian scenarios only.

This module implements heuristics to identify valid outdoor walking scenarios:
1. Metadata-based filtering (scene_attributes in SANPO)
2. Motion detection (depth variance across frames)
3. Brightness check (outdoor typically well-lit)
4. Valid duration (minimum frames to be useful)

Why we filter:
- SANPO contains indoor sessions (offices), static shots, vehicle footage
- We only want egocentric pedestrian navigation in outdoor urban environments
- Reduces dataset to ~45-50% relevant scenes, keeping training focused

Filtering logic:
    If (scene_type == "outdoor" AND 
        traffic_density IN ["low", "medium", "high"] AND
        environment == "sidewalk" OR "street" AND
        brightness > 50% AND
        motion_score > 0.1 AND
        duration > 10 sec):
        KEEP
    Else:
        DISCARD
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of applying filters to a session."""
    session_id: str
    is_valid: bool
    reason: str  # Why valid or why rejected
    confidence: float  # [0, 1] confidence in decision
    filters_passed: Dict[str, bool]  # Individual filter results


class ScenarioFilter:
    """
    Determine if a SANPO session is valid for outdoor pedestrian training.
    
    Uses multi-level filtering:
    1. Metadata-based (fast, deterministic)
    2. Motion/brightness heuristics (requires frame sampling)
    """
    
    def __init__(self, 
                 min_frames: int = 150,  # ~10 sec at 15 fps
                 brightness_threshold: float = 50.0,
                 motion_threshold: float = 0.1):
        """
        Args:
            min_frames: Minimum frame count to keep session
            brightness_threshold: % brightness (0-100) required for outdoor
            motion_threshold: Depth variance threshold for motion detection
        """
        self.min_frames = min_frames
        self.brightness_threshold = brightness_threshold
        self.motion_threshold = motion_threshold
    
    def filter_by_metadata(self, session_meta: Dict) -> Dict[str, bool]:
        """
        Fast metadata-based filtering. No frame processing needed.
        
        Args:
            session_meta: Session metadata from SANPO
        
        Returns:
            Dict of filter results:
            {
                'has_scene_attributes': bool,
                'is_outdoor': bool,
                'valid_environment': bool,
                'valid_traffic': bool,
                'valid_duration': bool
            }
        """
        filters = {}
        
        # 1. Check if metadata is complete
        scene_attrs = session_meta.get("scene_attributes", {})
        filters['has_scene_attributes'] = len(scene_attrs) > 0
        
        # 2. Check outdoor indicator
        # SANPO attributes: 'indoor_flag' (True=indoor) or 'location_type' (outdoor/indoor)
        indoor_flag = scene_attrs.get("indoor_flag", False)
        location_type = scene_attrs.get("location_type", "").lower()
        filters['is_outdoor'] = (not indoor_flag) and (location_type == "outdoor")
        
        # 3. Check valid environment (sidewalk, street, plaza)
        environment = scene_attrs.get("environment", "").lower()
        valid_envs = {"sidewalk", "street", "plaza", "crosswalk", "pedestrian_zone"}
        filters['valid_environment'] = environment in valid_envs
        
        # 4. Check traffic level (we want scenarios with pedestrians/vehicles)
        # Low, medium, or high traffic are all interesting
        traffic = scene_attrs.get("traffic_level", "").lower()
        filters['valid_traffic'] = traffic in {"low", "medium", "high"}
        
        # 5. Check minimum duration
        num_frames = session_meta.get("num_frames", 0)
        filters['valid_duration'] = num_frames >= self.min_frames
        
        return filters
    
    def compute_brightness(self, rgb_frame: np.ndarray) -> float:
        """
        Compute brightness of frame as % (0-100).
        
        Outdoor scenes typically have brightness > 50%.
        Indoor scenes typically < 50% or extremely overexposed.
        
        Args:
            rgb_frame: [H, W, 3] uint8 image
        
        Returns:
            Brightness percentage (0-100)
        """
        # Convert to grayscale and normalize
        gray = np.dot(rgb_frame[..., :3], [0.299, 0.587, 0.114])
        brightness = (gray.mean() / 255.0) * 100.0
        return brightness
    
    def compute_motion_score(self, 
                            depth_frames: List[np.ndarray],
                            sample_size: int = 10) -> float:
        """
        Estimate motion from depth variance across sampled frames.
        
        High variance = motion (person walking through scene)
        Low variance = static scene (standing still or very gradual movement)
        
        Args:
            depth_frames: List of [H, W] depth arrays
            sample_size: Number of frames to sample (doesn't require all)
        
        Returns:
            Motion score (0-1): 0 = static, 1 = high motion
        """
        if len(depth_frames) < 2:
            return 0.0
        
        # Sample frames to test (don't need to check all)
        indices = np.linspace(0, len(depth_frames) - 1, min(sample_size, len(depth_frames)), dtype=int)
        sampled_frames = [depth_frames[i] for i in indices]
        
        # Compute frame-to-frame depth changes
        depth_diffs = []
        for i in range(len(sampled_frames) - 1):
            # Absolute difference in depth values (centroid-based)
            diff = np.abs(sampled_frames[i].mean() - sampled_frames[i + 1].mean())
            depth_diffs.append(diff)
        
        # Motion score: normalized variance of depth changes
        if len(depth_diffs) > 0:
            motion_score = min(1.0, np.std(depth_diffs) / 2.0)  # Normalize
        else:
            motion_score = 0.0
        
        return motion_score
    
    def filter_session(self,
                      session_id: str,
                      session_meta: Dict,
                      frame_sample: Optional[List[np.ndarray]] = None,
                      depth_sample: Optional[List[np.ndarray]] = None) -> FilterResult:
        """
        Apply all filters to determine if session is valid.
        
        Args:
            session_id: Session ID
            session_meta: Session metadata
            frame_sample: List of RGB frames (optional, for brightness check)
            depth_sample: List of depth frames (optional, for motion check)
        
        Returns:
            FilterResult with decision and reasoning
        """
        filters_passed = {}
        reasons = []
        
        # Stage 1: Metadata filtering
        metadata_filters = self.filter_by_metadata(session_meta)
        filters_passed.update(metadata_filters)
        
        if not metadata_filters['is_outdoor']:
            return FilterResult(
                session_id=session_id,
                is_valid=False,
                reason="Not marked as outdoor",
                confidence=1.0,
                filters_passed=filters_passed
            )
        
        if not metadata_filters['valid_environment']:
            return FilterResult(
                session_id=session_id,
                is_valid=False,
                reason=f"Invalid environment: {session_meta.get('scene_attributes', {}).get('environment')}",
                confidence=1.0,
                filters_passed=filters_passed
            )
        
        if not metadata_filters['valid_duration']:
            return FilterResult(
                session_id=session_id,
                is_valid=False,
                reason=f"Too short: {session_meta.get('num_frames', 0)} < {self.min_frames}",
                confidence=1.0,
                filters_passed=filters_passed
            )
        
        # Stage 2: Heuristic filtering (if samples provided)
        if frame_sample:
            brightness_scores = [self.compute_brightness(f) for f in frame_sample]
            mean_brightness = np.mean(brightness_scores)
            filters_passed['brightness_ok'] = mean_brightness > self.brightness_threshold
            
            if not filters_passed['brightness_ok']:
                return FilterResult(
                    session_id=session_id,
                    is_valid=False,
                    reason=f"Too dark: {mean_brightness:.1f}% < {self.brightness_threshold}%",
                    confidence=0.8,
                    filters_passed=filters_passed
                )
        
        if depth_sample:
            motion_score = self.compute_motion_score(depth_sample)
            filters_passed['motion_ok'] = motion_score > self.motion_threshold
            
            if not filters_passed['motion_ok']:
                return FilterResult(
                    session_id=session_id,
                    is_valid=False,
                    reason=f"Low motion: {motion_score:.3f} < {self.motion_threshold}",
                    confidence=0.7,
                    filters_passed=filters_passed
                )
        
        # All filters passed
        return FilterResult(
            session_id=session_id,
            is_valid=True,
            reason="All filters passed",
            confidence=0.95,
            filters_passed=filters_passed
        )


if __name__ == "__main__":
    # Example: Test filter on mock metadata
    mock_session = {
        "session_id": "001",
        "num_frames": 450,
        "fps": 15,
        "scene_attributes": {
            "indoor_flag": False,
            "location_type": "outdoor",
            "environment": "sidewalk",
            "traffic_level": "medium"
        }
    }
    
    filter = ScenarioFilter()
    result = filter.filter_session("001", mock_session)
    print(f"Session {result.session_id}: {result.is_valid}")
    print(f"Reason: {result.reason}")
    print(f"Filters: {result.filters_passed}")
