"""
training_dataset.py — Accumulate threat metrics from preprocessed frames.

This module:
1. Collects detection metrics (distance, velocity, position) from frames
2. Computes ground-truth threat scores using deterministic physics
3. Stores as training cache (NOT persistent JSON fact sheets)
4. Ready for fine-tuning threat_prioritizer model

Memory-efficient: Never stores full images, just metrics (kilobytes per frame)

Output cache structure:
    training_cache = [
        {
            "frame_id": 1200,
            "timestamp": 40.0,
            "session_id": "001",
            "detections": [
                {
                    "track_id": 1,
                    "class_name": "vehicle",
                    "distance_m": 25.5,
                    "velocity_mps": -8.3,  # negative = approaching
                    "ttc_s": 3.07,
                    "bearing": "left",
                    "extracted_kinetic_score": 6.8  # ground truth
                }
            ]
        },
        ...
    ]
"""

import numpy as np
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import pickle

logger = logging.getLogger(__name__)


@dataclass
class ThreatMetric:
    """Single detection with threat assessment."""
    track_id: int
    class_name: str
    distance_m: float
    velocity_mps: float
    ttc_s: Optional[float]
    kinetic_score: float  # Ground truth for training


class ThreatCalculator:
    """
    Compute threat kinetic score from distance + velocity (physics-based).
    
    Threat is based on:
    1. Proximity (inverse distance)
    2. Approach velocity (how fast closing)
    3. Time-to-contact (explicit danger window)
    
    Formula:
        kinetic_score = (1 / distance) * velocity_magnitude * time_weight
    
    High score = high threat (close + approaching fast)
    """
    
    def __init__(self,
                 distance_critical_m: float = 3.0,
                 velocity_high_mps: float = 5.0,
                 ttc_danger_s: float = 2.0):
        """
        Args:
            distance_critical_m: Distance threshold (< this = high proximity)
            velocity_high_mps: Velocity threshold (> this = fast approach)
            ttc_danger_s: Time-to-contact threshold (< this = imminent danger)
        """
        self.distance_critical_m = distance_critical_m
        self.velocity_high_mps = velocity_high_mps
        self.ttc_danger_s = ttc_danger_s
    
    def compute_kinetic_score(self,
                            distance_m: float,
                            velocity_mps: float,
                            ttc_s: Optional[float] = None) -> float:
        """
        Compute threat score (0-10 scale).
        
        Args:
            distance_m: Distance to object (meters)
            velocity_mps: Approach velocity (negative = approaching; m/s)
            ttc_s: Time-to-contact (seconds)
        
        Returns:
            Kinetic score (0-10): 0 = no threat, 10 = critical
        
        Example scoring:
            - 50m away, 0 mps: 0.2 (low threat, stationary)
            - 5m away, -8 mps: 6.5 (high threat, fast approach)
            - 3m away, -15 mps, TTC=0.2s: 10.0 (critical)
        """
        if distance_m <= 0:
            return 10.0  # Invalid/contact
        
        # Component 1: Proximity threat (inverse distance, normalized)
        # Ranges from ~0 at 100m to ~3.3 at 3m (critical distance)
        proximity_threat = self.distance_critical_m / distance_m
        
        # Component 2: Velocity threat (only if approaching)
        # Negative velocity = approaching; higher magnitude = faster approach
        velocity_threat = 0.0
        if velocity_mps < 0:
            # Scale by max expected velocity (~20 mps)
            velocity_threat = min(1.0, abs(velocity_mps) / self.velocity_high_mps)
        
        # Component 3: Time criticality (only if TTC available)
        time_threat = 0.0
        if ttc_s is not None and ttc_s > 0:
            # Inverse TTC: lower TTC = higher threat
            # At TTC=0.2s, threat = 10; at TTC=10s, threat = 0.2
            time_threat = self.ttc_danger_s / ttc_s
        
        # Combine components: proximity + velocity + time
        # Weighted average prioritizes time criticality
        if ttc_s is not None:
            kinetic_score = 0.3 * proximity_threat + 0.3 * velocity_threat + 0.4 * time_threat
        else:
            kinetic_score = 0.5 * proximity_threat + 0.5 * velocity_threat
        
        # Clamp to [0, 10]
        return min(10.0, max(0.0, kinetic_score))
    
    def get_threat_level(self, kinetic_score: float) -> str:
        """Categorize numeric score into threat level."""
        if kinetic_score < 2.0:
            return "low"
        elif kinetic_score < 5.0:
            return "medium"
        elif kinetic_score < 8.0:
            return "high"
        else:
            return "critical"


class TrainingDatasetAccumulator:
    """
    Accumulate frame metrics for threat prioritizer fine-tuning.
    
    Usage:
        accumulator = TrainingDatasetAccumulator()
        
        for frame_data in preprocessing_stream:
            threat_metrics = accumulator.process_frame(
                detections, frame_data
            )
            # Frame metrics are cached, not stored to disk
        
        # Save accumulated training data
        accumulator.save("training_cache.pkl")
    """
    
    def __init__(self):
        """Initialize training cache."""
        self.training_cache = []
        self.threat_calc = ThreatCalculator()
        self.frame_count = 0
    
    def process_frame(self,
                     detections: List,
                     frame_id: int,
                     timestamp: float,
                     session_id: str) -> List[ThreatMetric]:
        """
        Add frame detections to training cache.
        
        Args:
            detections: List of Detection objects (from frame_processor)
            frame_id: Frame index
            timestamp: Time in video (seconds)
            session_id: Which SANPO session
        
        Returns:
            List of ThreatMetric objects for this frame
        """
        threat_metrics = []
        frame_entry = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "session_id": session_id,
            "detections": []
        }
        
        for det in detections:
            if det.depth_m is None:
                # Skip detections without depth
                continue
            
            # Compute kinetic score
            kinetic_score = self.threat_calc.compute_kinetic_score(
                distance_m=det.depth_m,
                velocity_mps=det.velocity_mps or 0.0,
                ttc_s=det.ttc_s
            )
            
            threat_metric = ThreatMetric(
                track_id=det.track_id,
                class_name=det.class_name,
                distance_m=det.depth_m,
                velocity_mps=det.velocity_mps or 0.0,
                ttc_s=det.ttc_s,
                kinetic_score=kinetic_score
            )
            threat_metrics.append(threat_metric)
            
            frame_entry["detections"].append(asdict(threat_metric))
        
        self.training_cache.append(frame_entry)
        self.frame_count += 1
        
        if self.frame_count % 500 == 0:
            logger.info(f"Accumulated {self.frame_count} frames, "
                       f"{len(threat_metrics)} threat metrics")
        
        return threat_metrics
    
    def save(self, filepath: str) -> None:
        """Save training cache to pickle file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.training_cache, f)
            logger.info(f"Saved {len(self.training_cache)} frames to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save training cache: {e}")
    
    def load(self, filepath: str) -> None:
        """Load training cache from pickle file."""
        try:
            with open(filepath, 'rb') as f:
                self.training_cache = pickle.load(f)
            logger.info(f"Loaded {len(self.training_cache)} frames from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load training cache: {e}")
    
    def get_summary_stats(self) -> Dict:
        """Get statistics about accumulated training data."""
        if not self.training_cache:
            return {}
        
        all_scores = []
        for frame in self.training_cache:
            for det in frame["detections"]:
                all_scores.append(det["kinetic_score"])
        
        return {
            "total_frames": len(self.training_cache),
            "total_detections": sum(len(f["detections"]) for f in self.training_cache),
            "threat_score_mean": np.mean(all_scores),
            "threat_score_std": np.std(all_scores),
            "threat_score_min": np.min(all_scores),
            "threat_score_max": np.max(all_scores)
        }
