"""v2 threat engine: Kalman fusion, safety envelope, fused priority, announcement manager (docs/architecture.md §6.2-6.5)."""

from src.threat_engine.announcer import Announcement, AnnouncementManager
from src.threat_engine.engine import EngineFrame, ThreatEngine
from src.threat_engine.fusion import EgoPose, FusedTrack, GroundTracker, Measurement
from src.threat_engine.priority import (Arrival, GRUArrivalPredictor, PhysicsArrivalPredictor, RankedThreat,
                                        SceneContext, rank)

__all__ = [
    "Announcement", "AnnouncementManager", "EngineFrame", "ThreatEngine", "EgoPose", "FusedTrack",
    "GroundTracker", "Measurement", "Arrival", "GRUArrivalPredictor", "PhysicsArrivalPredictor",
    "RankedThreat", "SceneContext", "rank",
]
