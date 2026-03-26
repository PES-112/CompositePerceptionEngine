"""
sanpo_preprocessing_pipeline.py — End-to-end SANPO stream preprocessing pipeline.

This orchestrator combines all modules:
1. SANPOLoader: Load video + depth frames
2. ScenarioFilter: Filter outdoor pedestrian sessions
3. FrameProcessor: YOLO → Track → Depth extraction
4. TrainingDatasetAccumulator: Accumulate threat metrics

STREAM PROCESSING ARCHITECTURE:
- Load → Detect → Extract → Accumulate → Discard
- Never persist intermediate frame data
- Only store compressed training metrics

Example usage:
    pipeline = SANPOPreprocessingPipeline(
        sanpo_root="/path/to/sanpo_data",
        session_ids=["001", "002", "003"],  # Subset only
        output_dir="/path/to/outputs"
    )
    pipeline.run()
    # Outputs: training_cache.pkl + preprocessing_stats.json
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional
import time

from sanpo_loader import SANPOLoader
from scenario_filter import ScenarioFilter
from frame_processor import FrameProcessor
from training_dataset import TrainingDatasetAccumulator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SANPOPreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for SANPO → Training data.
    
    Design: Stream processing with zero persistent intermediate storage.
    Output: training_cache.pkl + processing statistics
    """
    
    def __init__(self,
                 sanpo_root: str,
                 session_ids: Optional[List[str]] = None,
                 output_dir: str = "./preprocessing_outputs",
                 device: str = "cpu",
                 yolo_model: str = "yolov8n"):
        """
        Args:
            sanpo_root: Path to SANPO-Real root directory
            session_ids: Specific sessions to process (None = all)
            output_dir: Where to save training cache + stats
            device: Compute device ("cpu", "cuda", "mps")
            yolo_model: YOLO model size ("yolov8n", "yolov8s", etc.)
        """
        self.sanpo_root = sanpo_root
        self.session_ids = session_ids
        self.output_dir = Path(output_dir)
        self.device = device
        self.yolo_model = yolo_model
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.loader = SANPOLoader(sanpo_root)
        self.filter = ScenarioFilter()
        self.processor = FrameProcessor(model_name=yolo_model, device=device)
        self.accumulator = TrainingDatasetAccumulator()
        
        # Stats tracking
        self.stats = {
            "total_sessions": 0,
            "valid_sessions": 0,
            "invalid_sessions": 0,
            "total_frames_processed": 0,
            "total_detections": 0,
            "processing_time_sec": 0.0,
            "session_summaries": []
        }
    
    def preprocess_session(self, session_id: str, sample_frame_count: int = 20) -> Dict:
        """
        Preprocess a single SANPO session.
        
        Steps:
        1. Load session metadata
        2. Apply scenario filter (metadata + heuristics)
        3. If valid, stream process all frames
        4. Accumulate threat metrics
        
        Args:
            session_id: Session to process
            sample_frame_count: Frames to sample for heuristic filtering
        
        Returns:
            Session summary dict
        """
        session_meta = self.loader.get_session_info(session_id)
        session_start = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing session: {session_id}")
        logger.info(f"Metadata: {session_meta.get('num_frames')} frames @ "
                   f"{session_meta.get('fps')} fps")
        
        # Stage 1: Metadata filtering (fast)
        filter_result = self.filter.filter_session(session_id, session_meta)
        
        if not filter_result.is_valid:
            logger.info(f"❌ Session {session_id} REJECTED: {filter_result.reason}")
            self.stats["invalid_sessions"] += 1
            return {
                "session_id": session_id,
                "valid": False,
                "reason": filter_result.reason,
                "confidence": filter_result.confidence
            }
        
        logger.info(f"✓ Metadata filters passed: {filter_result.filters_passed}")
        
        # Stage 2: Heuristic filtering (optional, for validation)
        # Sample first 20 frames for brightness/motion checks
        frame_sample = []
        depth_sample = []
        frame_count = 0
        
        try:
            for frame in self.loader.iter_frames(session_id):
                frame_sample.append(frame.rgb)
                depth_sample.append(frame.depth)
                frame_count += 1
                if frame_count >= sample_frame_count:
                    break
        except Exception as e:
            logger.error(f"Error loading sample frames: {e}")
            self.stats["invalid_sessions"] += 1
            return {
                "session_id": session_id,
                "valid": False,
                "reason": f"Failed to load frames: {str(e)}",
                "confidence": 0.0
            }
        
        # Apply heuristic filters
        filter_result = self.filter.filter_session(
            session_id, session_meta,
            frame_sample=frame_sample,
            depth_sample=depth_sample
        )
        
        if not filter_result.is_valid:
            logger.info(f"❌ Session {session_id} REJECTED (heuristics): {filter_result.reason}")
            self.stats["invalid_sessions"] += 1
            return {
                "session_id": session_id,
                "valid": False,
                "reason": filter_result.reason,
                "confidence": filter_result.confidence
            }
        
        logger.info(f"✓ Heuristic filters passed: {filter_result.filters_passed}")
        
        # Stage 3: Stream process all frames
        logger.info(f"Processing all {session_meta.get('num_frames')} frames...")
        
        self.processor.setup()  # Initialize YOLO if needed
        session_detections = 0
        
        try:
            for frame in self.loader.iter_frames(session_id):
                # Process frame
                detections = self.processor.process_frame(
                    frame.rgb,
                    frame.depth,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp
                )
                
                # Accumulate metrics (NOT persistent JSON)
                threat_metrics = self.accumulator.process_frame(
                    detections,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    session_id=session_id
                )
                
                session_detections += len(detections)
                self.stats["total_frames_processed"] += 1
                self.stats["total_detections"] += len(detections)
                
                # Frame data is implicitly freed here (generator pattern)
                if self.stats["total_frames_processed"] % 100 == 0:
                    logger.info(f"  Progress: {self.stats['total_frames_processed']} frames")
        
        except Exception as e:
            logger.error(f"Error processing session frames: {e}")
            self.stats["invalid_sessions"] += 1
            return {
                "session_id": session_id,
                "valid": False,
                "reason": f"Processing failed: {str(e)}",
                "confidence": 0.0
            }
        
        session_time = time.time() - session_start
        self.stats["valid_sessions"] += 1
        self.stats["processing_time_sec"] += session_time
        
        summary = {
            "session_id": session_id,
            "valid": True,
            "num_frames": session_meta.get("num_frames"),
            "detections": session_detections,
            "processing_time_sec": session_time,
            "fps": session_detections / session_time if session_time > 0 else 0
        }
        
        logger.info(f"✓ Session {session_id} COMPLETE: "
                   f"{session_detections} detections in {session_time:.1f}s")
        
        return summary
    
    def run(self):
        """Execute full preprocessing pipeline."""
        logger.info("\n" + "="*60)
        logger.info("SANPO PREPROCESSING PIPELINE START")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info("="*60)
        
        pipeline_start = time.time()
        
        # Determine which sessions to process
        if self.session_ids is None:
            self.session_ids = self.loader.list_sessions()
        
        logger.info(f"Processing {len(self.session_ids)} sessions")
        
        # Process each session
        for session_id in self.session_ids:
            summary = self.preprocess_session(session_id)
            self.stats["session_summaries"].append(summary)
        
        # Finalize stats
        total_time = time.time() - pipeline_start
        self.stats["processing_time_sec"] = total_time
        
        # Save outputs
        self._save_outputs()
        
        # Print summary
        self._print_summary()
    
    def _save_outputs(self):
        """Save training cache and statistics."""
        # Save training cache
        cache_path = self.output_dir / "training_cache.pkl"
        self.accumulator.save(str(cache_path))
        
        # Save statistics
        stats_path = self.output_dir / "preprocessing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"\n✓ Saved training cache: {cache_path}")
        logger.info(f"✓ Saved statistics: {stats_path}")
    
    def _print_summary(self):
        """Print preprocessing summary."""
        logger.info("\n" + "="*60)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total sessions: {self.stats['total_sessions']}")
        logger.info(f"Valid sessions: {self.stats['valid_sessions']}")
        logger.info(f"Invalid sessions: {self.stats['invalid_sessions']}")
        logger.info(f"Total frames processed: {self.stats['total_frames_processed']}")
        logger.info(f"Total detections: {self.stats['total_detections']}")
        logger.info(f"Processing time: {self.stats['processing_time_sec']:.1f}s")
        
        if self.stats["total_frames_processed"] > 0:
            avg_fps = self.stats["total_frames_processed"] / self.stats["processing_time_sec"]
            logger.info(f"Average FPS: {avg_fps:.1f}")
        
        # Training dataset stats
        training_stats = self.accumulator.get_summary_stats()
        if training_stats:
            logger.info(f"\nTraining data statistics:")
            logger.info(f"  Threat score mean: {training_stats['threat_score_mean']:.2f}")
            logger.info(f"  Threat score std: {training_stats['threat_score_std']:.2f}")
            logger.info(f"  Threat score range: [{training_stats['threat_score_min']:.2f}, "
                       f"{training_stats['threat_score_max']:.2f}]")
        
        logger.info("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python sanpo_preprocessing_pipeline.py <sanpo_root> [session_ids...]")
        print("Example: python sanpo_preprocessing_pipeline.py /path/to/sanpo 001 002 003")
        sys.exit(1)
    
    sanpo_root = sys.argv[1]
    session_ids = sys.argv[2:] if len(sys.argv) > 2 else None
    
    pipeline = SANPOPreprocessingPipeline(
        sanpo_root=sanpo_root,
        session_ids=session_ids,
        device="cpu"  # Change to "cuda" or "mps" if GPU available
    )
    pipeline.run()
