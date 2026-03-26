"""
demo_inference.py — Run full CPE pipeline on a custom video (e.g., iPhone recording).

This script:
1. Loads custom video (MP4, MOV, etc.)
2. Extracts depth (mock for custom videos without depth sensor)
3. Runs YOLO detections + threat scoring
4. Generates audio warnings via TTS

For demonstration purposes:
- Simulates egocentric perspective (full frame = FOV)
- Generates mock depth (uses focus heuristics if available)
- Streams predictions with real-time TTS playback

Example usage:
    python demo_inference.py /path/to/demo_video.mp4 --model threat_prioritizer_finetuned.pt
"""

import cv2
import numpy as np
import logging
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
import time

import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

from frame_processor import FrameProcessor, Detection
from training_dataset import ThreatCalculator

logger = logging.getLogger(__name__)


class GeneratedDepthEstimator:
    """
    Generate plausible depth maps for custom videos without depth sensors.
    
    Strategy:
    1. Use brightness as proxy for distance (darker = farther for outdoor)
    2. Add object size heuristics (larger objects = closer for known classes)
    3. Temporal smoothing (depth doesn't jump between frames)
    """
    
    def __init__(self):
        self.prev_depth = None
    
    def estimate_depth(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Generate plausible depth map for frame.
        
        Args:
            frame_rgb: [H, W, 3] RGB image
        
        Returns:
            [H, W] depth map in meters (range ~0.5 to 100m)
        """
        h, w = frame_rgb.shape[:2]
        
        # Brightness-based depth: darker = farther
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        brightness = gray / 255.0  # [0, 1]
        
        # Map brightness to depth: bright=close (1m), dark=far (50m)
        depth = 1.0 + brightness * 49.0
        
        # Add distance gradient (center = closer, edges = farther)
        y, x = np.ogrid[0:h, 0:w]
        cy, cx = h / 2, w / 2
        distance_from_center = np.sqrt((y - cy)**2 + (x - cx)**2)
        distance_from_center = distance_from_center / np.sqrt(h**2 + w**2)
        
        depth = depth * (1.0 + distance_from_center * 0.5)
        
        # Temporal smoothing (if previous frame available)
        if self.prev_depth is not None:
            depth = 0.7 * self.prev_depth + 0.3 * depth
        
        self.prev_depth = depth.copy()
        
        return depth.astype(np.float32)


class ThreatTTSGenerator:
    """Generate audio warnings for detected threats."""
    
    def __init__(self, use_tts: bool = False):
        """
        Args:
            use_tts: Use real TTS (requires TTS package) or text-only
        """
        self.use_tts = use_tts and TTS_AVAILABLE
        if self.use_tts:
            try:
                # Use lightweight model for speed
                self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", 
                              gpu=False)
            except:
                logger.warning("Failed to load TTS; falling back to text-only")
                self.use_tts = False
    
    def generate_warning(self, threat_text: str) -> str:
        """
        Generate audio warning for threat.
        
        Args:
            threat_text: Description of threat (e.g., "Vehicle approaching from left")
        
        Returns:
            Audio file path (if using TTS) or text message
        """
        if not self.use_tts:
            logger.info(f"🔊 THREAT WARNING: {threat_text}")
            return threat_text
        
        try:
            output_path = "/tmp/threat_warning.wav"
            self.tts.tts_to_file(text=threat_text, file_path=output_path)
            logger.info(f"🔊 Generated audio: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            logger.info(f"🔊 THREAT WARNING: {threat_text}")
            return threat_text


class CPEDemoInference:
    """Run full CPE pipeline on custom video for demonstration."""
    
    def __init__(self,
                 video_path: str,
                 model_path: Optional[str] = None,
                 device: str = "cpu",
                 output_video_path: Optional[str] = None,
                 use_tts: bool = False):
        """
        Args:
            video_path: Path to input video
            model_path: Path to fine-tuned threat_prioritizer.pt (optional)
            device: "cpu", "cuda", or "mps"
            output_video_path: Where to save annotated video (optional)
            use_tts: Enable audio warnings (requires TTS)
        """
        self.video_path = Path(video_path)
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.output_video_path = Path(output_video_path) if output_video_path else None
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Initialize components
        self.frame_processor = FrameProcessor(device=device)
        self.frame_processor.setup()
        
        self.depth_estimator = GeneratedDepthEstimator()
        self.threat_calc = ThreatCalculator()
        self.tts_gen = ThreatTTSGenerator(use_tts=use_tts)
        
        # Load custom threat model if provided
        self.threat_model = None
        if model_path:
            self._load_threat_model(model_path, device)
        
        # Video reading
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {self.width}x{self.height} @ {self.fps} fps, "
                   f"{self.total_frames} frames")
        
        # Video writer (if saving output)
        self.writer = None
        if self.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                str(self.output_video_path),
                fourcc, self.fps,
                (self.width, self.height)
            )
    
    def _load_threat_model(self, model_path: str, device: str):
        """Load fine-tuned threat prioritizer model."""
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Import threat model class
            from threat_prioritizer_finetuner import ThreatPrioritizerMLP
            
            num_classes = checkpoint.get("num_classes", 5)
            self.threat_model = ThreatPrioritizerMLP(num_classes=num_classes)
            self.threat_model.load_state_dict(checkpoint["model_state"])
            self.threat_model = self.threat_model.to(device)
            self.threat_model.eval()
            
            logger.info(f"Loaded threat model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load threat model: {e}")
            self.threat_model = None
    
    def _compute_threat_with_model(self, detections: List[Detection]) -> List[dict]:
        """Compute threat scores using fine-tuned model (if available)."""
        if not self.threat_model:
            # Fall back to deterministic scoring
            return [
                {
                    "kinetic_score": self.threat_calc.compute_kinetic_score(
                        det.depth_m or 50.0,
                        det.velocity_mps or 0.0,
                        det.ttc_s
                    ),
                    "threat_level": self.threat_calc.get_threat_level(
                        self.threat_calc.compute_kinetic_score(
                            det.depth_m or 50.0,
                            det.velocity_mps or 0.0,
                            det.ttc_s
                        )
                    )
                }
                for det in detections
            ]
        
        # Use neural model
        threats = []
        for det in detections:
            # Normalize inputs
            distance_norm = min(1.0, (det.depth_m or 50.0) / 100.0)
            velocity_norm = np.tanh((det.velocity_mps or 0.0) / 10.0)
            ttc_norm = min(1.0, (det.ttc_s or 5.0) / 10.0)
            
            # For simplicity, use generic class vector (in practice, one-hot)
            class_vector = np.zeros(5, dtype=np.float32)
            class_vector[0 if det.class_name == "vehicle" else 1] = 1.0
            
            features = np.concatenate([
                [distance_norm], [velocity_norm], [ttc_norm], class_vector
            ])
            
            with torch.no_grad():
                features_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
                score_norm = self.threat_model(features_t).cpu().numpy()[0, 0]
            
            kinetic_score = score_norm * 10.0  # Denormalize
            
            threats.append({
                "kinetic_score": kinetic_score,
                "threat_level": self.threat_calc.get_threat_level(kinetic_score)
            })
        
        return threats
    
    def _get_bearing(self, x_center: int) -> str:
        """Estimate bearing from pixel x-coordinate."""
        mid_x = self.width / 2
        if x_center < mid_x * 0.33:
            return "left"
        elif x_center > mid_x * 0.67:
            return "right"
        else:
            return "center"
    
    def _annotate_frame(self, frame_rgb: np.ndarray, detections: List[Detection],
                       threats: List[dict]) -> np.ndarray:
        """Draw detections and threat annotations on frame."""
        frame_annotated = frame_rgb.copy()
        
        for det, threat in zip(detections, threats):
            x1, y1, x2, y2 = det.bbox
            
            # Color based on threat level
            threat_level = threat["threat_level"]
            if threat_level == "critical":
                color = (0, 0, 255)  # Red
            elif threat_level == "high":
                color = (0, 165, 255)  # Orange
            elif threat_level == "medium":
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with threat score
            label = f"{det.class_name} {threat['kinetic_score']:.1f}"
            cv2.putText(frame_annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw depth and velocity info
            if det.depth_m:
                info = f"d={det.depth_m:.1f}m v={det.velocity_mps or 0:.1f}m/s"
                cv2.putText(frame_annotated, info, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame_annotated
    
    def run(self, max_frames: Optional[int] = None):
        """
        Run CPE pipeline on video.
        
        Args:
            max_frames: Maximum frames to process (None = all)
        """
        logger.info(f"\n{'='*60}")
        logger.info("CPE DEMO INFERENCE START")
        logger.info(f"Video: {self.video_path}")
        logger.info(f"{'='*60}")
        
        frame_id = 0
        processed_frames = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame_bgr = self.cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp = frame_id / self.fps
                
                # Generate depth map (mock)
                depth_map = self.depth_estimator.estimate_depth(frame_rgb)
                
                # Process frame: YOLO + Tracking + Depth
                detections = self.frame_processor.process_frame(
                    frame_rgb, depth_map, frame_id, timestamp
                )
                
                # Compute threat scores
                threats = self._compute_threat_with_model(detections)
                
                # Generate warnings for high-threat objects
                for det, threat in zip(detections, threats):
                    if threat["threat_level"] in ["high", "critical"]:
                        bearing = self._get_bearing(int(det.centroid[0]))
                        distance = det.depth_m or 50.0
                        warning = (f"{threat['threat_level'].upper()}: "
                                 f"{det.class_name} from {bearing}, "
                                 f"{distance:.1f}m away")
                        self.tts_gen.generate_warning(warning)
                
                # Annotate frame
                frame_annotated = self._annotate_frame(frame_rgb, detections, threats)
                
                # Write to output video
                if self.writer:
                    frame_annotated_bgr = cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR)
                    self.writer.write(frame_annotated_bgr)
                
                processed_frames += 1
                if processed_frames % 30 == 0:
                    logger.info(f"Processed {processed_frames} frames "
                               f"({processed_frames/self.total_frames*100:.1f}%)")
                
                frame_id += 1
                if max_frames and frame_id >= max_frames:
                    break
        
        finally:
            elapsed = time.time() - start_time
            self.cap.release()
            if self.writer:
                self.writer.release()
            
            logger.info(f"\n{'='*60}")
            logger.info("CPE DEMO INFERENCE COMPLETE")
            logger.info(f"Frames processed: {processed_frames}")
            logger.info(f"Processing time: {elapsed:.1f}s")
            logger.info(f"FPS: {processed_frames/elapsed:.1f}")
            if self.output_video_path:
                logger.info(f"Output video: {self.output_video_path}")
            logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPE demo on custom video")
    parser.add_argument("video", help="Input video path (MP4, MOV, etc.)")
    parser.add_argument("--model", help="Path to threat_prioritizer_finetuned.pt")
    parser.add_argument("--output", help="Output video path (optional)")
    parser.add_argument("--max-frames", type=int, help="Max frames to process")
    parser.add_argument("--device", default="cpu", help="cpu/cuda/mps")
    parser.add_argument("--tts", action="store_true", help="Enable audio warnings")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    demo = CPEDemoInference(
        video_path=args.video,
        model_path=args.model,
        device=args.device,
        output_video_path=args.output,
        use_tts=args.tts
    )
    
    demo.run(max_frames=args.max_frames)
