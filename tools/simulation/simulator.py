"""
Simulation Harness

End-to-end simulation for testing therapy system.

Phase 5: End-to-End Integration
See docs/therapy_system_implementation_plan.md for implementation details.

NOTE: This is a simplified simulator. For production use, see:
- tools/simulation/web_simulator.py (MaxSightSession) for multi-user web interface
- tools/simulation/comprehensive_simulator.py for full-featured simulation
"""

from typing import Dict, List, Optional, Any
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class TherapySimulator:
    """
    End-to-end simulation harness for therapy system.
    
    Built with Pygame or recorded sessions:
    - Run model
    - Display overlays
    - Accept simulated taps
    - Log outputs
    
    NOTE: This is a basic implementation. For production use, integrate with
    MaxSightSession from web_simulator.py or use ComprehensiveSimulator.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, model=None):
        """
        Initialize simulator.
        
        Arguments:
            config: Optional configuration dictionary
            model: Optional MaxSightCNN model instance (if None, will need to be set later)
        """
        self.config = config or {}
        self.model = model
        self.is_running = False
        self.frame_count = 0
        self.logs = []
        self.processing_times = []  # Track processing times for summary
    
    def start_simulation(self, video_source: Optional[str] = None):
        """
        Start simulation.
        
        Arguments:
            video_source: Optional video file path or camera index
            NOTE: Video source handling not implemented - use ComprehensiveSimulator for video support
        """
        self.is_running = True
        self.frame_count = 0
        self.logs = []
        self.processing_times = []
        
        if video_source:
            logger.warning(
                "Video source parameter provided but not implemented. "
                "Use ComprehensiveSimulator for video support."
            )
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame.
        
        Arguments:
            frame: Input frame [H, W, 3] as numpy array
        
        Returns:
            Processing results dictionary with:
                - frame_number: Frame index
                - timestamp: Timestamp in seconds
                - model_output: Model outputs (if model available)
                - overlays: List of overlay data
                - user_input: User input data (if any)
                - processing_time_ms: Processing time in milliseconds
        """
        if not self.is_running:
            logger.warning("Simulation not started. Call start_simulation() first.")
            return {}
        
        start_time = time.perf_counter()
        self.frame_count += 1
        
        # Run model inference if model is available
        model_output = {}
        if self.model is not None:
            try:
                import torch
                from PIL import Image
                
                # Convert numpy array to PIL Image
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)
                
                # Convert to tensor (simplified - should use proper preprocessing)
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(pil_image).unsqueeze(0)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    model_output = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                                   for k, v in outputs.items()}
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                model_output = {'error': str(e)}
        else:
            logger.debug("No model available - skipping inference")
        
        # Generate overlays (simplified - should use overlay_engine)
        overlays = []
        if model_output and 'boxes' in model_output:
            # Basic overlay generation
            overlays = [{'type': 'detection', 'data': model_output.get('boxes', [])}]
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(processing_time_ms)
        
        result = {
            'frame_number': self.frame_count,
            'timestamp': self.frame_count / 30.0,  # Assuming 30 FPS
            'model_output': model_output,
            'overlays': overlays,
            'user_input': None,  # User input handling not implemented
            'processing_time_ms': processing_time_ms
        }
        
        self.logs.append(result)
        return result
    
    def stop_simulation(self) -> Dict[str, Any]:
        """
        Stop simulation and return summary.
        
        Returns:
            Simulation summary with:
                - total_frames: Total frames processed
                - logs: All frame logs
                - summary: Summary statistics
        """
        self.is_running = False
        
        return {
            'total_frames': self.frame_count,
            'logs': self.logs,
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate simulation summary with statistics.
        
        Returns:
            Summary dictionary with:
                - frames_processed: Number of frames processed
                - avg_processing_time_ms: Average processing time
                - min_processing_time_ms: Minimum processing time
                - max_processing_time_ms: Maximum processing time
                - errors: List of errors encountered
        """
        errors = []
        for log in self.logs:
            if 'error' in log.get('model_output', {}):
                errors.append({
                    'frame': log['frame_number'],
                    'error': log['model_output']['error']
                })
        
        avg_time = 0.0
        min_time = 0.0
        max_time = 0.0
        
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            min_time = min(self.processing_times)
            max_time = max(self.processing_times)
        
        return {
            'frames_processed': self.frame_count,
            'avg_processing_time_ms': avg_time,
            'min_processing_time_ms': min_time,
            'max_processing_time_ms': max_time,
            'errors': errors
        }
    
    def save_logs(self, filepath: str):
        """
        Save simulation logs to file.
        
        Arguments:
            filepath: Path to save JSON log file
        """
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'config': self.config,
                'logs': self.logs,
                'summary': self._generate_summary()
            }, f, indent=2)

