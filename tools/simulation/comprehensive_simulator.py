"""Comprehensive MaxSight Simulator."""

import torch
import torch.cuda  # For torch.cuda.synchronize()
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json
from PIL import Image
import cv2
import logging

# Add parent directory to path.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.utils.preprocessing import ImagePreprocessor
from ml.utils.output_scheduler import OutputScheduler
from ml.therapy.session_manager import SessionManager
from ml.therapy.task_generator import TaskGenerator
from ml.utils.logging_config import setup_logging

# Setup logging.
logger = setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


class ComprehensiveSimulator:
    """Comprehensive simulator for MaxSight system with real-world data support."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        condition_mode: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """Initialize simulator."""
        self.verbose = verbose
        self.condition_mode = condition_mode
        
        # Device setup.
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        if self.verbose:
            logger.info(f"Initializing simulator on device: {self.device}")
        
        # Load model.
        if model_path and Path(model_path).exists():
            if self.verbose:
                logger.info(f"Loading model from: {model_path}")
            self.model = torch.load(model_path, map_location=self.device)
        else:
            if self.verbose:
                logger.info("Creating new model...")
            self.model = create_model(condition_mode=condition_mode)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessing.
        self.preprocessor = ImagePreprocessor(condition_mode=condition_mode)
        
        # Initialize output scheduler.
        self.scheduler = OutputScheduler()
        
        # Initialize therapy components.
        self.session_manager = SessionManager()
        self.task_generator = TaskGenerator()
        
        # Statistics.
        self.stats = {
            'frames_processed': 0,
            'total_inference_time': 0.0,
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'fps': 0.0
        }
        
        # Session data.
        self.session_logs = []
        self.is_running = False
    
    def process_image(
        self,
        image_path: str,
        save_output: bool = False,
        output_path: Optional[str] = None,
        show_overlay: bool = True
    ) -> Dict[str, Any]:
        """Process a single image file."""
        if self.verbose:
            print(f"\nProcessing image: {image_path}")
        
        # Load image.
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Preprocess.
        start_time = time.perf_counter()
        preprocessed = self.preprocessor.preprocess(image)
        preprocess_time = time.perf_counter() - start_time
        
        # Convert to tensor.
        if isinstance(preprocessed, Image.Image):
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            image_tensor = to_tensor(preprocessed).unsqueeze(0).to(self.device)
        else:
            image_tensor = preprocessed.unsqueeze(0).to(self.device)
        
        # Inference.
        inference_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(image_tensor)
        inference_time = time.perf_counter() - inference_start
        
        # Post-process detections.
        detections = self.model.get_detections(outputs, confidence_threshold=0.3)
        
        # Update statistics.
        self.stats['frames_processed'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['total_detections'] += len(detections[0]) if detections else 0
        self.stats['avg_latency_ms'] = (self.stats['total_inference_time'] / 
                                        self.stats['frames_processed'] * 1000)
        
        # Create result.
        result = {
            'image_path': image_path,
            'original_size': original_size,
            'preprocess_time_ms': preprocess_time * 1000,
            'inference_time_ms': inference_time * 1000,
            'total_time_ms': (preprocess_time + inference_time) * 1000,
            'num_detections': len(detections[0]) if detections else 0,
            'detections': detections[0] if detections else [],
            'outputs': {
                'classifications': outputs['classifications'].cpu().numpy(),
                'boxes': outputs['boxes'].cpu().numpy(),
                'objectness': outputs['objectness'].cpu().numpy(),
                'urgency_scores': outputs['urgency_scores'].cpu().numpy(),
                'distance_zones': outputs['distance_zones'].cpu().numpy(),
                'scene_embedding': outputs['scene_embedding'].cpu().numpy(),
            },
            'timestamp': time.time()
        }
        
        # Schedule outputs.
        scheduled = self.scheduler.schedule_outputs(
            detections=detections[0] if detections else [],
            urgency_scores=outputs['urgency_scores'][0].cpu().numpy(),
            uncertainty=None
        )
        result['scheduled_outputs'] = scheduled
        
        # Log session.
        self.session_logs.append(result)
        
        # Create visualization if requested.
        if show_overlay or save_output:
            vis_image = self._create_visualization(image, detections[0] if detections else [], outputs)
            
            if save_output:
                if output_path is None:
                    output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_output.jpg")
                vis_image.save(output_path)
                if self.verbose:
                    print(f"Saved output to: {output_path}")
        
        if self.verbose:
            print(f"  Detections: {result['num_detections']}")
            print(f"  Inference time: {result['inference_time_ms']:.2f}ms")
            print(f"  Total time: {result['total_time_ms']:.2f}ms")
        
        return result
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        save_frames: bool = False
    ) -> Dict[str, Any]:
        """Process a video file frame by frame."""
        if self.verbose:
            print(f"\nProcessing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.verbose:
            print(f"  Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
        
        # Setup output video writer if requested.
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_count = 0
        
        self.is_running = True
        
        while self.is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            # Convert BGR to RGB.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Process frame.
            result = self.process_image_frame(frame_pil, frame_number=frame_count)
            frame_results.append(result)
            
            # Create visualization.
            vis_frame = self._create_visualization(
                frame_pil,
                result['detections'],
                result['outputs']
            )
            
            # Convert back to BGR for OpenCV.
            vis_frame_bgr = cv2.cvtColor(np.array(vis_frame), cv2.COLOR_RGB2BGR)
            
            # Write to output video.
            if out_writer:
                out_writer.write(vis_frame_bgr)
            
            # Save individual frames if requested.
            if save_frames:
                frame_output_path = Path(output_path).parent / f"frame_{frame_count:06d}.jpg"
                vis_frame.save(str(frame_output_path))
            
            frame_count += 1
            
            if self.verbose and frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames "
                      f"({self.stats['avg_latency_ms']:.1f}ms avg latency)")
        
        cap.release()
        if out_writer:
            out_writer.release()
        
        self.is_running = False
        
        # Calculate video statistics.
        video_stats = {
            'total_frames': frame_count,
            'video_fps': fps,
            'processing_fps': frame_count / (self.stats['total_inference_time'] + 1e-6),
            'avg_latency_ms': self.stats['avg_latency_ms'],
            'total_detections': self.stats['total_detections'],
            'avg_detections_per_frame': self.stats['total_detections'] / max(frame_count, 1)
        }
        
        if self.verbose:
            print(f"\nVideo processing complete:")
            print(f"  Frames processed: {frame_count}")
            print(f"  Processing FPS: {video_stats['processing_fps']:.2f}")
            print(f"  Average latency: {video_stats['avg_latency_ms']:.2f}ms")
            if output_path:
                print(f"  Output saved to: {output_path}")
        
        return {
            'video_path': video_path,
            'frame_results': frame_results,
            'statistics': video_stats,
            'session_logs': self.session_logs[-frame_count:]
        }
    
    def process_image_frame(
        self,
        image: Image.Image,
        frame_number: int = 0
    ) -> Dict[str, Any]:
        """Process a single image frame (internal method)."""
        # Preprocess.
        preprocessed = self.preprocessor.preprocess(image)
        
        # Convert to tensor.
        if isinstance(preprocessed, Image.Image):
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            image_tensor = to_tensor(preprocessed).unsqueeze(0).to(self.device)
        else:
            image_tensor = preprocessed.unsqueeze(0).to(self.device)
        
        # Inference.
        inference_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(image_tensor)
        inference_time = time.perf_counter() - inference_start
        
        # Post-process.
        detections = self.model.get_detections(outputs, confidence_threshold=0.3)
        
        # Update statistics.
        self.stats['frames_processed'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['total_detections'] += len(detections[0]) if detections else 0
        self.stats['avg_latency_ms'] = (self.stats['total_inference_time'] / 
                                        self.stats['frames_processed'] * 1000)
        self.stats['fps'] = 1.0 / (inference_time + 1e-6)
        
        return {
            'frame_number': frame_number,
            'inference_time_ms': inference_time * 1000,
            'num_detections': len(detections[0]) if detections else 0,
            'detections': detections[0] if detections else [],
            'outputs': {
                'classifications': outputs['classifications'].cpu().numpy(),
                'boxes': outputs['boxes'].cpu().numpy(),
                'objectness': outputs['objectness'].cpu().numpy(),
                'urgency_scores': outputs['urgency_scores'].cpu().numpy(),
                'distance_zones': outputs['distance_zones'].cpu().numpy(),
            }
        }
    
    def process_directory(
        self,
        directory: str,
        pattern: str = "*.jpg",
        output_dir: Optional[str] = None,
        max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process all images in a directory."""
        if self.verbose:
            print(f"\nProcessing directory: {directory}")
        
        dir_path = Path(directory)
        image_files = sorted(list(dir_path.glob(pattern)))
        
        if max_images:
            image_files = image_files[:max_images]
        
        if self.verbose:
            print(f"  Found {len(image_files)} images")
        
        results = []
        
        for i, image_path in enumerate(image_files):
            if self.verbose:
                print(f"\n[{i+1}/{len(image_files)}] Processing: {image_path.name}")
            
            output_path = None
            if output_dir:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                output_path = str(output_dir_path / f"{image_path.stem}_output.jpg")
            
            result = self.process_image(
                str(image_path),
                save_output=output_dir is not None,
                output_path=output_path,
                show_overlay=False
            )
            results.append(result)
        
        if self.verbose:
            print(f"\nDirectory processing complete:")
            print(f"  Images processed: {len(results)}")
            print(f"  Average latency: {self.stats['avg_latency_ms']:.2f}ms")
            print(f"  Total detections: {self.stats['total_detections']}")
        
        return {
            'directory': directory,
            'results': results,
            'statistics': {
                'total_images': len(results),
                'avg_latency_ms': self.stats['avg_latency_ms'],
                'total_detections': self.stats['total_detections'],
                'avg_detections_per_image': self.stats['total_detections'] / max(len(results), 1)
            }
        }
    
    def _create_visualization(
        self,
        image: Image.Image,
        detections: List[Dict],
        outputs: Dict[str, np.ndarray]
    ) -> Image.Image:
        """Create visualization with overlays."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Rectangle
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.axis('off')
        
        # Draw detections.
        for det in detections:
            if 'bbox' in det:
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # Color by urgency.
                urgency = det.get('urgency', 0)
                colors = ['green', 'yellow', 'orange', 'red']
                color = colors[min(urgency, 3)]
                
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Label.
                label = det.get('class_name', 'object')
                confidence = det.get('confidence', 0.0)
                ax.text(x1, y1 - 5, f"{label} ({confidence:.2f})",
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                       fontsize=8, color='white', weight='bold')
        
        # Convert to PIL Image.
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(buf)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            **self.stats,
            'fps': self.stats['fps'] if self.stats['frames_processed'] > 0 else 0.0
        }
    
    def save_session(self, filepath: str):
        """Save session logs to file."""
        session_data = {
            'config': {
                'condition_mode': self.condition_mode,
                'device': str(self.device)
            },
            'statistics': self.get_statistics(),
            'logs': self.session_logs
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        if self.verbose:
            print(f"Session saved to: {filepath}")
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'frames_processed': 0,
            'total_inference_time': 0.0,
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'fps': 0.0
        }
        self.session_logs = []


def main():
    """Command-line interface for simulator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MaxSight Comprehensive Simulator')
    parser.add_argument('input', type=str, help='Input: image file, video file, or directory')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--condition', type=str, help='Visual condition mode')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], help='Device')
    parser.add_argument('--max-frames', type=int, help='Max frames for video')
    parser.add_argument('--max-images', type=int, help='Max images for directory')
    parser.add_argument('--save-session', type=str, help='Save session logs to file')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode')
    
    args = parser.parse_args()
    
    # Initialize simulator.
    simulator = ComprehensiveSimulator(
        condition_mode=args.condition,
        device=args.device,
        verbose=not args.quiet
    )
    
    input_path = Path(args.input)
    
    # Process based on input type.
    if input_path.is_file():
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Image file.
            result = simulator.process_image(
                str(input_path),
                save_output=args.output is not None,
                output_path=args.output
            )
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # Video file.
            result = simulator.process_video(
                str(input_path),
                output_path=args.output,
                max_frames=args.max_frames
            )
        else:
            print(f"Unsupported file type: {input_path.suffix}")
            return
    elif input_path.is_dir():
        # Directory.
        result = simulator.process_directory(
            str(input_path),
            output_dir=args.output,
            max_images=args.max_images
        )
    else:
        print(f"Input not found: {args.input}")
        return
    
    # Save session if requested.
    if args.save_session:
        simulator.save_session(args.save_session)
    
    # Print final statistics.
    stats = simulator.get_statistics()
    print("\n" + "=" * 50)
    print("Final Statistics:")
    print(f"  Frames processed: {stats['frames_processed']}")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  FPS: {stats['fps']:.2f}")
    print(f"  Total detections: {stats['total_detections']}")


if __name__ == "__main__":
    main()






