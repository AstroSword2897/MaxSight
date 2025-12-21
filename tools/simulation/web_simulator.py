"""
MaxSight Web-Based Product Simulator
Complete end-to-end simulation of the MaxSight product on a local web server.

This simulator integrates ALL components:
- Model inference (MaxSightCNN)
- Preprocessing (condition-specific)
- OCR integration
- Output scheduling
- Therapy system
- Description generation
- Spatial memory
- Path planning
- Voice feedback
- Haptic feedback
- Visual overlays
- Session management

Run with: python tools/simulation/web_simulator.py
Access at: http://localhost:5001
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json
import base64
from io import BytesIO
import asyncio
from queue import Queue
import threading
from PIL import Image
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Flask for web server
from flask import Flask, render_template, request, jsonify, send_from_directory  # type: ignore
from flask_cors import CORS  # type: ignore

# Import ALL MaxSight components
from ml.models.maxsight_cnn import create_model
from ml.utils.preprocessing import ImagePreprocessor
from ml.utils.output_scheduler import CrossModalScheduler, OutputConfig
from ml.utils.ocr_integration import OCRIntegration
from ml.utils.description_generator import DescriptionGenerator
from ml.utils.spatial_memory import SpatialMemory
from ml.utils.path_planning import PathPlanner
from ml.therapy.session_manager import SessionManager
from ml.therapy.task_generator import TaskGenerator
from ml.therapy.therapy_integration import TherapyTaskIntegrator
from app.overlays.overlay_engine import OverlayEngine
from app.ui.voice_feedback import VoiceFeedback
from app.ui.haptic_feedback import HapticFeedback, HapticPattern
from ml.utils.logging_config import setup_logging
from ml.utils.output_scheduler import (
    OutputMode, Severity, RuntimeOutput, 
    create_patient_output, create_clinician_output, create_dev_output
)

# Setup logging
logger = setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


app = Flask(__name__, 
            template_folder=Path(__file__).parent / 'templates',
            static_folder=Path(__file__).parent / 'static')
CORS(app)


class MaxSightSimulator:
    """
    Complete MaxSight product simulator integrating all components.
    """
    
    # Configuration constants
    _CONFIG = {
        'confidence_threshold': 0.3,
        'max_ocr_texts_in_description': 3,
        'therapy_difficulty': 0.5,
        'urgency_warning_threshold': 2,
        'haptic_intensity_high': 0.7,
        'haptic_intensity_low': 0.3,
        'baseline_save_frame': 1
    }
    
    def __init__(self, device: Optional[str] = None, output_mode: OutputMode = OutputMode.PATIENT):
        """Initialize all MaxSight components."""
        logger.info("Initializing MaxSight Simulator...")
        
        # Output mode
        self.output_mode = output_mode
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Device: {self.device}")
        
        # Initialize model
        logger.info("Loading model...")
        self.model = create_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize all components
        logger.info("Initializing components...")
        self.preprocessor = None  # Will be set per user condition
        self.scheduler = CrossModalScheduler(OutputConfig())
        self.ocr = OCRIntegration()
        self.description_gen = DescriptionGenerator()
        self.spatial_memory = SpatialMemory()
        self.path_planner = PathPlanner()
        self.session_manager = SessionManager()
        self.task_generator = TaskGenerator()
        self.therapy = TherapyTaskIntegrator()
        self.overlay_engine = OverlayEngine()
        self.voice_feedback = VoiceFeedback()
        self.haptic_feedback = HapticFeedback()
        
        # User state
        self.current_condition = None
        self.current_scenario = None
        self.session_active = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'total_inference_time': 0.0
        }
        
        # Baseline output path for regression testing
        self.baseline_output_path = Path(__file__).parent / 'baseline_output.json'
        
        # Initialize async queues for voice and haptic feedback (thread safety)
        self.voice_queue: Queue = Queue()
        self.haptic_queue: Queue = Queue()
        self._voice_worker_running = False
        self._haptic_worker_running = False
        
        logger.info("Simulator initialized")
        self._start_async_workers()
    
    def set_user_condition(self, condition: str):
        """Set user's visual condition."""
        self.current_condition = condition
        self.preprocessor = ImagePreprocessor(condition_mode=condition)
        logger.info(f"Condition set to: {condition}")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: PIL Image in RGB format
        
        Returns:
            Preprocessed tensor [1, 3, H, W] on device
        """
        import torchvision.transforms as T
        if self.preprocessor:
            preprocessed_tensor = self.preprocessor(image)
            image_tensor = preprocessed_tensor.unsqueeze(0).to(self.device)
        else:
            to_tensor = T.ToTensor()
            image_tensor = to_tensor(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def _postprocess_outputs(self, outputs: Dict[str, Any], confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Post-process model outputs to extract detections.
        
        Args:
            outputs: Raw model outputs dictionary
            confidence_threshold: Minimum confidence for detections
        
        Returns:
            List of detection dictionaries
        """
        detections = self.model.get_detections(outputs, confidence_threshold=confidence_threshold)
        detections_list: List[Dict[str, Any]] = detections[0] if detections else []
        return detections_list
    
    def _run_inference(self, image_tensor: torch.Tensor, audio_features: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], float]:
        """
        Run model inference.
        
        Args:
            image_tensor: Preprocessed image tensor
            audio_features: Optional audio features
        
        Returns:
            Tuple of (raw_outputs dict, inference_time_seconds)
        """
        inference_start = time.perf_counter()
        with torch.no_grad():
            if audio_features is not None:
                audio_tensor = torch.from_numpy(audio_features).unsqueeze(0).to(self.device)
                outputs = self.model(image_tensor, audio_tensor)
            else:
                outputs = self.model(image_tensor)
        inference_time = time.perf_counter() - inference_start
        return outputs, inference_time
    
    def _run_ocr(self, image: Image.Image, outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Run OCR text detection.
        
        Args:
            image: Original PIL image
            outputs: Model outputs dictionary
        
        Returns:
            List of OCR result dictionaries
        """
        ocr_results = []
        try:
            text_scores = outputs.get('text_regions', torch.zeros(1, 196))
            boxes = outputs.get('boxes', torch.zeros(1, 196, 4))
            ocr_results = self.ocr.process_image_for_ocr(
                image=image,
                text_scores=text_scores[0],
                boxes=boxes[0]
            )
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            ocr_results = []
        return ocr_results
    
    def _generate_description(self, detections_list: List[Dict[str, Any]], outputs: Dict[str, Any], ocr_results: List[Dict[str, Any]]) -> str:
        """
        Generate scene description from detections and OCR.
        
        Args:
            detections_list: List of detection dictionaries
            outputs: Model outputs dictionary
            ocr_results: List of OCR result dictionaries
        
        Returns:
            Scene description string
        """
        urgency_score = outputs.get('urgency_scores', torch.zeros(1, 4))
        urgency_level = int(urgency_score.argmax(dim=1).item()) if urgency_score.numel() > 0 else 0
        
        scene_detections = []
        for det in detections_list:
            if 'bbox' in det and 'class_name' in det:
                scene_detections.append({
                    'class_name': det.get('class_name', 'object'),
                    'box': torch.tensor(det.get('bbox', [0.5, 0.5, 0.1, 0.1]), dtype=torch.float32),
                    'distance': det.get('distance', 1),
                    'urgency': det.get('urgency', urgency_level),
                    'priority': det.get('confidence', 0.5) * 100
                })
        
        scene_description = self.description_gen.generate_scene_description(
            detections=scene_detections,
            urgency_score=urgency_level
        )
        
        if ocr_results:
            ocr_texts = [r.get('text', '') for r in ocr_results if r.get('text')]
            if ocr_texts:
                max_texts = self._CONFIG['max_ocr_texts_in_description']
                scene_description += f" Text detected: {', '.join(ocr_texts[:max_texts])}"
        
        return scene_description
    
    def _update_memory(self, detections_list: List[Dict[str, Any]]) -> None:
        """
        Update spatial memory with current detections.
        
        Args:
            detections_list: List of detection dictionaries
        """
        spatial_detections = []
        for det in detections_list:
            if 'bbox' in det and 'class_name' in det:
                spatial_detections.append({
                    'class_name': det['class_name'],
                    'bbox': det['bbox'],
                    'confidence': det.get('confidence', 0.0),
                    'distance': det.get('distance', 1)
                })
        if spatial_detections:
            self.spatial_memory.update(
                detections=spatial_detections,
                timestamp=time.time()
            )
    
    def _plan_path(self, detections_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Plan navigation path if navigation scenario.
        
        Args:
            detections_list: List of detection dictionaries
        
        Returns:
            Path info dictionary or None
        """
        path_info = None
        if self.current_scenario == 'navigation':
            path_result = self.path_planner.plan_path(
                detections=detections_list,
                target_direction='forward'
            )
            if path_result is not None:
                # Convert PathSuggestion to dict if needed
                if hasattr(path_result, '__dict__'):
                    path_info = path_result.__dict__
                else:
                    path_info = {'path': str(path_result)}
        return path_info
    
    def _schedule_outputs(self, detections_list: List[Dict[str, Any]], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Schedule cross-modal outputs (voice, haptic, visual).
        
        Args:
            detections_list: List of detection dictionaries
            outputs: Model outputs dictionary
        
        Returns:
            Scheduled outputs dictionary
        """
        model_outputs: Dict[str, Any] = {}
        urgency_scores = outputs.get('urgency_scores')
        uncertainty = outputs.get('uncertainty')
        if urgency_scores is not None:
            model_outputs['urgency_scores'] = urgency_scores
        if uncertainty is not None:
            model_outputs['uncertainty'] = uncertainty
        
        scheduled_outputs = self.scheduler.schedule_outputs(
            detections=detections_list,
            model_outputs=model_outputs,
            timestamp=time.time()
        )
        # Convert list to dict for consistency
        if isinstance(scheduled_outputs, list):
            return {'outputs': scheduled_outputs, 'count': len(scheduled_outputs)}
        return scheduled_outputs if isinstance(scheduled_outputs, dict) else {'outputs': scheduled_outputs}
    
    def _render_overlay(self, image: Image.Image, detections_list: List[Dict[str, Any]], ocr_results: List[Dict[str, Any]], path_info: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Render visual overlays on image.
        
        Args:
            image: Original PIL image
            detections_list: List of detection dictionaries
            ocr_results: List of OCR result dictionaries
            path_info: Optional path planning information
        
        Returns:
            Base64 encoded overlay image or None
        """
        try:
            urgency_scores = None
            if detections_list and 'urgency' in detections_list[0]:
                urgency_scores = np.array([det.get('urgency', 0) for det in detections_list])
            
            overlay_image = self.overlay_engine.create_overlay(
                base_image=image,
                detections=detections_list,
                urgency_scores=urgency_scores,
                text_regions=ocr_results
            )
            
            # Convert to base64 for web display
            buffered = BytesIO()
            overlay_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.warning(f"Overlay rendering error: {e}")
            return None
    
    def _start_async_workers(self) -> None:
        """Start background workers for async voice and haptic processing."""
        def voice_worker():
            self._voice_worker_running = True
            while self._voice_worker_running:
                try:
                    item = self.voice_queue.get(timeout=0.1)
                    if item is None:
                        break
                    text, priority = item
                    self.voice_feedback.speak_custom(text, priority=priority)
                    self.voice_queue.task_done()
                except:
                    pass
        
        def haptic_worker():
            self._haptic_worker_running = True
            while self._haptic_worker_running:
                try:
                    item = self.haptic_queue.get(timeout=0.1)
                    if item is None:
                        break
                    pattern, intensity = item
                    self.haptic_feedback.trigger(pattern, intensity=intensity)
                    self.haptic_queue.task_done()
                except:
                    pass
        
        # Start worker threads (daemon=True ensures they stop when main thread exits)
        self.voice_thread = threading.Thread(target=voice_worker, daemon=True)
        self.haptic_thread = threading.Thread(target=haptic_worker, daemon=True)
        self.voice_thread.start()
        self.haptic_thread.start()
    
    def shutdown(self) -> None:
        """Gracefully shutdown async workers."""
        self._voice_worker_running = False
        self._haptic_worker_running = False
        # Signal workers to stop
        self.voice_queue.put(None)
        self.haptic_queue.put(None)
        # Wait for threads to finish (with timeout)
        if hasattr(self, 'voice_thread'):
            self.voice_thread.join(timeout=1.0)
        if hasattr(self, 'haptic_thread'):
            self.haptic_thread.join(timeout=1.0)
    
    def _queue_outputs(self, scene_description: str, outputs: Dict[str, Any], detections_list: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Queue voice and haptic outputs asynchronously.
        
        Args:
            scene_description: Generated scene description
            outputs: Model outputs dictionary
            detections_list: List of detection dictionaries
        
        Returns:
            Tuple of (voice_announcements list, haptic_patterns list)
        """
        voice_announcements = []
        haptic_patterns = []
        
        if scene_description:
            self.voice_queue.put((scene_description, 0))
            voice_announcements.append(scene_description)
        
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        if urgency_scores.numel() > 0:
            urgency_level = int(urgency_scores.argmax(dim=1).item())
            warning_threshold = self._CONFIG['urgency_warning_threshold']
            if urgency_level >= warning_threshold:
                self.voice_queue.put((f"Warning: High urgency detected", urgency_level))
                voice_announcements.append(f"Warning: High urgency detected")
                self.haptic_queue.put((
                    HapticPattern.LONG_PULSE,
                    self._CONFIG['haptic_intensity_high']
                ))
                haptic_patterns.append({'pattern': 'long_pulse', 'intensity': self._CONFIG['haptic_intensity_high']})
            elif len(detections_list) > 0:
                self.haptic_queue.put((
                    HapticPattern.MICRO_PULSE,
                    self._CONFIG['haptic_intensity_low']
                ))
                haptic_patterns.append({'pattern': 'micro_pulse', 'intensity': self._CONFIG['haptic_intensity_low']})
        
        return voice_announcements, haptic_patterns
    
    def process_frame(
        self,
        image: Image.Image,
        audio_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process a single frame through the complete MaxSight pipeline.
        
        This integrates ALL components:
        1. Preprocessing (condition-specific)
        2. Model inference
        3. OCR text detection
        4. Description generation
        5. Spatial memory update
        6. Path planning
        7. Output scheduling
        8. Therapy integration
        9. Overlay generation
        10. Voice/haptic feedback
        
        Returns response shaped by output_mode (patient/clinician/dev).
        """
        start_time = time.perf_counter()
        
        # 1. Preprocessing
        image_tensor = self._preprocess_image(image)
        
        # 2. Model inference
        outputs, inference_time = self._run_inference(image_tensor, audio_features)
        
        # 3. Post-process detections
        detections_list = self._postprocess_outputs(outputs, confidence_threshold=self._CONFIG['confidence_threshold'])
        
        # 4. OCR text detection
        ocr_results = self._run_ocr(image, outputs)
        
        # 5. Description generation
        scene_description = self._generate_description(detections_list, outputs, ocr_results)
        
        # 6. Spatial memory update
        self._update_memory(detections_list)
        
        # 7. Path planning
        path_info = self._plan_path(detections_list)
        
        # 8. Output scheduling
        scheduled_outputs = self._schedule_outputs(detections_list, outputs)
        
        # 9. Therapy integration
        therapy_feedback = None
        if self.session_active and detections_list:
            target_objects = [det.get('class_name', 'object') for det in detections_list[:3]]
            therapy_feedback = self.therapy.create_attention_task(
                scene_description=scene_description or "Scene with objects",
                target_objects=target_objects,
                difficulty=self._CONFIG['therapy_difficulty']
            )
        
        # 10. Generate overlays
        overlay_image_b64 = self._render_overlay(image, detections_list, ocr_results, path_info)
        
        # 11. Queue outputs (voice and haptic)
        voice_announcements, haptic_patterns = self._queue_outputs(scene_description, outputs, detections_list)
        
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['total_detections'] += len(detections_list)
        self.stats['avg_latency_ms'] = (self.stats['total_inference_time'] / 
                                        self.stats['frames_processed'] * 1000)
        
        total_time = time.perf_counter() - start_time
        
        # Shape response based on output mode
        result = self._shape_response(
            detections_list=detections_list,
            outputs=outputs,
            scene_description=scene_description,
            ocr_results=ocr_results,
            voice_announcements=voice_announcements,
            haptic_patterns=haptic_patterns,
            path_info=path_info,
            therapy_feedback=therapy_feedback,
            overlay_image_b64=overlay_image_b64,
            inference_time_ms=inference_time * 1000,
            total_time_ms=total_time * 1000
        )
        
        # Save baseline output for regression testing (first frame only)
        self._save_baseline_output(result)
        
        return result
    
    def _shape_response(
        self,
        detections_list: List[Dict[str, Any]],
        outputs: Dict[str, Any],
        scene_description: str,
        ocr_results: List[Dict[str, Any]],
        voice_announcements: List[str],
        haptic_patterns: List[Dict[str, Any]],
        path_info: Optional[Dict[str, Any]],
        therapy_feedback: Optional[Dict[str, Any]],
        overlay_image_b64: Optional[str],
        inference_time_ms: float,
        total_time_ms: float
    ) -> Dict[str, Any]:
        """
        Shape response based on output mode.
        
        Patient mode: minimal, actionable only
        Clinician mode: adds metrics and component breakdown
        Dev mode: full debug information
        """
        # Extract urgency for patient safety
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        urgency_level = int(urgency_scores.argmax(dim=1).item()) if urgency_scores.numel() > 0 else 0
        
        # Determine severity
        if urgency_level >= 3:
            severity = Severity.CRITICAL
        elif urgency_level >= 2:
            severity = Severity.HAZARD
        elif urgency_level >= 1:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        
        # Compute confidence
        objectness = outputs.get('objectness', torch.zeros(1, 196))
        avg_confidence = objectness.max().item() if objectness.numel() > 0 else 0.0
        
        # Patient mode: minimal, calm, actionable
        if self.output_mode == OutputMode.PATIENT:
            # Only top hazard + one instruction
            top_hazards = [d for d in detections_list if d.get('urgency', 0) >= 2]
            if top_hazards:
                message = f"{top_hazards[0]['class_name']} detected"
            elif scene_description:
                # Truncate to first sentence
                message = scene_description.split('.')[0] + '.'
            else:
                message = "Scene clear"
            
            return {
                'mode': 'patient',
                'severity': severity.value,
                'message': message,
                'confidence': round(avg_confidence, 2),
                'cooldown_applied': False,
                'overlay_image': overlay_image_b64
            }
        
        # Clinician mode: adds metrics and breakdown
        elif self.output_mode == OutputMode.CLINICIAN:
            return {
                'mode': 'clinician',
                'severity': severity.value,
                'message': scene_description or "No description",
                'confidence': round(avg_confidence, 2),
                'cooldown_applied': False,
                
                # Clinician-specific fields
                'latency_ms': round(inference_time_ms, 1),
                'total_time_ms': round(total_time_ms, 1),
                'num_detections': len(detections_list),
                'num_hazards': len([d for d in detections_list if d.get('urgency', 0) >= 2]),
                'ocr_texts': [r.get('text', '') for r in ocr_results],
                'component_breakdown': {
                    'detections': len(detections_list),
                    'ocr': len(ocr_results),
                    'voice': len(voice_announcements),
                    'haptic': len(haptic_patterns)
                },
                'overlay_image': overlay_image_b64
            }
        
        # Dev mode: full information
        else:
            return {
                'mode': 'dev',
                'severity': severity.value,
                'message': scene_description or "No description",
                'confidence': round(avg_confidence, 2),
                'cooldown_applied': False,
                
                # Dev-specific fields
                'frame_number': self.stats['frames_processed'],
                'timestamp': time.time(),
                'processing_time_ms': total_time_ms,
                'inference_time_ms': inference_time_ms,
                
                # Full model outputs
                'detections': detections_list,
                'num_detections': len(detections_list),
                'urgency_scores': urgency_scores[0].cpu().tolist(),
                'distance_zones': outputs['distance_zones'][0].cpu().tolist(),
                'scene_embedding': outputs['scene_embedding'][0].cpu().tolist(),
                
                # OCR results
                'text_regions': ocr_results,
                'num_text_regions': len(ocr_results),
                
                # Generated content
                'scene_description': scene_description,
                'scheduled_outputs': scheduled_outputs,
                'voice_announcements': voice_announcements,
                'haptic_patterns': haptic_patterns,
                'path_info': path_info,
                'therapy_feedback': therapy_feedback,
                'overlay_image': overlay_image_b64,
                
                # Statistics
                'stats': self.stats.copy(),
                
                # Debug info
                'debug_info': {
                    'condition': self.current_condition,
                    'scenario': self.current_scenario,
                    'session_active': self.session_active
                }
            }
    
    def _save_baseline_output(self, result: Dict[str, Any]) -> None:
        """
        Save baseline output for regression testing.
        Only saves first frame output to establish baseline.
        """
        if self.stats['frames_processed'] == self._CONFIG['baseline_save_frame']:
            try:
                # Convert tensors to lists for JSON serialization
                baseline = {
                    'frame_number': result['frame_number'],
                    'num_detections': result['num_detections'],
                    'num_text_regions': result['num_text_regions'],
                    'processing_time_ms': result['processing_time_ms'],
                    'inference_time_ms': result['inference_time_ms'],
                    'scene_description': result.get('scene_description', ''),
                    'urgency_scores': result.get('urgency_scores', []),
                    'stats': result.get('stats', {})
                }
                with open(self.baseline_output_path, 'w') as f:
                    json.dump(baseline, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save baseline output: {e}")


# Global simulator instance
simulator = None


def init_simulator():
    """Initialize simulator on first use."""
    global simulator
    if simulator is None:
        simulator = MaxSightSimulator()
    return simulator


# ============================================================================
# Web Routes
# ============================================================================

@app.route('/')
def index():
    """Main simulator interface."""
    return render_template('simulator.html')


@app.route('/api/init', methods=['POST'])
def api_init():
    """Initialize simulator with user settings."""
    data = request.json
    condition = data.get('condition', 'normal')
    scenario = data.get('scenario', 'general')
    output_mode_str = data.get('output_mode', 'patient')
    
    # Parse output mode
    if output_mode_str == 'clinician':
        output_mode = OutputMode.CLINICIAN
    elif output_mode_str == 'dev':
        output_mode = OutputMode.DEV
    else:
        output_mode = OutputMode.PATIENT
    
    sim = init_simulator()
    sim.output_mode = output_mode
    sim.set_user_condition(condition)
    sim.current_scenario = scenario
    sim.session_active = data.get('start_session', False)
    
    if sim.session_active:
        sim.session_manager.start_session()
    
    return jsonify({
        'status': 'initialized',
        'condition': condition,
        'scenario': scenario,
        'output_mode': output_mode_str,
        'session_active': sim.session_active
    })


@app.route('/api/process', methods=['POST'])
def api_process():
    """Process image through complete pipeline."""
    sim = init_simulator()
    
    # Get image from request
    if 'image' in request.files:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
    elif 'image_data' in request.json:
        # Base64 encoded image
        image_data = request.json['image_data']
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    else:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get audio features if provided
    audio_features = None
    if 'audio_features' in request.json:
        audio_features = np.array(request.json['audio_features'])
    
    # Process frame
    result = sim.process_frame(image, audio_features)
    
    # Overlay image is already in result from process_frame
    # If overlay_image is None, fallback to original image
    if not result.get('overlay_image'):
        image_buffer = BytesIO()
        image.save(image_buffer, format='PNG')
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        result['overlay_image'] = f"data:image/png;base64,{image_base64}"
    
    return jsonify(result)


@app.route('/api/scenarios', methods=['GET'])
def api_scenarios():
    """Get available test scenarios."""
    scenarios = [
        {
            'id': 'general',
            'name': 'General Environment',
            'description': 'Standard object detection and scene understanding'
        },
        {
            'id': 'navigation',
            'name': 'Navigation Assistance',
            'description': 'Path planning and obstacle avoidance'
        },
        {
            'id': 'text_reading',
            'name': 'Text Reading',
            'description': 'OCR and text-to-speech focus'
        },
        {
            'id': 'therapy',
            'name': 'Vision Therapy',
            'description': 'Therapy session with task generation'
        },
        {
            'id': 'safety',
            'name': 'Safety Alerts',
            'description': 'Urgency detection and hazard warnings'
        },
        {
            'id': 'accessibility',
            'name': 'Accessibility Features',
            'description': 'Condition-specific adaptations'
        }
    ]
    return jsonify({'scenarios': scenarios})


@app.route('/api/conditions', methods=['GET'])
def api_conditions():
    """Get available visual conditions."""
    conditions = [
        {'id': 'normal', 'name': 'Normal Vision'},
        {'id': 'myopia', 'name': 'Myopia'},
        {'id': 'hyperopia', 'name': 'Hyperopia'},
        {'id': 'astigmatism', 'name': 'Astigmatism'},
        {'id': 'cataracts', 'name': 'Cataracts'},
        {'id': 'glaucoma', 'name': 'Glaucoma'},
        {'id': 'amd', 'name': 'AMD (Age-Related Macular Degeneration)'},
        {'id': 'diabetic_retinopathy', 'name': 'Diabetic Retinopathy'},
        {'id': 'retinitis_pigmentosa', 'name': 'Retinitis Pigmentosa'},
        {'id': 'color_blindness', 'name': 'Color Blindness'},
        {'id': 'cvi', 'name': 'CVI (Cortical Visual Impairment)'},
        {'id': 'amblyopia', 'name': 'Amblyopia'},
        {'id': 'strabismus', 'name': 'Strabismus'}
    ]
    return jsonify({'conditions': conditions})


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get current statistics."""
    sim = init_simulator()
    return jsonify(sim.stats)


@app.route('/api/session/start', methods=['POST'])
def api_session_start():
    """Start therapy session."""
    sim = init_simulator()
    sim.session_active = True
    sim.session_manager.start_session()
    return jsonify({'status': 'session_started'})


@app.route('/api/session/stop', methods=['POST'])
def api_session_stop():
    """Stop therapy session."""
    sim = init_simulator()
    sim.session_active = False
    session_summary = sim.session_manager.end_session()
    return jsonify({
        'status': 'session_stopped',
        'summary': session_summary
    })


@app.route('/api/session/status', methods=['GET'])
def api_session_status():
    """Get session status."""
    sim = init_simulator()
    return jsonify({
        'active': sim.session_active,
        'stats': sim.stats,
        'output_mode': sim.output_mode.value
    })


@app.route('/api/mode', methods=['POST'])
def api_set_mode():
    """Set output mode (patient/clinician/dev)."""
    data = request.json
    mode_str = data.get('mode', 'patient')
    
    if mode_str == 'clinician':
        mode = OutputMode.CLINICIAN
    elif mode_str == 'dev':
        mode = OutputMode.DEV
    else:
        mode = OutputMode.PATIENT
    
    sim = init_simulator()
    sim.output_mode = mode
    
    return jsonify({
        'status': 'mode_updated',
        'output_mode': mode_str
    })


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("MaxSight Product Simulator")
    logger.info("=" * 60)
    logger.info("Starting web server...")
    logger.info("Access the simulator at: http://localhost:5001")
    logger.info("Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5001, debug=True)

