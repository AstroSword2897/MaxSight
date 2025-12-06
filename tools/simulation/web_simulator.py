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
Access at: http://localhost:5000
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json
import base64
from io import BytesIO
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Flask for web server
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Import ALL MaxSight components
from ml.models.maxsight_cnn import create_model
from ml.utils.preprocessing import ImagePreprocessor
from ml.utils.output_scheduler import OutputScheduler
from ml.utils.ocr_integration import OCRIntegration
from ml.utils.description_generator import DescriptionGenerator
from ml.utils.spatial_memory import SpatialMemory
from ml.utils.path_planning import PathPlanner
from ml.therapy.session_manager import SessionManager
from ml.therapy.task_generator import TaskGenerator
from ml.therapy.therapy_integration import TherapyIntegration
from app.overlays.overlay_engine import OverlayEngine
from app.ui.voice_feedback import VoiceFeedback
from app.ui.haptic_feedback import HapticFeedback


app = Flask(__name__, 
            template_folder=Path(__file__).parent / 'templates',
            static_folder=Path(__file__).parent / 'static')
CORS(app)


class MaxSightSimulator:
    """
    Complete MaxSight product simulator integrating all components.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize all MaxSight components."""
        print("Initializing MaxSight Simulator...")
        
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
        
        print(f"  Device: {self.device}")
        
        # Initialize model
        print("  Loading model...")
        self.model = create_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize all components
        print("  Initializing components...")
        self.preprocessor = None  # Will be set per user condition
        self.scheduler = OutputScheduler()
        self.ocr = OCRIntegration()
        self.description_gen = DescriptionGenerator()
        self.spatial_memory = SpatialMemory()
        self.path_planner = PathPlanner()
        self.session_manager = SessionManager()
        self.task_generator = TaskGenerator()
        self.therapy = TherapyIntegration(
            session_manager=self.session_manager,
            task_generator=self.task_generator
        )
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
        
        print("✅ Simulator initialized!")
    
    def set_user_condition(self, condition: str):
        """Set user's visual condition."""
        self.current_condition = condition
        self.preprocessor = ImagePreprocessor(condition_mode=condition)
        print(f"  Condition set to: {condition}")
    
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
        """
        start_time = time.perf_counter()
        
        # 1. Preprocessing (condition-specific)
        if self.preprocessor:
            preprocessed = self.preprocessor.preprocess(image)
        else:
            preprocessed = image
        
        # Convert to tensor
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        image_tensor = to_tensor(preprocessed).unsqueeze(0).to(self.device)
        
        # 2. Model inference
        inference_start = time.perf_counter()
        with torch.no_grad():
            if audio_features is not None:
                audio_tensor = torch.from_numpy(audio_features).unsqueeze(0).to(self.device)
                outputs = self.model(image_tensor, audio_tensor)
            else:
                outputs = self.model(image_tensor)
        inference_time = time.perf_counter() - inference_start
        
        # 3. Post-process detections
        detections = self.model.get_detections(outputs, confidence_threshold=0.3)
        detections_list = detections[0] if detections else []
        
        # 4. OCR text detection
        try:
            ocr_results = self.ocr.process_image_for_ocr(image)
        except Exception as e:
            print(f"  OCR error: {e}")
            ocr_results = []
        
        # 5. Description generation
        scene_description = self.description_gen.generate_scene_description(
            detections=detections_list,
            urgency_scores=outputs['urgency_scores'][0].cpu().numpy(),
            distance_zones=outputs['distance_zones'][0].cpu().numpy(),
            text_regions=ocr_results
        )
        
        # 6. Spatial memory update
        for det in detections_list:
            if 'bbox' in det and 'class_name' in det:
                self.spatial_memory.update(
                    detection={
                        'class_name': det['class_name'],
                        'bbox': det['bbox'],
                        'confidence': det.get('confidence', 0.0)
                    },
                    frame_id=self.stats['frames_processed']
                )
        
        # 7. Path planning (if navigation scenario)
        path_info = None
        if self.current_scenario == 'navigation':
            # Get spatial context
            spatial_context = self.spatial_memory.get_spatial_summary()
            path_info = self.path_planner.plan_path(
                current_position=(0, 0),  # Would come from GPS/sensors
                obstacles=spatial_context.get('objects', [])
            )
        
        # 8. Output scheduling
        scheduled_outputs = self.scheduler.schedule_outputs(
            detections=detections_list,
            urgency_scores=outputs['urgency_scores'][0].cpu().numpy(),
            uncertainty=None
        )
        
        # 9. Therapy integration
        therapy_feedback = None
        if self.session_active:
            therapy_feedback = self.therapy.process_frame(
                detections=detections_list,
                user_interaction=None
            )
        
        # 10. Generate overlays
        overlay_image = self.overlay_engine.create_overlay(
            base_image=image,
            detections=detections_list,
            urgency_scores=outputs['urgency_scores'][0].cpu().numpy(),
            text_regions=ocr_results
        )
        
        # 11. Generate voice feedback
        voice_announcements = self.voice_feedback.generate_announcements(
            detections=detections_list,
            urgency_scores=outputs['urgency_scores'][0].cpu().numpy(),
            scene_description=scene_description,
            scheduled_outputs=scheduled_outputs
        )
        
        # 12. Generate haptic feedback
        haptic_patterns = self.haptic_feedback.generate_patterns(
            detections=detections_list,
            urgency_scores=outputs['urgency_scores'][0].cpu().numpy(),
            path_info=path_info
        )
        
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['total_detections'] += len(detections_list)
        self.stats['avg_latency_ms'] = (self.stats['total_inference_time'] / 
                                        self.stats['frames_processed'] * 1000)
        
        total_time = time.perf_counter() - start_time
        
        # Compile complete result
        result = {
            'frame_number': self.stats['frames_processed'],
            'timestamp': time.time(),
            'processing_time_ms': total_time * 1000,
            'inference_time_ms': inference_time * 1000,
            
            # Model outputs
            'detections': detections_list,
            'num_detections': len(detections_list),
            'urgency_scores': outputs['urgency_scores'][0].cpu().tolist(),
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
            
            # Statistics
            'stats': self.stats.copy()
        }
        
        return result, overlay_image


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
    
    sim = init_simulator()
    sim.set_user_condition(condition)
    sim.current_scenario = scenario
    sim.session_active = data.get('start_session', False)
    
    if sim.session_active:
        sim.session_manager.start_session(user_id='simulator_user')
    
    return jsonify({
        'status': 'initialized',
        'condition': condition,
        'scenario': scenario,
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
    result, overlay_image = sim.process_frame(image, audio_features)
    
    # Convert overlay to base64
    overlay_buffer = BytesIO()
    overlay_image.save(overlay_buffer, format='PNG')
    overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode('utf-8')
    
    result['overlay_image'] = f"data:image/png;base64,{overlay_base64}"
    
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
    sim.session_manager.start_session(user_id='simulator_user')
    return jsonify({'status': 'session_started'})


@app.route('/api/session/stop', methods=['POST'])
def api_session_stop():
    """Stop therapy session."""
    sim = init_simulator()
    sim.session_active = False
    session_summary = sim.session_manager.end_session('simulator_user')
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
        'stats': sim.stats
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MaxSight Product Simulator")
    print("=" * 60)
    print("\nStarting web server...")
    print("Access the simulator at: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

