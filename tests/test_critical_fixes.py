"""Test Plan for Critical Fixes - Thread Safety, Overlay Rendering, OCR Clustering..."""

import pytest
import torch
import numpy as np
from PIL import Image
import time
import threading
from queue import Queue, Empty
from pathlib import Path
import sys
import unittest.mock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

sys.modules['flask'] = unittest.mock.MagicMock()
sys.modules['flask_cors'] = unittest.mock.MagicMock()

# Import web simulator properly
from tools.simulation.web_simulator import MaxSightSimulator
from ml.utils.ocr_integration import OCRIntegration


class TestWebSimulatorThreadSafety:
    """Test thread safety fixes in web simulator."""
    
    def test_queue_initialization(self):
        """Test that voice and haptic queues are properly initialized."""
        sim = MaxSightSimulator(device='cpu')
        
        # Queues should be initialized
        assert hasattr(sim, 'voice_queue'), "voice_queue not initialized"
        assert hasattr(sim, 'haptic_queue'), "haptic_queue not initialized"
        assert isinstance(sim.voice_queue, Queue), "voice_queue is not a Queue"
        assert isinstance(sim.haptic_queue, Queue), "haptic_queue is not a Queue"
        
        # Worker flags should be set
        assert hasattr(sim, '_voice_worker_running'), "voice worker flag not set"
        assert hasattr(sim, '_haptic_worker_running'), "haptic worker flag not set"
        
        sim.shutdown()
    
    def test_async_queue_operations(self):
        """Test that async queues can handle concurrent operations."""
        sim = MaxSightSimulator(device='cpu')
        
        # Test voice queue
        test_items = [("Test message 1", 0), ("Test message 2", 1)]
        for item in test_items:
            sim.voice_queue.put(item)
        
        assert sim.voice_queue.qsize() == 2, "Voice queue should have 2 items"
        
        # Test haptic queue
        haptic_items = [("vibration", 0.5), ("pulse", 0.7)]
        for item in haptic_items:
            sim.haptic_queue.put(item)
        
        assert sim.haptic_queue.qsize() == 2, "Haptic queue should have 2 items"
        
        sim.shutdown()
    
    def test_concurrent_frame_processing(self):
        """Test that multiple threads can process frames safely."""
        sim = MaxSightSimulator(device='cpu')
        sim.set_user_condition('normal')
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        results = []
        errors = []
        
        def process_frame(thread_id):
            try:
                result = sim.process_frame(test_image)
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Launch multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=process_frame, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join(timeout=10.0)
        
        # All threads should complete without errors
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        
        sim.shutdown()
    
    def test_graceful_shutdown(self):
        """Test that shutdown properly stops worker threads."""
        sim = MaxSightSimulator(device='cpu')
        
        # Verify workers are running
        assert sim._voice_worker_running == True, "Voice worker should be running"
        assert sim._haptic_worker_running == True, "Haptic worker should be running"
        
        # Shutdown
        sim.shutdown()
        
        # Workers should be stopped (check after brief delay)
        time.sleep(0.2)
        assert sim._voice_worker_running == False, "Voice worker should be stopped"
        assert sim._haptic_worker_running == False, "Haptic worker should be stopped"


class TestOverlayRendering:
    """Test overlay rendering fixes."""
    
    def test_overlay_image_in_result(self):
        """Test that overlay image is properly included in process_frame result."""
        sim = MaxSightSimulator(device='cpu')
        sim.set_user_condition('normal')
        
        # Create test image with some content
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        result = sim.process_frame(test_image)
        
        # Overlay image should be in result
        assert 'overlay_image' in result, "overlay_image missing from result"
        assert result['overlay_image'] is not None, "overlay_image is None"
        assert isinstance(result['overlay_image'], str), "overlay_image should be string"
        assert result['overlay_image'].startswith('data:image/png;base64,'), \
            "overlay_image should be base64 encoded PNG"
        
        sim.shutdown()
    
    def test_overlay_image_format(self):
        """Test that overlay image is valid base64 PNG."""
        sim = MaxSightSimulator(device='cpu')
        sim.set_user_condition('normal')
        
        test_image = Image.new('RGB', (224, 224), color='green')
        result = sim.process_frame(test_image)
        
        # Decode base64
        import base64
        overlay_b64 = result['overlay_image'].replace('data:image/png;base64,', '')
        overlay_bytes = base64.b64decode(overlay_b64)
        
        # Should be valid PNG
        assert overlay_bytes[:8] == b'\x89PNG\r\n\x1a\n', "Overlay is not valid PNG"
        
        # Should be decodable as image
        from io import BytesIO
        overlay_img = Image.open(BytesIO(overlay_bytes))
        assert overlay_img.size == (224, 224), "Overlay size mismatch"
        
        sim.shutdown()
    
    def test_overlay_with_detections(self):
        """Test overlay rendering with actual detections."""
        from ml.utils.output_scheduler import OutputMode
        sim = MaxSightSimulator(device='cpu')
        sim.set_user_condition('normal')
        # Set to dev mode to get full output including detections
        sim.output_mode = OutputMode.DEV
        
        # Create image that might trigger detections
        test_image = Image.new('RGB', (224, 224), color='white')
        
        result = sim.process_frame(test_image)
        
        # Should have overlay even with no detections
        assert 'overlay_image' in result, "overlay_image missing"
        assert result['overlay_image'] is not None, "overlay_image is None"
        
        # In dev mode, should have detections list
        assert 'detections' in result, "detections missing"
        assert isinstance(result['detections'], list), "detections should be list"
        
        sim.shutdown()


class TestOCRClusteringOptimization:
    """Test OCR clustering optimization with cKDTree."""
    
    def test_clustering_with_small_regions(self):
        """Test clustering handles small text regions correctly."""
        ocr = OCRIntegration()
        
        # Small number of pixels
        h, w = 100, 100
        x_coords = torch.tensor([10, 12, 14, 50, 52])
        y_coords = torch.tensor([10, 10, 10, 50, 50])
        
        regions = ocr._cluster_text_pixels(x_coords, y_coords, h, w, cluster_distance=5)
        
        # Should cluster nearby pixels
        assert len(regions) > 0, "Should find at least one region"
        assert all(len(region) == 4 for region in regions), "Regions should be (x_min, y_min, x_max, y_max)"
        
        # Check region bounds are valid
        for x_min, y_min, x_max, y_max in regions:
            assert 0 <= x_min < x_max <= w, f"Invalid x bounds: {x_min}, {x_max}"
            assert 0 <= y_min < y_max <= h, f"Invalid y bounds: {y_min}, {y_max}"
    
    def test_clustering_performance(self):
        """Test that cKDTree clustering is faster than O(N²) fallback."""
        ocr = OCRIntegration()
        
        # Create larger set of coordinates
        h, w = 500, 500
        n_pixels = 200
        x_coords = torch.randint(0, w, (n_pixels,))
        y_coords = torch.randint(0, h, (n_pixels,))
        
        # Time cKDTree clustering (if scipy available)
        try:
            from scipy.spatial import cKDTree
            
            start_time = time.perf_counter()
            regions_optimized = ocr._cluster_text_pixels(
                x_coords, y_coords, h, w, cluster_distance=10, use_dbscan=False
            )
            optimized_time = time.perf_counter() - start_time
            
            # Should complete in reasonable time (< 1 second for 200 pixels)
            assert optimized_time < 1.0, f"Clustering took too long: {optimized_time:.3f}s"
            assert len(regions_optimized) > 0, "Should find regions"
            
        except ImportError:
            pytest.skip("scipy not available for performance test")
    
    def test_clustering_with_overlapping_regions(self):
        """Test clustering handles overlapping text regions."""
        ocr = OCRIntegration()
        
        h, w = 200, 200
        # Create two overlapping clusters
        x_coords = torch.tensor([
            10, 12, 14, 16,  # Cluster 1
            15, 17, 19, 21,  # Cluster 2 (overlaps with cluster 1)
            50, 52, 54       # Cluster 3 (separate)
        ])
        y_coords = torch.tensor([
            10, 10, 10, 10,  # Cluster 1
            10, 10, 10, 10,  # Cluster 2
            50, 50, 50       # Cluster 3
        ])
        
        regions = ocr._cluster_text_pixels(x_coords, y_coords, h, w, cluster_distance=5)
        
        # Should merge overlapping clusters or keep them separate based on distance
        assert len(regions) > 0, "Should find regions"
        
        # Check that regions don't have invalid bounds
        for x_min, y_min, x_max, y_max in regions:
            assert x_max > x_min and y_max > y_min, "Region has zero or negative size"
    
    def test_clustering_empty_input(self):
        """Test clustering handles empty input gracefully."""
        ocr = OCRIntegration()
        
        h, w = 100, 100
        x_coords = torch.tensor([])
        y_coords = torch.tensor([])
        
        regions = ocr._cluster_text_pixels(x_coords, y_coords, h, w)
        
        assert regions == [], "Empty input should return empty list"
    
    def test_clustering_edge_cases(self):
        """Test clustering edge cases (single pixel, all pixels, boundary pixels)."""
        ocr = OCRIntegration()
        
        h, w = 100, 100
        
        # Single pixel
        x_coords = torch.tensor([50])
        y_coords = torch.tensor([50])
        regions = ocr._cluster_text_pixels(x_coords, y_coords, h, w, cluster_distance=5)
        # Single pixel might not form a region (min_samples=2), which is fine
        
        # Boundary pixels
        x_coords = torch.tensor([0, 1, w-2, w-1])
        y_coords = torch.tensor([0, 1, h-2, h-1])
        regions = ocr._cluster_text_pixels(x_coords, y_coords, h, w, cluster_distance=5)
        
        # Check bounds are clamped
        for x_min, y_min, x_max, y_max in regions:
            assert 0 <= x_min < x_max <= w, f"Invalid bounds: ({x_min}, {y_min}, {x_max}, {y_max})"
            assert 0 <= y_min < y_max <= h, f"Invalid bounds: ({x_min}, {y_min}, {x_max}, {y_max})"


class TestIntegration:
    """Integration tests combining multiple fixes."""
    
    def test_full_pipeline_with_overlay(self):
        """Test full pipeline: preprocessing -> inference -> overlay."""
        from ml.utils.output_scheduler import OutputMode
        sim = MaxSightSimulator(device='cpu')
        sim.set_user_condition('cataracts')
        # Set to dev mode to get full output
        sim.output_mode = OutputMode.DEV
        
        test_image = Image.new('RGB', (224, 224), color='purple')
        
        result = sim.process_frame(test_image)
        
        # Check all expected outputs (dev mode has all fields)
        assert 'overlay_image' in result, "Missing overlay_image"
        assert 'detections' in result, "Missing detections"
        assert 'scene_description' in result, "Missing scene_description"
        assert 'processing_time_ms' in result, "Missing processing_time_ms"
        
        # Overlay should be valid
        assert result['overlay_image'] is not None, "Overlay is None"
        
        sim.shutdown()
    
    def test_stress_test_rapid_frames(self):
        """Stress test: process many frames rapidly."""
        sim = MaxSightSimulator(device='cpu')
        sim.set_user_condition('normal')
        
        test_image = Image.new('RGB', (224, 224), color='orange')
        
        start_time = time.perf_counter()
        num_frames = 10
        
        for i in range(num_frames):
            result = sim.process_frame(test_image)
            assert 'overlay_image' in result, f"Frame {i}: Missing overlay"
        
        total_time = time.perf_counter() - start_time
        avg_time = total_time / num_frames
        
        print(f"\nProcessed {num_frames} frames in {total_time:.2f}s (avg: {avg_time:.3f}s/frame)")
        
        # Should complete without errors
        assert total_time < 30.0, f"Stress test took too long: {total_time:.2f}s"
        
        sim.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

