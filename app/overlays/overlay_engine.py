"""
Overlay Engine

Renders visual overlays for therapy guidance.

Phase 4: Overlay Engine & UX Guidance
See docs/therapy_system_implementation_plan.md for implementation details.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class OverlayEngine:
    """
    Renders visual overlays for therapy guidance.
    
    Overlay types:
    - Subtle halo
    - Edge glow (contrast reinforcement)
    - Depth "soft fog" for near objects
    - Motion trace for tracking tasks
    - Gaze indicator
    - Gentle arrows for guidance
    
    Safety constraints:
    - Never obscure more than 10% of screen
    - No bright colors
    - Overlays must fade after task ends
    """
    
    def __init__(self, screen_size: Tuple[int, int] = (224, 224)):
        self.screen_size = screen_size
        self.max_overlay_percent = 0.10  # 10% max screen coverage
        self.active_overlays = []
    
    def add_halo(
        self,
        center: Tuple[float, float],
        radius: float,
        intensity: float = 0.3
    ) -> Dict[str, Any]:
        """
        Add subtle halo overlay.
        
        Args:
            center: (x, y) center position [0, 1]
            radius: Radius in pixels
            intensity: Opacity [0, 1]
        
        Returns:
            Overlay configuration
        """
        overlay = {
            'type': 'halo',
            'center': center,
            'radius': radius,
            'intensity': min(intensity, 0.5),  # Cap at 50% opacity
            'color': (255, 255, 255)  # White, subtle
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_edge_glow(
        self,
        edges: List[Tuple[float, float]],
        width: float = 2.0,
        intensity: float = 0.4
    ) -> Dict[str, Any]:
        """
        Add edge glow for contrast reinforcement.
        
        Args:
            edges: List of (x, y) edge points
            width: Glow width in pixels
            intensity: Opacity [0, 1]
        
        Returns:
            Overlay configuration
        """
        overlay = {
            'type': 'edge_glow',
            'edges': edges,
            'width': width,
            'intensity': min(intensity, 0.5),
            'color': (200, 200, 255)  # Subtle blue-white
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_depth_fog(
        self,
        depth_map: np.ndarray,
        near_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Add depth "soft fog" for near objects.
        
        Args:
            depth_map: Depth map [H, W] with values [0, 1]
            near_threshold: Threshold for "near" objects
        
        Returns:
            Overlay configuration
        """
        overlay = {
            'type': 'depth_fog',
            'depth_map': depth_map,
            'near_threshold': near_threshold,
            'intensity': 0.2,  # Very subtle
            'color': (150, 150, 150)  # Gray fog
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_motion_trace(
        self,
        path: List[Tuple[float, float]],
        width: float = 3.0
    ) -> Dict[str, Any]:
        """
        Add motion trace for tracking tasks.
        
        Args:
            path: List of (x, y) positions
            width: Trace width in pixels
        
        Returns:
            Overlay configuration
        """
        overlay = {
            'type': 'motion_trace',
            'path': path,
            'width': width,
            'intensity': 0.6,
            'color': (100, 200, 255)  # Light blue
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_gaze_indicator(
        self,
        position: Tuple[float, float],
        size: float = 10.0
    ) -> Dict[str, Any]:
        """
        Add gaze position indicator.
        
        Args:
            position: (x, y) gaze position [0, 1]
            size: Indicator size in pixels
        
        Returns:
            Overlay configuration
        """
        overlay = {
            'type': 'gaze_indicator',
            'position': position,
            'size': size,
            'intensity': 0.5,
            'color': (255, 200, 100)  # Orange-yellow
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_guidance_arrow(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        width: float = 5.0
    ) -> Dict[str, Any]:
        """
        Add gentle arrow for guidance.
        
        Args:
            start: (x, y) start position [0, 1]
            end: (x, y) end position [0, 1]
            width: Arrow width in pixels
        
        Returns:
            Overlay configuration
        """
        overlay = {
            'type': 'guidance_arrow',
            'start': start,
            'end': end,
            'width': width,
            'intensity': 0.4,
            'color': (150, 255, 150)  # Light green
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def render_overlays(self, base_image: np.ndarray) -> np.ndarray:
        """
        Render all active overlays onto base image.
        
        Args:
            base_image: Base image [H, W, 3]
        
        Returns:
            Image with overlays [H, W, 3]
        """
        # TODO: Implement actual rendering
        # For now, return base image
        return base_image.copy()
    
    def clear_overlays(self):
        """Clear all active overlays."""
        self.active_overlays = []
    
    def fade_overlays(self, fade_factor: float = 0.9):
        """Fade all overlays (called after task ends)."""
        for overlay in self.active_overlays:
            overlay['intensity'] *= fade_factor
            if overlay['intensity'] < 0.01:
                self.active_overlays.remove(overlay)

