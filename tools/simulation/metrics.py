"""Metrics and monitoring for MaxSight Web Simulator. Tracks performance, errors, and system health."""
import time
from typing import Dict, Any, Optional
from collections import defaultdict
from threading import Lock
from dataclasses import dataclass, field
from .config import config


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    total_requests: int = 0
    total_errors: int = 0
    total_sessions_created: int = 0
    total_sessions_expired: int = 0
    total_images_processed: int = 0
    total_inference_time: float = 0.0
    total_processing_time: float = 0.0
    
    # Error counts by type.
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Request counts by endpoint.
    endpoint_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Timestamps.
    start_time: float = field(default_factory=time.time)
    last_request_time: Optional[float] = None
    
    lock: Lock = field(default_factory=Lock)
    
    def record_request(self, endpoint: str, processing_time: float = 0.0):
        """Record a successful request."""
        with self.lock:
            self.total_requests += 1
            self.endpoint_counts[endpoint] += 1
            self.last_request_time = time.time()
            if processing_time > 0:
                self.total_processing_time += processing_time
    
    def record_error(self, error_type: str):
        """Record an error."""
        with self.lock:
            self.total_errors += 1
            self.error_counts[error_type] += 1
    
    def record_inference(self, inference_time: float):
        """Record inference time."""
        with self.lock:
            self.total_inference_time += inference_time
            self.total_images_processed += 1
    
    def record_session_created(self):
        """Record session creation."""
        with self.lock:
            self.total_sessions_created += 1
    
    def record_session_expired(self):
        """Record session expiration."""
        with self.lock:
            self.total_sessions_expired += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.lock:
            uptime = time.time() - self.start_time
            avg_processing_time = (
                self.total_processing_time / self.total_requests
                if self.total_requests > 0 else 0.0
            )
            avg_inference_time = (
                self.total_inference_time / self.total_images_processed
                if self.total_images_processed > 0 else 0.0
            )
            error_rate = (
                self.total_errors / self.total_requests * 100
                if self.total_requests > 0 else 0.0
            )
            
            return {
                'uptime_seconds': uptime,
                'total_requests': self.total_requests,
                'total_errors': self.total_errors,
                'error_rate_percent': round(error_rate, 2),
                'total_sessions_created': self.total_sessions_created,
                'total_sessions_expired': self.total_sessions_expired,
                'total_images_processed': self.total_images_processed,
                'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
                'avg_inference_time_ms': round(avg_inference_time * 1000, 2),
                'requests_per_second': round(self.total_requests / uptime, 2) if uptime > 0 else 0.0,
                'error_counts': dict(self.error_counts),
                'endpoint_counts': dict(self.endpoint_counts),
                'last_request_time': self.last_request_time
            }
    
    def reset(self):
        """Reset all metrics (for testing)."""
        with self.lock:
            self.total_requests = 0
            self.total_errors = 0
            self.total_sessions_created = 0
            self.total_sessions_expired = 0
            self.total_images_processed = 0
            self.total_inference_time = 0.0
            self.total_processing_time = 0.0
            self.error_counts.clear()
            self.endpoint_counts.clear()
            self.start_time = time.time()
            self.last_request_time = None


# Global metrics instance.
metrics = SystemMetrics()


def get_health_status() -> Dict[str, Any]:
    """Get system health status. Returns: Health status dictionary."""
    summary = metrics.get_summary()
    
    # Determine health status.
    error_rate = summary['error_rate_percent']
    if error_rate > 10:
        health = 'unhealthy'
    elif error_rate > 5:
        health = 'degraded'
    else:
        health = 'healthy'
    
    return {
        'status': health,
        'timestamp': time.time(),
        'metrics': summary
    }







