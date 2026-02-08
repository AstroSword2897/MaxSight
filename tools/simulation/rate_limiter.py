"""Rate limiting for MaxSight Web Simulator.
Prevents abuse and ensures fair resource usage."""
import time
from typing import Dict, Optional
from collections import defaultdict
from threading import Lock
from .exceptions import RateLimitError


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""
    
    def __init__(self, requests_per_minute: int, window_seconds: int = 60):
        """Args:
            requests_per_minute: Maximum requests allowed per minute
            window_seconds: Time window in seconds (default 60)"""
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)  # Session_id -> timestamps.
        self.lock = Lock()
    
    def check_rate_limit(self, session_id: str, identifier: Optional[str] = None) -> None:
        """Check if request is within rate limit...."""
        key = f"{session_id}:{identifier}" if identifier else session_id
        now = time.time()
        
        with self.lock:
            # Clean old requests outside window.
            cutoff = now - self.window_seconds
            self.requests[key] = [ts for ts in self.requests[key] if ts > cutoff]
            
            # Check limit.
            if len(self.requests[key]) >= self.requests_per_minute:
                raise RateLimitError(
                    f"Rate limit exceeded: {self.requests_per_minute} requests per {self.window_seconds} seconds"
                )
            
            # Record request for rate limiting.
            self.requests[key].append(now)
    
    def get_remaining(self, session_id: str, identifier: Optional[str] = None) -> int:
        """Get remaining requests in current window."""
        key = f"{session_id}:{identifier}" if identifier else session_id
        now = time.time()
        cutoff = now - self.window_seconds
        
        with self.lock:
            self.requests[key] = [ts for ts in self.requests[key] if ts > cutoff]
            return max(0, self.requests_per_minute - len(self.requests[key]))


class GlobalRateLimiter:
    """Global rate limiter across all sessions."""
    
    def __init__(self, requests_per_minute: int):
        self.limiter = RateLimiter(requests_per_minute)
    
    def check_rate_limit(self, identifier: str) -> None:
        """Check global rate limit."""
        self.limiter.check_rate_limit("global", identifier)
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining global requests."""
        return self.limiter.get_remaining("global", identifier)



