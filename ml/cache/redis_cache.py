"""Redis Caching Utilities for MaxSight Provides Redis-based caching for model outputs and responses with TTL support."""

import hashlib
import json
import os
import pickle
from collections.abc import Callable
from typing import Any

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCache:
    """Redis-based cache with TTL support."""

    def __init__(self, redis_url: str | None = None, default_ttl: int = 60):
        """Initialize Redis cache. Args: redis_url: Redis connection URL (defaults to REDIS_URL env var) default_ttl: Default TTL in seconds."""
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Install with: pip install redis")

        redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.from_url(redis_url)
        self.default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        try:
            value = self.client.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with TTL."""
        try:
            ttl = ttl or self.default_ttl
            serialized = pickle.dumps(value)
            return self.client.setex(key, ttl, serialized)
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            return bool(self.client.delete(key))
        except Exception:
            return False

    def clear(self):
        """Clear all cache entries."""
        try:
            self.client.flushdb()
        except Exception:
            pass


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments. Args: *args: Positional arguments **kwargs: Keyword arguments Returns: MD5 hash of serialized arguments."""
    key_data = {"args": args, "kwargs": kwargs}
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(ttl: int = 60, redis_url: str | None = None):
    """Decorator for caching function results."""
    cache = RedisCache(redis_url=redis_url, default_ttl=ttl) if REDIS_AVAILABLE else None

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if cache is None:
                # Redis not available, just call function.
                return func(*args, **kwargs)

            # Generate cache key.
            key = f"{func.__name__}:{cache_key(*args, **kwargs)}"

            # Get from cache when present.
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Compute and cache.
            result = func(*args, **kwargs)
            cache.set(key, result, ttl=ttl)
            return result

        return wrapper

    return decorator
