# Caching Strategy

## Overview

Redis-based caching for model outputs and responses to improve performance.

## Setup

1. Install Redis: `brew install redis` (macOS) or `apt-get install redis` (Linux)
2. Start Redis: `redis-server`
3. Set environment variable: `export REDIS_URL=redis://localhost:6379/0`

## Usage

### Basic Caching

```python
from ml.cache.redis_cache import RedisCache

cache = RedisCache(default_ttl=300)  # 5 minute TTL

# Set value
cache.set('model_output_123', output_dict, ttl=600)

# Get value
cached_output = cache.get('model_output_123')
```

### Function Decorator

```python
from ml.cache.redis_cache import cached

@cached(ttl=300)
def expensive_computation(x, y):
    # This result will be cached for 5 minutes
    return x + y
```

## Cache Keys

Cache keys are automatically generated from function arguments using MD5 hashing.

## TTL Strategy

- **Model outputs**: 5-10 minutes (scene doesn't change quickly)
- **User preferences**: 1 hour (changes infrequently)
- **Session data**: 30 minutes (session timeout)

## Cache Invalidation

- **On model update**: Clear all model output caches
- **On user preference change**: Clear user-specific caches
- **On session expiry**: Automatic via TTL

## Best Practices

1. **Use appropriate TTLs**: Balance freshness vs. performance
2. **Version cache keys**: Include model version in key
3. **Monitor cache hit rate**: Optimize TTL based on hit rate
4. **Handle Redis failures**: Gracefully degrade if Redis unavailable

