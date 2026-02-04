# Security Guide

## Overview

MaxSight implements comprehensive security measures for production deployment.

## Authentication

### HMAC-Signed Session Tokens

MaxSight uses stateless HMAC-SHA256 signed tokens for session management.

**Token Format**: `<payload_base64>.<signature_base64>`

**Features**:
- Expiration time (default: 1 hour)
- Tamper detection via HMAC signature
- Stateless (no server-side session storage)

**Usage**:
```python
from ml.auth.token import make_token, verify_token

# Generate token
token = make_token({'session_id': 'abc123', 'user_id': 'user1'})

# Verify token
payload = verify_token(token)
```

**Configuration**:
- `MAXSIGHT_SECRET_KEY`: Secret key for HMAC signing (MUST be set in production)
- `MAXSIGHT_SESSION_TIMEOUT`: Token TTL in seconds (default: 3600)

## Input Validation

### File Upload Validation

All uploaded images are validated using magic number detection:

```python
from ml.security.validation import decode_and_validate_image

is_valid, image_bytes, error = decode_and_validate_image(
    base64_str,
    max_size_mb=10,
    allowed_types=('jpg', 'png', 'gif', 'bmp', 'webp', 'tiff')
)
```

**Supported Formats**: JPEG, PNG, GIF, BMP, WEBP, TIFF

**Size Limits**: Configurable (default: 10MB)

### Base64 Validation

Base64 strings are validated before decoding to prevent injection attacks.

## Security Headers

All HTTP responses include security headers:

- `X-Content-Type-Options: nosniff` - Prevents MIME type sniffing
- `X-Frame-Options: DENY` - Prevents clickjacking
- `X-XSS-Protection: 1; mode=block` - XSS protection
- `Content-Security-Policy` - Restricts resource loading
- `Strict-Transport-Security` - HTTPS enforcement (if HTTPS enabled)

## Rate Limiting

Rate limiting is implemented per-session and globally:

- **Per-session**: Limits requests per session ID
- **Global**: Limits requests per IP address

**Configuration**: See `tools/simulation/config.py`

## Error Sanitization

Error messages are sanitized in production to prevent information leakage:

- **Debug mode**: Full error details (set `DEBUG=1`)
- **Production mode**: Generic error messages

**Usage**:
```python
from ml.middleware.error_sanitizer import sanitize_error, log_error

try:
    # Operation
    pass
except Exception as e:
    log_error(e, context={'endpoint': '/api/process'})  # Log server-side
    error_response = sanitize_error(e, debug=False)  # Sanitize for client
```

## Secure Configuration

### Environment Variables

All secrets and configuration are stored in environment variables:

- `MAXSIGHT_SECRET_KEY`: Authentication secret (REQUIRED)
- `MAXSIGHT_SESSION_TIMEOUT`: Session timeout
- `MAXSIGHT_CORS_ORIGINS`: Allowed CORS origins
- `REDIS_URL`: Redis connection URL
- `DEBUG`: Debug mode flag

**Never commit secrets to version control!**

### .env File

Create a `.env` file (not tracked in git) with your configuration:

```bash
MAXSIGHT_SECRET_KEY=your_strong_random_secret_here
MAXSIGHT_SESSION_TIMEOUT=3600
REDIS_URL=redis://localhost:6379/0
DEBUG=0
```

## Best Practices

1. **Use strong secrets**: Generate random secrets for `MAXSIGHT_SECRET_KEY`
2. **Enable HTTPS**: Set `HTTPS=true` in production
3. **Restrict CORS**: Only allow trusted origins
4. **Monitor rate limits**: Adjust limits based on usage patterns
5. **Regular security audits**: Review security headers and validation regularly

## Security Checklist

- [ ] `MAXSIGHT_SECRET_KEY` set to strong random value
- [ ] HTTPS enabled in production
- [ ] CORS origins restricted
- [ ] Rate limiting configured
- [ ] Error sanitization enabled (DEBUG=0)
- [ ] Security headers verified
- [ ] Input validation tested
- [ ] Token expiration tested

