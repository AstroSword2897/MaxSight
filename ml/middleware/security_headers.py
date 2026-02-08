"""Security Headers Middleware for Flask/FastAPI

Adds security headers to HTTP responses to prevent common web vulnerabilities."""

from typing import Callable
import os


def add_security_headers(response) -> None:
    """Add security headers to Flask response object.
    
    Args:
        response: Flask response object"""
    # Prevent MIME type sniffing.
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Prevent clickjacking.
    response.headers['X-Frame-Options'] = 'DENY'
    
    # XSS protection (legacy, but still useful)
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Content Security Policy.
    # Allow self, data URIs for images, but no inline scripts.
    csp = "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self' 'unsafe-inline'"
    response.headers['Content-Security-Policy'] = csp
    
    # Strict Transport Security (if HTTPS)
    if os.environ.get('HTTPS', '').lower() == 'true':
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'


def security_headers_middleware(app):
    """Flask middleware to add security headers to all responses...."""
    @app.after_request
    def after_request(response):
        add_security_headers(response)
        return response
    
    return app



