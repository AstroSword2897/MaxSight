# MaxSight Web Simulator – User Guide v2.0

**Version:** 2.0  
**Last Updated:** December 2025  
**Status:** Production-Hardened / Multi-User Safe

---

## Overview

The **MaxSight Web Simulator** is a complete end-to-end simulation platform for testing and demonstrating the MaxSight assistive vision system. It provides a web-based interface for processing images through the full MaxSight pipeline, including:

* Model inference (MaxSightCNN)
* Scene understanding
* OCR integration
* Therapy and feedback integration
* Multi-modal output (voice, haptic, visual)

**v2.0 Upgrades:**

* Multi-user safety and per-session isolation
* Thread-safe GPU access and queue management
* Rate limiting and automatic cleanup
* Production deployment guidelines
* Enhanced logging and monitoring

---

## Intended Use Cases

1. **Clinical Testing & Evaluation**

   * Test patient-facing scenarios with real-world images
   * Assess therapy task integration and multi-modal feedback
   * Monitor system health, session metrics, and performance

2. **Development & Debugging**

   * Test new model versions safely in a multi-user environment
   * Validate inference pipeline, preprocessing, and post-processing
   * Profile performance and identify resource bottlenecks

3. **Demonstrations & Prototyping**

   * Showcase system capabilities to stakeholders
   * Demonstrate session management, multi-user support, and output modes
   * Visualize model reasoning and decision-making traces

4. **Research & Validation**

   * Collect baseline outputs for regression testing
   * Study spatial memory, path planning, and therapy algorithms
   * Analyze degraded-mode behavior, confidence gating, and safety mechanisms

---

## Architecture & Multi-User Safety

### Multi-User Design

* **MaxSightCore**: Shared compute resources (model, OCR, schedulers)
* **MaxSightSession**: Per-user isolated state (memory, therapy, stats, queues)
* **SessionRegistry**: Thread-safe session management with automatic cleanup

### Key Hardening Features

1. **Thread-Safe Session Access**

   * Each session has a dedicated lock
   * All reads/writes to queues, memory, and stats are atomic

2. **Logical Resource Isolation**

   * Sessions share compute (GPU/MPS) but have separate state
   * Memory, queues, and therapy state are per-session

3. **Automatic Cleanup**

   * Sessions expire after 30 minutes of inactivity
   * Background janitor thread cleans queues, memory, and timers

4. **Rate Limiting**

   * **Per-session**: 60 requests/minute
   * **Global**: 1000 requests/minute
   * Returns HTTP 429 if exceeded

5. **Health Monitoring**

   * Track per-session and global metrics
   * Log degraded components, queue drops, and memory pruning events

---

## Getting Started

### Starting the Server

```bash
# From the project root
python -m tools.simulation.web_simulator
```

Default URL: **[http://localhost:8002](http://localhost:8002)**

**Production Deployment:**

```bash
gunicorn -w 1 -t 120 tools.simulation.web_simulator:app \
  --bind 0.0.0.0:8002 \
  --access-logfile - \
  --error-logfile -
```

> ⚠️ Use 1 worker only. Model inference is serialized; multiple workers will duplicate memory.

---

### Creating a Session

1. **Web UI:** Click **Initialize Session**

2. **API:**

```bash
curl -X POST http://localhost:8002/api/init \
  -H "Content-Type: application/json" \
  -d '{
    "condition": "myopia",
    "scenario": "navigation",
    "output_mode": "patient"
  }'
```

3. Save `session_id` for all subsequent requests

---

### Processing an Image

**API Endpoint:** `POST /api/process`

**Request Options:**

* Form Data: `image` (file upload)
* JSON: `{"image_data": "<base64>", "frame_id": 123}`

**Headers:** `X-Session-ID: <session_id>` (required)

**Responses** vary by mode: **patient**, **clinician**, **dev**. See detailed examples below.

---

## API Endpoints (v2.0)

### Session Management

| Endpoint              | Method | Purpose                |
| --------------------- | ------ | ---------------------- |
| `/api/init`           | POST   | Initialize session     |
| `/api/process`        | POST   | Process image          |
| `/api/session/status` | GET    | Session health & stats |
| `/api/session/start`  | POST   | Start therapy          |
| `/api/session/stop`   | POST   | Stop therapy           |
| `/api/session/abort`  | POST   | Hard kill switch       |

### System Monitoring

| Endpoint       | Method | Purpose                      |
| -------------- | ------ | ---------------------------- |
| `/api/health`  | GET    | System health check          |
| `/api/metrics` | GET    | Per-session & global metrics |

---

## Output Modes

| Mode        | Purpose                | Content                                       | Safety                               |
| ----------- | ---------------------- | --------------------------------------------- | ------------------------------------ |
| `patient`   | End-user experience    | Simplified confidence-gated outputs           | Low-confidence results suppressed    |
| `clinician` | Medical evaluation     | Detailed detections with confidence           | All outputs shown, warnings included |
| `dev`       | Development & research | Full technical data, model traces, debug info | No filtering, raw data exposed       |

---

## Visual Conditions & Scenarios

**Supported Conditions:** `normal`, `myopia`, `hyperopia`, `astigmatism`, `cataracts`, `glaucoma`, `amd`, `diabetic_retinopathy`, `retinitis_pigmentosa`, `color_blindness`, `cvi`, `amblyopia`, `strabismus`

**Supported Scenarios:** `general`, `navigation`, `text_reading`, `therapy`, `safety`, `accessibility`

---

## Safety Features

### Confidence Gating

* Patient outputs filtered by minimum confidence (default: 0.5)
* Critical alerts require higher confidence (default: 0.7)
* Low-confidence results: `"Unable to confirm objects in view"`

### Degraded Mode Tracking

* Tracks component failures (`VISION_UNSTABLE`, `AUDIO_UNAVAILABLE`, `HAPTIC_UNAVAILABLE`, `MEMORY_FULL`, `PROCESSING_SLOW`)
* Returned in `degraded_status` per session

### Output Authority Hierarchy

1. SAFETY_ALERTS
2. NAVIGATION_GUIDANCE
3. THERAPY_PROMPTS
4. DESCRIPTIVE_NARRATION

### Hard Kill Switch

* Stops all outputs and flushes queues immediately
* Session marked as **aborted** to reject future requests

---

## Resource Management

* **Memory Caps:** 1000 spatial memory entries per session
* **History Depth:** 100 frames per session
* **Queue Caps:** 10 items per voice/haptic queue
* **Memory Budget:** 500 MB per session (soft enforcement)
* **Session Timeout:** 30 minutes inactivity triggers cleanup

---

## Best Practices

1. **Session Management**

   * Always initialize before processing
   * Protect session IDs (bearer tokens)
   * Let sessions expire or clean up manually

2. **Error Handling**

   * Check `degraded_status`
   * Handle 429 rate limit errors gracefully
   * Validate input formats and sizes

3. **Performance**

   * Use frame IDs for sequential ordering
   * Batch image requests when possible
   * Monitor `/api/metrics` for bottlenecks

4. **Testing**

   * Test multiple visual conditions and edge cases
   * Test empty, corrupted, or large images
   * Verify confidence gating works as expected

5. **Security**

   * Do not expose to public internet
   * Enforce HTTPS at reverse proxy
   * Validate all inputs and monitor usage

---

## Logging

* Structured JSON logs for tracing sessions

```json
{
  "timestamp": "2025-12-22T17:01:50",
  "level": "INFO",
  "component": "session",
  "message": "Session initialized",
  "session_id": "uuid-here"
}
```

* Include `session_id` in every log
* Track degraded events, memory pruning, queue drops

---

## Example: Multi-User Safe Image Processing

```python
import requests

# Initialize session
response = requests.post('http://localhost:8002/api/init', json={
    'condition': 'myopia',
    'scenario': 'navigation',
    'output_mode': 'patient'
})
session_id = response.json()['session_id']

# Process image safely
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8002/api/process',
        files={'image': f},
        headers={'X-Session-ID': session_id}
    )

result = response.json()
print(result['perspectives']['user_view'])
```

---

## Known Limitations & Assumptions

* Single image source per session
* Extreme lighting may reduce accuracy
* No adversarial input protection
* Base64 overlays may bloat responses
* Multi-camera setups not supported
* Debug mode enabled by default; sensitive patient data **should not** be used in development

---

## Configuration (`tools/simulation/config.py`)

```python
# Server
host = '0.0.0.0'
port = 8002
debug = False  # ⚠️ Disable for production

# Session Management
session_timeout_seconds = 30 * 60
multi_user_enabled = True

# Rate Limiting
rate_limit_per_session = 60
rate_limit_global = 1000

# Confidence Thresholds
confidence_threshold = 0.3
min_confidence_for_patient_output = 0.5
min_confidence_for_critical_alert = 0.7

# Resource Caps
max_spatial_memory_entries = 1000
max_history_depth = 100
max_memory_mb_per_session = 500

# Queue Settings
voice_queue_maxsize = 10
haptic_queue_maxsize = 10
```

---

## Detailed API Response Examples

### Patient Mode Response

```json
{
  "mode": "patient",
  "severity": "info",
  "message": "Object detected",
  "confidence": 0.85,
  "cooldown_applied": false,
  "overlay_image": "data:image/png;base64,...",
  "stats": {
    "frames_processed": 1,
    "avg_latency_ms": 25.5,
    "total_detections": 3
  },
  "perspectives": {
    "user_view": {
      "overlay_image": "data:image/png;base64,...",
      "scene_description": "Scene with objects",
      "voice_announcements": ["Object detected"]
    },
    "model_reasoning": {
      "feature_extraction": {...},
      "attention_weights": {...},
      "confidence_scores": {...},
      "decision_path": [...]
    },
    "final_judgment": {
      "final_score": 0.82,
      "urgency_level": 1,
      "urgency_confidence": 0.75,
      "objectness_confidence": 0.85,
      "num_detections": 3,
      "weighted_detections": [...],
      "decision": "normal"
    }
  },
  "degraded_status": {
    "is_degraded": false,
    "active_modes": [],
    "reasons": {},
    "status_message": "All systems operational"
  },
  "resource_usage": {
    "spatial_memory_count": 3,
    "memory_usage_mb": 0.5,
    "queue_dropped_voice": 0,
    "queue_dropped_haptic": 0
  }
}
```

### Clinician Mode Response

```json
{
  "mode": "clinician",
  "severity": "warning",
  "message": "Scene with objects",
  "confidence": 0.85,
  "cooldown_applied": false,
  "latency_ms": 25.5,
  "total_time_ms": 30.2,
  "inference_time_ms": 25.5,
  "num_detections": 3,
  "num_hazards": 1,
  "ocr_texts": ["Text detected"],
  "component_breakdown": {
    "detections": 3,
    "ocr": 1,
    "voice": 1,
    "haptic": 1
  },
  "overlay_image": "data:image/png;base64,...",
  "stats": {...},
  "perspectives": {
    "user_view": {...},
    "model_reasoning": {...},
    "final_judgment": {...}
  },
  "degraded_status": {...},
  "resource_usage": {...}
}
```

### Dev Mode Response

```json
{
  "mode": "dev",
  "severity": "warning",
  "message": "Scene with objects",
  "confidence": 0.85,
  "cooldown_applied": false,
  "frame_number": 1,
  "timestamp": 1234567890.123,
  "processing_time_ms": 30.2,
  "inference_time_ms": 25.5,
  "detections": [...],
  "num_detections": 3,
  "urgency_scores": [...],
  "distance_zones": [...],
  "scene_embedding": [...],
  "text_regions": [...],
  "num_text_regions": 1,
  "scene_description": "...",
  "scheduled_outputs": {...},
  "voice_announcements": [...],
  "haptic_patterns": [...],
  "path_info": {...},
  "therapy_feedback": {...},
  "overlay_image": "data:image/png;base64,...",
  "stats": {...},
  "debug_info": {
    "condition": "myopia",
    "scenario": "navigation",
    "session_active": false,
    "session_id": "uuid-here"
  },
  "perspectives": {
    "user_view": {...},
    "model_reasoning": {...},
    "final_judgment": {...}
  },
  "degraded_status": {...},
  "resource_usage": {...}
}
```

### Health Check Response

```json
{
  "status": "healthy",
  "core": {
    "device": "mps",
    "model_loaded": true,
    "model_mode": "eval"
  },
  "sessions": {
    "active": 2,
    "expired": 0,
    "total": 2
  },
  "metrics": {
    "uptime_seconds": 3600,
    "total_requests": 150,
    "total_errors": 2,
    "error_rate_percent": 1.33,
    "total_sessions_created": 5,
    "total_images_processed": 120,
    "avg_processing_time_ms": 28.5,
    "avg_inference_time_ms": 25.2,
    "requests_per_second": 0.04
  },
  "timestamp": 1234567890.123
}
```

---

## Troubleshooting

### Server Won't Start
- Check if port 8002 is already in use: `lsof -i :8002`
- Verify Python environment is activated
- Run as module: `python -m tools.simulation.web_simulator`
- Check logs for import errors

### Import Errors
- Ensure you're running from project root
- Verify all dependencies are installed
- Check that `__init__.py` files exist in simulation directory

### Model Loading Issues
- Verify model files exist
- Check device availability (CUDA/MPS/CPU)
- Review error logs for specific issues

### Session Not Found
- Verify session was initialized
- Check session hasn't expired (30 min timeout)
- Ensure `X-Session-ID` header is sent correctly

### Rate Limit Errors
- Wait before retrying
- Check rate limit configuration
- Use health endpoint to monitor system load

### Degraded Mode Warnings
- Check `degraded_status` in response
- Review logs for component failures
- Verify resource caps aren't exceeded
- Check queue drop counts in `resource_usage`

---

## Production Deployment Checklist

- [ ] Set `debug = False` in `config.py`
- [ ] Use Gunicorn with 1 worker
- [ ] Configure reverse proxy (nginx/Apache) with HTTPS
- [ ] Set up proper firewall rules
- [ ] Configure CORS origins appropriately
- [ ] Enable structured logging to file
- [ ] Set up monitoring/alerting for health endpoint
- [ ] Configure session timeout based on use case
- [ ] Review and adjust rate limits
- [ ] Test degraded mode handling
- [ ] Verify resource caps are appropriate
- [ ] Document any custom configuration

---

**Version:** 2.0  
**Last Updated:** December 2025  
**Status:** Production-Hardened / Multi-User Safe
