# Deployment Bottlenecks and Recovery

This document covers production hardening: monitoring, recovery procedures, alert logic, and known edge cases.

## Thermal Throttling

### Detection

- **ThermalThrottleDetector** (in `tools/simulation/simulator/inference_engine.py`) maintains a 30s sliding window of inference latencies.
- Baseline = average of first 5 samples; current = average of last 10 samples.
- If `current_avg > baseline * 2.0`, thermal throttling is reported and the engine transitions to **DEGRADED**.

### Recovery (step-by-step)

1. **Confirm**: Check logs for `"Thermal throttling detected"` and `"transitioning to DEGRADED"`.
2. **Reduce load**: Lower frame rate or batch size; skip non-critical stages (overlay/audio) when pipeline > 200 ms or CPU > 80%.
3. **Cooling**: Improve device cooling; reduce ambient temperature if possible.
4. **Reset**: Restart the inference engine (or process) to clear state and re-establish baseline after cooling.
5. **Monitor**: Re-enable pipeline latency logging and watch for repeated degradation.

### Optional: GPU temperature

- If using CUDA, log or sample GPU temperature (e.g. `nvidia-smi`) and correlate with latency spikes.
- Fallback: reduce frame rate before skipping detection stages.

---

## Alert Suppression Logic and Scoring

### Priority score (per detection)

Used by **PriorityBudgetFilter** to cap alerts per frame:

- `priority_score = urgency * confidence * (1 / (distance_ordinal + 1))`
- **Urgency**: 0–3 (safe, caution, warning, danger); internally scaled 1–4 for the formula.
- **Confidence**: model detection score in [0, 1].
- **Distance ordinal**: near=0, medium=1, far=2; closer objects rank higher.

Detections are sorted by this score descending; only the top **max_alerts_per_frame** (default 5) are kept.

### Alert cooldown

- **AlertCooldownFilter** suppresses repeat alerts for the same object.
- Object ID = hash of (class_name, rounded bbox).
- If an object was alerted in the last **alert_cooldown_frames** (default 5) frames, it is filtered out this frame.
- Reduces alert spam without dropping new high-priority hazards.

### Safety bias (urgency)

- **Hazard classes** (e.g. car, truck, bus, motorcycle, train, fire, hazard, emergency, siren, alarm) with confidence > 0.3 get at least urgency 2 (warning).
- **Large/close objects** (normalized box area > 0.2) get urgency +1 (capped at 3).
- Ensures over-warning on potential hazards rather than under-warning.

---

## Known Edge Cases and Fallbacks

| Scenario | Behavior | Fallback |
|----------|----------|----------|
| Pipeline total > 200 ms or CPU > 80% | Overlay and audio stages skipped for that frame; logged. | Detection and postprocess still run; user gets no overlay/voice for that frame. |
| Thermal throttling detected | Inference engine state → DEGRADED. | Reduce load or restart after cooling. |
| Priority filter drops detections | Top N (default 5) kept by priority score; dropped count logged at DEBUG. | Increase `max_alerts_per_frame` in config if needed (trade-off: more alerts). |
| Alert cooldown filters object | Object not re-alerted for `alert_cooldown_frames` frames. | Lower `alert_cooldown_frames` for more responsive re-alerts (trade-off: more spam). |
| Scene graph invalid | Stage B outputs skipped; `skip_stage_b = True`. | No partial/corrupted Stage B output. |
| Circuit breaker (failure rate, fallback rate, latency, uncertainty) | Engine can transition to DEGRADED or HALTED. | Follow circuit breaker config; fix underlying cause; reset engine. |

---

## Monitoring Hooks

- **Pipeline timing**: Per-stage breakdown in `result['pipeline_breakdown']` (preprocess, model, postprocess, overlay, audio, total_ms). Log at DEBUG or to a metrics endpoint.
- **Detection flicker**: Compare object IDs across consecutive frames (e.g. from temporal smoother); compute % reappearing/disappearing for dashboards.
- **Safety bias audit**: Log detections that received boosted urgency (hazard class or large box).
- **Thermal**: Log when ThermalThrottleDetector returns True; optionally integrate with existing metrics/alerting.

---

## Rollback and Deployment

- **Rollback**: Keep previous container/image tag; revert to it if pipeline or stability regresses.
- **Alerting**: Trigger an alert if pipeline total consistently exceeds a threshold (e.g. p95 > 300 ms) or if DEGRADED/HALTED state is entered.

---

## Quick run (what to do)

### Tests

```bash
pytest tests/test_production_hardening.py tests/test_critical_fixes.py tests/test_model.py -v
```

### Full production rehearsal (stress + pipeline + alerts)

```bash
python scripts/full_production_rehearsal.py --device cpu --num-frames 3 --log-dir logs
# On M3 Pro: --device mps
```

Logs: `logs/production_rehearsal.log`, `logs/production_rehearsal_results.json`.

### M3 Pro / Apple Silicon

- **Latency**: `torch.mps.synchronize()` is used in inference_engine, benchmark_tiers, validate_forward_passes so Stage A/B timing is accurate on MPS.
- **GradNorm**: Task-weight update runs on CPU when device is MPS; no code change needed.
- **Production training**: Use cloud GPU; M3 Pro is for dev, small fine-tuning, and forward-pass validation.
- **M3 Pro dev readiness**: Supported for inference, benchmarks, rehearsal, and short training; see `REQUIREMENTS.md` for hardware summary.

### CoreML export

- **Image-only.** Audio and temporal inputs are not yet supported. To add them, extend `ml/training/export.py` with fixed-shape inputs and re-run export tests.
