# MaxSight Safety-First V1 Release Gates

## Purpose
Define non-negotiable safety thresholds and release criteria for V1 pilot readiness.

These gates override feature completeness. If any mandatory gate fails, release is blocked.

## Scope
Applies to the critical runtime path:
- Hazard detection and urgency classification.
- Direction and distance cues.
- Alert generation and output scheduling.

Non-critical features (OCR, rich summaries, retrieval) cannot bypass these gates.

## Mandatory Safety Gates (V1)

| Gate ID | Metric | Threshold | Dataset/Scenario | Blocker if Failed |
|---|---|---|---|---|
| SG-01 | Hazard recall (critical hazards) | >= 0.95 | Street + indoor safety scenarios | Yes |
| SG-02 | False-safe rate (critical hazards incorrectly marked safe or not alerted) | <= 0.01 | Same as SG-01 | Yes |
| SG-03 | Time-to-alert p95 | <= 600 ms | Target hybrid hardware, live stream | Yes |
| SG-04 | Time-to-alert median | <= 350 ms | Target hybrid hardware, live stream | Yes |
| SG-05 | Directional cue correctness | >= 0.90 | Left/center/right lane and obstacle scenarios | Yes |
| SG-06 | Distance zone accuracy (near/medium/far) | >= 0.85 | Measured-distance route set | Yes |
| SG-07 | Alert suppression correctness under uncertainty | 100% critical hazards still surfaced | High-uncertainty replay suite | Yes |
| SG-08 | Overload guardrail | <= 12 alerts/min average in dense scenes, unless emergency | High-clutter stress scenes | Yes |

## Secondary Readiness Metrics (Non-Blocker for initial V1, but tracked)
- OCR task success >= 0.85 on pilot signage set.
- Object-finding task success >= 0.80 without facilitator hints.
- User confidence uplift against baseline questionnaire.

## Gate Definitions

### SG-01 Hazard recall
Critical hazards include:
- Moving vehicles in crossing context.
- Immediate collision obstacles in path.
- Curb/drop-off hazards.

Recall is measured over annotated critical events.

### SG-02 False-safe rate
False-safe is any event where a critical hazard exists but:
- System emits safe/no-warning state, or
- Fails to emit warning within alert deadline.

### SG-03/SG-04 Time-to-alert
Time-to-alert starts at frame timestamp where hazard first meets alert criterion and ends at emitted output event (voice/haptic event enqueue).

### SG-07 Uncertainty behavior
If uncertainty suppression logic is active, critical detections must bypass suppression and still alert.

## Validation Environments
- Controlled route replay with annotated events.
- Hardware-in-the-loop run on target glasses + companion phone architecture.
- In-the-wild closed pilot shadow runs with incident tagging.

## Release Decision Flow
1. Run gate suite on candidate model + scheduler config.
2. Generate signed gate report.
3. Block release on any failed mandatory gate.
4. Approve release only with safety owner signoff.

## Required Evidence Artifacts per Candidate
- Gate metrics report (raw + summary).
- Incident log with false-safe/late-alert analysis.
- Versioned model package hash and scheduler config.
- Test environment metadata (hardware, build, dataset version).

## Roles
- Safety owner: final release gate authority.
- ML owner: model performance and regression response.
- Runtime owner: alert latency and scheduling integrity.
- Product owner: verifies claims remain within validated scope.
