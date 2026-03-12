# MaxSight Pilot Validation Protocol

## Purpose
Define a real-world pilot validation framework that measures:
- Safety outcomes (mandatory for release).
- Awareness impact.
- Independence gains.
- User trust and cognitive load.

This protocol is designed for a closed pilot before broader rollout.

## Pilot Objectives
1. Verify safety-critical performance in real environments.
2. Demonstrate meaningful gains in environmental awareness.
3. Demonstrate reduced task dependence on guides/caregivers.
4. Confirm output modality is usable and not cognitively overwhelming.

## Participant Profile
- Adults with low vision or blindness across representative conditions.
- Mix of mobility aid users (white cane users, guide dog users, mixed).
- Include varied familiarity with assistive tech to capture onboarding burden.

## Pilot Structure
- Duration: 4 weeks.
- Stages:
  - Week 0: Baseline assessment (without MaxSight).
  - Week 1: Controlled environment sessions.
  - Week 2-3: Real-route supervised usage.
  - Week 4: Follow-up assessment and interviews.

## Scenario Suite

### Safety-Critical Scenarios (Mandatory)
1. Street crossing with moving vehicles and curb transitions.
2. Obstacle approach in sidewalk/corridor with varied clutter density.
3. Sudden hazard appearance (bike/scooter/person crossing trajectory).
4. Low-light transitions and high-glare sections.

### Independence Scenarios
1. Sign and label reading (bus stop, room sign, shelf label).
2. Object-finding tasks (pill bottle, keys, checkout kiosk controls).
3. Indoor wayfinding to target points (exit, restroom, platform marker).

### Social/Context Awareness Scenarios
1. Meeting/cafe setting with room layout summary.
2. Queue/checkout context understanding.
3. Basic scene understanding in unfamiliar environments.

## Metrics Framework

### A. Safety-First Gates (Release Blocking)
- Critical hazard recall.
- False-safe rate.
- Time-to-alert (median and p95).
- Direction cue correctness under motion.

### B. Awareness Impact Metrics
- Awareness task score: proportion of correctly identified hazards/signals in route.
- Missed hazard count per km/session.
- Time to orient after entering unfamiliar area.

### C. Independence Metrics
- Task completion without human assistance (read/find/navigate tasks).
- Assistance request frequency per session.
- Time-to-task completion (baseline vs assisted).

### D. Trust and Usability Metrics
- User confidence score (pre/post).
- Cognitive load score (e.g., brief NASA-TLX style scale).
- Alert usefulness rating (actionable vs noisy).
- Modality preference stability (voice/haptic/hybrid).

## Data Collection
- Automated telemetry:
  - Alert events, priority, timestamps, latency, suppressions.
  - Mode changes and degraded mode activation.
  - Failure events and fallbacks.
- Human annotations:
  - Ground-truth hazard timeline.
  - Observer notes for near-miss and confusion moments.
- Participant reports:
  - Session-level confidence and fatigue.
  - Qualitative feedback on trust and burden.

## Incident Classification
- I1 Critical: false-safe on immediate hazard.
- I2 Major: late alert causing unsafe reaction window.
- I3 Moderate: repeated non-actionable alert spam.
- I4 Minor: informational errors with no safety consequence.

Any I1 incident pauses rollout pending review.

## Weekly Review Loop
1. Review safety gate trend lines.
2. Review incident stack and root causes.
3. Approve remediation actions (model, scheduler, UX policy).
4. Re-run affected scenario subset before continuing pilot.

## Success Criteria

### Mandatory for pilot graduation
- All safety-first gates from `02_safety_first_release_gates.md` pass.
- No unresolved I1 incidents.

### Target impact outcomes
- >= 20% reduction in missed hazards versus baseline sessions.
- >= 25% increase in independent task completion for read/find scenarios.
- Positive confidence trend in majority of participants.

## Ethics and Participant Safety
- Supervised sessions for all outdoor mobility scenarios.
- Immediate intervention protocol for unsafe situations.
- Clear informed consent: assistive tool, not autonomous navigation.
- Opt-out and pause controls available in all sessions.

## Output Artifacts
- Pilot summary report with safety and impact dashboards.
- Incident postmortem pack.
- Recommendation memo: scale, hold, or iterate.
