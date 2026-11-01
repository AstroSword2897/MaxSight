# ADR-013: No additional sensors beyond what is explicitly decided

- **Register ID:** DR-017
- **Status:** Confirmed
- **Source:** Directive

## Context / rationale
Scope and privacy control

## Alternatives considered
Use IMU/LiDAR freely

## Decision
No additional sensors beyond what is explicitly decided

## Consequences
V1 inputs = camera (perception) + microphone (spoken requests). IMU, LiDAR, depth and any other sensor are **excluded from V1** (DR-210/211 closed); any future sensor needs a new, separate architecture decision

## Dependencies
—

## Validation required
Code review + entitlement audit

## Approver
_pending signature_
