# MaxSight Product Scope and Claims Matrix

## Purpose
Define intended use, in-scope capabilities, explicit non-claims, and evidence expectations for a full life-assistant product for visually impaired users.

This document is the product baseline for:
- Runtime priorities.
- Safety and quality gates.
- Pilot validation and user messaging.

## Intended Use (V1)
MaxSight is an assistive smart-glasses system that helps visually impaired users understand nearby hazards, orientation cues, and everyday context through spoken and haptic guidance.

V1 focus:
- Safety-critical awareness (hazards, obstacle proximity, directional cues).
- Daily independence tasks (text reading, finding objects/signs, route cues).
- Low-verbosity situational summaries in unfamiliar environments.

## Primary User Profiles
- Low-vision users (e.g., AMD, glaucoma, diabetic retinopathy, CVI).
- Blind users relying primarily on audio/haptic outputs.
- Users needing both passive safety alerts and on-demand detailed descriptions.

## Product Modes
- Continuous mode: quiet background monitoring, safety alerts prioritized.
- On-demand mode: explicit command/tap triggers richer descriptions.
- Channel preferences: voice only, haptic only, or hybrid.

## Claims Matrix

| Claim ID | Claim | Claim Type | V1 Status | Required Evidence |
|---|---|---|---|---|
| C-01 | System alerts users to high-priority nearby hazards. | Safety-critical | In scope | Hazard recall, false-safe rate, time-to-alert tests |
| C-02 | System provides directional and distance cues for navigation decisions. | Safety-critical | In scope | Directional accuracy, distance-zone accuracy, user task outcomes |
| C-03 | System reads visible text (signs, labels, menus) on demand. | Assistive | In scope | OCR recognition quality + task completion success |
| C-04 | System helps users find key objects in daily contexts. | Assistive | In scope | Object-finding task success without assistance |
| C-05 | System provides concise scene summaries in unfamiliar spaces. | Assistive | In scope | Human-rated usefulness + low overload rates |
| C-06 | System runs with privacy-preserving on-device processing for core path. | Platform | In scope | Architecture and telemetry verification |
| C-07 | System safely functions as a replacement for white cane/guide dog. | Disallowed claim | Out of scope | Never claim |
| C-08 | System guarantees collision-free/autonomous travel. | Disallowed claim | Out of scope | Never claim |
| C-09 | System makes medical diagnoses or treatment decisions. | Disallowed claim | Out of scope | Never claim |
| C-10 | System improves medical vision acuity. | Disallowed claim | Out of scope | Never claim |

## Explicit Non-Claims (Must appear in product and pilot materials)
- MaxSight is assistive guidance, not autonomous navigation.
- Users should not rely on MaxSight as their only mobility safety aid.
- MaxSight does not provide medical diagnosis or treatment advice.

## Capability Priority Tiers (Product)
- Tier A (must pass for release): hazard alerts, urgency, direction, distance, stable voice/haptic output.
- Tier B (high value, secondary): OCR, findability, context summaries.
- Tier C (defer if stability risk): advanced scene graph/retrieval-heavy features.

## User Impact Goals
- Fewer near-collision moments in routine routes.
- Higher independence on reading/finding tasks.
- Lower dependence on bystander/caregiver help in common tasks.
- Increased confidence without output overload.

## Ownership and Review Cadence
- Product owner: approves scope and user-facing claims.
- Safety owner: approves safety-critical claims and non-claims.
- ML/platform owners: provide evidence artifacts before release decision.
- Review frequency: every release candidate and every pilot incident review.
