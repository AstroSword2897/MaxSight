<!-- Part of the MaxSight V1 Technical Architecture Specification. See 00-overview.md. Source: approved plan 2026-10-31. -->

# D5: FVP and Assistance Policy Specification

### D5.1 FVP schema (`fvp.v1`, JSON; Swift Codable; stored with `NSFileProtectionComplete`)

| Field | Type | Req | Default | Notes |
|---|---|---|---|---|
| schema_version | string | yes | "maxsight.fvp/1.0" | |
| profile_id | uuid | yes | generated | |
| preferred_modality | enum speech/haptic/both | yes | both | |
| alert_verbosity | enum minimal/standard/detailed | yes | standard | affects wording length only |
| speech | {rate_wpm: 120–300, voice_id: string\|null, volume: 0–1} | yes | {200, null, 1.0} | |
| haptic | {enabled: bool, intensity: 0.2–1.0} | yes | {true, 0.8} | |
| assistance | {proactive_context: bool, repeat_on_request: bool, describe_on_degraded: bool} | yes | {false, true, true} | |
| accessibility | {voiceover_aware: bool, large_text: bool, high_contrast: bool} | yes | from system settings | |
| functional_needs | {central_vision_difficulty, peripheral_vision_difficulty, light_sensitivity, low_light_difficulty, contrast_difficulty}: each {value: none/some/severe/unknown, source: user_reported/professional, updated_at} | no | all `unknown` | informational; affects presentation suggestions only |
| assessments | [{source, assessed_at, summary}] | no | [] | never merged (Q67); flagged `possibly_outdated` after 365 days (Q68) |
| condition_ids | string[] | no | [] | evaluation/compatibility only (DR-013) |
| consent | {c1: bool, c2: bool, c3: bool, sync: bool, updated_at} | yes | {true(disclosed), false, false, false} | |
| updated_at | ISO-8601 | yes | now | |

**Lifecycle:**

| Step | Behaviour |
|---|---|
| Create | Spoken onboarding (D5.5) → defaults + answers |
| Update | Settings or voice ("speak faster") → new version written atomically; previous version kept for undo (1) |
| Persistence | Local file plus Keychain item for consent flags |
| Versioning | `schema_version` migration functions; unknown major → reset to defaults with notice |
| Deletion | "Delete my profile" → local wipe + queued server delete if synced (Q210) |
| Sync | Only if `consent.sync`; DynamoDB `prefs` (PK user_id) with `updated_at` last-writer-wins **for preference fields only** (Q209: safety config is never synced) |

**Separation:** the safety configuration (`configs/stage_a/*.yaml`, rules) is bundled with the model package, not in the FVP (Q123).

### D5.2 Policy inputs

| Input | Required |
|---|---|
| `EnvironmentalState` (latest, non-stale) | yes |
| system_status, connectivity | yes |
| current output activity (speaking/idle, current urgency) | yes |
| event table (`HazardEvent[]` with delivery) | yes |
| FVP | yes |
| Stage B `TaskResponse` | optional |
| user commands (repeat, stop, more) | optional |

### D5.3 Policy outcomes and conditions

| Action | Condition | Priority tier |
|---|---|---|
| **alert** | event NEW/UPDATED-with-urgency-increase and urgency ≥ 2; or urgency 3 and last delivery > 4 s ago while still ACTIVE | Safety (1) |
| **inform** | urgency ≤ 1 hazards when `proactive_context=true` and within the non-critical cap; Stage B responses; status announcements (degraded/offline) | Request (2) / Context (3) |
| **ask** | ambiguous request (e.g. "find it" with no object) | Request (2) |
| **offer** | INSUFFICIENT_EVIDENCE > 5 s with `describe_on_degraded=true` → "Want me to try describing the scene?" (only if Stage B is available) | Context (3) |
| **defer** | any tier-2/3 output while a tier-1 alert is being delivered or queued | — |
| **stay_quiet** | duplicate within cooldown; urgency 0 with `proactive_context=false`; non-critical cap reached | — |
| **refuse** | asked for a distance that is null or unvalidated; Stage B `supported=false`; offline request needing Stage B | Request (2) |

**Priority hierarchy:** Safety (urgency 3 > 2) > status announcements of UNAVAILABLE/DEGRADED > user request responses > proactive context.

**Interruption:**
- Urgency 3 interrupts anything immediately.
- Urgency 2 waits for the current utterance to finish if it is tier-1, and otherwise interrupts.
- Tier 2/3 never interrupt.

**Caps:** 12 per minute for tier 3 (DR-107); tiers 1–2 are never capped.

**Invariant:** FVP is applied **after** tier-1 selection and only formats it (INV-3).

### D5.4 Output selection rules

| preferred_modality | Urgency 3 | Urgency 2 | ≤1 / responses |
|---|---|---|---|
| both | haptic critical pattern **and** speech (interrupt) | haptic high + speech | speech |
| speech | speech (interrupt) **+ haptic critical pattern** (safety minimum) | speech | speech |
| haptic | haptic critical pattern **+ short speech** ("Stop") (safety minimum) | haptic high | haptic info; speech only for requested answers |

**Safety minimum:** urgency 3 always uses ≥ 2 channels when available, whatever the preference (DR-014). If a channel fails, the remaining channel is used and DEGRADED is announced.

**UI:** every spoken message is also posted as a `UIAccessibility` announcement only when VoiceOver is off (to avoid double speech); the status banner always reflects `system_status`.

### D5.5 Onboarding (spoken, ≤ 4 min)

1. Welcome, purpose and limits (Q250 warning).
2. Consent C1 disclosure, C2 opt-in question.
3. Modality.
4. Verbosity.
5. Speech-rate trial.
6. Haptic test (feel 4 patterns).
7. Optional functional-needs questions (skippable).
8. Status-indicator demo.
9. "Hazard detection is on."

Diagnosis is never asked as required (Q252).

### D5.6 Same event, different preferences (classification identical)

Event: static obstacle (chair), path confirmed, 1.6 m, centre-right → urgency 2 (R4).

| FVP | Delivered |
|---|---|
| both / standard / 200 wpm | haptic "high" pattern on the right side (iOS: pattern only; direction in speech) + "Chair slightly right, about 2 metres." |
| speech / minimal / 260 wpm | "Chair, slightly right." (faster rate; distance omitted by minimal verbosity) |
| haptic / minimal | haptic "high" pattern only; asking "what was that?" repeats "Chair slightly right, about 2 metres." |
| any profile, same chair at 0.8 m (R2, urgency 3) | critical haptic + "Stop. Chair ahead." with interruption: **identical across profiles** apart from rate |

**Vocabulary keys** (`app/vocabulary/en.json`, DR-207 experimental):

| Key | Text |
|---|---|
| `alert.critical.static` | "Stop. {object} {dir}." |
| `alert.high.static` | "{Object} {dir}, about {dist} metres." *(dist only if validated)* |
| `alert.dynamic` | "{Object} approaching {dir}." |
| `alert.qualified` | "Possible obstacle {dir}." |
| `status.degraded.camera` | "Camera view is poor. Hazard detection is limited." |
| `status.unavailable` | "Hazard detection unavailable. Use your primary mobility aid." |
| `status.offline` | "You're offline. Hazard detection is still on." |

**Haptic placeholders:**

| Level | Pattern |
|---|---|
| info | 1 × 40 ms |
| caution | 2 × 40 ms |
| high | 120 ms + 40 ms |
| critical | 3 × 150 ms at full intensity |
