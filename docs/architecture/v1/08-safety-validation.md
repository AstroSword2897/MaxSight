<!-- Part of the MaxSight V1 Technical Architecture Specification. See 00-overview.md. Source: approved plan 2026-10-31. -->

# D8: Safety and Validation Plan

### D8.1 Safety gates (DR-025; thresholds Open, DR-205)

| Gate | Objective | Environment | Dataset/scenario | Metric(s) | Threshold | Statistics | Failure condition | Evidence | Release-blocking |
|---|---|---|---|---|---|---|---|---|---|
| SG-01 | Frame-quality validation | reference + device | 500 frames labelled by quality | per-label P/R | pending (exp. target recall ≥0.9 for obstructed/dark) | Wilson CI | below threshold or n insufficient | `sg01.json` | yes |
| SG-02 | Detection performance | device (exported artifact) | Hazard v1 test + env-holdout | per-class recall/precision/AP | pending | cluster bootstrap | any safety class below floor | `sg02.json` + PR curves | yes |
| SG-03 | Tracking integrity | device replay | test clips with track IDs | IDF1, ID switches/100 tracks, survival through ≤5-frame occlusion | pending | bootstrap | below threshold | `sg03.json` | yes |
| SG-04 | Spatial reliability | device | tape keyframes | MAE/bias per bucket, valid-rate, σ coverage | pending, **per bucket**; buckets without a pass are never spoken in metres | bootstrap | spoken bucket fails | `sg04.json` | yes (for distance claims) |
| SG-05 | Hazard-assessment correctness | device replay real-time | test clips (events) | event recall per category, FNR, false alerts/min, rule-trace completeness | pending | bootstrap over clips | any category below floor; untraceable decision | `sg05.json` | yes |
| SG-06 | Uncertainty and degraded states | device | ambiguous suite: 50 clips (covered lens, glare, fast pan, low light, conflicting detections) | % correct perception/system state; silent failures | silent failures = 0; others pending | exact counts | any silent failure | `sg06.json` | yes |
| SG-07 | Alert delivery and prioritisation | device | scripted scenarios + timing harness | preemption correctness; camera-to-alert p95; channel fallback | preemption 100%; latency pending | p95 with CI | violation | `sg07.json` | yes |
| SG-08 | Offline and resilience | device + CI | fault-injection suite | pass/fail per fault (network, Stage B hang, corrupt model, both invalid, audio off, BT drop, thermal) | all pass | — | any fail | `sg08.json` | yes |

**Gate cell status:** `passed | failed | insufficient_evidence | threshold_pending`. A release requires every cell `passed`. `[E] ml/evaluation/safety_gates.py` and `scripts/infra/run_safety_gate_ci.py` are extended to compute cells from `--metrics` (today they use a boolean flag; `run_safety_gate_ci.py:38-61`).

**Experimental vs approved:** `configs/stage_a/v1.yaml` and `gates.yaml` carry `status: experimental|approved` per value. `gate_release.py` refuses `experimental` thresholds.

### D8.2 Evaluation layers

| Layer | What | Where |
|---|---|---|
| 1 Model | detector metrics, calibration | Python + exported artifact |
| 2 Module | each C-02…C-08 unit + module metrics | pytest / XCTest |
| 3 System | camera-to-alert replay, faults, states | device replay harness (recorded clips fed through `PerceptionActor`) |
| 4 Human factors | comprehension, workload, usability | supervised formative study (DR-039, blocked by DR-209) |
| 5 Privacy and security | egress, auth, storage, signing | tests + review |
| 6 Mobile operations | battery, thermal, memory, lifecycle interruptions | device sessions (30 min) |

### D8.3 Latency boundaries (all measured on device; `MetricsRecorder` timestamps from `t_mono_ns`)

| Boundary | From → to | Use |
|---|---|---|
| L0–L7a | as defined in D3.1b (capture wait, preprocessing, inference, postprocessing, assessment, Stage A total, dispatch, camera-to-alert, perceptibility) | the 80 ms target applies to **one** of these only after DR-204 is closed; until then all are reported, none is pass/fail against 80 ms |
| L8 cloud | request send → response received (+ server-side Lambda duration) | 4 s cap (PR-022) |
| L9 user interaction | end of user utterance (ASR final) → start of spoken answer | usability |

### D8.4 Test catalogue (IDs referenced in D3/D9/D10)

| Group | Tests |
|---|---|
| T-Q-01…05 | quality labels (black, white, blur, uniform cover, normal) |
| T-D-01…04 | strict load fails on wrong class count; output schema; empty-frame batch; parity Py↔Core ML |
| T-T-01…04 | birth; 4-miss survival; 6-miss expiry; class majority |
| T-S-01…10 | geometry exact (h=1.3, pitch=15° → Z within 1 cm); 7 validity flags; azimuth sectors; approach rate |
| T-U-01…03 | uncertainty ranges; evidence rule; calibration flag |
| T-H-01…10 | R1–R7; de-escalation after 3 frames; no single-measurement decision; context category |
| T-C-01…08 | dedup 4 s; re-alert on increase; degraded announce once; refusal once per period; four perception states; stale decision expiry; fast path; status transitions |
| T-P-01…06 | policy safety priority over 1,000 FVPs; cap non-critical only; dismissal; stale Stage B; refuse unknown distance; offer on insufficient evidence |
| T-O-01…04 | preemption; channel failure fallback; route change; VoiceOver no double-speech |
| T-B-01…08 | Stage B schema validation; idempotency 409; quota 429; deadline 504; stale rejection; image field rejected (egress v1); no-input logging; kill switch |
| T-M-01…05 | model store: corrupt → LKG; unsigned → LKG; both invalid → UNAVAILABLE; incompatible manifest refused; rollback offline |
| INV-1…8 | D3.7 |
| T-W-01 | Playwright: simulator renders overlay + alert log; degraded banner |
| T-GV | golden-vector parity (Python ↔ Swift) |

### D8.5 Cross-platform parity

The same 500 golden frames and 30 golden clips run on reference / iOS (and Android later):
- detection agreement (matched IoU ≥ 0.9 for ≥ 99% of boxes)
- hazard-state agreement ≥ 99%
- latency and memory compared per device

### D8.6 V1 definition of done (Q280)

1. Stage A runs locally and offline (SG-08).
2. The selected model meets approved criteria (SG-02).
3. Tracking and hazard interfaces are validated (SG-03/05).
4. Uncertainty and degraded states are explicit (SG-06).
5. All SG cells are passed with approved thresholds.
6. The policy delivers accessible prioritised notifications (SG-07, usability).
7. The iOS runtime is reliable on the device matrix.
8. Stage B is delivered for the approved tasks and cannot disable local function (INV-1).
9. Privacy, security and consent are implemented (egress tests, review).
10. Integration and release tests pass.
11. Intended use and limitations are documented.
12. Engineering and independent safety approvals are recorded.
