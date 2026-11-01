<!-- Part of the MaxSight V1 Technical Architecture Specification. See 00-overview.md. Source: approved plan 2026-10-31. -->

# D1: Architecture Decision Register

Each row: ID · Decision · Status · Rationale · Alternatives · Consequences · Dependencies · Validation · Source.

**Rules:**
- An ADR file (`docs/adr/NNN-*.md`) is created for every *Confirmed* decision with system-wide consequences (marked ★).
- *Proposed* rows become Confirmed only by your approval, recorded in the ADR's "Approver" field.

### D1.1 Confirmed decisions

| ID | Decision | Status | Rationale | Alternatives | Consequences | Dependencies | Validation | Source |
|---|---|---|---|---|---|---|---|---|
| DR-001★ | iOS is the initial platform | Confirmed | Native camera, Core ML, accessibility | Android first; cross-platform | The Swift app is the V1 product; the Python pipeline is reference and evaluation tooling only | DR-206 device matrix | iOS device tests (D8) | Directive §1 |
| DR-002 | Indoor and outdoor use; daylight, low light, crowded, moving environments | Confirmed | Mobility anywhere is the product vision | Indoor only | Datasets and tests must cover all strata | D7 capture plan | SG-02/05 per-stratum results | Q3, Q6 |
| DR-003 | V1 = environmental awareness + on-demand understanding + personalised assistance | Confirmed | Differentiator vs single-image apps (Q9) | Awareness only | Requires Stage A, Stage B and FVP/policy | DR-005/006/008 | V1 DoD (D8.8) | Directive, Q1, Q9 |
| DR-004★ | Population = **low vision**; no 13-condition segmentation; complete blindness is out of V1 | Confirmed | Simulated conditions were ineffective; broader usability | 13 condition groups | No per-condition models or production inputs | — | Pilot recruits low-vision adults | Q4–5 |
| DR-005★ | Stage A is local; essential functions work offline | Confirmed | Safety must not depend on the network | Cloud perception | Model must fit on device | DR-206 | SG-08 offline suite | Q7, Q20 |
| DR-006★ | Stage B is cloud-based and **required** in the complete V1 | Confirmed (correction) | Contextual understanding exceeds on-device capacity | On-device only; optional | Needs cloud API, auth, privacy boundary | DR-201, DR-208, DR-213 | D4 task evaluations | Directive (corrects Q8 scope) |
| DR-007 | Build order: Stage A first, vertically; Stage B integrated after Stage A contracts are stable | Confirmed | Test the testable first | Parallel big-bang | D9 phase order | — | Phase exit criteria | Q8 |
| DR-008★ | Three layers: perception → FVP → assistance policy | Confirmed | Separation of truth, needs and communication | One end-to-end model | Strict module boundaries | — | Isolation tests (D8 T-INV) | Q11 |
| DR-009★ | Detection, hazard assessment and assistance policy are separate responsibilities | Confirmed | Hazard depends on context and the user, not just the class | Learned hazard head | Rule-based assessor between detector and policy | DR-010 | Rule-trace audit (SG-05) | Q2, Q26, directive |
| DR-010★ | Stage A = 7 components in order: quality → detection → tracking → spatial → uncertainty → hazard → local safety/alert controller. Feature extraction lives inside detection | Confirmed | Questionnaire decision; diagram order | 8 components | D3 module layout | — | Pipeline contract tests | Q21, conflict 5 |
| DR-011★ | SSD-MobileNet is the initial detection benchmark | Confirmed | ~3× faster than ResNet50+FPN in Google benchmarks; latency matters | ResNet50+FPN first | First trained model | DR-112 data | D7 benchmark | Q22, Q79 |
| DR-012 | ResNet50+FPN and EfficientDet-Lite0 are comparison candidates, not approved production models | Confirmed | Evidence-based selection | Single model | Same evaluation protocol for all | DR-011 | D7 selection procedure | Q23, Q81 |
| DR-013★ | Perception is condition-independent | Confirmed | A diagnosis doesn't change environmental truth | Condition-conditioned model | Remove condition inputs from production paths | — | `test_condition_free_stage_a` | Q76, ADR-006 |
| DR-014★ | FVP cannot alter perception truth, hazard classes or urgency | Confirmed | Safety invariance | Personalised thresholds | FVP read only by policy and output | DR-008 | `test_fvp_invariance` | Q72 |
| DR-015★ | Stage B cannot override or delay Stage A | Confirmed | Cloud is advisory and can fail or hallucinate | Fusion with equal weight | Policy priority rules; stale rejection | — | `test_stage_b_cannot_override` | Q112–113 |
| DR-016★ | Raw video stays local; any visual egress only through an approved data policy | Confirmed | Privacy | Cloud frames by default | Stage B visual capability depends on DR-201 | DR-201 | Egress tests (D4.6) | Q201, directive |
| DR-017★ | No additional sensors beyond what is explicitly decided | Confirmed | Scope and privacy control | Use IMU/LiDAR freely | V1 inputs = camera (perception) + microphone (spoken requests). IMU, LiDAR, depth and any other sensor are **excluded from V1** (DR-210/211 closed); any future sensor needs a new, separate architecture decision | — | Code review + entitlement audit | Directive |
| DR-018 | Hazard taxonomy: Dynamic, Static, Environmental, Context-dependent | Confirmed (classes Open, DR-202) | Mobility-relevant grouping | Class-only | Assessor maps tracks → categories | DR-202 | SG-05 per category | Q24 |
| DR-019 | Urgency 0 Informational / 1 Caution / 2 High / 3 Critical; DEGRADED/UNKNOWN is a separate system state | Confirmed | Clear priority semantics | 3 levels | Policy priority rules | — | Policy tests | Q46 |
| DR-020 | Four perception states never collapsed: CLEAR_VALID, POTENTIAL_HAZARD_UNCERTAIN, INSUFFICIENT_EVIDENCE, PERCEPTION_UNAVAILABLE | Confirmed | Silence ≠ safety | Empty list = clear | `EnvironmentalState` schema | — | SG-06 | Q54 |
| DR-021★ | Unknown values are explicit (null + reason); never 0, safe or confirmed | Confirmed | Prevents fabricated measurements | Defaults | Schema rules (D3.3) | — | Schema tests | Q32, Q37 |
| DR-022★ | No silent failure; degraded/unavailable states are announced | Confirmed | User must not read silence as safety | Best effort | Status machine + announcements | DR-031 | SG-06/08 | Q53, Q62 |
| DR-023 | Urgency is computed by interpretable rules, not an opaque learned model | Confirmed | Auditability | Learned urgency | Rule table versioned | — | Rule tests | Q47 |
| DR-024 | 0.7 uncertainty threshold is experimental only | Confirmed | Uncalibrated | Production use | Only changes wording to "possible" | D7 calibration | ECE report | Q49 |
| DR-025 | SG-01…08 = quality, detection, tracking, spatial, hazard, uncertainty/degraded, alert delivery, offline/resilience; thresholds pending | Confirmed | Area coverage | Old SG table | Update `02_safety_first_release_gates.md` | DR-205 | D8 | Q57–58 |
| DR-026 | Thresholds proposed by engineering with evidence; approved with independent accessibility/O&M review | Confirmed | Not self-certified | Developer-only | Reviewer recruitment | DR-209 | Approval records | Q59 |
| DR-027 | FVP is user-provided; diagnosis optional; professional input optional later | Confirmed | Access without diagnosis | Mandatory diagnosis | D5 schema | DR-214 | Onboarding test | Q63, Q74, Q252 |
| DR-028 | Assistance policy is deterministic: alert, inform, ask, offer, defer, stay_quiet, refuse | Confirmed | Predictable behaviour | Learned policy | D5.3 rules | — | Policy tests | Q117 |
| DR-029 | Event lifecycle NEW → ACTIVE → UPDATED → RESOLVED → EXPIRED; delivery status separate | Confirmed | Dismissal ≠ resolution | Single flag | `HazardEvent` schema | — | State tests | Q125–126 |
| DR-030 | Native TTS (`AVSpeechSynthesizer` on iOS) | Confirmed | Offline, no dependency | Cloud TTS | Voice availability varies | — | Output tests | Q132–134 |
| DR-031 | App states READY / DEGRADED / OFFLINE / UNAVAILABLE; connectivity ONLINE_FULL / ONLINE_DEGRADED / OFFLINE | Confirmed | Explicit status | Implicit | Status indicator | — | State tests | Q167, Q197 |
| DR-032 | Startup sequence per Q166 | Confirmed | Fail-closed start | Lazy init | D6 startup | DR-034 | Startup tests | Q166 |
| DR-033 | Release package = artifact + manifest (Q169 fields) + checksum + signature + evaluation evidence | Confirmed | Traceability | File only | Release tooling | — | `gate_release` | Q168–169 |
| DR-034★ | Signed artifacts; bundled last-known-good model; offline rollback; refuse when both models are invalid | Confirmed | Integrity | Unsigned | KMS + on-device verify | — | Integrity tests | Q170–177 |
| DR-035 | Native apps: Swift (iOS), Kotlin (Android); models in Python/PyTorch; AWS for training/deployment | Confirmed | Native capabilities | Cross-platform UI | Two codebases + shared contracts | — | Parity tests | Q17, Q262 |
| DR-036 | Branching `feature/* → dev → main → release/*` | Confirmed | Controlled release | Trunk-only | CI rules | — | Branch protection | Q266 |
| DR-037 | Architecture changes via ADRs | Confirmed | Traceable decisions | Ad hoc | `docs/adr/` | — | Review | Q276 |
| DR-038 | Intended-use statement (Q244 text); no promise of collision prevention or guaranteed detection; does not replace cane, guide dog or O&M | Confirmed (wording under review) | Honest claims | — | Onboarding and store copy | DR-209 | Claims review | Q244–245, Q250 |
| DR-039 | Pilot: 5–10 formative → 15–30 pilot after prerequisites | Confirmed | Staged risk | Direct pilot | Protocol work | DR-209 | Pilot checklist | Q254–256 |
| DR-040 | Cloud controls: 4 s Stage B timeout, per-user quotas, AWS budget alarms, scoped IAM, **no embedded cloud credentials** | Confirmed | Backlog requirement | — | Cognito-issued tokens | DR-208 | D4.4 tests | Backlog, Q191–195 |
| DR-041 | Model selection is benchmark-driven, multi-metric; never by aggregate mAP alone; a weak safety category disqualifies | Confirmed | Safety-critical selection | Best mAP | D7.9 procedure | — | Selection report | Q79, Q96, Q100, directive §8 |
| DR-042 | Splits by session/video/environment/participant, never adjacent frames | Confirmed | Leakage | Random frames | Split tooling | — | `gate_pretrain` | Q91 |
| DR-043 | Simulated impairments are a separate research category | Confirmed | Not real evidence | Use as evidence | Excluded from gates | — | Report tagging | Q94 |
| DR-044 | Telemetry: minimal diagnostics; research telemetry off by default with explicit consent; accessible withdrawal | Confirmed | Privacy | Opt-out | Consent manager | — | Consent tests | Q213–216 |
| DR-045 | Pilot data never trains models without consent and governance review | Confirmed | Ethics | Implicit reuse | Data tagging | — | Data audit | Q222 |

### D1.2 Proposed engineering solutions (need your approval)

| ID | Proposal | Status | Rationale | Alternatives | Consequences | Dependencies | Validation | Source |
|---|---|---|---|---|---|---|---|---|
| DR-101 | Stage A target 15 fps on the reference iPhone; 30 fps stretch | Proposed | Balances latency and thermal | Fixed 30 fps | Frame budget 66.7 ms | DR-206 | Device benchmark | Recommendation, Part 9 |
| DR-102 | Adaptive processing: ACTIVE / CALM (5 fps when no tracks and a static scene) / THERMAL_REDUCED (10 fps + DEGRADED) | Proposed | Battery and thermal | Always max | Missed-onset risk tested | DR-101 | Onset-latency test | Q14 (you flagged research) |
| DR-103 | Tracking: greedy IoU tracking-by-detection (V1 baseline); SORT and ByteTrack evaluated for production | Proposed (approach Confirmed Q33) | Simple and testable | DeepSORT | IDF1 benchmark | — | SG-03 | Q33 |
| DR-104 | Spatial baseline: geometric ground-plane estimate with validity flags; monocular depth benchmarked; final method Open (DR-203) | Proposed | Simplest measurable (Q30) | Mono only | Distance often unknown | DR-203 | SG-04 | Q30–32 |
| DR-105 | Walking corridor ±0.45 m, 0.3–4.0 m ahead *(exp.)* | Proposed | Shoulder width + margin | Image-centre band | Path relevance rule | DR-104 | Path-relevance audit | Q28–29 |
| DR-106 | Urgency rule table R1–R7 with experimental thresholds (D3.4) | Proposed | Interpretable (DR-023) | Learned | Tunable YAML | DR-018 | SG-05 | Q47 |
| DR-107 | 12 non-critical alerts/min cap; 0.3 s channel gap *(exp.)* | Proposed (experimental status Confirmed) | Overload control | None | Critical bypass | — | Usability | Q146–147 |
| DR-108 | Schema family of D3.3 (11 schemas, versioned `maxsight.stage_a/2.0`) | Proposed (Q36 core fields Confirmed) | Implementable contracts | Ad hoc dicts | Swift/Kotlin/Python codegen from JSON Schema | — | Schema tests | Q36–38 |
| DR-109 | On-device OCR (Apple Vision `VNRecognizeTextRequest`), on-device speech recognition (`SFSpeechRecognizer`, on-device mode); cloud receives derived text only | Proposed | Delivers OCR without visual egress | Cloud OCR on images | Text egress needs consent class "derived text" | DR-201 | Stage B task eval | Q101–105 |
| DR-110 | Cloud stack: API Gateway (HTTP API) + Lambda (Python 3.12) + Bedrock (one approved model) + DynamoDB + Cognito + CloudWatch | Proposed | Serverless, scoped IAM, no embedded credentials | ECS service | AWS inventory | DR-208, DR-213 | D4 tests | Q184–194 |
| DR-111 | Two-stage training: pretrain on a harmonised public mix, fine-tune and evaluate on MaxSight-captured data only | Proposed | Public data lacks hazard labels | Public only | Capture program | DR-202 | D7 | Q84–87 |
| DR-112 | Datasets: COCO 2017, Open Images V7, Mapillary Vistas v2 (non-commercial) for pretraining; MaxSight Hazard v1 for fine-tuning and all evaluation; BDD100K eval-only | Proposed (final list Open, Q85) | Coverage + licence | Single dataset | Licence review | DR-202 | Licence register | Q85 |
| DR-113 | Class balance: cap 8,000 instances per class; no class > 15% of instances; car/bus/truck merged as `vehicle` | Proposed | Your Q11 car-skew finding | Unbalanced | Balanced sampler | DR-112 | `gate_pretrain` | Q11 answer |
| DR-114 | Golden test vectors (recorded inputs → expected outputs) for Python ↔ Swift ↔ Kotlin parity | Proposed | Q154–155 parity | Manual comparison | Vector export tool | DR-108 | Parity CI | Q154–155 |
| DR-115 | Python reference pipeline on Mac = development and evaluation tool (`platform=reference`), never product evidence | Proposed | Fast iteration | iOS only | Dual implementation | DR-114 | Parity tests | Previous rounds |
| DR-116 | Precision: FP32 reference; FP16/INT8 adopted only if per-class relevant recall drops ≤ 1 pt *(exp.)* | Proposed (approach Confirmed Q98) | Safety over size | INT8 by default | Core ML quantisation | — | D7.7 | Q98, Q156 |
| DR-117 | Retention defaults: frames RAM only; tracks ≤ 10 s; session events local, 30 days; FVP until deletion | Proposed | Minimisation | Longer | Retention doc | DR-201 | Privacy tests | Q127, Q211 |

### D1.3 Open, experiment-dependent and blocked decisions

| ID | Decision needed | Status | Default until decided | Blocks | Source |
|---|---|---|---|---|---|
| DR-201 | **Visual data egress for Stage B** (frames, crops, thumbnails, embeddings, derived text) | **Open (most consequential)** | Nothing visual leaves; derived text only with consent (DR-109) | Cloud OCR on images, image scene understanding, visual Q&A (D4.6) | Q104, Q202–203 |
| DR-202 | Final hazard taxonomy and detector class list | Open | 17 classes (D7.2) | Label-map freeze (P2-08) | Q24, Q85 |
| DR-203 | Distance method (geometric vs monocular vs other) | Experiment + Open | Geometric baseline; distance often unknown | Any metre claim | Q30 |
| DR-204 | 80 ms measurement boundary | Open | **No default.** All boundaries L0–L7 (D3.1b) are measured and reported; no pass/fail against 80 ms until you close this decision | Latency gate | Q234, conflict 3 |
| DR-205 | Numeric SG thresholds | Experiment + Open | Gates report values; release blocked (`threshold_pending`) | Any release | Q58 |
| DR-206 | Device and OS matrix | Open | iPhone 13 or newer, iOS 17+ *(proposed)* | Device gates | Q149–151 |
| DR-207 | Audio and haptic vocabulary | Experiment (usability) | Dev placeholders (D5.6) | SG-07 sign-off | Q137–138 |
| DR-208 | AWS service inventory and cost limits | Open | Training-only S3 + SageMaker; Stage B stack per DR-110 in dev account | Stage B production | Q184, Q195 |
| DR-209 | Pilot approval, regulatory review, independent reviewers | **Blocked** (external) | No participant testing | Phase 6 human factors | Q246–248, Q255 |
| DR-210 | IMU in V1 | **Closed: excluded** (DR-017) | Camera-only spatial estimation with image-based calibration-validity monitoring (D3.6.2) | — (future sensor = new ADR) | DR-017 |
| DR-211 | LiDAR/depth sensors in V1 (including ground-truth capture) | **Closed: excluded** (DR-017) | Ground truth by tape-measured floor marks (D7.4) | — | DR-017 |
| DR-212 | Additional architecture layers beyond three | Open | Three layers | — | Q12 |
| DR-213 | Stage B task allocation and cloud model choice | Open | D4 task table, model TBD | Stage B implementation | Q82 |
| DR-214 | Professional assessment in the V1 FVP | Open | User-provided only | — | Q63 |
| DR-215 | Android in V1 scope | Open | After the iOS V1; architecture specified (D6.2) | Android tasks | Directive §7 |
| DR-216 | Wearable/glasses camera hardware | **Blocked** (hardware) | Phone camera on a chest mount | Wearable claims | Prior rounds |
| DR-217 | Subsystem owners | Open | You own all | Review routing | Q277 |

### D1.4 Parameter registry (`configs/parameters.yaml`, the single source read by Python, Swift and Kotlin)

**Rules:**
- Every tunable value has an ID and a status: `confirmed` (you decided), `engineering-start` (an initial value for development only) or `approved` (after experiment and review).
- `gate_release.py` refuses any **safety-relevant** (S) parameter that is not `approved`.
- Code may not hard-code these values (lint rule: a magic-number check in the Stage A/policy packages).

| ID | Parameter | Initial | Status | S? | Rationale | Experiment (D7/D8) | Acceptance evidence | Approver |
|---|---|---|---|---|---|---|---|---|
| PR-001 | Stage A target rate | 15 fps | engineering-start | S | thermal/latency balance | device L5 + thermal runs (P6-06) | L5 p95 + 30-min thermal report | you + reviewer |
| PR-002 | CALM rate / THERMAL rate | 5 / 10 fps | engineering-start | S | power | onset-latency test after CALM→ACTIVE | onset ≤ 1 frame at ACTIVE rate | you |
| PR-003 | Detector score threshold | 0.35 | engineering-start | S | recall-leaning | PR sweep on val (per class) | per-class recall/FP curve | you + reviewer |
| PR-004 | NMS IoU | 0.55 | engineering-start | — | standard | val sweep | mAP/recall | eng |
| PR-005 | Tracker match IoU | 0.3 | engineering-start | — | low-fps association | IDF1 sweep | SG-03 | eng |
| PR-006 | Track max misses / decay | 5 frames / 0.8 | engineering-start | S | occlusion survival vs ghosts | occlusion clips | SG-03 survival | you |
| PR-007 | Confirm hits for `confirmed` path | 3 | engineering-start | S | evidence before claims | path audit | SG-05 | you |
| PR-008 | Corridor half-width / range | 0.45 m / 0.3–4.0 m | engineering-start | S | shoulder + margin | path-relevance audit vs annotations | SG-05 | you + O&M reviewer |
| PR-009 | Distance bins for urgency | 1.0 / 2.0 / 4.0 m | engineering-start | S | walking reaction time | event replay | SG-05 | you + reviewer |
| PR-010 | Looming-τ thresholds (indicator only) | 2.0 / 4.0 s | engineering-start | S | see D3.6.3 corroboration rule | τ vs ground-truth TTC on dynamic clips | SG-05 | you + reviewer |
| PR-011 | De-escalation frames / resolve time | 3 frames / 1.0 s | engineering-start | S | stability | flicker audit | SG-05/07 | you |
| PR-012 | Duplicate-suppression cooldown | 4.0 s | engineering-start | — | overload | usability | SG-07 + study | you |
| PR-013 | Non-critical cap | 12/min | confirmed-experimental (Q146) | — | overload | usability study | study report | you + accessibility reviewer |
| PR-014 | Channel gap | 0.3 s | confirmed-experimental (Q146) | — | perception of two channels | usability | study | same |
| PR-015 | Stale state / decision age | 300 ms | engineering-start | S | freshness | timing harness | SG-07 | you |
| PR-016 | Quality thresholds (luma 40, clip 0.20, lap 60, uniform 0.60/15, flow 25 px) | as listed | engineering-start | S | D3.5 | 500-frame quality set | SG-01 | you |
| PR-017 | Uncertainty wording threshold | 0.7 | confirmed-experimental (Q49) | S | wording only | calibration (ECE) | SG-06 | you + reviewer |
| PR-018 | Evidence-reliability cut-offs | 0.3/0.2 sufficient; 0.6/0.5 insufficient | engineering-start | S | D3.6.4 | qualified-alert precision | SG-06 | you |
| PR-019 | Calibration validity: max pitch deviation / max rotation rate / min flow inliers | 5° / 30°/s / 0.5 ratio & ≥50 features | engineering-start | S | D3.6.2 | walking clips with tape ground truth | SG-04 valid-rate vs error | you |
| PR-020 | Haptic patterns | D5.6 placeholders | engineering-start | S | distinguishability | usability (DR-207) | recognition ≥ target | you + accessibility reviewer |
| PR-021 | Default speech rate | 200 wpm | engineering-start | — | comprehension | usability | study | you |
| PR-022 | Stage B client deadline | 4.0 s | **confirmed** (backlog, DR-040) | — | responsiveness | — | — | you |
| PR-023 | Stage B server budget / Bedrock budget | 3.5 s / 3.0 s | engineering-start | — | under client deadline | load test | L6 p95 | eng |
| PR-024 | Per-user daily quota | 200 tasks | engineering-start | — | cost | usage telemetry (consented) | budget report | you |
| PR-025 | Readiness timeout at startup | 2.0 s | engineering-start | S | fail-closed start | device startup tests | startup log | eng |

# Appendix Q: Questionnaire answer index (source of truth)

| Q | Your answer / accepted default | DR |
|---|---|---|
| 1 | All of awareness, hazard detection, navigation assistance, object recognition | DR-003 |
| 2 | Detect-hazardous-object and decide-it's-a-hazard are separate layers; hazard depends on the user | DR-009 |
| 3, 6 | Outdoor mobility / anywhere; all conditions | DR-002 |
| 4–5 | Low vision broadly; not the 13 simulated conditions; complete blindness too hard for V1 | DR-004 |
| 7 | Only Stage A must work offline | DR-005 |
| 8 | Stage A first, vertical (Stage B required in full V1 per correction) | DR-006/007 |
| 9 | Streaming hazard and navigation assistance is the differentiator | DR-003 |
| 10 | Claims per C-06, C-09 | DR-038 |
| 11 | Three layers; past 13-model ResNet50+FPN had high latency; car-skewed data → single-stage MobileNetV2 + balancing | DR-008, DR-011, DR-113 |
| 12 | Extra layers: needs more thought | DR-212 |
| 13 | Camera → quality → local layer (detect, track, matrix, uncertainty, distance, speed) → alert; cloud layer synthesises; assistance layer chooses output and whether help is needed | D2, D3 |
| 14 | Unsure how to avoid continuous streaming | DR-102, DR-219→DR-101/102 |
| 15 | Skipped | D2.3 |
| 16 | Can't decide before the model exists | DR-108 (proposed) |
| 17 | As native as possible; model dev/deploy on AWS | DR-035 |
| 18 | Tests, regression tests, Playwright | D8.4 |
| 19 | Switch to alternative components; to research | D2.10 |
| 20 | Stage A fully independent | DR-005 |
| 21 | Seven components fine | DR-010 |
| 22 | SSD-MobileNet first (≈3× faster than ResNet50-FPN) | DR-011 |
| 23 | Run all; one first as benchmark, then compare | DR-012 |
| 23b | Same input for all | D7.9 |
| 24–39 | Taxonomy, prioritisation, separation, path evidence, spatial benchmarking, calibration, unknown distance, tracking, occlusion, motion, contract, mandatory fields, timestamps, low quality | DR-018…021, D3 |
| 44–62 | Hazard definition, urgency levels, rule-based urgency, uncertainty handling, 0.7 experimental, states, FN/FP, conflicts, SG mapping, pending thresholds, approvals, evidence, no 13 models, disable conditions | DR-019…026, D8 |
| 63–78 | FVP user-provided, fields, unknowns, conflicts, staleness, editable, presentation-only, immutable safety, multiple conditions, no diagnosis, condition IDs eval-only, condition-independent perception | DR-013/014/027, D5 |
| 79–100 | Model selection process, candidates, Stage B modular, task-specific, fine-tuning, datasets TBD, mobility data, annotation, review, leakage, volume by statistics, simulation separate, metrics, release-blocking, calibration, quantisation, versioning, regression | DR-011/012/041…043, D7 |
| 101–116 | Stage B scope, deferrals, cloud enrichment, egress undecided, consent, priority, deadlines, cancellation, staleness, qualified language, no override, schemas, budgets, testing | DR-006/015/201, D4 |
| 117–130 | Policy actions, inputs, event history, dedup, priority, prefs storage, separation, safety constraint, dismissal, state machine, retention, simple stats before LSTM, measurement | DR-028/029, D5 |
| 131–148 | Output manager, native TTS, routing, bone conduction, haptic vocabulary, sync, interruption, BT/mute fallback, VoiceOver/TalkBack, controls, 12/min and 0.3 s experimental, overload | DR-030/107, D5.4 |
| 149–167 | Device matrix TBD, LiteRT eval, Core ML, preprocessing parity, numerical parity, budgets, permission/camera loss, lifecycle, background, status indicator, startup, states | DR-031/032/206, D6 |
| 168–183 | Release package, manifest, signing, keys, rotation, verification, corrupt/LKG/unavailable, rollback offline, staging, promotion, recertification, compatibility, update-failure tests | DR-033/034 |
| 184–200 | AWS inventory TBD, SageMaker, API GW/Lambda, Bedrock advisory, S3, DynamoDB, CloudWatch, auth, tokens, IAM, Bedrock restriction, budgets, AWS down, connectivity states, isolation, deadlines, logging | DR-040/110/208 |
| 201–222 | Raw video local, crops/frames/thumbnails/embeddings separate, local data, sync, source of truth, queue, idempotency, conflicts, offline deletion, retention, export/delete, diagnostics, telemetry off, consent, withdrawal, encryption, storage, secrets, breach, processors, pilot data | DR-016/044/045/117, D4.6 |
| 223–243 | Test matrix, levels, events vs frames, composition, conditions, devices, metrics, FNR TBD, latency boundaries, thermal, refusal, parity, integrity, gate failure, release blockers, evidence, regressions, prototype vs validated | D8 |
| 244–260 | Intended use, claims, regulatory review, professionals, risks, warnings, onboarding, diagnosis optional, accessible consent, formative protocol, pilot prerequisites, supervision, incidents, stop criteria, findings → issues | DR-038/039/209 |
| 261–280 | Repo modularity, languages, packages, coding standard, testing strategy, branching, CI, blockers to audit, quality gates, versioning, reproducibility, lineage, ADRs, ownership, review of safety changes, documentation template, V1 DoD | DR-035…037, D6, D8.6, D9 |
