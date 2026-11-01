<!-- Part of the MaxSight V1 Technical Architecture Specification. See 00-overview.md. Source: approved plan 2026-10-31. -->

# D10: Requirements Traceability Matrix

Questionnaire → requirement → component → task → test → acceptance.

| Source | Requirement | Component | Task | Test | Acceptance |
|---|---|---|---|---|---|
| Q7, Q20, DR-005 | Stage A works offline, no network imports | C-02…C-08 | P2-*, P3-*, P4-04 | INV-1, INV-7, SG-08 | identical state stream offline; ban list clean |
| Q2, Q26, DR-009 | Detection ≠ hazard | C-03, C-07 | P3-03 | T-H, rule trace | 100% decisions traceable |
| Q21, DR-010 | 7 components in order | C-02…C-08 | P2-01…P3-04 | pipeline contract test | stage order asserted |
| Q22, Q79, DR-011 | SSD-MobileNet first benchmark | C-03 | P2-02, P2-09/10 | T-D, D7.9 | M-SSD report exists first |
| Q23, DR-012, DR-041 | Fair comparison, safety-first selection | C-27 | P6-01, P6-04 | `test_decision_matrix` | hard-filter rejection proven |
| Q24, DR-018 | Four hazard categories | C-07 | P3-03 | T-H-10 | categories per D3.4 |
| Q28–29 | Conservative path claims | C-05, C-07 | P3-01, P3-03 | T-H, T-S | `confirmed` needs 3 hits + distance |
| Q30–32, DR-021 | Unknown distance explicit, never fabricated | C-05, C-12 | P3-01, P4-06 | T-S flags, INV-4 | no metres when null/unvalidated |
| Q33–35 | Tracking through misses; camera-motion handling | C-04, C-05 | P2-03, P3-01 | T-T, T-S | 4-miss survival; ego flag |
| Q36–38 | Versioned result contract; stale rejection | schemas, C-09 | P1-06, P3-04 | schema tests, T-C-06 | round-trip; stale dropped |
| Q46–47, DR-019, DR-023 | Urgency 0–3 by interpretable rules | C-07 | P3-03 | T-H-01…07 | rule table tests |
| Q48–52 | Separate uncertainties; qualified alerts | C-06, C-07 | P3-02 | T-U, INV-4 | qualified wording |
| Q53–54, DR-020/022 | No silent failure; four states | C-08, C-12 | P3-04, P4-09 | INV-5/6, SG-06 | 0 silent failures |
| Q57–60, DR-025 | SG-01…08 with evidence | C-27/28 | P3-05, P6-05, P6-07 | gate computation tests | cells computed; release blocked if pending |
| Q63–74, DR-014, DR-027 | FVP user-provided; cannot alter safety | C-10, C-11 | P4-07, P4-08 | INV-3, T-P-01 | 1,000-profile invariance |
| Q76, DR-013 | Condition-independent perception | C-03 | P2-02 | INV-8 | no condition input in graph |
| Q86–94 | Mobility data separate from public; no leakage; simulations separate | C-27 | P2-05/06/08 | `gate_pretrain` | no overlap; legacy excluded |
| Q95–100 | Multi-metric eval; regression | C-27 | P3-05, P6-01 | eval tests | reports with CIs |
| Q101–116, DR-006, DR-015 | Stage B required, bounded, non-overriding | C-13…C-14, C-20…C-23 | P5-* | T-B, INV-1/2 | deadline 4 s; override impossible |
| Q104, Q201–203, DR-016, DR-201 | Egress per approved policy | C-17, C-21 | P5-02, P5-05 | T-B-06 (image field rejected) | matrix enforced |
| Q117–130, DR-028/029 | Deterministic policy; event lifecycle; dismissal semantics | C-11, C-08 | P4-08 | T-P, T-C | state and dismissal tests |
| Q131–148, DR-030 | Native TTS, haptics, preemption, overload control | C-12 | P4-06 | T-O, SG-07 | preemption 100% |
| Q149–167, DR-001, DR-031/032 | iOS runtime, states, startup | C-01, C-16, app | P4-* | T-C status, T-M | device tests pass |
| Q168–183, DR-033/034 | Signed releases, LKG, rollback | C-16, C-28 | P4-05, P5-06 | T-M | tamper → LKG/UNAVAILABLE |
| Q184–200, DR-040 | Cloud controls, auth without embedded creds | C-20…C-26 | P5-01/02 | T-B-02/03/08 | quotas, alarms, Cognito |
| Q204–222, DR-044/045 | Privacy, consent, retention | C-17, C-19 | P4-07, P7-02 | consent tests | review passed |
| Q223–243 | Test matrix, events not frames, release evidence | C-27/28 | P3-05, P6-05, P7-04 | D8 | evidence retained per release |
| Q244–260, DR-038/039 | Claims, pilot prerequisites | docs, onboarding | P7-01, P6-08, P7-05 | claims review | approvals |
| Q261–280 | Governance, CI, ADRs, DoD | repo | P1-04/05, P7-04 | CI rules | D8.6 met |
