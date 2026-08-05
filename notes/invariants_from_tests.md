# Invariants from tests (Block 3.3–3.7)

One line per test function from the four named suites, then a compiled CI-facing list.

---

## 3.3 `tests/test_runtime_safety_gates.py`

### TestRuntimeConstants
- `test_critical_urgency_threshold` — CRITICAL_URGENCY_THRESHOLD is exactly 3.
- `test_latency_budgets` — LATENCY_MEDIAN_MS and LATENCY_P95_MS are both ≤ 80 ms.
- `test_alerts_per_minute_cap` — ALERTS_PER_MINUTE_CAP is exactly 12.
- `test_safety_gate_thresholds` — HAZARD_RECALL_MIN ≥ 0.95, FALSE_SAFE_RATE_MAX ≤ 0.01, DIRECTION_CORRECTNESS_MIN ≥ 0.90, DISTANCE_ZONE_ACCURACY_MIN ≥ 0.85.
- `test_check_safety_gate_report_pass` — metrics meeting thresholds yield pass with empty failure list.
- `test_check_safety_gate_report_fail` — low hazard recall / high false-safe fail report and include SG-01 and SG-02.

### TestSchedulerCriticalPath
- `test_critical_detections_always_included_under_uncertainty` — urgency ≥ CRITICAL_URGENCY_THRESHOLD still produces ≥1 scheduled output under high uncertainty (SG-07).
- `test_scheduler_uses_runtime_constants` — CrossModalScheduler.min_channel_interval equals MIN_CHANNEL_INTERVAL_S.

### TestMvpRuntimeContract
- `test_filter_mvp_keeps_only_allowed_keys` — ModelOutputContract.filter(training=False) drops non-MVP keys (e.g. scene_embedding/description) and keeps only allowed_keys.
- `test_filter_mvp_passes_through_when_training` — filter(training=True) returns the full outputs unchanged.

---

## 3.4 `tests/test_timing_enforcement.py`

- `test_timing_import` — `time` is available to the MaxSightCNN module for latency measurement.
- `test_timing_flag` — `_enable_timing` can be set on the model.
- `test_timing_tracking` — with timing enabled, forward exposes stage completion fields (and optionally stage_a_latency_ms).
- `test_timing_enforcement` — forward always exposes `stage_a_completed`, `stage_b_completed`, and `skip_stage_b_reason` for Stage B skip control.
- `test_timing_disabled` — with timing disabled, inference still returns stage completion fields.
- `test_actual_latency` — with timing enabled, stage_a_latency_ms can be measured over repeated forwards (soft check vs 200 ms hard limit).

---

## 3.5 `tests/test_production_rag_and_therapy_contracts.py`

- `test_rag_retrieval_is_advisory_only_in_forward_outputs` — with retrieval + mvp_runtime, forward outputs must not contain retrieval_results / retrieval / distances / indices.
- `test_rag_async_retrieve_non_blocking_contract` — AsyncRetrievalSystem.retrieve(..., blocking=False) returns immediately (None) even if stage1 ANN is missing.
- `test_therapy_is_independent_of_retrieval_keys` — TherapyEngine decisions ignore injected retrieval_results keys when perception signals are unchanged.
- `test_forward_triggers_retrieval_with_blocking_false` — enabled retrieval is invoked with blocking=False and still does not pollute tensor outputs (no retrieval artefacts / scene_description).

---

## 3.6 `tests/test_integration_constraints.py`

- `test_audio_attention_preserves_channels` — audio spatial attention fusion must preserve feature channel count (256).
- `test_depth_uncertainty_encapsulated` — DepthHead emits uncertainty with matching spatial shape via an uncertainty_conv module.
- `test_temporal_transformer_exports_timesformer` — TimeSformer is importable from ml.models.temporal.temporal_transformer as an nn.Module.
- `test_temporal_spatial_alignment` — TemporalEncoder motion_features spatial HxW matches Stage A feature map resolution.
- `test_scene_graph_top_k` — scene-graph object selection is capped by max_scene_graph_objects (top-K, not full H×W).
- `test_personalization_normalized` — user embedding vectors are L2-normalized.
- `test_depth_vectorized` — depth-at-box sampling is batched via grid_sample (no per-box Python loops required).

---

## 3.7 Invariants this repo already enforces in CI today

Compiled (deduplicated) product/architecture invariants covered by the suites above when those tests run in CI:

1. **Safety gate numeric floors/ceilings** — hazard recall ≥ 0.95, false-safe ≤ 0.01, direction ≥ 0.90, distance-zone ≥ 0.85; report helper encodes SG-01/SG-02 failures.
2. **Latency budget constants** — published median/p95 budgets ≤ 80 ms; alert rate cap = 12/min; critical urgency threshold = 3.
3. **Critical alerts cannot be silenced by uncertainty** — SG-07: high-urgency detections still schedule output under high model uncertainty.
4. **Scheduler channel pacing** — min channel interval comes from runtime constants.
5. **MVP runtime output surface is closed** — inference filter keeps only allowed keys; training mode may pass full tensors.
6. **Two-stage timing contract** — Stage A/B completion and skip_stage_b_reason are part of the forward contract; Stage B can be skipped on time budget.
7. **RAG is advisory-only on the tensor path** — forward must not export retrieval artefacts; async retrieve is non-blocking.
8. **Therapy is retrieval-key independent** — therapy actions do not change solely because retrieval_results were injected.
9. **Forward retrieval is non-blocking** — when enabled, retrieval is requested with `blocking=False`.
10. **Multimodal fusion must not reshape feature channels** — audio attention preserves C.
11. **Depth uncertainty is first-class head output** — same spatial shape as depth_map.
12. **Temporal / scene-graph / personalization structural bounds** — TimeSformer public path exists; temporal motion maps align spatially; scene graph is top-K capped; user embeddings normalized; depth sampling is vectorized.

*Note:* Collect-only shows 611 tests repo-wide; this list is limited to the four files named in Block 3 (plus the structural invariants they assert). Broader CI (gold, run_config, Stage A, flake8, pyright) adds more contracts outside this note.
