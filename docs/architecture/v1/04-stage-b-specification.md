<!-- Part of the MaxSight V1 Technical Architecture Specification. See 00-overview.md. Source: approved plan 2026-10-31. -->

# D4: Stage B Technical Specification

**Status:** required for the complete V1 (DR-006). Model choice and AWS inventory are Open (DR-208/213). Visual egress is Open (DR-201), so every task below lists its egress dependency.

### D4.0 Capability matrix: V1 product scope follows the approved data policy

| # | Capability | Category | Works under egress policy v1 (no images)? | V1 product commitment |
|---|---|---|---|---|
| 1 | Read visible text aloud (on-device OCR → speech) | **(1) Fully on device** | Yes | **Committed** (works offline) |
| 2 | Interpret/summarise text (sign type, menu summary) | **(2) Derived text** (C2 consent) | Yes, with C2 | **Committed** |
| 3 | Find a requested object among detector classes, with direction | (1) on device, (2) synonym mapping in the cloud | Yes | **Committed** for the 17 detector classes only |
| 4 | Scene summary of detected objects and layout | **(2) Structured metadata** (C1) | Yes | **Committed**, limited to detected classes (stated in onboarding) |
| 5 | Contextual Q&A about detected objects and read text | (2) | Yes | **Committed**; unsupported questions get an explicit refusal |
| 6 | Knowledge retrieval from approved sources (text query) | (2) | Yes | **Committed** after corpus approval |
| 7 | Find objects outside the detector taxonomy (open vocabulary) | **(3) Requires authorised image transmission** | No | **Not in V1** unless DR-201 approves C4 |
| 8 | Free-form visual description of unrecognised things, colours, layouts | (3) | No | **Not in V1** unless DR-201 |
| 9 | Pixel-level visual Q&A ("what does this picture show?") | (3) | No | **Not in V1** unless DR-201 |
| 10 | Persistent environmental memory, conversational agent, navigation routing | **(4) Excluded from V1** | — | Excluded (Q102) |

**Advertising rule:** the App Store description, onboarding and help text list only the **Committed** rows. If DR-201 later approves C4, rows 7–9 move to Committed with a per-request authorisation prompt, and this matrix is updated through an ADR.

### D4.1 Task catalogue

| Task ID | Trigger | Permitted input (default, DR-201 unresolved) | Processing | Output | Timeout | Failure fallback | Privacy class | Cost per task (est.) | Evaluation |
|---|---|---|---|---|---|---|---|---|---|
| SB-OCR | "Read this" (voice/button) | **Derived text** from on-device Vision OCR (≤ 2,000 chars) + layout boxes | Bedrock: clean up, order and summarise text; detect sign type | `{text_read, summary, confidence}` | 4 s client (3.5 s server) | Speak the raw on-device OCR text locally; "Couldn't interpret the text" | C2 derived text (consent) | ~1.0k in / 200 out tokens + Lambda ≈ (in·p_in + out·p_out) + $0.0000004 per invoke (approx.); priced once the model is chosen | WER of spoken output vs ground truth; task success; latency p95 |
| SB-FIND | "Where is my <object>?" | Structured: Stage A detections (class, sector, distance bucket) + optional OCR text | Local first: match class in current detections → direction. Cloud only for synonym mapping (text) | `{found: bool, direction, distance_bucket\|null, guidance}` | 4 s | Local answer only ("I can't see a <object>") | C1 structured metadata | ~300 in / 60 out tokens | Find success on scripted tasks; false "found" rate |
| SB-SCENE | "What's around me?" | Structured scene graph from Stage A (classes, sectors, distances if valid, counts) | Bedrock: concise description constrained to the provided objects (no invention) | `{description, objects_mentioned[]}` | 4 s | Template description from local state | C1 | ~800 in / 120 out tokens | Groundedness: 100% of mentioned objects present in input; human usefulness rating |
| SB-VQA | Free question | Request text + structured scene + OCR text | Bedrock answer constrained to inputs; must answer "I can't tell from what I can detect" when unsupported | `{answer, supported: bool}` | 4 s | "I can't answer that right now" | C1/C2 | ~1.2k in / 150 out tokens | Unsupported-answer rate; refusal correctness |
| SB-KNOW | Question needing facts (e.g. "What does this sign symbol mean?") | Request text (+ OCR text) | C-23 retrieval over an **approved corpus** (licence-registered) + Bedrock answer with citations | `{answer, sources[]}` | 4 s | "I don't have information on that" | C1/C2 | ~1.5k in / 200 out tokens + retrieval | Citation correctness; no-source handling |
| SB-VIS (blocked) | Open-vocabulary visual description or visual Q&A on pixels | **Image crop/frame**, only if DR-201 approves class C4 | Bedrock multimodal | as SB-SCENE/VQA | 4 s | Fall back to SB-SCENE | C4 image (explicit per-request consent) | image tokens + text | as above + hallucination audit |

**Dependency statement:**
- Cloud OCR **on images**, open-vocabulary object finding and pixel-level visual Q&A are **unavailable** unless DR-201 approves image egress.
- Without that approval, V1 Stage B delivers OCR *interpretation* (on-device extraction), grounded scene description, metadata-based finding and Q&A, and knowledge retrieval.

### D4.2 API contract (`/v1`, OpenAPI file `docs/contracts/openapi/stageb.v1.yaml` [P])

**`POST /v1/tasks`**
- **Headers:** `Authorization: Bearer <Cognito JWT>`, `Idempotency-Key: <request_id uuid>`, `X-Schema-Version: maxsight.stageb/1.0`.
- **Request:**

  ```json
  {"request_id":"uuid","session_id":"uuid","task_type":"ocr|find|scene|vqa|know",
   "client_ts":"ISO-8601","frame_ref":{"frame_id":123,"t_mono_ns":0},
   "deadline_ms":4000,"inputs":{"text":"…","ocr":{"text":"…","lines":[…]},"scene":{"objects":[{"class":"door","sector":"left","distance_bucket":"1-2m|null","urgency":0}]}},
   "fvp_hints":{"verbosity":"brief|standard|detailed","language":"en-US"},"app_version":"1.0.0","model_version":"stagea-…"}
  ```

- **Response 200:**

  ```json
  {"request_id":"uuid","task_type":"…","status":"ok|refused|timeout|error","result":{…},"supported":true,
   "uncertainty":"low|medium|high","server_ts":"…","frame_ref":{…},"service_model":"bedrock:<model-id>","schema_version":"maxsight.stageb/1.0"}
  ```

- **Errors:**

  | Code | Meaning |
  |---|---|
  | 400 | schema invalid |
  | 401/403 | auth |
  | 409 | duplicate `request_id` with a different body |
  | 413 | input too large |
  | 429 | quota/throttle (`Retry-After`) |
  | 503 | upstream unavailable |
  | 504 | deadline |

  The body is always `{request_id, status, error_code, message}`.

**`GET /v1/quota`** returns `{daily_limit, used, resets_at}`.

**Validation:** server-side JSON Schema; unknown fields are rejected. The `inputs` allow-list is enforced against `egress_policy_version`, and any image field is rejected until DR-201.

### D4.3 Coordination semantics

| Aspect | Rule |
|---|---|
| **Idempotency** | DynamoDB `idempotency` table (PK `request_id`, TTL 24 h); a repeat with the same body returns the cached response; a different body → 409 |
| **Timeouts** | Client hard deadline 4.0 s (DR-040). Server budget 3.5 s: Bedrock call ≤ 3.0 s, retrieval ≤ 0.5 s |
| **Retries** | **No automatic client retry** for user tasks (Q14.3 rule: no repeat without user permission). Server: one retry on Bedrock throttling only if ≥ 1.0 s budget remains |
| **Rate limits** | API Gateway 2 req/s per user (burst 4); per-user daily quota *(proposed: 200 tasks/day)*, stored in DynamoDB `quota` (PK `user_id#date`) |
| **Stale rejection (client)** | Discard a response when `now − client_ts > deadline_ms`, or a newer request of the same `task_type` was issued, or `session_id` changed |
| **Versioning** | URL `/v1`; `X-Schema-Version`; the server supports N and N−1 |
| **Observability** | Structured JSON logs (`request_id`, `task_type`, `status`, `latency_ms`, `model_id`, `tokens_in/out`, `quota_remaining`), **no input text logged by default** (sampled redacted logging only with consent). EMF metrics `StageB_P95_Latency`, `StageB_Error_Rate`, `Bedrock_Invocation_Count`, `Quota_Rejections`. ADOT tracing on the Lambda only |
| **Cost controls** | Per-user quota; Bedrock IAM restricted to one model ARN; `max_tokens` per task (out ≤ 300); AWS Budget alarms at 50/80/100% of the monthly limit (DR-208); kill switch: SSM parameter `stageb_enabled=false` → 503 + client message |
| **Auth** | Cognito user pool (email or Sign in with Apple), access token 15 min, refresh in Keychain; **no AWS credentials in the app** (DR-040); IAM role per Lambda, least privilege |

### D4.4 Stage B failure modes

| Case | Client behaviour | Server behaviour |
|---|---|---|
| Offline | Speak `STAGE_B_OFFLINE_MESSAGE` (`[E] app/stage_b/messages.py:4`); local fallback where defined | — |
| Timeout | "That request timed out" + local fallback | Return 504, log |
| Quota | Spoken limit message with reset time | 429 |
| Upstream error | Generic failure message | 503, alarm on error rate > 5% over 5 min |
| Unsupported question | Speak the refusal text | `supported=false` |

### D4.5 Stage B evaluation

| Task | Dataset | Metrics | Acceptance |
|---|---|---|---|
| SB-OCR | 200 sign/label photos with transcripts | WER, task success, p95 latency | thresholds Open |
| SB-SCENE / SB-VQA | 150 scene graphs with gold descriptions and answerable/unanswerable questions | groundedness (hallucinated objects = 0 target), refusal correctness | Open |
| SB-KNOW | 100 questions with known sources | citation precision | Open |
| SB-FIND | 60 scripted find tasks | success, false-found rate | Open |

### D4.6 Data-egress matrix (default policy `egress_policy_version = 1`)

| Data type | Collected? | Stored locally | Leaves device | Retention | Notes |
|---|---|---|---|---|---|
| Raw video | Frames in RAM only | No | **Never** (DR-016) | frame lifetime | — |
| Individual frames | RAM | No | **No** unless DR-201 approves class C4 with per-request authorisation | — | Blocks SB-VIS |
| Image crops | RAM | No | **No** unless DR-201 approves C4 | — | — |
| Thumbnails | Not generated | No | No | — | — |
| Visual embeddings | Not generated for egress | No | **No** unless DR-201 | — | — |
| Structured detections (class, sector, distance bucket, urgency) | Yes | session RAM | **Yes (C1)** on a user-initiated Stage B request | server: not stored (logs exclude inputs) | allow-listed fields only |
| Hazard metadata | Yes | session RAM | C1 only inside a user request; research telemetry only with C3 consent | — | — |
| OCR-derived text | Yes (on device) | RAM | **C2:** with one-time consent "send recognised text for interpretation" | not stored server-side | — |
| Request audio | RAM | No | **Never** (on-device ASR) | — | — |
| Request text | Yes | session RAM | C1 with the request | not stored | — |
| FVP | Yes | Yes (file protection) | Only with sync consent | until deletion | — |
| Session events | Yes | 30 days (DR-117) | Only with sync consent, idempotent | per policy | — |
| Research telemetry | Off | — | Only with C3 opt-in | per research protocol | Withdrawal → deletion job |
| Crash diagnostics | OS | OS | Opt-in (Apple) | OS | — |

**Consent classes:**

| Class | Meaning |
|---|---|
| C1 | Structured metadata; part of using Stage B; disclosed in onboarding |
| C2 | Derived text; one-time explicit opt-in |
| C3 | Research telemetry; explicit opt-in |
| C4 | Images; **blocked pending DR-201** |
