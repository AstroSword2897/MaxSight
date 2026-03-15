# MaxSight Therapy System Architecture

The therapy subsystem is a **closed-loop behavioral feedback system**, not just another neural network. It is a **decision + adaptation** layer on top of perception and context.

## Pipeline

```
Perception (Vision / Audio / Context)
        ↓
Situation Understanding
        ↓
Therapy Decision Engine
        ↓
Intervention Generator
        ↓
Output Scheduler (Audio / Haptics)
        ↓
User Response
        ↓
Response Evaluation
        ↓
Adaptation Engine
        ↓
Therapy Memory
```

This forms a **continuous feedback loop**.

---

## 1. Situation Understanding Layer

**Module:** `ml/therapy/situation_understanding.py`

Converts raw perception outputs into **psychological context**.

**Inputs (from perception stack):** object detection, navigation risk, crowd density, motion, audio environment, uncertainty.

**Derived features:** `environment_stress_level`, `cognitive_load_estimate`, `task_difficulty`, `uncertainty`, `user_motion_state`, `crowd_density`, `noise_level`, `navigation_complexity`.

**Implementation:** Deterministic (no NN). Feeds the therapy decision engine.

---

## 2. Therapy Decision Engine

**Module:** `ml/therapy/therapy_decision_engine.py`

Core brain: determines **should we intervene?**, **what intervention?**, **how strong?**

**Architecture:** Rule + ML hybrid (rules first; policy/bandit can be added later).

- Rule layer: safety constraints (e.g. do not override hazard alerts).
- Policy: high stress → calming/grounding; navigation complexity → reassurance; high cognitive load → attention redirection.
- Uses **AdaptationEngine** to prefer intervention types that have worked before.

**Output:** `TherapyDecision(should_intervene, intervention_type, strength, reason)`.

---

## 3. Intervention Generator

**Module:** `ml/therapy/intervention_generator.py`

Converts decisions into **specific therapeutic actions**.

**Intervention types:** grounding prompts, navigation reassurance, breathing guidance, cognitive reframing, attention redirection, calming prompt, rest suggestion.

**Outputs:** `TherapeuticAction` with channel (audio/haptic/visual), content text, intensity, duration, priority. Content is rule-based (no medical claims).

---

## 4. Output Scheduler

**Module:** `ml/utils/output_scheduler.py` (existing)

Therapy output is **merged with hazard/navigation alerts** here. Therapy must not overload the user.

**Therapy-specific limits (in `ml/runtime_constants.py`):**

- `THERAPY_MAX_PROMPTS_PER_MINUTE = 2`
- `THERAPY_MIN_GAP_BETWEEN_PROMPTS_S = 10.0`
- `THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD = 0.7` (suppress therapy when perception uncertainty is high)

---

## 5. User Response Monitoring

The system infers response from **next perception snapshot**: e.g. stress level drops, movement stabilizes. No separate “user response” sensor required; we use **before_state** (context when we intervened) vs **after_state** (context after a short delay).

---

## 6. Response Evaluation Model

**Module:** `ml/therapy/response_evaluation.py`

**Inputs:** `before_state`, `intervention_type`, `after_state`.

**Output:** `effectiveness_score` (0–1), `stress_reduction`, `reason`.

**Implementation:** Lightweight rule-based (stress delta). Can be replaced by a small MLP later.

---

## 7. Adaptation Engine

**Module:** `ml/therapy/adaptation_engine.py`

Personalizes therapy from outcomes: which prompts work, which fail, preferred channel (audio/haptic/visual). Updates **TherapyMemorySystem** (long-term success rates, preferred channel). Decision engine can then prefer high-success intervention types.

---

## 8. Therapy Memory System

**Module:** `ml/therapy/therapy_memory.py`

**Short-term:** last intervention, recent stress levels, recent prompts (for cooldowns and evaluation).

**Long-term:** trigger patterns, successful interventions (per type), therapy preferences, user tolerance level.

---

## 9. Safety Layer

**Module:** `ml/therapy/therapy_safety.py`

- **Max intervention rate:** enforced via `THERAPY_MAX_PROMPTS_PER_MINUTE` and `THERAPY_MIN_GAP_BETWEEN_PROMPTS_S`.
- **Suppress when uncertainty high:** `THERAPY_UNCERTAINTY_SUPPRESS_THRESHOLD`.
- **No medical claims:** `sanitize_content()` strips diagnostic/prescriptive language.
- **Fail-safe:** when in doubt, do not prompt.

---

## 10. Integration: TherapyEngine

**Module:** `ml/therapy/therapy_engine.py`

Single entry point:

1. **`update(perception, current_time)`**  
   Perception → situation context → decision → safety check → intervention generation. Returns a list of `TherapeuticAction` for the output manager to schedule (audio/haptic).

2. **`on_user_response(perception_after, current_time)`**  
   Call after the user has had time to respond. Runs response evaluation and updates adaptation and memory.

**Usage:** App or simulator calls `therapy_engine.update(perception)` each frame (or every N frames). Deliver returned actions via the existing output scheduler. After a short delay, call `therapy_engine.on_user_response(perception_after)` so the loop can learn.

---

## Design Principles

Therapy is:

- **Adaptive** — learns which interventions work for this user.
- **Non-intrusive** — rate limits and cooldowns; never overrides safety alerts.
- **Context-aware** — driven by situation understanding, not raw detections only.
- **Safety-constrained** — uncertainty suppress, no medical language, fail-safe silence.

This keeps therapy **separate from perception** and makes it a true **behavioral feedback control** layer.
