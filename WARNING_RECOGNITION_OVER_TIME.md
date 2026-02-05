# Helping the User Over Time Recognize Warnings

**In simple terms:** The system is designed so that users can **learn what each warning means** and react faster over time—instead of relying on the same verbal descriptions forever.

---

## How It Works (Three Parts)

### 1. **Consistent cues (same hazard → same cue)**

- Each **type of hazard** and **urgency level** is mapped to a **stable** sound or haptic pattern.
- Examples:
  - **Stairs** → always the same tone or vibration pattern
  - **Vehicle approaching** → a different, but always the same, pattern
  - **Danger** (urgency 3) → faster, stronger pattern than **caution** (urgency 1)
- Because the mapping doesn’t change, the user can learn: *“That sound = stairs”* or *“That pattern = danger.”*

**Where in the code:**  
Output scheduling uses **urgency** and **priority** to set intensity, frequency, and duration of alerts (`ml/utils/output_scheduler.py`). The app should use a **fixed mapping** from (hazard type, urgency) → (audio pattern / haptic pattern) so the same situation always produces the same cue.

---

### 2. **Adaptive alert level (more alerts when learning, fewer when skilled)**

- The system tracks **hazard awareness**: how well the user notices and responds to hazards (e.g. from therapy or usage).
- **When the user is still learning:**
  - **More alerts:** caution, warning, and danger are all announced (so they hear and learn the cues).
- **When the user is skilled:**
  - **Fewer alerts:** only higher-urgency hazards (e.g. warning and danger) trigger alerts, so they aren’t overwhelmed and can rely on the cues they’ve learned.

So over time, the user needs **less verbal repetition** and can rely **more on the consistent cues** they’ve learned.

**Where in the code:**  
`ml/utils/adaptive_assistance.py`  
- `hazard_awareness` (0–1): higher = user is better at recognizing hazards.  
- `get_adaptive_hazard_threshold()`: returns which urgency levels to alert on (0 = all, 1 = caution+, 2 = warning and danger only).  
- When **hazard_awareness** is high → threshold goes up → only high-urgency alerts.  
- When **hazard_awareness** is low → threshold stays low → all levels announced so the user can learn.

---

### 3. **Warning recognition practice (optional drills)**

- We can run **warning recognition** tasks: e.g. “This is the cue for **stairs**. When you hear/feel this in real use, it means stairs are present.”
- The user hears or feels the cue, sees or is told the hazard name, and over time builds a strong link between **cue** and **meaning**.
- This is implemented as a therapy task type so it can be used in the same flow as other skill-building exercises.

**Where in the code:**  
`ml/therapy/therapy_integration.py`  
- `TherapyTaskType.WARNING_RECOGNITION`  
- `create_warning_recognition_task(hazard_type, urgency_level, cue_description, difficulty)`  
- Used in `generate_task_from_scene()` when the task type is warning recognition.

---

## Summary Table

| Goal | How we support it |
|------|--------------------|
| **Same hazard → same cue** | Output scheduler uses urgency/priority; app should map (hazard, urgency) to a fixed audio/haptic pattern. |
| **More cues when learning** | Low hazard_awareness → low hazard threshold → alert on more levels (caution, warning, danger). |
| **Fewer cues when skilled** | High hazard_awareness → high hazard threshold → alert only on warning/danger. |
| **Practice recognizing cues** | Therapy task type `WARNING_RECOGNITION` and `create_warning_recognition_task()` for drills. |

---

## For product / app integration

1. **Define a stable cue map**  
   For each (hazard_type, urgency_level), choose one audio pattern and one haptic pattern and use it everywhere (e.g. in `CrossModalScheduler` or the app layer).

2. **Feed hazard_awareness into the scheduler**  
   Call `AdaptiveAssistance.get_adaptive_hazard_threshold()` and only schedule alerts for detections with urgency ≥ that threshold.

3. **Offer optional “Learn this warning” drills**  
   Use `TherapyTaskIntegrator.create_warning_recognition_task(...)` (and optionally `generate_task_from_scene` with `TherapyTaskType.WARNING_RECOGNITION`) to run short exercises that play the cue and say what it means.

Together, this is how the system **helps the user over time recognize warnings**: consistent cues, adaptive alert level, and optional recognition practice.
