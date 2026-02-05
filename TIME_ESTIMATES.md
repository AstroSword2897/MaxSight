# Time Estimates: Training, Validation, Testing, Inference

Rough estimates for **14 condition models**, **batch size 16**, **20 epochs** each, and for validation/testing/inference. Times depend on dataset size and GPU (Colab T4 vs Pro A100).

---

## Assumptions

| Setting | Value |
|--------|--------|
| Models | 14 (one per vision condition) |
| Epochs per model | 20 |
| Batch size | 16 |
| Grad accumulation | 2 (effective batch 32) |
| Validation | Every epoch (included in training time below) |
| GPU | Colab free **T4** (16 GB) or Colab Pro **A100** (40 GB) |

---

## 1. Training (one model)

Time per epoch depends on **number of training samples**.

| Train samples | Batches/epoch (÷16) | Per epoch (T4) | Per epoch (A100) | 20 epochs (T4) | 20 epochs (A100) |
|---------------|----------------------|----------------|------------------|----------------|------------------|
| ~2,000        | 125                  | ~2–3 min       | ~0.5–1 min       | **~45–60 min** | **~15–20 min**   |
| ~10,000       | 625                  | ~12–18 min     | ~3–5 min         | **~4–6 h**     | **~1–1.5 h**     |
| ~50,000       | 3,125                | ~60–90 min     | ~15–25 min       | **~20–30 h**   | **~5–8 h**       |
| ~118,000 (full COCO train) | ~7,375 | ~2.5–4 h   | ~40–70 min       | **~50–80 h**   | **~13–23 h**     |

- **T4**: ~1–2 sec per batch (T5, batch 16, FP32).
- **A100**: ~0.2–0.5 sec per batch.
- Validation each epoch adds typically **5–15%** (already included in the ranges above).

---

## 2. Training all 14 conditions

Run **14 separate jobs** (one per condition), one after the other.

| Train samples | 14 models on T4   | 14 models on A100 |
|---------------|-------------------|-------------------|
| ~2,000        | **~10–14 h**      | **~3.5–5 h**      |
| ~10,000       | **~2–3.5 days**   | **~14–21 h**      |
| ~50,000       | **~12–18 days**   | **~3–4 days**     |
| ~118,000      | **~29–47 days**   | **~7.5–13 days** |

Notes:

- Colab free often **disconnects after ~12 h**; for long runs use **checkpointing** and **resume** (e.g. `--resume-from` for each `checkpoints_{cond}`), or run one condition per session.
- Pro/Pro+ with A100 and longer sessions can do 2–4 conditions per run for medium-sized data.

---

## 3. Validation (during vs after training)

- **During training**: One validation pass per epoch (same data loader as training). Already counted in the “per epoch” and “20 epochs” times above. Typically **1–5 min per epoch** on T4 for small/medium val sets.
- **After training** (single full validation run per model):
  - Val size ~1,000–5,000: **~2–10 min per model** on T4.
  - For 14 models: **~30 min–2.5 h** total (T4).

So: validation time is either **included** (during training) or **~30 min–2.5 h** for a one-off full validation of all 14 models.

---

## 4. Testing (full test set, one pass per model)

- One pass over test set, no backward pass. Similar to validation, slightly longer if test set is bigger.
- **Per model** (test set ~2,000–10,000 images): **~5–20 min** on T4.
- **14 models**: **~1–5 h** on T4 (sequential).

---

## 5. Inference (single image or real-time)

| Scenario        | T4 (FP32) | A100 (FP32) | Notes                    |
|-----------------|-----------|-------------|--------------------------|
| Single image   | ~30–80 ms | ~10–30 ms   | One forward pass         |
| Batch 16        | ~0.2–0.5 s| ~0.05–0.15 s| Throughput ~30–80 img/s  |
| “Real-time” 10 fps | ~100 ms/image | OK | Camera at 10 fps needs &lt;100 ms/frame |

So: **training** and **validation** dominate; **inference** is sub-second per image once the model is loaded.

---

## 6. Summary table (your setup: 14 conditions, batch 16, 20 epochs)

| Phase        | What it is                          | Typical time (ballpark)                    |
|--------------|--------------------------------------|--------------------------------------------|
| **Training** | 14 models × 20 epochs each           | **~10 h – 47 days** (see table above)       |
| **Validation** | During training (included) or one full val run | **Included** or **~30 min–2.5 h** (all 14) |
| **Testing**  | One test pass per model (14 total)   | **~1–5 h** (all 14, T4)                    |
| **Inference**| Per image after load                | **~30–80 ms** (T4)                         |

---

## 7. Practical recommendations

1. **Check your dataset size**  
   Look at `Train samples: …` and `Val samples: …` in the first training log to see which row of the tables applies.

2. **Colab free (T4)**  
   - Prefer **&lt; ~10k train samples** per condition so one model fits in one session (e.g. **&lt; ~6 h** per model).  
   - For 14 conditions, run **one or a few conditions per session**, save checkpoints to Drive, resume next time.

3. **Colab Pro (A100)**  
   - Can do **~10k samples × 20 epochs** in **~1–1.5 h** per model; **14 models ≈ 14–21 h** (e.g. 2–3 long sessions with resume).

4. **Resume**  
   Use the same `--checkpoint-dir` and add:
   ```bash
   --resume-from /content/drive/MyDrive/MaxSight/checkpoints_{cond}/last_checkpoint.pt
   ```
   when re-running a condition so you don’t redo finished epochs.

5. **Testing**  
   Run a single test pass per saved model when training for a condition is done; budget **~5–20 min per model** (T4).

6. **Inference**  
   After loading a checkpoint, expect **~30–80 ms per image** on T4; fine for real-time use at moderate frame rates.

---

## 8. Quick reference (medium dataset: ~10k train)

| Activity              | Time (single model) | Time (14 models)   |
|-----------------------|---------------------|--------------------|
| Training (T4)         | ~4–6 h              | ~2–3.5 days*       |
| Training (A100)       | ~1–1.5 h            | ~14–21 h*           |
| Validation (one full run) | ~2–5 min        | ~30 min–1 h        |
| Testing (one full run)| ~5–15 min           | ~1–3.5 h           |
| Inference (per image) | ~30–80 ms           | —                  |

\* Sequential; use resume if Colab disconnects.
