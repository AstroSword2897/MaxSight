# Google Colab Setup Guide - MaxSight Training

**Goal:** Train MaxSight model on free GPU in ~90 minutes (vs 23 hours on CPU)

---

## 📦 Step 1: Prepare Your Code (On Mac)

### Option A: Small Package (Recommended if you have good internet)
```bash
cd /Users/nani/2026-Prototype

# Package just the code and annotations (no images)
tar -czf maxsight_code.tar.gz \
  ml/ \
  scripts/ \
  datasets/cleaned_splits/ \
  requirements_colab.txt \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  --exclude="*.pth"

# Size: ~50-100 MB
ls -lh maxsight_code.tar.gz
```

**Then:** Colab will download COCO images directly (~20 min, but only once)

### Option B: Full Package (If you want everything)
```bash
cd /Users/nani/2026-Prototype

# Package everything including COCO images
tar -czf maxsight_full.tar.gz \
  ml/ \
  scripts/ \
  datasets/ \
  requirements_colab.txt \
  --exclude="*.pyc" \
  --exclude="__pycache__" \
  --exclude="*.pth" \
  --exclude="checkpoints/"

# Size: ~10-15 GB (includes all images)
ls -lh maxsight_full.tar.gz
```

**Choose based on your upload speed:**
- Fast internet (>50 Mbps): Use Option A
- Slow internet: Use Option A anyway (Colab downloads images faster from COCO)

---

## ☁️ Step 2: Upload to Google Drive

1. **Open Google Drive** in browser
2. **Create folder:** `MaxSight/`
3. **Upload** your `maxsight_code.tar.gz` (or `maxsight_full.tar.gz`)
4. **Wait** for upload to complete

**Upload time estimate:**
- 100 MB file: ~5-10 minutes on typical home internet
- 15 GB file: ~2-3 hours (not recommended)

---

## 🚀 Step 3: Open Colab Notebook

1. **Go to:** https://colab.research.google.com
2. **Click:** File → Upload notebook
3. **Upload:** `MaxSight_Colab_Training.ipynb` (I created this for you)

**Or create new notebook and copy cells from the .ipynb file**

---

## ⚙️ Step 4: Enable GPU

**CRITICAL:** Must enable GPU!

1. **Click:** Runtime → Change runtime type
2. **Hardware accelerator:** Select **"T4 GPU"** (free tier)
3. **Click:** Save

**Verify GPU:**
```python
!nvidia-smi
```

Should show: `Tesla T4` with 15 GB memory

---

## 🎯 Step 5: Run Training

**Execute cells in order:**

1. ✅ Check GPU (verify T4 is available)
2. ✅ Mount Google Drive (authorize when prompted)
3. ✅ Extract project (update path to your tar.gz)
4. ✅ Install dependencies (~2 min)
5. ✅ Download COCO images (skip if using Option B, ~20 min)
6. ✅ Patch missing images (~2 min)
7. ✅ **Start training** (~90 min for 50 epochs)
8. ✅ Download results

**Total time:** ~2 hours including setup

---

## 📊 What to Expect

### Training Progress
```
Epoch 1/50:   0%|          | 0/2500 [00:00<?, ?it/s]
Epoch 1/50:   4%|▍         | 100/2500 [00:12<04:52, 8.21it/s, loss=3.4, lr=1.0e-06]
Epoch 1/50:  20%|██        | 500/2500 [01:00<04:00, 8.32it/s, loss=3.2, lr=2.5e-06]
Epoch 1/50: 100%|██████████| 2500/2500 [05:00<00:00, 8.33it/s, loss=3.1, lr=5.0e-06]

Epoch 1 Complete - Val Loss: 2.95, mAP: 0.12
```

### Speed Comparison
| Metric | M3 CPU | Colab T4 GPU |
|--------|--------|--------------|
| it/s | 1.5-2.0 | 25-35 |
| Per Epoch | 20-28 min | 1-2 min |
| 50 Epochs | 23 hours | 90 min |

---

## 💾 Step 6: Download Results

**After training completes:**

1. **Cell 8** copies checkpoints to Google Drive
2. **Open Google Drive** in browser
3. **Navigate to:** `maxsight_results/`
4. **Download folder** (or individual checkpoints)
5. **On Mac:** Copy to `/Users/nani/2026-Prototype/checkpoints/`

**Files to download:**
- `best_model.pth` - Best performing model
- `checkpoint_epoch_X.pth` - Periodic checkpoints
- `training_history.json` - Loss curves, metrics

---

## ⚠️ Important Tips

### 1. Colab Session Limits
- **Free tier:** ~12 hours max session
- **Your training:** ~90 minutes (well within limit)
- **Solution if disconnected:** Code auto-saves checkpoints every 5 epochs

### 2. Resuming Training
If session disconnects, restart and run:
```python
# Add --resume-from flag
!python scripts/train_maxsight.py \
  --resume-from checkpoints/checkpoint_epoch_10.pth \
  [... other args ...]
```

### 3. Monitor Progress
**Keep tab open** or use Colab's built-in monitoring:
```python
# In separate cell, run periodically:
!tail -20 logs/*.log
!ls -lth checkpoints/ | head -3
```

### 4. Save Checkpoints Frequently
**Modify checkpoint interval if worried:**
```python
--checkpoint-interval 3  # Save every 3 epochs instead of 5
```

---

## 🐛 Troubleshooting

### "No GPU available"
**Problem:** Forgot to enable GPU or free tier exhausted  
**Solution:**
1. Runtime → Change runtime type → T4 GPU
2. If still no GPU, you've hit daily limit (wait 12-24 hours)
3. Alternative: Use Kaggle Notebooks (also free GPU)

### "Out of memory"
**Problem:** Batch size too large for GPU  
**Solution:** Reduce batch size:
```python
--batch-size 8  # Instead of 16
```

### "Permission denied" for Drive
**Problem:** Drive mounting failed  
**Solution:**
1. Re-run drive.mount() cell
2. Click authorization link
3. Allow Colab access

### Training is slow
**Problem:** Not using GPU  
**Check:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name())  # Should show "Tesla T4"
```

---

## 📈 After Training

### Download Your Model
```bash
# On Mac, after downloading from Drive:
cd /Users/nani/2026-Prototype
mv ~/Downloads/best_model.pth checkpoints/
```

### Test Your Model
```bash
python scripts/test_model.py \
  --checkpoint checkpoints/best_model.pth \
  --test-image path/to/test/image.jpg
```

### Continue Training Locally (Optional)
```bash
# Resume on Mac if you want to fine-tune:
python scripts/train_maxsight.py \
  --resume-from checkpoints/best_model.pth \
  --epochs 60 \
  [... other args ...]
```

---

## 💰 Cost

**Everything is FREE!**
- Google Colab: Free T4 GPU
- COCO dataset: Free download
- Google Drive: 15GB free (enough for checkpoints)

**Upgrade options (if you want):**
- Colab Pro: $10/month → V100 GPU (~2x faster)
- Colab Pro+: $50/month → A100 GPU (~4x faster) + longer sessions

**For this project:** Free tier is perfect!

---

## ✅ Quick Start Checklist

- [ ] Create `maxsight_code.tar.gz` on Mac
- [ ] Upload to Google Drive
- [ ] Open `MaxSight_Colab_Training.ipynb` in Colab
- [ ] Enable T4 GPU (Runtime → Change runtime type)
- [ ] Run all cells in order
- [ ] Wait ~2 hours (mostly automated)
- [ ] Download results from Drive
- [ ] Copy checkpoints back to Mac

---

## 🎉 Success Metrics

**You'll know it worked when:**
- ✅ Training completes in ~90 minutes (not 23 hours)
- ✅ Loss decreases steadily across epochs
- ✅ Validation mAP increases (>0.3 is good)
- ✅ `best_model.pth` is saved
- ✅ You can download and use the trained model

---

## 📞 Need Help?

**Common questions:**

**Q: Can I close my laptop?**  
A: No! Keep tab open. Use "Keep-Alive" extension if needed.

**Q: What if I lose connection?**  
A: Training stops but checkpoints are saved. Resume from last checkpoint.

**Q: Can I train multiple models in parallel?**  
A: No, free tier = 1 GPU session at a time.

**Q: Is my data secure?**  
A: Yes, it's in your private Google Drive. Colab can only access what you mount.

---

## 🚀 You're Ready!

**Next:** Run the packaging command on your Mac, then follow the steps above.

**Estimated total time:** ~2 hours (mostly automated)  
**Your active time:** ~15 minutes (clicks and uploads)  
**Coffee breaks:** ~3-4 ☕

Good luck! 🎯
