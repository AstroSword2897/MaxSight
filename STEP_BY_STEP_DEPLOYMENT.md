# Step-by-Step Deployment Workflow

**How this works**: You run commands, share outputs, I analyze and give you the next step.

---

## Current Status ✅

- ✅ **Open Images V6**: 41,620 images ready (12 GB)
- ✅ **ADE20K**: 2,000 images ready (1 GB)  
- ✅ **COCO Training**: 102K+ images ready
- ✅ **Checkpoints**: 2 trained models ready
- ✅ **Export Scripts**: All formats ready

---

## Step 1: Install rclone

**Run this**:
```bash
brew install rclone
```

**Then share the output** - I'll check if it installed correctly and give you Step 2.

---

## Step 2: Configure rclone

**Run this**:
```bash
rclone config
```

**Follow the prompts** (I'll guide you through each one):
- Type `n` for new remote
- Name it `gdrive`
- Choose `drive` (Google Drive)
- Press Enter for defaults
- Press `y` for auto config (opens browser)
- Authorize in browser
- Press `y` to save

**Then share the output** - I'll verify configuration and give you Step 3.

---

## Step 3: Test Connection

**Run this**:
```bash
rclone lsd gdrive:
```

**Then share the output** - I'll verify it's working and give you Step 4.

---

## Step 4: Download Datasets in Colab

**Open Colab:** https://colab.research.google.com/

**Cell 1 - Mount Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 - Download Datasets:**
Copy entire contents of `COLAB_DATASETS_CELL.txt` and paste into Colab, then run.

**That's it!** Downloads Open Images V6, BDD100K, and ADE20K directly to Drive.

---

## Step 5: Upload Checkpoints (if needed)

```bash
rclone copy checkpoints "gdrive:MaxSight/checkpoints" --progress --transfers 4
```

**Done!** Datasets download in Colab, checkpoints upload via rclone.

---

## Next Steps After Deployment Setup

**Complete workflow:**

1. **Web Simulator** → [tools/simulation/README.md](tools/simulation/README.md)
2. **Verify** → [PRE_TRAIN_CHECKLIST.md](PRE_TRAIN_CHECKLIST.md)
3. **Train** → [TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md)
4. **Export for Xcode** → [EXPORT_FOR_XCODE.md](EXPORT_FOR_XCODE.md)
5. **You**: Update 2026 repo, then make Xcode app look good

**All guides are ready - follow them in order!**
