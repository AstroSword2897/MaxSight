# rclone Setup Guide for Google Drive Upload

**Purpose**: Upload locally downloaded datasets to Google Drive for Colab access

---

## Step 1: Install rclone

```bash
brew install rclone
```

Verify installation:
```bash
rclone version
```

---

## Step 2: Configure Google Drive

Run the configuration wizard:

```bash
rclone config
```

**Configuration Steps**:

1. **Create new remote**: Type `n` and press Enter
2. **Name**: Type `gdrive` and press Enter
3. **Storage type**: Type `drive` (number for Google Drive) and press Enter
4. **Client ID**: Press Enter (use default)
5. **Client Secret**: Press Enter (use default)
6. **Scope**: Press Enter (use default - full access)
7. **Service Account**: Press Enter (skip)
8. **Advanced config**: Press `n` (no)
9. **Auto config**: Press `y` (yes) - this opens your browser
10. **Authenticate**: Follow browser prompts to authorize rclone
11. **Configure as Shared Drive**: Press `n` (no)
12. **Done**: Press `y` (yes) to save

**Verify configuration**:
```bash
rclone listremotes
# Should show: gdrive:
```

---

## Step 3: Test Connection

```bash
# List files in your Drive root
rclone lsd gdrive:

# Should show your Drive folders
```

---

## Step 4: Upload Datasets

### Option A: Use the Setup Script (Recommended)

```bash
cd /Users/nani/2026-Prototype
./scripts/setup_rclone_upload.sh
```

This script will:
- Check if rclone is installed
- Guide you through configuration if needed
- Show what's available to upload
- Upload datasets and checkpoints

### Option B: Manual Upload Commands

**Upload Open Images V6**:
```bash
rclone copy datasets/open_images_v6 \
  "gdrive:MaxSight/datasets/open_images_v6" \
  --progress \
  --transfers 4
```

**Upload ADE20K**:
```bash
rclone copy datasets/ade20k \
  "gdrive:MaxSight/datasets/ade20k" \
  --progress \
  --transfers 4
```

**Upload Checkpoints**:
```bash
rclone copy checkpoints \
  "gdrive:MaxSight/checkpoints" \
  --progress
```

**Upload Dataset Splits**:
```bash
rclone copy datasets/cleaned_splits \
  "gdrive:MaxSight/datasets/cleaned_splits" \
  --progress
```

---

## Step 5: Verify Upload

```bash
# List uploaded files
rclone ls "gdrive:MaxSight/datasets/open_images_v6/validation" | wc -l
# Should show ~41,620

rclone ls "gdrive:MaxSight/datasets/ade20k/images/validation" | wc -l
# Should show ~2,000

rclone ls "gdrive:MaxSight/checkpoints"
# Should show your checkpoint files
```

---

## Upload Time Estimates

| Dataset | Size | Upload Time (typical) |
|---------|------|----------------------|
| Open Images V6 | ~2 GB | 15-20 minutes |
| ADE20K | ~1 GB | 5-10 minutes |
| Checkpoints | ~1.6 GB | 10-15 minutes |
| Splits | ~30 MB | <1 minute |
| **Total** | **~4.7 GB** | **30-45 minutes** |

---

## Troubleshooting

### "rclone: command not found"
**Fix**: Install rclone
```bash
brew install rclone
```

### "Failed to create file system"
**Fix**: Reconfigure rclone
```bash
rclone config
# Delete old remote and create new one
```

### Upload is slow
**Options**:
- Increase transfers: `--transfers 8`
- Use `--drive-chunk-size 128M` for large files
- Upload during off-peak hours

### "Access denied" errors
**Fix**: Re-authenticate
```bash
rclone config
# Select your remote, choose "Edit", re-authenticate
```

---

## Quick Reference

**Check what's uploaded**:
```bash
rclone ls "gdrive:MaxSight/" --recursive
```

**Sync (update only changed files)**:
```bash
rclone sync datasets/open_images_v6 "gdrive:MaxSight/datasets/open_images_v6" --progress
```

**Copy (upload everything)**:
```bash
rclone copy datasets/open_images_v6 "gdrive:MaxSight/datasets/open_images_v6" --progress
```

**Difference**: `sync` deletes files in destination that don't exist locally. `copy` only adds/updates.

---

## After Upload

In Colab, your datasets will be at:
```
/content/drive/MyDrive/MaxSight/datasets/
```

You can skip the download cell and use:
```python
from pathlib import Path
BASE_DIR = Path("/content/drive/MyDrive/MaxSight/datasets")

# Open Images V6
oi6_dir = BASE_DIR / "open_images_v6"

# ADE20K
ade20k_dir = BASE_DIR / "ade20k"

# Checkpoints
checkpoints_dir = Path("/content/drive/MyDrive/MaxSight/checkpoints")
```

---

**Ready to upload?** Run: `./scripts/setup_rclone_upload.sh`
