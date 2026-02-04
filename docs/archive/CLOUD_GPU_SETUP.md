# Cloud GPU Setup Guide - MaxSight 3.0

**Goal**: Get MaxSight training running on cloud GPU (CUDA)

**Time**: 30-60 minutes

---

## 🚀 Option 1: Google Colab (Easiest - Recommended)

### Why Colab?
- ✅ **Free tier available** (T4 GPU)
- ✅ **No credit card required** (free tier)
- ✅ **Instant access** - no setup
- ✅ **Pre-installed PyTorch/CUDA**
- ⚠️ **Limited hours** (disconnects after inactivity)
- ⚠️ **May need to upgrade** for long training runs

### Step 1: Access Colab

1. Go to: https://colab.research.google.com
2. Sign in with Google account
3. Create new notebook: **File → New notebook**

### Step 2: Enable GPU

1. Click **Runtime → Change runtime type**
2. Set **Hardware accelerator**: **GPU** (T4)
3. Click **Save**

### Step 3: Clone Repository

In a new cell, run:

```python
# Install git if needed
!apt-get update && apt-get install -y git

# Clone repository
!git clone https://github.com/AstroSword2897/2026-Prototype.git
!cd 2026-Prototype && git checkout feature/multimodal_refactor

# Verify
!cd 2026-Prototype && ls -la
```

### Step 4: Install Dependencies

```python
# Install PyTorch (usually pre-installed, but verify)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
!cd 2026-Prototype && pip install -r requirements.txt

# Install FAISS
!pip install faiss-cpu

# Verify CUDA
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Step 5: Verify Setup

```python
# Test import
import sys
sys.path.insert(0, '/content/2026-Prototype')

from ml.models.maxsight_cnn import MaxSightCNN, CapabilityTier
import torch

# Create model
model = MaxSightCNN(
    tier=CapabilityTier.T0_BASELINE_CNN,
    device='cuda'
)

# Test forward pass
dummy_input = torch.randn(1, 3, 224, 224).cuda()
output = model(dummy_input)

print("✅ Setup complete! Model loaded and tested.")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 6: Start Training

```python
# Run smoke training (2-3 hours)
!cd /content/2026-Prototype && python scripts/smoke_train.py --tier T0_BASELINE_CNN --epochs 2 --device cuda

# Or start full training
!cd /content/2026-Prototype && python scripts/train_maxsight.py \
  --config ml/training/configs/t0_baseline.yaml \
  --device cuda
```

### Colab Tips

**Prevent Disconnection**:
```python
# Run this in a cell to prevent timeout
import time
from IPython.display import display, HTML
import threading

def keep_alive():
    while True:
        time.sleep(60)
        display(HTML('<script>Jupyter.notebook.kernel.execute("");</script>'))

threading.Thread(target=keep_alive, daemon=True).start()
```

**Save Checkpoints to Google Drive**:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Train with checkpoints on Drive
!cd /content/2026-Prototype && python scripts/train_maxsight.py \
  --config ml/training/configs/t0_baseline.yaml \
  --device cuda \
  --checkpoint-dir /content/drive/MyDrive/maxsight_checkpoints
```

**Upgrade to Colab Pro** (if needed):
- Better GPUs (A100 available)
- More hours
- Less disconnection
- $10/month (Colab Pro) or $50/month (Colab Pro+)

---

## 🖥️ Option 2: AWS EC2 (Most Reliable)

### Why AWS EC2?
- ✅ **Reliable** - won't disconnect
- ✅ **Good for long training** runs
- ✅ **A100/H100 available**
- ⚠️ **Requires AWS account** (credit card)
- ⚠️ **More setup** required

### Step 1: Create AWS Account

1. Go to: https://aws.amazon.com
2. Sign up (requires credit card)
3. Navigate to **EC2** console

### Step 2: Launch GPU Instance

1. **EC2 Dashboard → Launch Instance**

2. **Configure Instance**:
   - **Name**: `maxsight-training`
   - **AMI**: **Deep Learning AMI (Ubuntu)** or **Deep Learning Base AMI**
     - Search: "Deep Learning AMI GPU PyTorch"
     - Select latest Ubuntu version
   - **Instance Type**: 
     - **g4dn.xlarge** (T4, 16GB) - ~$0.50/hour
     - **g5.xlarge** (A10G, 24GB) - ~$1.00/hour
     - **p3.2xlarge** (V100, 16GB) - ~$3.00/hour
     - **p4d.24xlarge** (A100, 40GB) - ~$32/hour
   - **Key Pair**: Create new or use existing
   - **Storage**: 100GB+ (for data + checkpoints)
   - **Security Group**: 
     - Allow SSH (port 22) from your IP
     - Allow HTTP/HTTPS if using web simulator

3. **Launch Instance**

### Step 3: Connect to Instance

**Option A: SSH (Terminal)**
```bash
# Download key pair (if new)
# Then connect:
ssh -i /path/to/key.pem ubuntu@<instance-ip>

# Or use AWS Systems Manager Session Manager (no key needed)
```

**Option B: EC2 Instance Connect (Browser)**
1. Select instance in EC2 console
2. Click **Connect**
3. Choose **EC2 Instance Connect**
4. Click **Connect**

### Step 4: Setup Environment

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Clone repository
git clone https://github.com/AstroSword2897/2026-Prototype.git
cd 2026-Prototype
git checkout feature/multimodal_refactor

# Deep Learning AMI already has CUDA/PyTorch, but verify:
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If PyTorch not installed:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip3 install -r requirements.txt
pip3 install faiss-cpu

# Verify CUDA
nvidia-smi
python3 -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### Step 5: Download Data (or mount S3)

**Option A: Download COCO to instance**
```bash
# Create data directory
mkdir -p datasets/coco_raw

# Download COCO (or upload from local)
# You can use S3, or scp from local machine
```

**Option B: Mount S3 (Recommended)**
```bash
# Install s3fs
sudo apt-get install s3fs

# Mount S3 bucket
echo ACCESS_KEY:SECRET_KEY > ~/.passwd-s3fs
chmod 600 ~/.passwd-s3fs
mkdir ~/s3-mount
s3fs your-bucket-name ~/s3-mount -o passwd_file=~/.passwd-s3fs
```

### Step 6: Start Training

```bash
# Smoke training
python3 scripts/smoke_train.py --tier T0_BASELINE_CNN --epochs 2 --device cuda

# Full training
python3 scripts/train_maxsight.py \
  --config ml/training/configs/t0_baseline.yaml \
  --device cuda \
  --checkpoint-dir /home/ubuntu/checkpoints
```

### Step 7: Monitor Training

**Option A: Use `tmux` (Recommended)**
```bash
# Install tmux
sudo apt-get install tmux

# Start tmux session
tmux new -s training

# Run training (inside tmux)
python3 scripts/train_maxsight.py --config ml/training/configs/t0_baseline.yaml --device cuda

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

**Option B: Use `screen`**
```bash
screen -S training
# Run training
# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

**Option C: Use `nohup`**
```bash
nohup python3 scripts/train_maxsight.py \
  --config ml/training/configs/t0_baseline.yaml \
  --device cuda \
  > training.log 2>&1 &

# Monitor
tail -f training.log
```

### AWS Cost Management

**Stop Instance When Not Training**:
```bash
# In AWS Console: EC2 → Instances → Stop
# Or via CLI:
aws ec2 stop-instances --instance-ids i-xxxxx
```

**Set Up Billing Alerts**:
1. AWS Console → Billing → Budgets
2. Create budget
3. Set alert threshold (e.g., $50)

---

## 🎯 Option 3: Paperspace Gradient (Easy + Reliable)

### Why Paperspace?
- ✅ **Easy setup** (similar to Colab)
- ✅ **Reliable** (won't disconnect)
- ✅ **Good pricing** (~$1/hour for A100)
- ⚠️ **Requires credit card**

### Step 1: Sign Up

1. Go to: https://www.paperspace.com
2. Sign up (requires credit card)
3. Navigate to **Gradient** → **Notebooks**

### Step 2: Create Notebook

1. **Create Notebook**
2. **Select**:
   - **Machine**: A100 (40GB) or H100 (80GB)
   - **Container**: PyTorch (latest)
   - **Storage**: 50GB+
3. **Create**

### Step 3: Setup (Same as Colab)

```python
# Clone repo
!git clone https://github.com/AstroSword2897/2026-Prototype.git
!cd 2026-Prototype && git checkout feature/multimodal_refactor

# Install dependencies
!cd 2026-Prototype && pip install -r requirements.txt
!pip install faiss-cpu

# Verify CUDA
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Step 4: Train

```python
!cd /notebooks/2026-Prototype && python scripts/train_maxsight.py \
  --config ml/training/configs/t0_baseline.yaml \
  --device cuda
```

---

## 🧪 Quick Test Script

Create this to verify setup:

```python
# test_setup.py
import torch
import sys
sys.path.insert(0, '/content/2026-Prototype')  # Adjust path for your setup

from ml.models.maxsight_cnn import MaxSightCNN, CapabilityTier

print("=" * 50)
print("MaxSight GPU Setup Test")
print("=" * 50)

# Test CUDA
print(f"\n✅ CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test Model
print("\n📦 Loading T0 model...")
model = MaxSightCNN(
    tier=CapabilityTier.T0_BASELINE_CNN,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Test Forward Pass
print("\n🧪 Testing forward pass...")
dummy_input = torch.randn(1, 3, 224, 224)
if torch.cuda.is_available():
    dummy_input = dummy_input.cuda()
    model = model.cuda()

with torch.no_grad():
    output = model(dummy_input)

print("✅ Forward pass successful!")
print(f"✅ Output keys: {list(output.keys())}")

print("\n" + "=" * 50)
print("🎉 Setup Complete! Ready for training.")
print("=" * 50)
```

---

## 📋 Setup Checklist

### Pre-Setup
- [ ] Choose cloud provider (Colab/AWS/Paperspace)
- [ ] Create account (if needed)
- [ ] Have repository URL ready

### Colab Setup
- [ ] Create new notebook
- [ ] Enable GPU runtime
- [ ] Clone repository
- [ ] Install dependencies
- [ ] Run test script
- [ ] Start training

### AWS EC2 Setup
- [ ] Create AWS account
- [ ] Launch GPU instance
- [ ] Connect via SSH
- [ ] Clone repository
- [ ] Install dependencies
- [ ] Setup data (S3 or download)
- [ ] Start training in tmux/screen

### Paperspace Setup
- [ ] Create account
- [ ] Create notebook
- [ ] Clone repository
- [ ] Install dependencies
- [ ] Run test script
- [ ] Start training

---

## 🆘 Troubleshooting

### CUDA Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. **Colab**: Make sure GPU runtime is enabled (Runtime → Change runtime type → GPU)
2. **AWS**: Use Deep Learning AMI (has CUDA pre-installed)
3. **Reinstall PyTorch with CUDA**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Out of Memory

**Problem**: `CUDA out of memory`

**Solutions**:
1. **Reduce batch size** in config:
   ```yaml
   data:
     batch_size: 4  # Reduce from 8 or 16
   ```
2. **Use gradient accumulation**:
   ```yaml
   training:
     accumulate_grad_batches: 8  # Increase
   ```
3. **Use mixed precision**:
   ```yaml
   training:
     mixed_precision: true
   ```

### Connection Lost (Colab)

**Problem**: Colab disconnects during training

**Solutions**:
1. **Use keep-alive script** (see Colab Tips above)
2. **Upgrade to Colab Pro** (more stable)
3. **Use AWS EC2** (won't disconnect)

### Slow Data Loading

**Problem**: Training is slow due to data loading

**Solutions**:
1. **Increase num_workers**:
   ```yaml
   data:
     num_workers: 8  # Increase from 4
   ```
2. **Use SSD storage** (AWS: gp3 volumes)
3. **Pre-load data to instance** (don't stream from S3)

---

## 💰 Cost Estimates

| Provider | GPU | Cost/Hour | 24h Cost | 1 Week Cost |
|----------|-----|-----------|----------|-------------|
| **Colab Free** | T4 | $0 | $0 | $0 (limited hours) |
| **Colab Pro** | T4/A100 | $10/month | - | - |
| **AWS g4dn.xlarge** | T4 | $0.50 | $12 | $84 |
| **AWS g5.xlarge** | A10G | $1.00 | $24 | $168 |
| **AWS p3.2xlarge** | V100 | $3.00 | $72 | $504 |
| **AWS p4d.24xlarge** | A100 | $32.00 | $768 | $5,376 |
| **Paperspace A100** | A100 | $1.10 | $26 | $185 |
| **Lambda Labs A100** | A100 | $1.10 | $26 | $185 |

**Recommendation**: Start with **Colab Free** or **AWS g4dn.xlarge** for testing, upgrade to A100 for full training.

---

## 🎯 Next Steps After Setup

1. **Run smoke training** (2-3 hours):
   ```bash
   python scripts/smoke_train.py --tier T0_BASELINE_CNN --epochs 2 --device cuda
   ```

2. **Start full T0 training** (1-2 days):
   ```bash
   python scripts/train_maxsight.py \
     --config ml/training/configs/t0_baseline.yaml \
     --device cuda
   ```

3. **Monitor training**:
   - Watch loss curves
   - Check validation metrics
   - Save checkpoints regularly

4. **Proceed to T1, T2, etc.** after T0 completes

---

**Last Updated**: 2025-01-30

