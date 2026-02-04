# Quick Start: Cloud GPU Setup

**Fastest way to get MaxSight training on cloud GPU**

---

## 🚀 Option 1: Google Colab (5 minutes)

### Step 1: Open Colab
1. Go to: https://colab.research.google.com
2. **File → New notebook**

### Step 2: Enable GPU
1. **Runtime → Change runtime type**
2. Set **Hardware accelerator**: **GPU**
3. Click **Save**

### Step 3: Run Setup (Copy-Paste This)

```python
# Clone repo and install deps (run in a Colab cell):
!git clone -q https://github.com/AstroSword2897/2026-Prototype.git
%cd 2026-Prototype
!pip install -q -r requirements.txt
```

### Step 4: Test GPU (optional)

```python
!python scripts/archive/test_gpu_setup.py
# If that fails (no script), skip or: !python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Step 5: Start Training

```python
# 12–18 hour run (same as local EPOCHS=28, but on GPU so much faster per epoch)
%cd /content/2026-Prototype
!DEVICE=cuda EPOCHS=28 BATCH_SIZE=16 NUM_WORKERS=4 ./scripts/run_production_training.sh --no-export

# Or run train_maxsight.py directly (use your data paths):
# !python scripts/train_maxsight.py --data-dir datasets/coco_raw --train-annotation datasets/cleaned_splits/maxsight_train.json --val-annotation datasets/cleaned_splits/maxsight_val.json --image-dir datasets/coco_raw --epochs 28 --device cuda --batch-size 16
```

**Done!** ✅ Checkpoints in `checkpoints/`. Download via Files panel or mount Google Drive and copy there.

---

## 🖥️ Option 2: AWS EC2 (30 minutes)

### Step 1: Launch Instance
1. AWS Console → EC2 → Launch Instance
2. **AMI**: Deep Learning AMI (Ubuntu)
3. **Instance**: g4dn.xlarge (T4 GPU)
4. **Storage**: 100GB
5. **Launch**

### Step 2: Connect
```bash
ssh -i your-key.pem ubuntu@<instance-ip>
```

### Step 3: Setup
```bash
# Clone repo
git clone https://github.com/AstroSword2897/2026-Prototype.git
cd 2026-Prototype
git checkout feature/multimodal_refactor

# Install dependencies
pip3 install -r requirements.txt
pip3 install faiss-cpu

# Verify CUDA
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 4: Test
```bash
python3 scripts/test_gpu_setup.py
```

### Step 5: Train (in tmux so it keeps running)
```bash
tmux new -s training

# One-shot script (env check + data check + training)
DEVICE=cuda EPOCHS=28 BATCH_SIZE=16 NUM_WORKERS=4 ./scripts/run_production_training.sh --no-export

# Or run train_maxsight.py directly (use your data paths)
# python3 scripts/train_maxsight.py --data-dir datasets/coco_raw --train-annotation datasets/cleaned_splits/maxsight_train.json --val-annotation datasets/cleaned_splits/maxsight_val.json --image-dir datasets/coco_raw --epochs 100 --device cuda

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

**Done!** ✅

---

## 🎯 Option 3: RunPod & Lambda (GPU by the hour)

**RunPod** (runpod.io) and **Lambda Labs** (lambdalabs.com) give you a Linux GPU instance over SSH. Same steps for both:

1. **Sign up** → Create a GPU instance (e.g. A100 40GB, or T4 for cheaper).
2. **SSH in**: `ssh user@<instance-ip>` (they show the command in the dashboard).
3. **Clone and run**:
```bash
git clone https://github.com/AstroSword2897/2026-Prototype.git
cd 2026-Prototype
git checkout feature/multimodal_refactor   # or your branch
pip install -r requirements.txt
# Upload your data or use their storage; then:
DEVICE=cuda EPOCHS=28 BATCH_SIZE=16 NUM_WORKERS=4 ./scripts/run_production_training.sh --no-export
```
4. **Keep it running**: run inside `tmux` or `screen` so you can disconnect. Download `checkpoints/` when done (e.g. `scp -r user@host:2026-Prototype/checkpoints ./`).

**Rough cost**: RunPod A100 ~\$0.20–0.80/h; Lambda similar. A 12–18 h run ≈ \$5–15.

---

## 🎯 Option 4: Paperspace (10 minutes)

### Step 1: Create Notebook
1. Go to: https://www.paperspace.com
2. **Gradient → Notebooks → Create**
3. **Machine**: A100 (40GB)
4. **Container**: PyTorch
5. **Create**

### Step 2: Setup (Same as Colab)
```python
!wget -q https://raw.githubusercontent.com/AstroSword2897/2026-Prototype/feature/multimodal_refactor/scripts/colab_setup.py
!python colab_setup.py
```

### Step 3: Test & Train
```python
!cd /notebooks/2026-Prototype && python scripts/test_gpu_setup.py
!cd /notebooks/2026-Prototype && python scripts/train_maxsight.py --data-dir datasets/coco_raw --train-annotation datasets/cleaned_splits/maxsight_train.json --val-annotation datasets/cleaned_splits/maxsight_val.json --image-dir datasets/coco_raw --epochs 100 --device cuda
```

**Done!** ✅

---

## 🆘 Troubleshooting

**CUDA not available?**
- Colab: Enable GPU runtime (Runtime → Change runtime type → GPU)
- AWS: Use Deep Learning AMI (has CUDA pre-installed)

**Out of memory?**
- Reduce batch size in config: `batch_size: 4`
- Use gradient accumulation: `accumulate_grad_batches: 8`

**Need help?**
- Full cloud guide: `docs/archive/CLOUD_GPU_SETUP.md`
- Training summary (local + cloud): `TRAINING_SETUP_SUMMARY.md`
- Run test script: `python scripts/archive/test_gpu_setup.py` (or `scripts/test_gpu_setup.py` if present)

---

**Last Updated**: 2026-02

