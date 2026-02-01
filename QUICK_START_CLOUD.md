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
# Run this in a Colab cell:
!wget -q https://raw.githubusercontent.com/AstroSword2897/2026-Prototype/feature/multimodal_refactor/scripts/colab_setup.py
!python colab_setup.py
```

### Step 4: Test Setup

```python
!cd /content/2026-Prototype && python scripts/test_gpu_setup.py
```

### Step 5: Start Training

```python
# Smoke training (2-3 hours)
!cd /content/2026-Prototype && python scripts/smoke_train.py --tier T0_BASELINE_CNN --epochs 2 --device cuda

# Full training (1-2 days)
!cd /content/2026-Prototype && python scripts/train_maxsight.py --config ml/training/configs/t0_baseline.yaml --device cuda
```

**Done!** ✅

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

### Step 5: Train (in tmux)
```bash
# Start tmux
tmux new -s training

# Run training
python3 scripts/train_maxsight.py --config ml/training/configs/t0_baseline.yaml --device cuda

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

**Done!** ✅

---

## 🎯 Option 3: Paperspace (10 minutes)

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
!cd /notebooks/2026-Prototype && python scripts/train_maxsight.py --config ml/training/configs/t0_baseline.yaml --device cuda
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
- See full guide: `docs/CLOUD_GPU_SETUP.md`
- Run test script: `python scripts/test_gpu_setup.py`

---

**Last Updated**: 2025-01-30

