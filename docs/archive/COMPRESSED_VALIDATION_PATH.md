# Compressed Validation Path: Risk-First Approach

**Date**: 2025-01-27  
**Status**: Ready to Execute  
**Timeline**: 4-5 days total

---

## 🎯 **The Path: Test → Smoke Train → Benchmark → Commit**

### **Phase 1: Hard Validation Sprint (48 hours max)**

**Goal**: Prove nothing is broken in integration.

**Scripts**:
- `scripts/validate_forward_passes.py` - Tests all tiers T0-T5

**What it does**:
1. Single batch forward pass for each tier
2. Random + real sample inputs
3. Logs shapes, memory, latency
4. End-to-end dry run (no training)

**Success criteria**:
- ✅ All tiers pass forward pass
- ✅ No crashes or errors
- ✅ Stage A latency logged (target <150ms)

**If this fails**: Stop and fix immediately. No exceptions.

---

### **Phase 2: Smoke Training (1-2 days)**

**Goal**: Verify gradients, loss flow, and stability.

**Scripts**:
- `scripts/smoke_train.py` - Minimal training proof of life

**What it does**:
- Tiny synthetic dataset (50-100 images worth)
- 1-2 epochs only
- Overfit on purpose
- Watch: loss decreasing, no NaNs, GPU memory stable

**Success criteria**:
- ✅ Loss decreases over epochs
- ✅ No NaN detected
- ✅ Memory stable
- ✅ Reasonable throughput

**If it can't overfit a tiny set**: Something is wrong. Fix it now, not later.

---

### **Phase 3: Benchmark Before Scaling (1 day)**

**Goal**: Avoid architectural regret.

**Scripts**:
- `scripts/benchmark_tiers.py` - Comprehensive tier benchmarking

**What it measures**:
- Stage A latency (target <150ms)
- Memory per tier
- Parameter counts
- Mobile export size (JIT, ONNX, CoreML)

**This tells you**:
- Which tiers are actually viable
- Where pruning / distillation will matter
- Whether mobile is realistic before long training

---

### **Phase 4: Commit to Full Training**

**Only after the above passes cleanly.**

Now you:
- Lock configs
- Expand dataset
- Schedule long runs
- Start logging like this is production

---

## 🚀 **Quick Start**

### **Step 1: Validate Forward Passes**
```bash
python scripts/validate_forward_passes.py
```

**Expected output**:
- All tiers pass
- Latency summary
- Memory summary
- ✅ "VALIDATION COMPLETE - PROCEED TO SMOKE TRAINING"

**If fails**: Fix issues before proceeding.

---

### **Step 2: Smoke Training**
```bash
# Default: T2_HYBRID_VIT, 2 epochs, 10 batches
python scripts/smoke_train.py

# Custom tier
python scripts/smoke_train.py --tier T0_MOBILE --epochs 2 --batches 10

# More aggressive test
python scripts/smoke_train.py --tier T5_TEMPORAL --epochs 3 --batches 20
```

**Expected output**:
- Loss decreases over epochs
- No NaN detected
- ✅ "SUCCESS: Model can learn - proceed to full training!"

**If fails**: Check learning rate, loss functions, or model architecture.

---

### **Step 3: Benchmark Tiers**
```bash
# Benchmark all tiers
python scripts/benchmark_tiers.py --runs 50

# Benchmark specific tier
python scripts/benchmark_tiers.py --tier T2_HYBRID_VIT --runs 100

# Include export sizes
python scripts/benchmark_tiers.py --export --output benchmark_results.json
```

**Expected output**:
- Latency per tier (mean, p50, p95, p99)
- Memory usage per tier
- Parameter counts
- Export sizes (if --export)
- Recommendations for viable tiers

**Use this to decide**: Which tier becomes the flagship.

---

## 📊 **Decision Points**

### **After Phase 1 (Validation)**
- ✅ All pass → Proceed to smoke training
- ❌ Any fail → Stop and fix

### **After Phase 2 (Smoke Training)**
- ✅ Loss decreases → Proceed to benchmarking
- ❌ Loss doesn't decrease → Fix learning rate / architecture

### **After Phase 3 (Benchmarking)**
- ✅ Tier meets targets → Commit to full training
- ⚠️ Tier close to targets → Consider optimization before training
- ❌ Tier far from targets → Reconsider architecture

---

## ⚠️ **What We Are NOT Doing Yet**

- ❌ Mobile deployment (after benchmarks)
- ❌ Full multi-week training (after smoke test passes)
- ❌ Aggressive optimization (after benchmarks show need)

Those come *after* we have real numbers.

---

## ✅ **Why This Path Works**

1. **Fast feedback**: Days, not weeks
2. **Risk-first**: Catches problems early
3. **Data-driven**: Benchmarks inform decisions
4. **Compounds**: Each step builds on the last

---

## 📝 **Checklist**

### **Day 1-2: Validation**
- [ ] Run `validate_forward_passes.py`
- [ ] Fix any failures
- [ ] Verify all tiers pass

### **Day 3-4: Smoke Training**
- [ ] Run `smoke_train.py` for target tier(s)
- [ ] Verify loss decreases
- [ ] Check for NaN
- [ ] Verify memory stability

### **Day 5: Benchmarking**
- [ ] Run `benchmark_tiers.py` for all tiers
- [ ] Review latency, memory, export sizes
- [ ] Decide on flagship tier
- [ ] Plan optimization if needed

### **Day 6+: Full Training**
- [ ] Lock configs
- [ ] Expand dataset
- [ ] Start long training runs
- [ ] Set up production logging

---

## 🎯 **Success Criteria**

**Phase 1 Complete When**:
- ✅ All tiers pass forward pass
- ✅ No crashes or errors
- ✅ Latency logged

**Phase 2 Complete When**:
- ✅ Loss decreases over epochs
- ✅ No NaN detected
- ✅ Memory stable

**Phase 3 Complete When**:
- ✅ Benchmarks complete for all tiers
- ✅ Flagship tier identified
- ✅ Optimization plan (if needed)

**Phase 4 Ready When**:
- ✅ All above phases pass
- ✅ Configs locked
- ✅ Dataset ready

---

## 🚀 **Ready to Start?**

```bash
# Step 1: Validate
python scripts/validate_forward_passes.py

# Step 2: Smoke train
python scripts/smoke_train.py

# Step 3: Benchmark
python scripts/benchmark_tiers.py --export
```

**Let's get signal fast and momentum rolling!** 🚀

