# Critical Limitations & Breaking Points

**Date:** December 2025  
**Status:** ⚠️ **MUST ADDRESS BEFORE PRODUCTION**

---

## Executive Summary

This document outlines **critical limitations** that will cause the system to break or degrade if not addressed. These are not theoretical concerns - they are **guaranteed failure modes** based on the current architecture.

---

## 🚨 CRITICAL ISSUE #1: Gradient Warfare (20 Heads)

### The Problem

**Current State**: 20 heads train simultaneously without proper task balancing.

**What Will Happen**:
- ✅ Detection accuracy will **slowly decay** (dominant task overwhelms others)
- ✅ Depth estimation will **oscillate** (conflicting gradients)
- ✅ Rare heads (fatigue, personalization) will **overfit silently**
- ✅ Training will appear to work but **quality degrades over time**

### Why This Happens

Multi-task learning with 20 heads creates gradient conflicts:

```
Detection Head (strong signal, lots of data)
    ↓ gradient conflicts with ↓
Fatigue Head (weak signal, sparse data)
    ↓ gradient conflicts with ↓
Personalization Head (user-specific, rare updates)
```

Without balancing, the **strongest gradients win**, causing:
- Dominant tasks (detection) to dominate training
- Rare tasks to be ignored or overfit
- Unstable training dynamics

### Current Mitigation (Insufficient)

- ✅ `UncertaintyWeightedLoss` exists but only for 6 tasks in `DetectionLoss`
- ❌ **No global task balancing** across all 20 heads
- ❌ **No GradNorm or PCGrad** implementation
- ❌ **No per-head loss monitoring**

### Required Fixes

**Immediate (This Week)**:
1. ✅ Implement `GradNormBalancer` (see `ml/training/task_balancing.py`)
2. ✅ Add per-head loss monitoring
3. ✅ Integrate into training loop

**Short-Term (Next 2-4 Weeks)**:
4. Add PCGrad as alternative
5. Implement alternating head freezing schedules
6. Add head-level kill switches (runtime control)

**Status**: ⚠️ **PARTIALLY ADDRESSED** - GradNorm implemented, needs integration

---

## 🚨 CRITICAL ISSUE #2: Scene Description Head Limitations

### The Problem

**Current State**: Scene description uses `[B, 512]` embedding + heuristic NLG.

**What Will Happen**:
- ✅ Descriptions will plateau as **"technically correct, cognitively awkward"**
- ✅ Spatial relationships will be **poorly expressed**
- ✅ Urgency-based ordering will be **suboptimal**
- ✅ Long-term: Users will find descriptions **unnatural**

### Why This Happens

Scene understanding is **graph-structured**, not just embedding-based:

```
Current: [B, 512] embedding → Heuristic NLG
    ↓
Problem: No explicit relationship modeling
    ↓
Result: "Door. Person. Stairs." (correct but awkward)
```

**What's Missing**:
- Scene graph (object nodes + spatial edges)
- Compositional language generation
- Urgency-aware ordering

### Current State

- ✅ Basic scene description works
- ❌ **No scene graph** (implicit or explicit)
- ❌ **No compositional language** model
- ❌ **Heuristic-based** (not learned)

### Required Fixes

**Medium-Term (Next 2-3 Months)**:
1. Implement lightweight scene graph (no GNN needed initially)
   - Object nodes: detections
   - Spatial edges: left/right/above/below/near/far
   - Feed into description generator
2. Add frozen small language decoder (conditioned on embeddings)
3. Improve urgency-based ordering

**Status**: ⚠️ **NOT ADDRESSED** - Future enhancement

---

## 🚨 CRITICAL ISSUE #3: Distance Estimation Limitations

### The Problem

**Current State**: Monocular depth via object size estimation.

**What Will Happen**:
- ✅ **Will fail badly indoors** with unknown objects
- ✅ **Fragile to camera FOV changes** (different phone models)
- ✅ **Will drift** across devices
- ✅ **Not navigation-grade** (assistive only)

### Why This Happens

Monocular depth estimation has fundamental limitations:

```
Object Size → Depth Estimate
    ↓
Problem: Assumes known object sizes
    ↓
Fails When:
- Unknown objects (indoor furniture)
- Camera FOV changes (different phones)
- Objects at angles (not frontal)
```

**Uncertainty helps but doesn't solve**:
- Uncertainty tells you "I'm not sure"
- But doesn't give you **better depth**

### Current Mitigation

- ✅ Uncertainty-weighted loss (properly implemented)
- ✅ Multi-scale features (helps)
- ⚠️ **Still monocular** (fundamental limitation)

### Required Documentation

**CRITICAL**: Document this limitation clearly:

> **Distance Estimation Disclaimer**:
> 
> MaxSight's distance estimation is **assistive, not navigation-grade**.
> 
> **Limitations**:
> - Works best outdoors with known objects (vehicles, signs)
> - Less reliable indoors with unknown furniture
> - Accuracy varies by device (camera FOV differences)
> - Use for **alerts and guidance**, not precise navigation
> 
> **When to Trust**:
> - ✅ High confidence detections (>0.8)
> - ✅ Known object classes (vehicles, people, signs)
> - ✅ Outdoor environments
> 
> **When to Be Cautious**:
> - ⚠️ Indoor environments
> - ⚠️ Unknown objects
> - ⚠️ High uncertainty (>0.7)
> - ⚠️ Different device models

**Status**: ⚠️ **DOCUMENTED** - Limitation acknowledged, not fixed (by design)

---

## 🚨 CRITICAL ISSUE #4: Therapy + Fatigue Head - Ethical/Regulatory Risk

### The Problem

**Current State**: Therapy and fatigue heads provide medical-adjacent feedback.

**What Will Happen**:
- ⚠️ **Regulatory scrutiny** if deployed without safeguards
- ⚠️ **Legal liability** if misused as diagnostic tool
- ⚠️ **User harm** if thresholds are wrong
- ⚠️ **No explainability** for decisions

### Why This Matters

**This is assistive technology, NOT a medical device**:

```
Therapy Head → Fatigue Detection → User Feedback
    ↓
Risk: User interprets as medical diagnosis
    ↓
Problem: No regulatory approval, no clinical validation
```

**Ethical Concerns**:
- Fatigue detection could be wrong
- Therapy recommendations could be inappropriate
- No clinical validation
- No explainability

### Required Safeguards

**Immediate (Before Any Deployment)**:

1. **Clear Disclaimers**:
   ```
   "MaxSight is an assistive technology tool, not a medical device.
   Fatigue and therapy features are experimental and should not be
   used for medical diagnosis or treatment decisions."
   ```

2. **Explainability**:
   - Log all fatigue/therapy decisions
   - Show confidence scores
   - Explain reasoning (why fatigue detected)

3. **Conservative Thresholds**:
   - Default to **low sensitivity** (fewer false positives)
   - User-adjustable thresholds
   - Clear "uncertain" states

4. **Audit Logs**:
   - Log all therapy/fatigue decisions
   - Track user interactions
   - Enable post-hoc analysis

5. **Non-Diagnostic Framing**:
   - Never claim "diagnosis"
   - Use "suggestions" not "prescriptions"
   - Emphasize "experimental" and "assistive"

**Status**: ⚠️ **NOT ADDRESSED** - Critical before any production deployment

---

## 📋 Implementation Roadmap

### Short-Term (Next 2-4 Weeks) - CRITICAL

1. ✅ **Task Balancing** (GradNorm)
   - Status: Implemented, needs integration
   - Priority: **HIGHEST**

2. ✅ **Head Kill Switches**
   - Status: Implemented, needs integration
   - Priority: **HIGH**

3. ⚠️ **Per-Head Loss Monitoring**
   - Status: Implemented, needs integration
   - Priority: **HIGH**

4. ⚠️ **Ethical Safeguards**
   - Status: **NOT STARTED**
   - Priority: **CRITICAL** (before deployment)

### Medium-Term (Next 2-3 Months)

5. **Scene Graph Lite**
   - Object nodes + spatial edges
   - Feed into description generator
   - Priority: **MEDIUM**

6. **Video Consistency**
   - Temporal smoothing on urgency & distance
   - Reduces cognitive noise
   - Priority: **MEDIUM**

7. **Battery-Aware Scheduler**
   - Tie head execution to thermal state
   - Drop motion & sound first
   - Priority: **MEDIUM**

### Long-Term (If Product)

8. **Separate Perception vs Policy**
   - Freeze perception core
   - Iterate policy layer faster
   - Regulatory isolation
   - Priority: **LOW** (future)

---

## 🎯 Immediate Action Items

### This Week

1. **Integrate GradNorm into training loop**
   ```python
   # In train_loop.py
   from ml.training.task_balancing import GradNormBalancer
   
   balancer = GradNormBalancer(num_tasks=20)
   # Use in loss computation
   ```

2. **Add head kill switches to MaxSightCNN**
   ```python
   # In maxsight_cnn.py
   from ml.training.task_balancing import HeadKillSwitch
   
   kill_switch = HeadKillSwitch(enabled_heads=['classification', 'box_regression', ...])
   # Check before head execution
   ```

3. **Add loss monitoring**
   ```python
   # In train_loop.py
   from ml.training.task_balancing import PerHeadLossMonitor
   
   monitor = PerHeadLossMonitor()
   # Update after each iteration
   ```

4. **Document distance limitations**
   - Add to user documentation
   - Add to API responses
   - Add to iOS app (if built)

### Next Week

5. **Add ethical safeguards**
   - Disclaimers in UI
   - Explainability logging
   - Conservative thresholds
   - Audit logs

6. **Test task balancing**
   - Monitor per-head losses
   - Verify balanced training
   - Adjust hyperparameters

---

## 📊 Risk Assessment

| Issue | Severity | Impact | Timeline | Status |
|-------|----------|--------|----------|--------|
| Gradient Warfare | 🔴 CRITICAL | Training degradation | Immediate | ⚠️ Partial |
| Scene Description | 🟡 MEDIUM | User experience | 2-3 months | ⚠️ Not Started |
| Distance Limitations | 🟡 MEDIUM | Feature accuracy | Documented | ✅ Documented |
| Ethical/Regulatory | 🔴 CRITICAL | Legal/regulatory | Before deploy | ❌ Not Started |

---

## 🚨 Breaking Point Timeline

**Without Fixes**:

- **Week 1-2**: Training appears normal
- **Week 3-4**: Detection accuracy starts decaying
- **Week 5-6**: Depth oscillates, rare heads overfit
- **Week 7+**: Model quality degrades significantly

**With Fixes**:

- **Week 1**: Task balancing integrated
- **Week 2**: Balanced training verified
- **Week 3+**: Stable multi-task learning

---

## References

- **GradNorm**: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (ICML 2018)
- **PCGrad**: Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
- **Multi-Task Learning**: Vandenhende et al., "Multi-Task Learning for Dense Prediction Tasks: A Survey" (TPAMI 2021)

---

**Last Updated**: December 2025  
**Priority**: 🔴 **CRITICAL** - Address before production deployment


