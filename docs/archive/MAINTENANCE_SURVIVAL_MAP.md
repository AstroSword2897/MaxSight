# MaxSight Maintenance Survival Map

**One-Page Guide for Long-Term System Health**

---

## 🎯 Critical Components to Monitor

### Tier 1 (Safety-Critical) - **Never Fail**
| Component | What to Check | When | Action if Fails |
|-----------|---------------|------|-----------------|
| **Objectness Head** | Detection rate drops <85% | Daily | **STOP DEPLOYMENT** - Rollback model |
| **Classification Head** | mAP drops >5% | Daily | **STOP DEPLOYMENT** - Rollback model |
| **Box Regression** | IoU drops <0.7 | Daily | **STOP DEPLOYMENT** - Rollback model |
| **Distance Zones** | False near/far predictions | Daily | **STOP DEPLOYMENT** - Rollback model |
| **Urgency Head** | False reassurance rate >1% | Daily | **STOP DEPLOYMENT** - Rollback model |
| **Uncertainty Head** | Calibration error >0.1 | Daily | **STOP DEPLOYMENT** - Rollback model |

**Rule**: If ANY Tier 1 head fails, the system is unsafe. Rollback immediately.

---

### Tier 2 (Navigation & Context) - **Can Degrade**
| Component | What to Check | When | Action if Fails |
|-----------|---------------|------|-----------------|
| **Motion Head** | Temporal consistency <0.8 | Weekly | Throttle to every N frames |
| **ROI Priority** | Ranking loss increases | Weekly | Fallback to uniform priority |
| **Scene Complexity** | Prediction variance spikes | Weekly | Disable, use default |
| **Spatial Memory** | Memory corruption detected | Daily | Clear cache, restart |
| **Path Planning** | Planning failures >10% | Weekly | Disable, use simple fallback |

**Rule**: Tier 2 failures degrade gracefully. System continues with reduced functionality.

---

### Tier 3 (Enhancement) - **Optional**
| Component | What to Check | When | Action if Fails |
|-----------|---------------|------|-----------------|
| **Scene Description** | Generation latency >500ms | Weekly | Disable, use template |
| **Retrieval** | Network/index failures | Real-time | Disable, continue without |
| **Therapy** | Task generation errors | Weekly | Disable, log for review |
| **Fatigue** | Prediction confidence <0.5 | Weekly | Disable, use default |

**Rule**: Tier 3 failures are silent. System continues normally.

---

## 📊 Health Checks (Automated)

### Daily Checks
```bash
# Run daily health check
python scripts/health_check.py --tier 1 --alert-on-failure

# Check logs for Tier 1 errors
python scripts/check_logs.py --severity critical --last 24h

# Monitor latency
python scripts/latency_monitor.py --threshold 150ms
```

### Weekly Checks
```bash
# Run full test suite
pytest tests/ --coverage

# Check model drift
python scripts/check_model_drift.py --baseline latest

# Backup code and models
python scripts/backup.py --models --code --data
```

### Monthly Checks
```bash
# Evaluate model performance
python scripts/evaluate_model.py --full-report

# Clean datasets
python scripts/clean_datasets.py --remove-duplicates

# Update dependencies (security)
pip-audit --requirement requirements.txt
```

---

## 🔄 Maintenance Schedule

| Frequency | Task | Owner | Critical? |
|-----------|------|-------|-----------|
| **Daily** | Check Tier 1 errors | On-call | ✅ YES |
| **Daily** | Monitor latency (<150ms Stage A) | DevOps | ✅ YES |
| **Weekly** | Run all tests | QA | ✅ YES |
| **Weekly** | Backup code + models | DevOps | ✅ YES |
| **Monthly** | Evaluate model drift | ML Engineer | ✅ YES |
| **Monthly** | Clean datasets | Data Engineer | ⚠️ MEDIUM |
| **Quarterly** | Retrain models | ML Engineer | ✅ YES |
| **Quarterly** | Full regression tests | QA | ✅ YES |
| **Bi-Annually** | Update dependencies | DevOps | ⚠️ MEDIUM |
| **Bi-Annually** | Architecture review | Tech Lead | ⚠️ MEDIUM |
| **Annually** | Deep audit (all systems) | Tech Lead | ✅ YES |

---

## 🚨 Alert Thresholds

### Critical (Immediate Action Required)
- Tier 1 head failure rate >1%
- Stage A latency >200ms
- False reassurance rate >1%
- Model uncertainty >0.8 (system-wide)
- Memory leak detected
- GPU/CPU overload (>90%)

### Warning (Investigate Within 24h)
- Tier 2 head failure rate >10%
- Stage B latency >500ms
- Model drift >5% from baseline
- Data quality issues detected
- Test coverage drops <80%

### Info (Monitor)
- Tier 3 head failures
- Retrieval latency spikes
- User feedback trends
- Dependency updates available

---

## 📦 Versioning Strategy

### Models
```
models/
  ├── v1.0.0/          # Initial release
  ├── v1.1.0/          # Bug fixes
  ├── v2.0.0/          # Architecture upgrade
  └── latest/          # Symlink to current
```

**Rule**: Never overwrite. Always version. Keep last 5 versions.

### Code
```
git tag -a v1.0.0 -m "Initial release"
git tag -a v1.1.0 -m "Tiered architecture"
```

**Rule**: Tag every release. Use semantic versioning.

### Data
```
datasets/
  ├── v1.0.0/          # COCO + accessibility
  ├── v1.1.0/          # + real-world feedback
  └── latest/          # Symlink to current
```

**Rule**: Immutable datasets. Tag with model version used.

---

## 🔍 Monitoring Dashboard

### Key Metrics to Track
1. **Tier 1 Availability**: >99.9% (target)
2. **Stage A Latency**: <150ms (target: <100ms)
3. **False Reassurance Rate**: <1%
4. **Model Uncertainty**: Well-calibrated
5. **Test Coverage**: >80%
6. **Code Quality**: Lint score >90%

### Logging Requirements
- Every inference logged (with frame_id)
- Every Tier 1 failure logged (with full context)
- Every model update logged (with version)
- Every data change logged (with hash)

---

## 🛠️ Quick Fixes

### If Tier 1 Fails
1. **STOP** all deployments
2. **ROLLBACK** to last known good model
3. **INVESTIGATE** logs for root cause
4. **FIX** issue in staging
5. **TEST** thoroughly
6. **DEPLOY** to 10% users first
7. **MONITOR** for 24h
8. **SCALE** to 100% if stable

### If Latency Spikes
1. Check GPU/CPU usage
2. Check memory leaks
3. Disable Tier 2/3 heads temporarily
4. Profile Stage A bottleneck
5. Optimize or rollback

### If Model Drifts
1. Compare predictions to baseline
2. Identify drift source (data? environment?)
3. Retrain with new data
4. Validate on held-out set
5. Deploy gradually

---

## 👥 Ownership Matrix

| Component | Owner | Backup Owner | Review Frequency |
|-----------|-------|-------------|------------------|
| **Tier 1 Heads** | ML Engineer A | ML Engineer B | Weekly |
| **Backbone** | ML Engineer B | ML Engineer A | Monthly |
| **Output Scheduler** | Software Engineer A | Software Engineer B | Monthly |
| **Therapy System** | ML Engineer C | ML Engineer A | Quarterly |
| **Retrieval** | ML Engineer D | ML Engineer B | Quarterly |
| **Infrastructure** | DevOps | Software Engineer A | Weekly |
| **Testing** | QA | Software Engineer B | Weekly |

**Rule**: Rotate ownership annually. No single-person dependencies.

---

## 📚 Documentation Requirements

### Must-Have Docs
- [x] Architecture diagram (README.md)
- [x] API documentation (code comments)
- [x] Training procedures (scripts/train_maxsight.py)
- [x] Deployment guide (docs/DEPLOYMENT.md)
- [x] Maintenance schedule (this file)
- [ ] Onboarding guide (docs/ONBOARDING.md)
- [ ] Troubleshooting guide (docs/TROUBLESHOOTING.md)

### Update Frequency
- Architecture: Update on major changes
- API: Update on every change
- Procedures: Update quarterly
- Troubleshooting: Update as issues arise

---

## ✅ Pre-Deployment Checklist

Before deploying ANY change:

- [ ] All tests pass
- [ ] Code reviewed by 2+ people
- [ ] Lint/type checks pass
- [ ] Model versioned and tagged
- [ ] Data versioned and tagged
- [ ] Documentation updated
- [ ] Rollback plan documented
- [ ] Monitoring alerts configured
- [ ] Staging environment tested
- [ ] Performance benchmarks met

**Rule**: If ANY checkbox is unchecked, don't deploy.

---

## 🎓 Onboarding Checklist

New developer should be able to:

- [ ] Clone repository
- [ ] Run tests (`pytest tests/`)
- [ ] Train model (`python scripts/train_maxsight.py`)
- [ ] Run inference (`python -m ml.models.maxsight_cnn`)
- [ ] Understand tiered architecture
- [ ] Know Tier 1 vs Tier 2 vs Tier 3
- [ ] Know when to rollback
- [ ] Know who to contact for issues

**Rule**: If onboarding takes >1 day, documentation needs work.

---

## 🔗 Quick Links

- **Architecture**: `README.md`
- **Training**: `scripts/train_maxsight.py`
- **Testing**: `tests/`
- **Monitoring**: `scripts/health_check.py`
- **Deployment**: `docs/DEPLOYMENT.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

---

## 📞 Emergency Contacts

| Role | Contact | When to Contact |
|------|---------|-----------------|
| **Tier 1 Failure** | On-call ML Engineer | Immediately |
| **Infrastructure** | DevOps | System down |
| **Data Issues** | Data Engineer | Data corruption |
| **Architecture** | Tech Lead | Major changes |

---

**Last Updated**: 2025-01-XX  
**Next Review**: Quarterly  
**Owner**: Tech Lead

---

## 🎯 Bottom Line

1. **Tier 1 never fails** - If it does, rollback immediately
2. **Monitor daily** - Catch issues before they become problems
3. **Version everything** - Models, code, data
4. **Test everything** - No untested code goes live
5. **Document everything** - Future you will thank you
6. **Own everything** - Clear ownership prevents decay

**If you follow this map, MaxSight will survive long-term.**

