# Codebase Cleanup Plan

**Date:** December 2025  
**Goal:** Make the codebase cleaner, more organized, and easier to maintain

---

## Cleanup Categories

### 1. Remove Backup/Redundant Files
- [ ] Remove `ml/models/maxsight_cnn_v1_backup.py` (backup in git history)
- [ ] Consolidate duplicate iOS bundles (3 directories → 1)
- [ ] Remove test export files if redundant

### 2. Organize Documentation
- [ ] Move analysis/report files from root to `docs/analysis/`
- [ ] Consolidate duplicate analysis files
- [ ] Organize documentation by category

### 3. Clean Up Deprecated Code
- [ ] Remove or move deprecated `MaxSightSimulator` class
- [ ] Update references to deprecated code

### 4. Remove Empty Directories
- [ ] Remove `app/session_manager/` if empty
- [ ] Remove `test_checkpoints/` if empty

### 5. Organize Log Files
- [ ] Move log files to `logs/` directory
- [ ] Ensure `.gitignore` properly excludes logs

### 6. Clean Up Temporary Files
- [ ] Remove temporary test files
- [ ] Clean up `__pycache__` directories (already in .gitignore)

---

## Detailed Actions

### Phase 1: File Removal

1. **Backup Files**
   - Delete: `ml/models/maxsight_cnn_v1_backup.py`
   - Reason: Backup is preserved in git history, no need for duplicate

2. **Duplicate iOS Bundles**
   - Keep: `test_ios_bundle/` (used for testing)
   - Archive: `ios_bundle/` and `maxsight_ios_bundle/` → consolidate to one
   - Action: Keep `test_ios_bundle/`, remove duplicates

3. **Empty Directories**
   - Remove: `app/session_manager/` (empty)
   - Remove: `test_checkpoints/` (empty)

### Phase 2: Documentation Organization

1. **Move Analysis Files to docs/**
   - Move: `COMPREHENSIVE_CODEBASE_ANALYSIS.md` → `docs/analysis/`
   - Move: `DEEP_ERROR_REPORT.md` → `docs/analysis/`
   - Move: `ERROR_FIXES_SUMMARY.md` → `docs/analysis/`
   - Move: `IMPLEMENTATION_ENHANCEMENTS.md` → `docs/analysis/`

2. **Consolidate Duplicate Documentation**
   - Review: Multiple export/status documents
   - Keep: Most recent/complete versions
   - Archive: Older versions

### Phase 3: Code Cleanup

1. **Deprecated Code**
   - Option A: Remove `MaxSightSimulator` class entirely
   - Option B: Move to `tools/simulation/deprecated/`
   - Recommendation: Option B (safer, preserves history)

2. **Log Files**
   - Move: `training_output_mps.log` → `logs/` or delete
   - Move: `training_output.log` → `logs/` or delete
   - Update: `.gitignore` already excludes `*.log` and `logs/`

### Phase 4: Structure Improvements

1. **Create Organized Structure**
   ```
   docs/
     analysis/          # Analysis and reports
     architecture/      # Architecture docs
     setup/            # Setup guides
     status/           # Status reports
   ```

2. **Clean Up Root Directory**
   - Keep only essential files in root
   - Move all documentation to appropriate subdirectories

---

## Execution Order

1. **Safe Removals** (no code changes)
   - Remove backup files
   - Remove empty directories
   - Move log files

2. **Documentation Organization** (no code impact)
   - Move analysis files
   - Organize docs structure

3. **Code Cleanup** (requires testing)
   - Handle deprecated code
   - Update imports if needed

4. **Final Verification**
   - Run tests to ensure nothing broke
   - Verify imports still work
   - Check git status

---

## Files to Remove

- `ml/models/maxsight_cnn_v1_backup.py`
- `ios_bundle/` (duplicate)
- `maxsight_ios_bundle/` (duplicate)
- `app/session_manager/` (empty)
- `test_checkpoints/` (empty)
- `training_output_mps.log` (move to logs/ or delete)
- `training_output.log` (move to logs/ or delete)

## Files to Move

- `COMPREHENSIVE_CODEBASE_ANALYSIS.md` → `docs/analysis/`
- `DEEP_ERROR_REPORT.md` → `docs/analysis/`
- `ERROR_FIXES_SUMMARY.md` → `docs/analysis/`
- `IMPLEMENTATION_ENHANCEMENTS.md` → `docs/analysis/`

## Code to Refactor

- `tools/simulation/web_simulator.py` - Move deprecated `MaxSightSimulator` class

---

**Estimated Impact:** Low risk - mostly file organization and removal of unused files

