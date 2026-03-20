# Git workflow (production-ready `main`)

Use **short-lived feature branches** for each coherent change, then merge into `main` so the default branch always matches what you would ship or run in CI.

## Branch naming

- `feature/<topic>` — new capability (e.g. `feature/s3-large-scale-hardening`)
- `fix/<topic>` — bugfix
- `docs/<topic>` — documentation-only

## Typical flow

```bash
git checkout main
git pull origin main
git checkout -b feature/my-change

# Work, stage, commit in logical chunks
git add -p
git commit -m "feat(scope): short imperative description"

# Keep main current before merge
git checkout main
git pull origin main
git merge feature/my-change --no-ff -m "merge: my-change"
git push origin main
```

## Rules of thumb

1. **`main` is deployable** — Do not merge broken tests or known regressions; run `pytest` before merge when touching code.
2. **One theme per branch** — Easier review, cleaner history, simpler reverts.
3. **Prefer `--no-ff` merges** — Preserves the branch boundary in history for larger features.
4. **Large WIP** — If many unrelated edits exist locally, split into multiple branches/commits rather than one giant merge.

## Staging

Use `git add -p` (patch mode) to stage only the hunks that belong to the current commit so unrelated workspace changes stay out of the branch.

## This repo and SageMaker / S3 work

Infrastructure and training scripts should land on `main` only after validation tests pass (e.g. `tests/test_ml_lifecycle.py` for S3 helpers).
