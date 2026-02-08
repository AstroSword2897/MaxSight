#!/usr/bin/env python3
"""Cleanup script for Colab/Cloud training artifacts.

Deletes old checkpoints, logs, and temporary files to free up space.
Safe to run - will ask for confirmation before deleting."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

def get_size_mb(path: Path) -> float:
    """Get size of file or directory in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = 0
        for p in path.rglob('*'):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024 * 1024)
    return 0.0

def find_checkpoints(checkpoint_dir: Path) -> List[Tuple[Path, float]]:
    """Find all checkpoint files and their sizes."""
    checkpoints = []
    
    # Find .pth, .pt files.
    for pattern in ['*.pth', '*.pt']:
        for ckpt in checkpoint_dir.glob(pattern):
            size_mb = get_size_mb(ckpt)
            checkpoints.append((ckpt, size_mb))
    
    # Find checkpoint directories.
    for ckpt_dir in checkpoint_dir.glob('checkpoint_*'):
        if ckpt_dir.is_dir():
            size_mb = get_size_mb(ckpt_dir)
            checkpoints.append((ckpt_dir, size_mb))
    
    # Find automl trial directories.
    automl_dir = checkpoint_dir / 'checkpoints_automl'
    if automl_dir.exists():
        for trial_dir in automl_dir.glob('trial_*'):
            if trial_dir.is_dir():
                size_mb = get_size_mb(trial_dir)
                checkpoints.append((trial_dir, size_mb))
    
    return sorted(checkpoints, key=lambda x: x[1], reverse=True)

def cleanup_checkpoints(
    checkpoint_dir: Path,
    keep_best: bool = True,
    keep_last: bool = True,
    keep_recent: int = 0,
    dry_run: bool = True
) -> Tuple[int, float]:
    """Clean up checkpoint files...."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"FAIL Checkpoint directory {checkpoint_dir} does not exist")
        return 0, 0.0
    
    checkpoints = find_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"OK No checkpoints found in {checkpoint_dir}")
        return 0, 0.0
    
    print(f"\nFound {len(checkpoints)} checkpoint(s) in {checkpoint_dir}")
    print("=" * 70)
    
    # Identify what to keep.
    to_keep = set()
    
    if keep_best:
        best_model = checkpoint_dir / 'best_model.pt'
        if best_model.exists():
            to_keep.add(best_model)
            print(f"OK KEEPING: {best_model.name} ({get_size_mb(best_model):.2f} MB)")
    
    if keep_last:
        last_checkpoint = checkpoint_dir / 'last_checkpoint.pt'
        if last_checkpoint.exists():
            to_keep.add(last_checkpoint)
            print(f"OK KEEPING: {last_checkpoint.name} ({get_size_mb(last_checkpoint):.2f} MB)")
    
    # Keep most recent N checkpoints.
    if keep_recent > 0:
        recent = sorted(checkpoints, key=lambda x: x[0].stat().st_mtime, reverse=True)[:keep_recent]
        for ckpt, size in recent:
            to_keep.add(ckpt)
            print(f"OK KEEPING (recent): {ckpt.name} ({size:.2f} MB)")
    
    # Identify what to delete.
    to_delete = []
    total_size = 0.0
    
    for ckpt, size in checkpoints:
        if ckpt not in to_keep:
            to_delete.append((ckpt, size))
            total_size += size
    
    if not to_delete:
        print("\nOK Nothing to delete - all checkpoints are being kept")
        return 0, 0.0
    
    print(f"\n Will delete {len(to_delete)} checkpoint(s) ({total_size:.2f} MB):")
    print("-" * 70)
    for ckpt, size in to_delete:
        print(f"  - {ckpt.name}: {size:.2f} MB")
    
    if dry_run:
        print(f"\nWARNING  DRY RUN - No files deleted. Run with --execute to actually delete.")
        return 0, total_size
    
    # Confirm deletion.
    print(f"\nWARNING  About to delete {len(to_delete)} file(s) ({total_size:.2f} MB)")
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("FAIL Deletion cancelled")
        return 0, 0.0
    
    # Delete files.
    deleted_count = 0
    deleted_size = 0.0
    
    for ckpt, size in to_delete:
        try:
            if ckpt.is_file():
                ckpt.unlink()
            elif ckpt.is_dir():
                shutil.rmtree(ckpt)
            deleted_count += 1
            deleted_size += size
            print(f"OK Deleted: {ckpt.name}")
        except Exception as e:
            print(f"FAIL Failed to delete {ckpt.name}: {e}")
    
    print(f"\nOK Deleted {deleted_count} checkpoint(s), freed {deleted_size:.2f} MB")
    return deleted_count, deleted_size

def cleanup_logs(log_dir: Path = Path('logs'), dry_run: bool = True) -> Tuple[int, float]:
    """Clean up log files."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return 0, 0.0
    
    log_files = list(log_dir.glob('*.log'))
    if not log_files:
        return 0, 0.0
    
    total_size = sum(get_size_mb(f) for f in log_files)
    
    if dry_run:
        print(f"\n Found {len(log_files)} log file(s) ({total_size:.2f} MB)")
        print("WARNING  DRY RUN - No files deleted. Run with --execute to actually delete.")
        return 0, total_size
    
    response = input(f"\nDelete {len(log_files)} log file(s) ({total_size:.2f} MB)? (yes/no): ").strip().lower()
    if response != 'yes':
        return 0, 0.0
    
    deleted = 0
    for log_file in log_files:
        try:
            log_file.unlink()
            deleted += 1
        except Exception as e:
            print(f"FAIL Failed to delete {log_file.name}: {e}")
    
    return deleted, total_size

def cleanup_temp_files(dry_run: bool = True) -> Tuple[int, float]:
    """Clean up temporary files (__pycache__, .pyc, etc.)."""
    temp_patterns = ['__pycache__', '*.pyc', '*.pyo', '.pytest_cache', '.mypy_cache']
    deleted_count = 0
    deleted_size = 0.0
    
    for pattern in temp_patterns:
        for path in Path('.').rglob(pattern):
            if path.is_dir():
                size = get_size_mb(path)
                if not dry_run:
                    try:
                        shutil.rmtree(path)
                        deleted_count += 1
                        deleted_size += size
                    except Exception as e:
                        print(f"FAIL Failed to delete {path}: {e}")
                else:
                    deleted_size += size
            elif path.is_file():
                size = get_size_mb(path)
                if not dry_run:
                    try:
                        path.unlink()
                        deleted_count += 1
                        deleted_size += size
                    except Exception as e:
                        print(f"FAIL Failed to delete {path}: {e}")
                else:
                    deleted_size += size
    
    if deleted_size > 0:
        if dry_run:
            print(f"\n Would delete {deleted_count} temp file(s) ({deleted_size:.2f} MB)")
        else:
            print(f"\nOK Deleted {deleted_count} temp file(s), freed {deleted_size:.2f} MB")
    
    return deleted_count, deleted_size

def main():
    parser = argparse.ArgumentParser(description="Cleanup Colab/Cloud training artifacts")
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=Path('checkpoints'),
        help='Checkpoint directory to clean (default: checkpoints)'
    )
    parser.add_argument(
        '--keep-best',
        action='store_true',
        default=True,
        help='Keep best_model.pt (default: True)'
    )
    parser.add_argument(
        '--keep-last',
        action='store_true',
        default=True,
        help='Keep last_checkpoint.pt (default: True)'
    )
    parser.add_argument(
        '--keep-recent',
        type=int,
        default=0,
        help='Keep N most recent checkpoints (default: 0)'
    )
    parser.add_argument(
        '--clean-logs',
        action='store_true',
        help='Also clean log files'
    )
    parser.add_argument(
        '--clean-temp',
        action='store_true',
        help='Also clean temporary files (__pycache__, .pyc, etc.)'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete files (default: dry run)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Delete everything except best_model.pt and last_checkpoint.pt'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧹 Cloud Checkpoint Cleanup Script")
    print("=" * 70)
    
    if args.all:
        args.keep_recent = 0
        args.clean_logs = True
        args.clean_temp = True
    
    total_deleted = 0
    total_freed = 0.0
    
    # Clean checkpoints.
    deleted, freed = cleanup_checkpoints(
        args.checkpoint_dir,
        keep_best=args.keep_best,
        keep_last=args.keep_last,
        keep_recent=args.keep_recent,
        dry_run=not args.execute
    )
    total_deleted += deleted
    total_freed += freed
    
    # Clean logs.
    if args.clean_logs:
        deleted, freed = cleanup_logs(dry_run=not args.execute)
        total_deleted += deleted
        total_freed += freed
    
    # Clean temp files.
    if args.clean_temp:
        deleted, freed = cleanup_temp_files(dry_run=not args.execute)
        total_deleted += deleted
        total_freed += freed
    
    print("\n" + "=" * 70)
    if args.execute:
        print(f"OK Cleanup complete: {total_deleted} file(s) deleted, {total_freed:.2f} MB freed")
    else:
        print(f" Dry run complete: Would delete {total_deleted} file(s), free {total_freed:.2f} MB")
        print("   Run with --execute to actually delete files")
    print("=" * 70)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())


