"""Memory Profiling Utilities for MaxSight

Provides memory profiling tools for debugging and optimization."""

import torch
from typing import Dict, Optional


def report_memory(device: Optional[str] = None) -> Dict[str, float]:
    """Report current memory usage.
    
    Args:
        device: Device to check ('cuda', 'mps', 'cpu', or None for auto-detect)
    
    Returns:
        Dictionary with memory statistics in MB"""
    stats = {}
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    if device == 'cuda':
        stats['allocated'] = torch.cuda.memory_allocated() / (1024 ** 2)  # MB.
        stats['reserved'] = torch.cuda.memory_reserved() / (1024 ** 2)  # MB.
        stats['max_allocated'] = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB.
        stats['max_reserved'] = torch.cuda.max_memory_reserved() / (1024 ** 2)  # MB.
    elif device == 'mps':
        stats['allocated'] = torch.mps.current_allocated_memory() / (1024 ** 2)  # MB.
        stats['driver_allocated'] = torch.mps.driver_allocated_memory() / (1024 ** 2)  # MB.
    else:
        # CPU - use system memory tracking if available.
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            stats['rss'] = mem_info.rss / (1024 ** 2)  # MB.
            stats['vms'] = mem_info.vms / (1024 ** 2)  # MB.
        except ImportError:
            stats['note'] = 'Install psutil for CPU memory profiling'
    
    return stats


def print_memory_summary(device: Optional[str] = None):
    """Print formatted memory summary."""
    stats = report_memory(device)
    print(f"\nMemory Usage ({device or 'auto'}):")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f} MB")
        else:
            print(f"  {key}: {value}")


def reset_peak_stats(device: Optional[str] = None):
    """Reset peak memory statistics."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    elif device == 'mps':
        # MPS doesn't have reset_peak_memory_stats, but we can track manually.
        pass



