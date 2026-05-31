"""Distributed training helpers for DDP and FSDP."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_DIST_INITIALIZED = False


def init_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize the process group from launcher env vars."""
    global _DIST_INITIALIZED
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size <= 1:
        return rank, world_size, local_rank
    if not _DIST_INITIALIZED:
        import torch.distributed as dist

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        _DIST_INITIALIZED = True
        logger.info(
            "distributed_initialized rank=%d world_size=%d local_rank=%d",
            rank,
            world_size,
            local_rank,
        )
    return rank, world_size, local_rank


def is_main_process() -> bool:
    """Return True on rank 0 or when distributed is not initialized."""
    try:
        import torch.distributed as dist
    except ImportError:
        return True
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def should_checkpoint() -> bool:
    """Only rank 0 writes checkpoints under distributed training."""
    return is_main_process()


def barrier() -> None:
    """Synchronize all ranks when distributed training is active."""
    try:
        import torch.distributed as dist
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def distributed_sampler(dataset: Any, rank: int, world_size: int, *, shuffle: bool = True) -> Any:
    """Build a DistributedSampler for the given dataset."""
    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)


def wrap_ddp(model: nn.Module, local_rank: int) -> nn.Module:
    """Wrap a model in DistributedDataParallel for single-node multi-GPU training."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    device_ids = [local_rank] if torch.cuda.is_available() else None
    return DDP(model, device_ids=device_ids)


def wrap_fsdp(model: nn.Module) -> nn.Module:
    """Wrap MaxSightCNN in FSDP with block-level auto-wrap when available."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    auto_wrap_policy = None
    try:
        from ml.models.backbone.hybrid_backbone import HybridBlock

        auto_wrap_policy = transformer_auto_wrap_policy({HybridBlock: set()})
    except Exception:
        logger.warning("fsdp_auto_wrap_unavailable — wrapping root module only")
    return FSDP(model, auto_wrap_policy=auto_wrap_policy)
