"""GPU memory reporting for router intervention experiments."""
from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def _bytes_to_gib(b: float) -> float:
    return b / (1024**3)


def get_gpu_memory_gib(device: int | None = None) -> Tuple[float, float, float]:
    """Return (total_gib, allocated_gib, free_gib) for the given device, or (0, 0, 0) if not CUDA."""
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    if device is None:
        device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    # "Free" as in not allocated; total - reserved is a better proxy for what we can still allocate
    free = total - reserved
    return _bytes_to_gib(total), _bytes_to_gib(allocated), _bytes_to_gib(free)


def get_all_gpus_allocated_gib() -> float:
    """Return sum of allocated GiB across all CUDA devices (for device_map='auto' models)."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(
        get_gpu_memory_gib(d)[1] for d in range(torch.cuda.device_count())
    )


def log_gpu_memory(label: str = "GPU memory") -> None:
    """Log GPU memory: current device (total, allocated, free) and total allocated across all devices."""
    if not torch.cuda.is_available():
        logger.info("%s: CUDA not available", label)
        return
    total, allocated, free = get_gpu_memory_gib()
    logger.info(
        "%s: total=%.2f GiB, allocated=%.2f GiB, free≈%.2f GiB",
        label, total, allocated, free,
    )
    if torch.cuda.device_count() > 1:
        total_alloc = get_all_gpus_allocated_gib()
        logger.info(
            "%s: total allocated across %d devices = %.2f GiB",
            label, torch.cuda.device_count(), total_alloc,
        )


def require_gpu_memory_gib(need_gib: float, label: str = "Operation") -> bool:
    """
    Check if at least need_gib GiB is free on GPU. Log and return True if OK, False otherwise.
    Use before model.to("cuda") or similar to avoid OOM.
    """
    total, allocated, free = get_gpu_memory_gib()
    if total == 0:
        logger.warning("CUDA not available; cannot check memory")
        return True
    if free < need_gib:
        logger.warning(
            "%s needs ~%.2f GiB but only ~%.2f GiB free (total=%.2f GiB, allocated=%.2f GiB). "
            "Try smaller batch_size/seq_len/num_samples or use --quantization 8bit.",
            label, need_gib, free, total, allocated,
        )
        return False
    logger.info("%s: need ~%.2f GiB, free ~%.2f GiB", label, need_gib, free)
    return True
