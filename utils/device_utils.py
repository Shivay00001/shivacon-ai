"""
Device and tensor utilities for cross-platform support.
"""

from __future__ import annotations

import torch
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    resolved = torch.device(device)
    logger.info(f"Using device: {resolved}")
    
    if resolved.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    return resolved


def move_to_device(
    data: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {
            k: move_to_device(v, device, non_blocking)
            for k, v in data.items()
        }
    elif isinstance(data, (list, tuple)):
        moved = [move_to_device(item, device, non_blocking) for item in data]
        return type(data)(moved)
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    else:
        return data


def get_autocast_context(
    device: torch.device,
    enabled: bool = True,
    dtype: Optional[torch.dtype] = None,
):
    if not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    
    if dtype is None:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def get_gradient_scaler(enabled: bool = True):
    return torch.amp.GradScaler(enabled=enabled)


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"allocated_gb": 0.0, "cached_gb": 0.0}
    
    allocated = torch.cuda.memory_allocated(device) / 1e9
    cached = torch.cuda.memory_reserved(device) / 1e9
    return {"allocated_gb": allocated, "cached_gb": cached}


def estimate_model_memory(model: torch.nn.Module, batch_size: int = 1) -> float:
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    param_memory = (param_size + buffer_size) / 1e9
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gradient_memory = trainable * 4 / 1e9 if trainable > 0 else 0
    
    optimizer_memory = trainable * 8 / 1e9
    
    total = param_memory + gradient_memory + optimizer_memory
    return total * batch_size
