"""
Learning rate schedulers for training.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    
    LR schedule:
        - Linear increase from 0 to base_lr during warmup
        - Cosine decay from base_lr to min_lr after warmup
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list:
        step = self.last_epoch
        
        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor
        
        return [base_lr * lr_factor for base_lr in self.base_lrs]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup followed by the base scheduler.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> list:
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        if self.base_scheduler is not None:
            return self.base_scheduler.get_last_lr()
        
        return self.base_lrs
    
    def step(self, epoch: Optional[int] = None) -> None:
        super().step(epoch)
        
        if self.last_epoch >= self.warmup_steps and self.base_scheduler is not None:
            self.base_scheduler.step()


def build_lr_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine_with_warmup",
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1,
) -> _LRScheduler:
    """
    Build a learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR ratio for cosine decay
    
    Returns:
        Configured LR scheduler
    """
    if scheduler_type == "cosine_with_warmup":
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )
    
    elif scheduler_type == "linear_with_warmup":
        base_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr_ratio,
            total_iters=total_steps - warmup_steps,
        )
        return LinearWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            base_scheduler=base_scheduler,
        )
    
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
