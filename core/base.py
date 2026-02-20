"""
Base classes for modality encoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ModalityEncoder(ABC, nn.Module):
    """
    Abstract base class for all modality encoders.
    
    Every encoder must:
    - Define modality_name (unique identifier)
    - Define output_dim (latent dimension)
    - Implement forward() returning (B, T, D) tensors
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    @property
    @abstractmethod
    def modality_name(self) -> str:
        ...
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...
    
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...
    
    def extra_repr(self) -> str:
        return f"modality={self.modality_name}, output_dim={self.output_dim}"


class ModalityDecoder(ABC, nn.Module):
    """
    Abstract base class for modality decoders (generative models).
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    @property
    @abstractmethod
    def modality_name(self) -> str:
        ...
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...
    
    @abstractmethod
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...
    
    @abstractmethod
    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...
