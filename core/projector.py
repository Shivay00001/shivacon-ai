"""
Modality projector for aligning encoder outputs to shared latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.settings import FusionConfig


class ModalityProjector(nn.Module):
    """
    Projects encoder outputs to shared latent dimension.
    
    Uses a 2-layer MLP with GELU for non-linear transformation.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        
        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, encoder_dim)
        Returns:
            (B, T, latent_dim)
        """
        return self.proj(x)
    
    def _init_weights(self) -> None:
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
