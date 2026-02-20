"""
Production-grade Vision Transformer (ViT) encoder for images.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import ModalityEncoder
from config.settings import ImageEncoderConfig


class PatchEmbedding(nn.Module):
    """
    Image to patch embedding with optional overlapping patches.
    """
    
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
        overlap: int = 0,
    ) -> None:
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.overlap = overlap
        
        stride = patch_size - overlap
        self.num_patches_h = (image_size - overlap) // stride
        self.num_patches_w = (image_size - overlap) // stride
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=stride,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N, D) where N = num_patches
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ImageEncoder(ModalityEncoder):
    """
    Vision Transformer encoder for images.
    
    Features:
    - Patch embedding via Conv2d
    - [CLS] token for global representation
    - Learned positional embeddings
    - Pre-LayerNorm Transformer
    """
    
    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self._config = config
        
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            d_model=config.d_model,
        )
        
        num_patches = self.patch_embed.num_patches
        
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            num_tokens = num_patches + 1
        else:
            self.cls_token = None
            num_tokens = num_patches
        
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_tokens, config.d_model)
        )
        self.pos_drop = nn.Dropout(config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        self._init_weights()
    
    @property
    def modality_name(self) -> str:
        return "image"
    
    @property
    def output_dim(self) -> int:
        return self._config.d_model
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
        
        Returns:
            (B, N+1, d_model) where N = num_patches
            Index 0 is [CLS] token if use_cls_token=True
        """
        if x.dim() != 4:
            raise ValueError(f"ImageEncoder expects (B, C, H, W), got {x.shape}")
        
        batch_size = x.shape[0]
        
        patches = self.patch_embed(x)
        
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, patches], dim=1)
        else:
            tokens = patches
        
        tokens = tokens + self.pos_embedding
        tokens = self.pos_drop(tokens)
        
        tokens = self.transformer(tokens)
        
        tokens = self.layer_norm(tokens)
        
        return tokens
    
    def _init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Get the CLS token representation for an image."""
        output = self.forward(x)
        if self.cls_token is not None:
            return output[:, 0]
        return output.mean(dim=1)
