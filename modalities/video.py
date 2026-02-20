"""
Production-grade Video Encoder with factorized spatial-temporal attention.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import ModalityEncoder
from config.settings import VideoEncoderConfig


class PatchEmbedding3D(nn.Module):
    """
    3D patch embedding for video (tubelet embedding).
    """
    
    def __init__(
        self,
        num_frames: int,
        image_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
        tubelet_size: int = 2,
    ) -> None:
        super().__init__()
        
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        
        self.num_temporal_patches = num_frames // tubelet_size
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_patches = self.num_temporal_patches * self.num_spatial_patches
        
        self.proj = nn.Conv3d(
            in_channels,
            d_model,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, N_t * N_s, D)
        """
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VideoEncoder(ModalityEncoder):
    """
    Factorized spatial-temporal video encoder.
    
    Architecture:
        1. Spatial attention within each frame
        2. Temporal attention across frames
        3. Global CLS token aggregation
    """
    
    def __init__(self, config: VideoEncoderConfig) -> None:
        super().__init__()
        self._config = config
        
        self.patch_proj = nn.Conv2d(
            config.in_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        
        num_patches = (config.image_size // config.patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        
        self.spatial_pos_emb = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.d_model)
        )
        
        self.temporal_pos_emb = nn.Parameter(
            torch.zeros(1, max(config.num_frames, 16), config.d_model)
        )
        
        self.pos_drop = nn.Dropout(config.dropout)
        
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer,
            num_layers=config.spatial_layers,
            enable_nested_tensor=False,
        )
        
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=config.temporal_layers,
            enable_nested_tensor=False,
        )
        
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        self._init_weights()
    
    @property
    def modality_name(self) -> str:
        return "video"
    
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
            x: (B, T, C, H, W) video tensor
        
        Returns:
            (B, 1 + T*N, d_model) where N = num_spatial_patches
        """
        if x.dim() != 5:
            raise ValueError(f"VideoEncoder expects (B, T, C, H, W), got {x.shape}")
        
        batch_size, num_frames, channels, height, width = x.shape
        
        # Calculate actual number of patches from input dimensions
        num_patches_h = height // self._config.patch_size
        num_patches_w = width // self._config.patch_size
        num_spatial_patches = num_patches_h * num_patches_w
        
        x_flat = x.reshape(batch_size * num_frames, channels, height, width)
        patches = self.patch_proj(x_flat)
        patches = patches.flatten(2).transpose(1, 2)
        
        # Get actual d_model from the patches
        actual_d_model = patches.shape[-1]
        
        patches = patches.reshape(batch_size, num_frames, num_spatial_patches, actual_d_model)
        
        cls_spatial = self.cls_token.expand(batch_size * num_frames, -1, -1)
        patches_flat = patches.reshape(batch_size * num_frames, num_spatial_patches, actual_d_model)
        tokens = torch.cat([cls_spatial, patches_flat], dim=1)
        
        # Use only the needed portion of positional embedding
        pos_emb = self.spatial_pos_emb[:, :num_spatial_patches + 1, :]
        tokens = self.pos_drop(tokens + pos_emb)
        
        tokens = self.spatial_transformer(tokens)
        tokens = tokens.reshape(batch_size, num_frames, num_spatial_patches + 1, actual_d_model)
        
        patch_tokens = tokens[:, :, 1:, :]
        # Use only needed portion of temporal positional embedding
        temporal_emb = self.temporal_pos_emb[:, :num_frames, :].unsqueeze(2)
        patch_tokens = patch_tokens + temporal_emb
        
        patch_tokens = patch_tokens.permute(0, 2, 1, 3).contiguous()
        patch_tokens = patch_tokens.view(
            batch_size * num_spatial_patches, num_frames, -1
        )
        patch_tokens = self.temporal_transformer(patch_tokens)
        patch_tokens = patch_tokens.view(
            batch_size, num_spatial_patches, num_frames, -1
        )
        patch_tokens = patch_tokens.permute(0, 2, 1, 3).contiguous()
        
        patch_tokens_flat = patch_tokens.view(batch_size, num_frames * num_spatial_patches, -1)
        
        cls_global = tokens[:, :, 0, :].mean(dim=1, keepdim=True)
        
        output = torch.cat([cls_global, patch_tokens_flat], dim=1)
        
        return self.layer_norm(output)
    
    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.spatial_pos_emb, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
