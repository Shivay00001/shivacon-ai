"""
Production-grade Audio Encoder with CNN + Transformer architecture.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import ModalityEncoder
from config.settings import AudioEncoderConfig


class AudioCNNFrontend(nn.Module):
    """
    CNN frontend for audio feature extraction.
    
    Processes mel-spectrogram through convolutional layers
    before the transformer.
    """
    
    def __init__(
        self,
        n_mels: int,
        d_model: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 2,
    ) -> None:
        super().__init__()
        
        layers = []
        in_channels = n_mels
        
        for i in range(num_layers):
            out_channels = d_model // 2 if i < num_layers - 1 else d_model
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_channels = out_channels
        
        self.conv = nn.Sequential(*layers)
        
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_mels, T) mel-spectrogram
        
        Returns:
            (B, T', d_model) features
        """
        x = self.conv(x)
        
        x = x.transpose(1, 2)
        
        x = self.out_proj(x)
        
        return x


class AudioEncoder(ModalityEncoder):
    """
    Audio encoder combining CNN frontend with Transformer.
    
    Architecture:
        Mel-spectrogram → CNN frontend → Positional encoding → Transformer
    """
    
    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self._config = config
        
        self.frontend = AudioCNNFrontend(
            n_mels=config.n_mels,
            d_model=config.d_model,
            num_layers=2,
        )
        
        max_audio_frames = config.max_frames // 4
        self.pos_embedding = nn.Embedding(max_audio_frames, config.d_model)
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
        return "audio"
    
    @property
    def output_dim(self) -> int:
        return self._config.d_model
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, n_mels, T) mel-spectrogram
            padding_mask: (B, T') mask for output frames
        
        Returns:
            (B, T', d_model) encoded features
        """
        if x.dim() != 3:
            raise ValueError(f"AudioEncoder expects (B, n_mels, T), got {x.shape}")
        
        features = self.frontend(x)
        
        batch_size, seq_len, _ = features.shape
        
        positions = torch.arange(seq_len, device=features.device).unsqueeze(0)
        features = features + self.pos_embedding(positions)
        features = self.pos_drop(features)
        
        if padding_mask is not None and padding_mask.shape[1] != seq_len:
            if self._config.use_spectrogram_augmentation:
                new_len = seq_len
                padding_mask = F.interpolate(
                    padding_mask.float().unsqueeze(1),
                    size=new_len,
                    mode="nearest",
                ).squeeze(1).bool()
        
        features = self.transformer(features, src_key_padding_mask=padding_mask)
        
        features = self.layer_norm(features)
        
        return features
    
    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
