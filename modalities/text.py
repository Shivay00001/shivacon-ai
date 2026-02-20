"""
Production-grade Text Encoder with Transformer architecture.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import ModalityEncoder
from config.settings import TextEncoderConfig


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better length generalization.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.shape[2] + offset
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :, offset:seq_len, :]
        sin = self.sin_cached[:, :, offset:seq_len, :]
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin


class TextEncoder(ModalityEncoder):
    """
    Production Transformer encoder for text.
    
    Features:
    - Pre-LayerNorm for training stability
    - RoPE or learned positional embeddings
    - Gradient checkpointing support
    - Efficient attention implementation
    """
    
    def __init__(
        self,
        config: TextEncoderConfig,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self._config = config
        self._use_rope = use_rope
        
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        
        if use_rope:
            self.pos_embedding = RotaryPositionalEmbedding(
                dim=config.d_model,
                max_seq_len=config.max_seq_len,
            )
        else:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
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
        return "text"
    
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
            x: Token IDs (B, T)
            padding_mask: Boolean mask (B, T), True = ignore
        
        Returns:
            Encoded representations (B, T, d_model)
        """
        if x.dim() != 2:
            raise ValueError(f"TextEncoder expects (B, T) input, got {x.shape}")
        
        batch_size, seq_len = x.shape
        
        # Truncate sequence if longer than max_seq_len
        if seq_len > self._config.max_seq_len:
            x = x[:, :self._config.max_seq_len]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :self._config.max_seq_len]
            seq_len = self._config.max_seq_len
        
        x = self.token_embedding(x)
        x = x * math.sqrt(self._config.d_model)
        
        if self._use_rope:
            x = self.pos_embedding(x)
        else:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)
        
        x = self.dropout(x)
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        x = self.layer_norm(x)
        
        return x
    
    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        if not self._use_rope:
            nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
