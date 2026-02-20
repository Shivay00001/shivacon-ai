"""
Cross-modal attention mechanisms.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from config.settings import FusionConfig


class CrossModalAttention(nn.Module):
    """
    Single cross-attention block between two modalities.
    
    Query from modality A, Key/Value from modality B.
    Includes gating mechanism to prevent modality starvation/attention hijacking.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.attn_drop = nn.Dropout(dropout)
        
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # Gating mechanism to prevent attention hijacking
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_padding_mask: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, D)
            context: (B, T_c, D)
            query_padding_mask: (B, T_q) True = ignore
            context_padding_mask: (B, T_c) True = ignore
        
        Returns:
            (B, T_q, D) updated query
        """
        q_norm = self.norm_q(query)
        c_norm = self.norm_kv(context)
        
        attn_out, _ = self.attn(
            q_norm,
            c_norm,
            c_norm,
            key_padding_mask=context_padding_mask,
        )
        
        # Apply modality balancing gate
        gate_weights = self.gate(torch.cat([query, attn_out], dim=-1))
        attn_gated = gate_weights * attn_out
        
        query = query + self.attn_drop(attn_gated)
        
        query = query + self.ffn(self.norm_ff(query))
        
        return query
    
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class CrossModalAttentionStack(nn.Module):
    """
    Stack of cross-attention layers.
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.layers = nn.ModuleList([
            CrossModalAttention(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_padding_mask: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, context, query_padding_mask, context_padding_mask)
        return query
