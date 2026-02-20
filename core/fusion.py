"""
Cross-modal fusion module for combining multiple modalities.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import FusionConfig


class CrossModalFusion(nn.Module):
    """
    Fuses multiple modality representations into a joint latent space.
    
    Strategy:
    1. Mean-pool each modality to get a single vector
    2. Add modality-type embeddings
    3. Apply inter-modality transformer
    4. Return fused vectors and global embedding
    """
    
    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        
        self.config = config
        self.latent_dim = config.latent_dim
        
        self._modality_to_idx: Dict[str, int] = {}
        self.modality_type_emb = nn.Embedding(
            num_embeddings=16,
            embedding_dim=config.latent_dim,
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_cross_attn_layers,
            enable_nested_tensor=False,
        )
        
        self.layer_norm = nn.LayerNorm(config.latent_dim)
        
        if config.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(config.latent_dim * 2, config.latent_dim),
                nn.Sigmoid(),
            )
    
    def _get_or_assign_idx(self, name: str) -> int:
        if name not in self._modality_to_idx:
            idx = len(self._modality_to_idx)
            if idx >= 16:
                raise RuntimeError("Maximum 16 modalities supported")
            self._modality_to_idx[name] = idx
        return self._modality_to_idx[name]
    
    def forward(
        self,
        modality_tensors: Dict[str, torch.Tensor],
    ) -> tuple:
        """
        Args:
            modality_tensors: {modality: (B, T, D)} projected modality outputs
        
        Returns:
            fused_dict: {modality: (B, 1, D)} per-modality fused vectors
            global_embedding: (B, D) global pooled embedding
        """
        if not modality_tensors:
            raise ValueError("No modalities provided to fusion")
        
        cls_tokens = []
        names = []
        
        for name, tensor in modality_tensors.items():
            if tensor.dim() == 2:
                pooled = tensor.unsqueeze(1)
            else:
                pooled = tensor.mean(dim=1, keepdim=True)
            
            idx = self._get_or_assign_idx(name)
            type_vec = self.modality_type_emb(
                torch.tensor(idx, device=tensor.device)
            )
            pooled = pooled + type_vec.view(1, 1, -1)
            
            cls_tokens.append(pooled)
            names.append(name)
        
        stacked = torch.cat(cls_tokens, dim=1)
        
        fused = self.transformer(stacked)
        fused = self.layer_norm(fused)
        
        fused_dict = {}
        for i, name in enumerate(names):
            fused_dict[name] = fused[:, i:i+1, :]
        
        global_embedding = fused.mean(dim=1)
        
        return fused_dict, global_embedding
