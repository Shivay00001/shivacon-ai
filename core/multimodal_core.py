"""
MultiModalCore: Central orchestrator for multi-modal intelligence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from core.base import ModalityEncoder
from core.projector import ModalityProjector
from core.cross_attention import CrossModalAttentionStack
from core.fusion import CrossModalFusion
from config.settings import FusionConfig


class MultiModalCore(nn.Module):
    """
    Central multi-modal intelligence core.
    
    Responsibilities:
    1. Register and manage modality encoders
    2. Project encoder outputs to shared latent space
    3. Enable cross-modal attention between modalities
    4. Fuse modalities into joint representations
    
    Usage:
        core = MultiModalCore(config.fusion)
        core.register_encoder(TextEncoder(config.text))
        core.register_encoder(ImageEncoder(config.image))
        
        outputs = core({
            "text": {"x": token_ids, "padding_mask": mask},
            "image": {"x": image_tensor},
        })
        
        global_emb = outputs["global_embedding"]
    """
    
    def __init__(self, fusion_config: FusionConfig) -> None:
        super().__init__()
        
        self._config = fusion_config
        self._latent_dim = fusion_config.latent_dim
        
        self._encoders: nn.ModuleDict = nn.ModuleDict()
        self._projectors: nn.ModuleDict = nn.ModuleDict()
        self._cross_attn_stacks: nn.ModuleDict = nn.ModuleDict()
        
        self.fusion = CrossModalFusion(fusion_config)
    
    def register_encoder(self, encoder: ModalityEncoder) -> "MultiModalCore":
        """
        Register a modality encoder with automatic projection.
        
        Args:
            encoder: ModalityEncoder instance
        
        Returns:
            self (for chaining)
        """
        name = encoder.modality_name
        
        if name in self._encoders:
            raise ValueError(f"Encoder '{name}' already registered")
        
        self._encoders[name] = encoder
        
        self._projectors[name] = ModalityProjector(
            encoder_dim=encoder.output_dim,
            latent_dim=self._latent_dim,
            dropout=self._config.dropout,
        )
        
        return self
    
    def unregister_encoder(self, name: str) -> "MultiModalCore":
        """Remove a registered encoder."""
        if name not in self._encoders:
            raise KeyError(f"Encoder '{name}' not registered")
        
        del self._encoders[name]
        del self._projectors[name]
        
        keys_to_remove = [
            k for k in self._cross_attn_stacks
            if k.startswith(f"{name}->") or k.endswith(f"->{name}")
        ]
        for k in keys_to_remove:
            del self._cross_attn_stacks[k]
        
        return self
    
    @property
    def registered_modalities(self) -> List[str]:
        return list(self._encoders.keys())
    
    def add_cross_attention(
        self,
        query_modality: str,
        context_modality: str,
    ) -> "MultiModalCore":
        """
        Add cross-attention from query to context modality.
        
        Args:
            query_modality: Name of query modality
            context_modality: Name of context modality
        
        Returns:
            self (for chaining)
        """
        for mod in (query_modality, context_modality):
            if mod not in self._encoders:
                raise ValueError(f"Modality '{mod}' not registered")
        
        key = f"{query_modality}->{context_modality}"
        
        if key in self._cross_attn_stacks:
            return self
        
        self._cross_attn_stacks[key] = CrossModalAttentionStack(
            num_layers=self._config.num_cross_attn_layers,
            d_model=self._latent_dim,
            num_heads=self._config.num_heads,
            dim_feedforward=self._config.dim_feedforward,
            dropout=self._config.dropout,
        )
        
        return self
    
    def cross_attend(
        self,
        query_modality: str,
        context_modality: str,
        latent_tensors: Dict[str, torch.Tensor],
        query_padding_mask: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-attention between modalities.
        
        Args:
            query_modality: Name of query modality
            context_modality: Name of context modality
            latent_tensors: Dict of projected tensors
            query_padding_mask: (B, T_q) mask
            context_padding_mask: (B, T_c) mask
        
        Returns:
            Updated query tensor (B, T_q, D)
        """
        key = f"{query_modality}->{context_modality}"
        
        if key not in self._cross_attn_stacks:
            raise KeyError(f"No cross-attention for '{key}'")
        
        if query_modality not in latent_tensors:
            raise KeyError(f"'{query_modality}' not in latent_tensors")
        if context_modality not in latent_tensors:
            raise KeyError(f"'{context_modality}' not in latent_tensors")
        
        stack = self._cross_attn_stacks[key]
        
        return stack(
            query=latent_tensors[query_modality],
            context=latent_tensors[context_modality],
            query_padding_mask=query_padding_mask,
            context_padding_mask=context_padding_mask,
        )
    
    def forward(
        self,
        inputs: Dict[str, Dict[str, Any]],
        active_modalities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full forward pass through the multi-modal core.
        
        Args:
            inputs: Nested dict of modality inputs
                {
                    "text": {"x": tensor, "padding_mask": mask, ...},
                    "image": {"x": tensor},
                    ...
                }
            active_modalities: Subset of modalities to process
        
        Returns:
            {
                "text": (B, T, D),
                "image": (B, T, D),
                ...
                "fused": {modality: (B, 1, D)},
                "global_embedding": (B, D),
            }
        """
        if active_modalities is None:
            active_modalities = [m for m in inputs if m in self._encoders]
        
        if not active_modalities:
            raise ValueError("No active modalities found in inputs")
        
        latent_tensors: Dict[str, torch.Tensor] = {}
        padding_masks: Dict[str, Optional[torch.Tensor]] = {}
        
        for name in active_modalities:
            if name not in self._encoders:
                raise KeyError(f"No encoder registered for '{name}'")
            
            encoder = self._encoders[name]
            projector = self._projectors[name]
            modal_inputs = inputs[name]
            
            x = modal_inputs["x"]
            kwargs = {k: v for k, v in modal_inputs.items() if k != "x"}
            padding_mask = kwargs.pop("padding_mask", None)
            
            encoded = encoder(x, padding_mask=padding_mask, **kwargs)
            projected = projector(encoded)
            
            latent_tensors[name] = projected
            padding_masks[name] = padding_mask
        
        fused_dict, global_embedding = self.fusion(latent_tensors)
        
        output: Dict[str, Any] = {**latent_tensors}
        output["fused"] = fused_dict
        output["global_embedding"] = global_embedding
        
        return output
    
    def get_encoder(self, name: str) -> ModalityEncoder:
        """Get a registered encoder by name."""
        if name not in self._encoders:
            raise KeyError(f"Encoder '{name}' not registered")
        return self._encoders[name]
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
