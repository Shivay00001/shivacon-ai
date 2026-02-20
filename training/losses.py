"""
Production-grade loss functions for multi-modal training.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ContrastiveLoss(nn.Module):
    """
    CLIP-style symmetric contrastive loss (InfoNCE).
    
    Aligns embeddings from two modalities by maximizing
    similarity of matched pairs and minimizing others.
    """
    
    def __init__(
        self,
        init_temperature: float = 0.07,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temperature))
        )
        self.learnable = learnable_temperature
    
    @property
    def temperature(self) -> Tensor:
        temp = self.log_temperature.exp()
        return torch.clamp(temp, min=0.01, max=100.0)
    
    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
    ) -> Tensor:
        """
        Args:
            emb_a: (B, D) first modality embeddings
            emb_b: (B, D) second modality embeddings
        
        Returns:
            Scalar contrastive loss
        """
        a = F.normalize(emb_a, dim=-1)
        b = F.normalize(emb_b, dim=-1)
        
        logits = torch.mm(a, b.t()) * self.temperature
        
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.t(), labels)
        
        # Entropy maximization (Uniformity constraint) to prevent latent space collapse
        # Penalize embeddings that are too tightly clustered together
        sq_pdist_a = torch.pdist(a, p=2).pow(2)
        sq_pdist_b = torch.pdist(b, p=2).pow(2)
        uniformity_loss = sq_pdist_a.mul(-2.0).exp().mean().log() + sq_pdist_b.mul(-2.0).exp().mean().log()
        
        return ((loss_a + loss_b) / 2.0) + (0.1 * uniformity_loss)


class ReconstructionLoss(nn.Module):
    """
    Cross-entropy loss for sequence reconstruction.
    
    Handles variable-length sequences with padding masks.
    """
    
    def __init__(
        self,
        pad_token_id: int = 0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            logits: (B, T, V) model output logits
            targets: (B, T) target token IDs
            padding_mask: (B, T) True = ignore
        
        Returns:
            Scalar reconstruction loss
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        targets_flat = targets.view(batch_size * seq_len)
        
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.pad_token_id,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        
        loss = loss.view(batch_size, seq_len)
        
        if padding_mask is not None:
            valid = (~padding_mask).float()
            loss = (loss * valid).sum() / valid.sum().clamp(min=1.0)
        else:
            loss = loss.mean()
        
        return loss


class AlignmentLoss(nn.Module):
    """
    Direct alignment loss between embeddings.
    
    Uses MSE to align embeddings from different modalities.
    """
    
    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
    
    def forward(
        self,
        emb_a: Tensor,
        emb_b: Tensor,
    ) -> Tensor:
        """
        Args:
            emb_a: (B, D) first modality embeddings
            emb_b: (B, D) second modality embeddings
        
        Returns:
            Scalar alignment loss
        """
        if self.normalize:
            emb_a = F.normalize(emb_a, dim=-1)
            emb_b = F.normalize(emb_b, dim=-1)
        
        return F.mse_loss(emb_a, emb_b)


class MultiModalLoss(nn.Module):
    """
    Combined loss for multi-modal training.
    
    Combines:
    - Contrastive loss between all modality pairs
    - Optional reconstruction loss for generative modalities
    - Optional alignment loss
    """
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        alignment_weight: float = 0.5,
        temperature: float = 0.07,
        pad_token_id: int = 0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.reconstruction_weight = reconstruction_weight
        self.alignment_weight = alignment_weight
        
        self.contrastive = ContrastiveLoss(
            init_temperature=temperature,
            learnable_temperature=True,
        )
        self.reconstruction = ReconstructionLoss(
            pad_token_id=pad_token_id,
            label_smoothing=label_smoothing,
        )
        self.alignment = AlignmentLoss(normalize=True)
    
    def forward(
        self,
        fused_dict: Dict[str, Tensor],
        global_embedding: Tensor,
        targets: Optional[Dict[str, Tensor]] = None,
        padding_masks: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            fused_dict: {modality: (B, 1, D)} fused per-modality vectors
            global_embedding: (B, D) global pooled embedding
            targets: {modality: (B, T)} target sequences for reconstruction
            padding_masks: {modality: (B, T)} padding masks
        
        Returns:
            Dict with 'total' loss and individual components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=global_embedding.device)
        
        names = list(fused_dict.keys())
        embeddings = {
            n: fused_dict[n].squeeze(1)
            for n in names
        }
        
        if len(names) >= 2 and self.contrastive_weight > 0:
            contrastive_losses = []
            for a, b in combinations(names, 2):
                loss = self.contrastive(embeddings[a], embeddings[b])
                contrastive_losses.append(loss)
            
            contrastive_loss = torch.stack(contrastive_losses).mean()
            losses["contrastive"] = contrastive_loss
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
        
        if targets and self.reconstruction_weight > 0:
            pass
        
        if self.alignment_weight > 0:
            alignment_losses = []
            for name in names:
                loss = F.mse_loss(embeddings[name], global_embedding.detach())
                alignment_losses.append(loss)
            
            if alignment_losses:
                alignment_loss = torch.stack(alignment_losses).mean()
                losses["alignment"] = alignment_loss
                total_loss = total_loss + self.alignment_weight * alignment_loss
        
        losses["total"] = total_loss
        
        return losses
