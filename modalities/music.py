"""
Production-grade Music Encoder and Decoder for symbolic music generation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import ModalityEncoder, ModalityDecoder
from config.settings import MusicEncoderConfig


class MusicEncoder(ModalityEncoder):
    """
    Transformer encoder for music token sequences.
    
    Encodes MIDI-like token sequences into latent representations.
    """
    
    def __init__(self, config: MusicEncoderConfig) -> None:
        super().__init__()
        self._config = config
        
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        
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
        return "music"
    
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
            x: (B, T) token IDs
            padding_mask: (B, T) boolean mask, True = ignore
        
        Returns:
            (B, T, d_model) encoded representations
        """
        if x.dim() != 2:
            raise ValueError(f"MusicEncoder expects (B, T), got {x.shape}")
        
        batch_size, seq_len = x.shape
        
        # Truncate sequence if longer than max_seq_len
        if seq_len > self._config.max_seq_len:
            x = x[:, :self._config.max_seq_len]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :self._config.max_seq_len]
            seq_len = self._config.max_seq_len
        
        x = self.token_embedding(x)
        x = x * math.sqrt(self._config.d_model)
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        x = self.layer_norm(x)
        
        return x
    
    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class MusicDecoder(ModalityDecoder):
    """
    Autoregressive Transformer decoder for music generation.
    
    Can be conditioned on any modality encoder output via cross-attention.
    """
    
    def __init__(self, config: MusicEncoderConfig) -> None:
        super().__init__()
        self._config = config
        
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers,
        )
        
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    @property
    def modality_name(self) -> str:
        return "music"
    
    @property
    def vocab_size(self) -> int:
        return self._config.vocab_size
    
    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (B, T_tgt) target token IDs
            memory: (B, T_mem, d_model) encoder output
            tgt_mask: causal mask
            memory_mask: cross-attention mask
            tgt_padding_mask: (B, T_tgt)
            memory_padding_mask: (B, T_mem)
        
        Returns:
            (B, T_tgt, vocab_size) logits
        """
        batch_size, seq_len = tgt.shape
        
        tgt_emb = self.token_embedding(tgt)
        tgt_emb = tgt_emb * math.sqrt(self._config.d_model)
        
        positions = torch.arange(seq_len, device=tgt.device).unsqueeze(0)
        tgt_emb = tgt_emb + self.pos_embedding(positions)
        tgt_emb = self.dropout(tgt_emb)
        
        if tgt_mask is None:
            tgt_mask = self._causal_mask(seq_len, tgt.device)
        
        output = self.transformer(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        
        output = self.layer_norm(output)
        
        logits = self.lm_head(output)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive music generation.
        
        Args:
            memory: (B, T_mem, d_model) conditioning
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling threshold
            bos_token_id: beginning of sequence token
            eos_token_id: end of sequence token
        
        Returns:
            (B, L) generated token IDs
        """
        bos = bos_token_id or self._config.bos_token_id
        eos = eos_token_id or self._config.eos_token_id
        
        batch_size = memory.shape[0]
        device = memory.device
        
        generated = torch.full(
            (batch_size, 1), bos, dtype=torch.long, device=device
        )
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_new_tokens):
            if finished.all():
                break
            
            logits = self.forward(generated, memory)
            next_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                top_k = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k)
                threshold = values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            finished = finished | (next_token.squeeze(-1) == eos)
        
        return generated
    
    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
