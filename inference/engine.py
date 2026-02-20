"""
Production-grade inference engine for MultiModal AI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from config.settings import Config, InferenceConfig
from core.multimodal_core import MultiModalCore
from modalities.music import MusicDecoder
from utils.device_utils import get_device, move_to_device

logger = logging.getLogger(__name__)


class MultiModalInference:
    """
    High-level inference wrapper for MultiModalCore.
    
    Handles:
    - Model loading and initialization
    - Device management
    - Batch inference
    - Music generation
    """
    
    def __init__(
        self,
        model: MultiModalCore,
        config: Config,
        music_decoder: Optional[MusicDecoder] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.inference_config = config.inference
        
        self.device = device or get_device(self.inference_config.device)
        self.model.to(self.device)
        self.model.eval()
        
        self.music_decoder = None
        if music_decoder is not None:
            self.music_decoder = music_decoder.to(self.device)
            self.music_decoder.eval()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ) -> "MultiModalInference":
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Config (loaded from checkpoint dir if not provided)
            device: Device to load model on
        
        Returns:
            Configured MultiModalInference instance
        """
        checkpoint_path = Path(checkpoint_path)
        
        if config is None:
            config_path = checkpoint_path.parent / "config.json"
            if config_path.exists():
                from config.settings import Config
                config = Config.load(config_path)
            else:
                config = Config()
        
        model = cls._build_model(config)
        
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        
        music_decoder = None
        if "music" in config.fusion.active_modalities:
            from modalities.music import MusicDecoder
            music_decoder = MusicDecoder(config.music)
        
        device = device or config.inference.device
        return cls(model, config, music_decoder, get_device(device))
    
    @staticmethod
    def _build_model(config: Config) -> MultiModalCore:
        from core.multimodal_core import MultiModalCore
        from modalities.text import TextEncoder
        from modalities.image import ImageEncoder
        from modalities.audio import AudioEncoder
        from modalities.video import VideoEncoder
        from modalities.music import MusicEncoder
        
        core = MultiModalCore(config.fusion)
        
        if "text" in config.fusion.active_modalities:
            core.register_encoder(TextEncoder(config.text))
        if "image" in config.fusion.active_modalities:
            core.register_encoder(ImageEncoder(config.image))
        if "audio" in config.fusion.active_modalities:
            core.register_encoder(AudioEncoder(config.audio))
        if "video" in config.fusion.active_modalities:
            core.register_encoder(VideoEncoder(config.video))
        if "music" in config.fusion.active_modalities:
            core.register_encoder(MusicEncoder(config.music))
        
        return core
    
    @torch.no_grad()
    def encode(
        self,
        inputs: Dict[str, Dict[str, Any]],
        active_modalities: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Encode inputs to latent representations.
        
        Args:
            inputs: Nested dict of modality inputs
            active_modalities: Modalities to process
        
        Returns:
            Dict with modality outputs and global_embedding
        """
        inputs = move_to_device(inputs, self.device)
        
        outputs = self.model(inputs, active_modalities=active_modalities)
        
        return outputs
    
    @torch.no_grad()
    def cross_attend(
        self,
        query_modality: str,
        context_modality: str,
        inputs: Dict[str, Dict[str, Any]],
    ) -> Tensor:
        """
        Apply cross-attention between modalities.
        
        Args:
            query_modality: Query modality name
            context_modality: Context modality name
            inputs: Nested dict of inputs
        
        Returns:
            Cross-attended query tensor
        """
        inputs = move_to_device(inputs, self.device)
        
        outputs = self.model(inputs)
        
        latent_tensors = {
            k: v for k, v in outputs.items()
            if k not in ("fused", "global_embedding")
        }
        
        return self.model.cross_attend(
            query_modality,
            context_modality,
            latent_tensors,
        )
    
    @torch.no_grad()
    def generate_music(
        self,
        conditioning_inputs: Dict[str, Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        Generate music conditioned on any modality.
        
        Args:
            conditioning_inputs: Input for conditioning
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        
        Returns:
            Generated music token sequence (B, L)
        """
        if self.music_decoder is None:
            raise RuntimeError("MusicDecoder not available")
        
        max_new_tokens = max_new_tokens or self.inference_config.music_max_tokens
        temperature = temperature or self.inference_config.music_temperature
        top_k = top_k or self.inference_config.music_top_k
        
        inputs = move_to_device(conditioning_inputs, self.device)
        outputs = self.model(inputs)
        
        memory = outputs["global_embedding"].unsqueeze(1)
        
        generated = self.music_decoder.generate(
            memory=memory,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        
        # Security: Inject micro-noise/truncate low bits to destroy cross-modal steganography
        if generated.is_floating_point():
            noise = torch.randn_like(generated) * 1e-4
            generated = generated + noise
            # Truncate to precision to further destroy covert encoded payloads
            generated = generated.to(torch.float16).to(generated.dtype)
        
        return generated
    
    def get_similarity(
        self,
        inputs_a: Dict[str, Dict[str, Any]],
        inputs_b: Dict[str, Dict[str, Any]],
    ) -> Tensor:
        """
        Compute similarity between two multi-modal inputs.
        
        Args:
            inputs_a: First input
            inputs_b: Second input
        
        Returns:
            Cosine similarity scores (B,)
        """
        import torch.nn.functional as F
        
        outputs_a = self.encode(inputs_a)
        outputs_b = self.encode(inputs_b)
        
        emb_a = F.normalize(outputs_a["global_embedding"], dim=-1)
        emb_b = F.normalize(outputs_b["global_embedding"], dim=-1)
        
        return (emb_a * emb_b).sum(dim=-1)
    
    def batch_encode(
        self,
        batch_inputs: List[Dict[str, Dict[str, Any]]],
        batch_size: int = 32,
    ) -> List[Dict[str, Tensor]]:
        """
        Encode a large batch of inputs.
        
        Args:
            batch_inputs: List of input dicts
            batch_size: Processing batch size
        
        Returns:
            List of output dicts
        """
        results = []
        
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i + batch_size]
            
            combined = {}
            for sample in batch:
                for mod, data in sample.items():
                    if mod not in combined:
                        combined[mod] = []
                    combined[mod].append(data)
            
            stacked = {}
            for mod, data_list in combined.items():
                if "x" in data_list[0]:
                    tensors = [d["x"] for d in data_list]
                    if isinstance(tensors[0], Tensor):
                        # Handle different tensor shapes
                        if tensors[0].dim() == 2:
                            # (B, T) or (T,) - pad to same length if needed
                            max_len = max(t.shape[1] if t.dim() > 1 else t.shape[0] for t in tensors)
                            padded = []
                            for t in tensors:
                                if t.dim() == 1:
                                    t = t.unsqueeze(0)
                                if t.shape[1] < max_len:
                                    pad = torch.zeros(t.shape[0], max_len - t.shape[1], dtype=t.dtype)
                                    t = torch.cat([t, pad], dim=1)
                                padded.append(t)
                            stacked[mod] = {"x": torch.cat(padded, dim=0)}
                        else:
                            stacked[mod] = {"x": torch.cat(tensors, dim=0)}
            
            if stacked:
                outputs = self.encode(stacked)
                
                batch_size_actual = next(iter(outputs.values())).shape[0]
                for j in range(batch_size_actual):
                    sample_output = {
                        k: v[j] if isinstance(v, Tensor) else v
                        for k, v in outputs.items()
                    }
                    results.append(sample_output)
                
                # Mitigate KV-Cache Fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
