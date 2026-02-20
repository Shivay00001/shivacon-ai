"""
Inference pipeline for production deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from inference.engine import MultiModalInference
from data.tokenizer import BPETokenizer
from data.image_processor import ImageProcessor, ImageProcessorConfig
from data.audio_processor import AudioProcessor, AudioProcessorConfig
from data.video_processor import VideoProcessor, VideoProcessorConfig
from data.music_processor import MusicProcessor, MusicProcessorConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for inference pipeline."""
    max_text_length: int = 512
    max_audio_frames: int = 512
    max_video_frames: int = 16
    max_music_tokens: int = 1024
    
    music_max_tokens: int = 512
    music_temperature: float = 0.9
    music_top_k: int = 50


class InferencePipeline:
    """
    Complete inference pipeline with preprocessing.
    
    Handles:
    - Text tokenization
    - Image preprocessing
    - Audio feature extraction
    - Video frame extraction
    - Music token processing
    """
    
    def __init__(
        self,
        inference_engine: MultiModalInference,
        tokenizer: Optional[BPETokenizer] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        self.engine = inference_engine
        self.config = pipeline_config or PipelineConfig()
        
        self.tokenizer = tokenizer
        
        config = inference_engine.config
        
        self.image_processor = None
        if "image" in config.fusion.active_modalities:
            self.image_processor = ImageProcessor(ImageProcessorConfig(
                image_size=config.image.image_size,
            ))
        
        self.audio_processor = None
        if "audio" in config.fusion.active_modalities:
            self.audio_processor = AudioProcessor(AudioProcessorConfig(
                sample_rate=config.audio.sample_rate,
                n_mels=config.audio.n_mels,
                max_frames=config.audio.max_frames,
            ))
        
        self.video_processor = None
        if "video" in config.fusion.active_modalities:
            self.video_processor = VideoProcessor()
        
        self.music_processor = None
        if "music" in config.fusion.active_modalities:
            self.music_processor = MusicProcessor()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        tokenizer_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> "InferencePipeline":
        """
        Create pipeline from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer
            device: Target device
        
        Returns:
            Configured InferencePipeline
        """
        engine = MultiModalInference.from_checkpoint(
            checkpoint_path,
            device=device,
        )
        
        tokenizer = None
        if tokenizer_path and tokenizer_path.exists():
            tokenizer = BPETokenizer.load(tokenizer_path)
        
        return cls(engine, tokenizer)
    
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text input."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        tokens = self.tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
            max_length=self.config.max_text_length,
        )
        
        return {"x": torch.tensor(tokens, dtype=torch.long).unsqueeze(0)}
    
    def process_image(
        self,
        image: Union[str, Path, bytes],
    ) -> Dict[str, torch.Tensor]:
        """Process image input."""
        if self.image_processor is None:
            raise RuntimeError("Image processor not initialized")
        
        tensor = self.image_processor.process(image, is_training=False)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return {"x": tensor}
    
    def process_audio(
        self,
        audio: Union[str, Path, bytes],
    ) -> Dict[str, torch.Tensor]:
        """Process audio input."""
        if self.audio_processor is None:
            raise RuntimeError("Audio processor not initialized")
        
        tensor = self.audio_processor.process(
            audio,
            is_training=False,
            max_frames=self.audio_processor.config.max_frames,
        )
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return {"x": tensor}
    
    def process_video(
        self,
        video: Union[str, Path],
    ) -> Dict[str, torch.Tensor]:
        """Process video input."""
        if self.video_processor is None:
            raise RuntimeError("Video processor not initialized")
        
        tensor = self.video_processor.process(video, is_training=False)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)
        return {"x": tensor}
    
    def process_music(
        self,
        music: Union[List[Dict], List[int]],
    ) -> Dict[str, torch.Tensor]:
        """Process music input."""
        if self.music_processor is None:
            raise RuntimeError("Music processor not initialized")
        
        tensor = self.music_processor.process(
            music,
            add_bos=True,
            add_eos=True,
            max_length=self.config.max_music_tokens,
        )
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return {"x": tensor}
    
    def encode_text_image(
        self,
        text: str,
        image: Union[str, Path, bytes],
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text and image together.
        
        Args:
            text: Text input
            image: Image input
        
        Returns:
            Encoding outputs including global embedding
        """
        inputs = {
            "text": self.process_text(text),
            "image": self.process_image(image),
        }
        
        return self.engine.encode(inputs)
    
    def encode_text_audio(
        self,
        text: str,
        audio: Union[str, Path, bytes],
    ) -> Dict[str, torch.Tensor]:
        """Encode text and audio together."""
        inputs = {
            "text": self.process_text(text),
            "audio": self.process_audio(audio),
        }
        
        return self.engine.encode(inputs)
    
    def encode_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, bytes]] = None,
        audio: Optional[Union[str, Path, bytes]] = None,
        video: Optional[Union[str, Path]] = None,
        music: Optional[Union[List[Dict], List[int]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple modalities.
        
        Args:
            text: Optional text input
            image: Optional image input
            audio: Optional audio input
            video: Optional video input
            music: Optional music input
        
        Returns:
            Encoding outputs
        """
        inputs = {}
        
        if text is not None:
            inputs["text"] = self.process_text(text)
        if image is not None:
            inputs["image"] = self.process_image(image)
        if audio is not None:
            inputs["audio"] = self.process_audio(audio)
        if video is not None:
            inputs["video"] = self.process_video(video)
        if music is not None:
            inputs["music"] = self.process_music(music)
        
        if not inputs:
            raise ValueError("At least one modality must be provided")
        
        return self.engine.encode(inputs)
    
    def generate_music_from_text(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[int]:
        """
        Generate music conditioned on text.
        
        Args:
            text: Conditioning text
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated music token IDs
        """
        inputs = {"text": self.process_text(text)}
        
        tokens = self.engine.generate_music(
            inputs,
            max_new_tokens=max_tokens or self.config.music_max_tokens,
            temperature=temperature or self.config.music_temperature,
        )
        
        return tokens[0].tolist()
    
    def compute_text_image_similarity(
        self,
        text: str,
        image: Union[str, Path, bytes],
    ) -> float:
        """
        Compute similarity between text and image.
        
        Args:
            text: Text input
            image: Image input
        
        Returns:
            Similarity score in [-1, 1]
        """
        inputs_a = {"text": self.process_text(text)}
        inputs_b = {"image": self.process_image(image)}
        
        similarity = self.engine.get_similarity(inputs_a, inputs_b)
        
        return similarity.item()
