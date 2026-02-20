"""
Video preprocessing with frame extraction and augmentation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    import torchvision.transforms as T
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not installed. Video loading may be limited. pip install opencv-python")

try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


@dataclass
class VideoProcessorConfig:
    num_frames: int = 16
    frame_rate: Optional[int] = None
    
    image_size: int = 224
    in_channels: int = 3
    
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    use_augmentation: bool = True
    random_crop: bool = True
    random_flip: bool = True
    random_rotation: bool = False
    rotation_degrees: int = 10
    
    temporal_jitter: bool = True
    temporal_jitter_range: int = 2
    
    frame_sampling: str = "uniform"
    center_crop: bool = False
    
    interpolation: str = "bilinear"


class VideoProcessor:
    """
    Production-grade video processor with frame extraction and augmentation.
    """
    
    def __init__(self, config: Optional[VideoProcessorConfig] = None):
        self.config = config or VideoProcessorConfig()
        self._build_transforms()
    
    def _build_transforms(self) -> None:
        if not TORCHVISION_AVAILABLE:
            self._use_manual_transforms = True
            return
        
        self._use_manual_transforms = False
        cfg = self.config
        
        if cfg.use_augmentation:
            transforms_list = [
                T.ToPILImage(),
                T.RandomResizedCrop(
                    cfg.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                ),
                T.RandomHorizontalFlip(p=0.5) if cfg.random_flip else T.Lambda(lambda x: x),
                T.ToTensor(),
                T.Normalize(mean=cfg.mean, std=cfg.std),
            ]
        else:
            transforms_list = [
                T.ToPILImage(),
                T.Resize(cfg.image_size),
                T.CenterCrop(cfg.image_size),
                T.ToTensor(),
                T.Normalize(mean=cfg.mean, std=cfg.std),
            ]
        
        self.train_transform = T.Compose([t for t in transforms_list if t is not None])
        
        self.eval_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(cfg.image_size),
            T.CenterCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.mean, std=cfg.std),
        ])
    
    def _manual_transform(self, frame: Tensor, is_training: bool = False) -> Tensor:
        if frame.dim() == 3 and frame.shape[2] == 3:
            frame = frame.permute(2, 0, 1)
        
        if frame.dtype == torch.uint8:
            frame = frame.float() / 255.0
        
        frame = F.interpolate(
            frame.unsqueeze(0),
            size=self.config.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        mean = torch.tensor(self.config.mean, device=frame.device).view(-1, 1, 1)
        std = torch.tensor(self.config.std, device=frame.device).view(-1, 1, 1)
        frame = (frame - mean) / std
        
        return frame
    
    def _load_with_decord(self, path: str) -> Tensor:
        if not DECORD_AVAILABLE:
            raise ImportError("decord is required for video loading. pip install decord")
        
        video_reader = decord.VideoReader(path)
        total_frames = len(video_reader)
        
        num_frames = min(self.config.num_frames, total_frames)
        
        if self.config.frame_sampling == "uniform":
            indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
        else:
            indices = list(range(num_frames))
        
        frames = video_reader.get_batch(indices).asnumpy()
        return torch.from_numpy(frames)
    
    def _load_with_cv2(self, path: str) -> Tensor:
        if not CV2_AVAILABLE:
            raise ImportError("opencv is required for video loading. pip install opencv-python")
        
        cap = cv2.VideoCapture(str(path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        num_frames = min(self.config.num_frames, total_frames)
        
        if self.config.frame_sampling == "uniform":
            indices = torch.linspace(0, total_frames - 1, num_frames).long().tolist()
        else:
            indices = list(range(num_frames))
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame))
            else:
                if frames:
                    frames.append(frames[-1].clone())
                else:
                    frames.append(torch.zeros(224, 224, 3, dtype=torch.uint8))
        
        cap.release()
        return torch.stack(frames)
    
    def load_video(self, source: Union[str, Path, Tensor]) -> Tensor:
        if isinstance(source, Tensor):
            return source
        
        path = str(source)
        
        if DECORD_AVAILABLE:
            return self._load_with_decord(path)
        elif CV2_AVAILABLE:
            return self._load_with_cv2(path)
        else:
            raise ImportError(
                "No video backend available. Install decord or opencv-python."
            )
    
    def process(
        self,
        source: Union[str, Path, Tensor],
        is_training: bool = False,
    ) -> Tensor:
        frames = self.load_video(source)
        
        num_frames = self.config.num_frames
        
        if frames.shape[0] > num_frames:
            if is_training and self.config.temporal_jitter:
                max_start = frames.shape[0] - num_frames
                start = torch.randint(0, max(1, max_start + 1), (1,)).item()
                frames = frames[start:start + num_frames]
            else:
                indices = torch.linspace(0, frames.shape[0] - 1, num_frames).long()
                frames = frames[indices]
        elif frames.shape[0] < num_frames:
            pad_count = num_frames - frames.shape[0]
            last_frame = frames[-1:].expand(pad_count, *frames.shape[1:])
            frames = torch.cat([frames, last_frame], dim=0)
        
        if self._use_manual_transforms:
            processed_frames = torch.stack([
                self._manual_transform(f, is_training) for f in frames
            ])
        else:
            transform = self.train_transform if is_training else self.eval_transform
            processed_frames = torch.stack([transform(f.numpy() if f.dtype == torch.uint8 else f) for f in frames])
        
        return processed_frames
    
    def process_batch(
        self,
        sources: List[Union[str, Path, Tensor]],
        is_training: bool = False,
    ) -> Tensor:
        videos = [self.process(s, is_training) for s in sources]
        return torch.stack(videos)
    
    def denormalize(self, tensor: Tensor) -> Tensor:
        mean = torch.tensor(self.config.mean, device=tensor.device).view(1, 1, -1, 1, 1)
        std = torch.tensor(self.config.std, device=tensor.device).view(1, 1, -1, 1, 1)
        return tensor * std + mean
