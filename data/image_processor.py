"""
Image preprocessing and augmentation for production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import io

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import PIL.Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not installed. Install with: pip install Pillow")

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision not installed. Install with: pip install torchvision")


@dataclass
class ImageProcessorConfig:
    image_size: int = 224
    in_channels: int = 3
    
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    use_augmentation: bool = True
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: bool = False
    color_jitter_params: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.1)
    
    min_scale: float = 0.08
    max_scale: float = 1.0
    
    interpolation: str = "bilinear"
    
    center_crop: bool = False


class ImageProcessor:
    """
    Production-grade image preprocessing with augmentation support.
    """
    
    def __init__(self, config: Optional[ImageProcessorConfig] = None):
        self.config = config or ImageProcessorConfig()
        
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required. Install with: pip install Pillow")
        
        self._build_transforms()
    
    def _build_transforms(self) -> None:
        if not TORCHVISION_AVAILABLE:
            self._use_manual_transforms = True
            return
        
        self._use_manual_transforms = False
        cfg = self.config
        
        transforms_list = []
        
        transforms_list.append(T.ToTensor())
        
        if cfg.use_augmentation and cfg.random_crop:
            transforms_list.append(
                T.RandomResizedCrop(
                    cfg.image_size,
                    scale=(cfg.min_scale, cfg.max_scale),
                    interpolation=self._get_interpolation(),
                )
            )
        elif cfg.center_crop:
            transforms_list.append(T.CenterCrop(cfg.image_size))
        else:
            transforms_list.append(T.Resize(cfg.image_size))
            transforms_list.append(T.CenterCrop(cfg.image_size))
        
        if cfg.use_augmentation and cfg.random_flip:
            transforms_list.append(T.RandomHorizontalFlip(p=0.5))
        
        if cfg.use_augmentation and cfg.color_jitter:
            transforms_list.append(
                T.ColorJitter(
                    brightness=cfg.color_jitter_params[0],
                    contrast=cfg.color_jitter_params[1],
                    saturation=cfg.color_jitter_params[2],
                    hue=cfg.color_jitter_params[3],
                )
            )
        
        transforms_list.append(T.Normalize(mean=cfg.mean, std=cfg.std))
        
        self.train_transform = T.Compose(transforms_list)
        
        eval_transforms = [
            T.ToTensor(),
            T.Resize(cfg.image_size),
            T.CenterCrop(cfg.image_size),
            T.Normalize(mean=cfg.mean, std=cfg.std),
        ]
        self.eval_transform = T.Compose(eval_transforms)
    
    def _get_interpolation(self):
        cfg = self.config
        if cfg.interpolation == "bilinear":
            return T.InterpolationMode.BILINEAR
        elif cfg.interpolation == "bicubic":
            return T.InterpolationMode.BICUBIC
        elif cfg.interpolation == "nearest":
            return T.InterpolationMode.NEAREST
        return T.InterpolationMode.BILINEAR
    
    def _manual_transform(
        self,
        image: "PIL.Image.Image",
        is_training: bool = False,
    ) -> Tensor:
        import numpy as np
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        if is_training:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=self.config.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        else:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=self.config.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        
        mean = torch.tensor(self.config.mean).view(-1, 1, 1)
        std = torch.tensor(self.config.std).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor
    
    def process(
        self,
        image: Union[str, Path, "PIL.Image.Image", bytes, Tensor],
        is_training: bool = False,
    ) -> Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Tensor):
            return self._process_tensor(image, is_training)
        
        if self._use_manual_transforms:
            return self._manual_transform(image, is_training)
        
        transform = self.train_transform if is_training else self.eval_transform
        return transform(image)
    
    def _process_tensor(self, tensor: Tensor, is_training: bool) -> Tensor:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).expand(3, -1, -1)
        
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=self.config.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        mean = torch.tensor(self.config.mean, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(self.config.std, device=tensor.device).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor
    
    def process_batch(
        self,
        images: List[Union[str, Path, "PIL.Image.Image", bytes, Tensor]],
        is_training: bool = False,
    ) -> Tensor:
        tensors = [self.process(img, is_training) for img in images]
        return torch.stack(tensors)
    
    def denormalize(self, tensor: Tensor) -> Tensor:
        mean = torch.tensor(self.config.mean, device=tensor.device).view(1, -1, 1, 1)
        std = torch.tensor(self.config.std, device=tensor.device).view(1, -1, 1, 1)
        return tensor * std + mean
    
    def to_pil(self, tensor: Tensor) -> "PIL.Image.Image":
        tensor = self.denormalize(tensor)
        tensor = tensor.clamp(0, 1)
        
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)
        
        numpy_array = (tensor.cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(numpy_array)
