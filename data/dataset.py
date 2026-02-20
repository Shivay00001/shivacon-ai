"""
Multi-modal dataset implementation for production training.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from data.tokenizer import BPETokenizer
from data.music_processor import MusicProcessor

logger = logging.getLogger(__name__)


@dataclass
class MultiModalSample:
    sample_id: str
    text: Optional[str] = None
    text_tokens: Optional[Tensor] = None
    image: Optional[Tensor] = None
    image_path: Optional[str] = None
    audio: Optional[Tensor] = None
    audio_path: Optional[str] = None
    video: Optional[Tensor] = None
    video_path: Optional[str] = None
    music: Optional[Tensor] = None
    music_events: Optional[List[Dict]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultiModalDataset(Dataset):
    """
    Production multi-modal dataset that loads and processes samples.
    
    Supports:
    - JSON/JSONL manifest files
    - Direct sample lists
    - Lazy loading from disk
    """
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path, List[Dict]]] = None,
        tokenizer: Optional[BPETokenizer] = None,
        image_processor: Optional[ImageProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        video_processor: Optional[VideoProcessor] = None,
        music_processor: Optional[MusicProcessor] = None,
        modalities: Optional[List[str]] = None,
        is_training: bool = True,
        transform: Optional[Callable] = None,
        max_text_length: int = 512,
        max_audio_frames: int = 512,
        max_video_frames: int = 16,
        max_music_tokens: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.video_processor = video_processor
        self.music_processor = music_processor
        self.modalities = modalities or ["text"]
        self.is_training = is_training
        self.transform = transform
        self.max_text_length = max_text_length
        self.max_audio_frames = max_audio_frames
        self.max_video_frames = max_video_frames
        self.max_music_tokens = max_music_tokens
        
        self.samples: List[Dict] = []
        
        if data_path is not None:
            self._load_data(data_path)
    
    def _load_data(self, data_path: Union[str, Path, List[Dict]]) -> None:
        if isinstance(data_path, list):
            self.samples = data_path
            return
        
        path = Path(data_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data path not found: {path}")
        
        if path.is_file():
            if path.suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.samples = data
                    elif isinstance(data, dict) and "samples" in data:
                        self.samples = data["samples"]
                    else:
                        self.samples = [data]
            elif path.suffix == ".jsonl":
                with open(path) as f:
                    self.samples = [json.loads(line) for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        elif path.is_dir():
            self.samples = self._scan_directory(path)
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def _scan_directory(self, directory: Path) -> List[Dict]:
        samples = []
        
        text_files = list(directory.glob("*.txt")) + list(directory.glob("**/*.txt"))
        for text_file in text_files:
            sample = {"text_path": str(text_file), "sample_id": text_file.stem}
            
            image_candidates = [
                text_file.with_suffix(".jpg"),
                text_file.with_suffix(".png"),
                text_file.with_suffix(".jpeg"),
            ]
            for img in image_candidates:
                if img.exists():
                    sample["image_path"] = str(img)
                    break
            
            audio_candidates = [
                text_file.with_suffix(".wav"),
                text_file.with_suffix(".mp3"),
                text_file.with_suffix(".flac"),
            ]
            for audio in audio_candidates:
                if audio.exists():
                    sample["audio_path"] = str(audio)
                    break
            
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_data = self.samples[idx]
        sample_id = sample_data.get("sample_id", str(idx))
        
        result = {"sample_id": sample_id}
        
        if "text" in self.modalities or "text_tokens" in sample_data:
            if self.tokenizer is not None:
                text = sample_data.get("text")
                if text is None and sample_data.get("text_path"):
                    with open(sample_data["text_path"]) as f:
                        text = f.read()
                
                if text:
                    tokens = self.tokenizer.encode(
                        text,
                        add_bos=True,
                        add_eos=True,
                        max_length=self.max_text_length,
                    )
                    result["text"] = {
                        "x": torch.tensor(tokens, dtype=torch.long),
                    }
        
        if "image" in self.modalities or "image_path" in sample_data or "image_type" in sample_data:
            if self.image_processor is not None:
                image_source = sample_data.get("image") or sample_data.get("image_path")
                
                # Handle synthetic image data
                if sample_data.get("image_type") == "synthetic":
                    h = sample_data.get("image_height", sample_data.get("height", 224))
                    w = sample_data.get("image_width", sample_data.get("width", 224))
                    c = sample_data.get("channels", 3)
                    image_tensor = torch.randn(c, h, w)
                    result["image"] = {"x": image_tensor}
                elif image_source:
                    try:
                        image_tensor = self.image_processor.process(
                            image_source, is_training=self.is_training
                        )
                        result["image"] = {"x": image_tensor}
                    except Exception as e:
                        logger.warning(f"Failed to load image for {sample_id}: {e}")
        
        if "audio" in self.modalities or "audio_path" in sample_data or "audio_type" in sample_data:
            if self.audio_processor is not None:
                audio_source = sample_data.get("audio") or sample_data.get("audio_path")
                
                # Handle synthetic audio data
                if sample_data.get("audio_type") == "synthetic":
                    n_mels = sample_data.get("n_mels", 80)
                    frames = sample_data.get("audio_frames", sample_data.get("frames", 128))
                    audio_tensor = torch.randn(n_mels, min(frames, self.max_audio_frames))
                    result["audio"] = {"x": audio_tensor}
                elif audio_source:
                    try:
                        audio_tensor = self.audio_processor.process(
                            audio_source,
                            is_training=self.is_training,
                            max_frames=self.max_audio_frames,
                        )
                        result["audio"] = {"x": audio_tensor}
                    except Exception as e:
                        logger.warning(f"Failed to load audio for {sample_id}: {e}")
        
        if "video" in self.modalities or "video_path" in sample_data or "video_type" in sample_data:
            if self.video_processor is not None:
                video_source = sample_data.get("video") or sample_data.get("video_path")
                
                # Handle synthetic video data
                if sample_data.get("video_type") == "synthetic":
                    t = sample_data.get("video_frames", sample_data.get("num_frames", 4))
                    h = sample_data.get("height", 64)
                    w = sample_data.get("width", 64)
                    c = sample_data.get("channels", 3)
                    video_tensor = torch.randn(t, c, h, w)
                    result["video"] = {"x": video_tensor}
                elif video_source:
                    try:
                        video_tensor = self.video_processor.process(
                            video_source, is_training=self.is_training
                        )
                        result["video"] = {"x": video_tensor}
                    except Exception as e:
                        logger.warning(f"Failed to load video for {sample_id}: {e}")
        
        if "music" in self.modalities or "music_events" in sample_data or "music_tokens" in sample_data:
            if self.music_processor is not None:
                music_source = sample_data.get("music") or sample_data.get("music_events") or sample_data.get("music_tokens")
                if music_source:
                    music_tensor = self.music_processor.process(
                        music_source,
                        add_bos=True,
                        add_eos=True,
                        max_length=self.max_music_tokens,
                    )
                    result["music"] = {"x": music_tensor}
        
        if self.transform:
            result = self.transform(result)
        
        # Ensure all active modalities are present with at least a dummy value
        for mod in self.modalities:
            if mod not in result:
                if mod == "text":
                    result["text"] = {"x": torch.zeros(1, dtype=torch.long)}
                elif mod == "image":
                    result["image"] = {"x": torch.zeros(3, 64, 64)}
                elif mod == "audio":
                    result["audio"] = {"x": torch.zeros(80, 32)}
                elif mod == "video":
                    result["video"] = {"x": torch.zeros(4, 3, 64, 64)}
                elif mod == "music":
                    result["music"] = {"x": torch.zeros(1, dtype=torch.long)}
        
        return result


class MultiModalCollator:
    """
    Collates multi-modal samples into batched tensors.
    """
    
    def __init__(
        self,
        pad_token_id: int = 0,
        modalities: Optional[List[str]] = None,
    ):
        self.pad_token_id = pad_token_id
        self.modalities = modalities
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Dict[str, Tensor]]:
        collated = {}
        
        all_modalities = set()
        for sample in batch:
            all_modalities.update(sample.keys())
        all_modalities.discard("sample_id")
        all_modalities.discard("metadata")
        
        for modality in all_modalities:
            modality_batch = []
            
            for sample in batch:
                if modality in sample:
                    modality_batch.append(sample[modality])
                else:
                    continue
            
            if not modality_batch:
                continue
            
            sample_keys = set()
            for item in modality_batch:
                sample_keys.update(item.keys())
            
            collated[modality] = {}
            
            for key in sample_keys:
                tensors = [item[key] for item in modality_batch if key in item]
                
                if not tensors:
                    continue
                
                if key == "x" and tensors[0].dim() == 1:
                    max_len = max(t.shape[0] for t in tensors)
                    padded = []
                    mask = []
                    
                    for t in tensors:
                        if t.shape[0] < max_len:
                            pad_len = max_len - t.shape[0]
                            if modality == "text" or modality == "music":
                                padded_t = torch.cat([
                                    t,
                                    torch.full((pad_len,), self.pad_token_id, dtype=t.dtype)
                                ])
                            else:
                                padded_t = torch.cat([t, torch.zeros(pad_len, dtype=t.dtype)])
                            mask_t = torch.cat([
                                torch.zeros(t.shape[0], dtype=torch.bool),
                                torch.ones(pad_len, dtype=torch.bool),
                            ])
                        else:
                            padded_t = t
                            mask_t = torch.zeros(t.shape[0], dtype=torch.bool)
                        padded.append(padded_t)
                        mask.append(mask_t)
                    
                    collated[modality]["x"] = torch.stack(padded)
                    collated[modality]["padding_mask"] = torch.stack(mask)
                elif key == "x" and tensors[0].dim() == 2:
                    # 2D tensors like audio (n_mels, T) - pad time dimension
                    max_t = max(t.shape[1] for t in tensors)
                    padded = []
                    mask = []
                    
                    for t in tensors:
                        if t.shape[1] < max_t:
                            pad_len = max_t - t.shape[1]
                            padded_t = F.pad(t, (0, pad_len), mode="constant", value=0)
                            mask_t = torch.cat([
                                torch.zeros(t.shape[1], dtype=torch.bool),
                                torch.ones(pad_len, dtype=torch.bool),
                            ])
                        else:
                            padded_t = t
                            mask_t = torch.zeros(t.shape[1], dtype=torch.bool)
                        padded.append(padded_t)
                        mask.append(mask_t)
                    
                    collated[modality]["x"] = torch.stack(padded)
                    collated[modality]["padding_mask"] = torch.stack(mask)
                elif key == "x" and tensors[0].dim() == 4:
                    # 4D tensors like video (T, C, H, W) - pad/truncate time dimension
                    max_t = max(t.shape[0] for t in tensors)
                    padded = []
                    
                    for t in tensors:
                        if t.shape[0] < max_t:
                            pad_len = max_t - t.shape[0]
                            # Repeat last frame to pad
                            last_frame = t[-1:].repeat(pad_len, 1, 1, 1)
                            padded_t = torch.cat([t, last_frame], dim=0)
                        else:
                            padded_t = t
                        padded.append(padded_t)
                    
                    collated[modality]["x"] = torch.stack(padded)
                else:
                    collated[modality][key] = torch.stack(tensors)
        
        return collated


def create_dataloader(
    dataset: MultiModalDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    pad_token_id: int = 0,
) -> DataLoader:
    collator = MultiModalCollator(pad_token_id=pad_token_id)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=drop_last,
    )
