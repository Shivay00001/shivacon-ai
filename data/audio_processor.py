"""
Audio preprocessing with mel-spectrogram extraction.

Features:
- Load audio from files (wav, mp3, flac, etc.)
- Resample to target sample rate
- Compute mel-spectrograms
- Audio augmentation (SpecAugment)
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
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("torchaudio not installed. Install with: pip install torchaudio")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class AudioProcessorConfig:
    sample_rate: int = 16000
    
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = None
    
    power: float = 2.0
    normalized: bool = False
    
    max_frames: int = 512
    
    use_spectrogram_augmentation: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 35
    n_freq_masks: int = 2
    n_time_masks: int = 2
    
    apply_log: bool = True
    log_offset: float = 1e-6
    
    normalize_spectrogram: bool = True
    mean: Optional[float] = None
    std: Optional[float] = None
    
    trim_silence: bool = True
    silence_threshold_db: float = -40.0


class AudioProcessor:
    """
    Production-grade audio processor with mel-spectrogram extraction.
    """
    
    def __init__(self, config: Optional[AudioProcessorConfig] = None):
        if not TORCHAUDIO_AVAILABLE:
            raise ImportError(
                "torchaudio is required for audio processing. "
                "Install with: pip install torchaudio"
            )
        
        self.config = config or AudioProcessorConfig()
        
        self._build_transforms()
    
    def _build_transforms(self) -> None:
        cfg = self.config
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max if cfg.f_max else cfg.sample_rate // 2,
            power=cfg.power,
            normalized=cfg.normalized,
        )
        
        if cfg.use_spectrogram_augmentation:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=cfg.freq_mask_param
            )
            self.time_mask = torchaudio.transforms.TimeMasking(
                time_mask_param=cfg.time_mask_param
            )
    
    def load_audio(
        self,
        source: Union[str, Path, bytes, Tensor, Tuple[Tensor, int]],
    ) -> Tuple[Tensor, int]:
        if isinstance(source, (str, Path)):
            try:
                waveform, sample_rate = torchaudio.load(str(source))
            except Exception as e:
                try:
                    import scipy.io.wavfile as wavfile
                    import numpy as np
                    sample_rate, data = wavfile.read(str(source))
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) / 32768.0
                    elif data.dtype == np.int32:
                        data = data.astype(np.float32) / 2147483648.0
                    elif data.dtype == np.uint8:
                        data = (data.astype(np.float32) - 128) / 128.0
                    if data.ndim == 1:
                        waveform = torch.from_numpy(data).unsqueeze(0)
                    else:
                        waveform = torch.from_numpy(data.T).float()
                except Exception as e2:
                    raise RuntimeError(f"Failed to load audio: {e}, {e2}")
        elif isinstance(source, bytes):
            import io
            try:
                waveform, sample_rate = torchaudio.load(io.BytesIO(source))
            except Exception:
                import scipy.io.wavfile as wavfile
                import numpy as np
                sample_rate, data = wavfile.read(io.BytesIO(source))
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                if data.ndim == 1:
                    waveform = torch.from_numpy(data).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(data.T).float()
        elif isinstance(source, Tensor):
            if source.dim() == 1:
                waveform = source.unsqueeze(0)
            else:
                waveform = source
            sample_rate = self.config.sample_rate
        elif isinstance(source, tuple):
            waveform, sample_rate = source
        else:
            raise ValueError(f"Unsupported audio source type: {type(source)}")
        
        if sample_rate != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.config.sample_rate
            )
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform, self.config.sample_rate
    
    def trim_silence(self, waveform: Tensor) -> Tensor:
        if not self.config.trim_silence:
            return waveform
        
        transformed = torchaudio.functional.DB_to_amplitude(
            torch.abs(waveform),
            ref=torch.abs(waveform).max(),
            power=0.5,
        )
        
        energy = transformed ** 2
        threshold = energy.max() * (10 ** (self.config.silence_threshold_db / 10))
        
        non_silent = (energy > threshold).squeeze(0)
        if not non_silent.any():
            return waveform
        
        indices = torch.where(non_silent)[0]
        start_idx = max(0, indices[0].item() - 100)
        end_idx = min(waveform.shape[1], indices[-1].item() + 100)
        
        return waveform[:, start_idx:end_idx]
    
    def compute_mel_spectrogram(
        self,
        waveform: Tensor,
        is_training: bool = False,
    ) -> Tensor:
        mel_spec = self.mel_transform(waveform)
        
        if self.config.apply_log:
            mel_spec = torch.log(mel_spec + self.config.log_offset)
        
        if self.config.use_spectrogram_augmentation and is_training:
            for _ in range(self.config.n_freq_masks):
                mel_spec = self.freq_mask(mel_spec)
            for _ in range(self.config.n_time_masks):
                mel_spec = self.time_mask(mel_spec)
        
        mel_spec = mel_spec.squeeze(0)
        
        if self.config.normalize_spectrogram:
            if self.config.mean is not None and self.config.std is not None:
                mel_spec = (mel_spec - self.config.mean) / self.config.std
            else:
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec
    
    def process(
        self,
        source: Union[str, Path, bytes, Tensor, Tuple[Tensor, int]],
        is_training: bool = False,
        max_frames: Optional[int] = None,
    ) -> Tensor:
        max_frames = max_frames or self.config.max_frames
        
        waveform, _ = self.load_audio(source)
        
        waveform = self.trim_silence(waveform)
        
        mel_spec = self.compute_mel_spectrogram(waveform, is_training)
        
        if mel_spec.shape[1] > max_frames:
            if is_training:
                start = torch.randint(0, mel_spec.shape[1] - max_frames + 1, (1,)).item()
                mel_spec = mel_spec[:, start:start + max_frames]
            else:
                mel_spec = mel_spec[:, :max_frames]
        elif mel_spec.shape[1] < max_frames:
            pad_length = max_frames - mel_spec.shape[1]
            mel_spec = F.pad(mel_spec, (0, pad_length), mode="constant", value=0)
        
        return mel_spec
    
    def process_batch(
        self,
        sources: List[Union[str, Path, bytes, Tensor, Tuple[Tensor, int]]],
        is_training: bool = False,
    ) -> Tensor:
        specs = [self.process(s, is_training) for s in sources]
        return torch.stack(specs)
    
    @staticmethod
    def compute_max_frames(
        duration_seconds: float,
        sample_rate: int = 16000,
        hop_length: int = 160,
    ) -> int:
        num_samples = int(duration_seconds * sample_rate)
        return math.ceil(num_samples / hop_length)
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.config.sample_rate,
            "n_fft": self.config.n_fft,
            "hop_length": self.config.hop_length,
            "n_mels": self.config.n_mels,
            "max_frames": self.config.max_frames,
        }
