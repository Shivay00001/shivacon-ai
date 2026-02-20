"""
Production-grade configuration management for MultiModal AI Core.

Features:
- Dataclass-based configs with validation
- Environment variable overrides
- YAML file loading
- Type safety and runtime validation
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def _get_env(name: str, default: Any = None, cast: type = str) -> Any:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        if cast == bool:
            return val.lower() in ("1", "true", "yes", "on")
        return cast(val)
    except (ValueError, TypeError):
        return default


@dataclass
class TextEncoderConfig:
    vocab_size: int = 32000
    max_seq_len: int = 512
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    
    tokenizer_type: str = "bpe"
    tokenizer_vocab_path: Optional[str] = None
    tokenizer_merges_path: Optional[str] = None

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be positive")


@dataclass
class ImageEncoderConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    
    normalization_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalization_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    
    use_cls_token: bool = True

    def __post_init__(self):
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )
        self.num_patches = (self.image_size // self.patch_size) ** 2


@dataclass
class AudioEncoderConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    max_frames: int = 512
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    f_min: float = 0.0
    f_max: Optional[float] = None
    power: float = 2.0
    norm: str = "slaney"
    mel_scale: str = "htk"
    
    use_spectrogram_augmentation: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 35

    @property
    def n_freq_bins(self) -> int:
        return self.n_fft // 2 + 1


@dataclass
class VideoEncoderConfig:
    num_frames: int = 16
    frame_rate: int = 30
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 512
    num_heads: int = 8
    spatial_layers: int = 4
    temporal_layers: int = 2
    dim_feedforward: int = 2048
    dropout: float = 0.1
    
    use_3d_conv: bool = False
    temporal_kernel_size: int = 3
    tubelet_size: int = 2

    def __post_init__(self):
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )
        self.num_spatial_patches = (self.image_size // self.patch_size) ** 2


@dataclass
class MusicEncoderConfig:
    vocab_size: int = 512
    max_seq_len: int = 1024
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    
    midi_min_pitch: int = 21
    midi_max_pitch: int = 108
    time_step_ms: int = 10
    velocity_bins: int = 32

    @property
    def pitch_range(self) -> int:
        return self.midi_max_pitch - self.midi_min_pitch + 1


@dataclass
class FusionConfig:
    latent_dim: int = 512
    num_cross_attn_layers: int = 2
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    active_modalities: List[str] = field(
        default_factory=lambda: ["text", "image", "audio", "video", "music"]
    )
    
    fusion_strategy: str = "attention_pool"
    use_gating: bool = True
    gate_per_dim: bool = True
    
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 1.0


@dataclass
class TokenRouterConfig:
    strategy: str = "perceiver"
    num_latents: int = 64
    top_k: int = 64
    routing_threshold: int = 128


@dataclass
class DataConfig:
    data_root: str = "data"
    cache_dir: str = ".cache"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    train_batch_size: int = 16
    eval_batch_size: int = 32
    
    max_text_length: int = 512
    max_audio_frames: int = 512
    max_video_frames: int = 16
    max_music_tokens: int = 1024
    
    image_augmentation: bool = True
    audio_augmentation: bool = True
    
    datasets: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    num_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    
    mixed_precision: bool = True
    precision: str = "bf16"
    
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 5
    
    seed: int = 42
    deterministic: bool = False
    
    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "contrastive": 1.0,
            "reconstruction": 1.0,
            "alignment": 0.5,
        }
    )
    
    optimizer: str = "adamw"
    optimizer_betas: Tuple[float, float] = (0.9, 0.98)
    optimizer_eps: float = 1e-8
    
    lr_scheduler: str = "cosine_with_warmup"
    min_lr_ratio: float = 0.1
    
    gradient_checkpointing: bool = False
    compile_model: bool = False
    
    use_wandb: bool = False
    wandb_project: str = "multimodal-ai"
    wandb_entity: Optional[str] = None
    
    use_tensorboard: bool = True


@dataclass
class InferenceConfig:
    device: str = "auto"
    batch_size: int = 1
    
    text_max_tokens: int = 256
    text_temperature: float = 0.8
    text_top_k: int = 50
    text_top_p: float = 0.95
    
    music_max_tokens: int = 512
    music_temperature: float = 0.9
    music_top_k: int = 50
    
    use_cache: bool = True
    cache_size: int = 1000
    
    quantization: Optional[str] = None
    quantization_bits: int = 8


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    cors_origins: List[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"]
    )
    
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    max_request_size_mb: int = 100
    request_timeout_seconds: int = 300
    
    api_key_header: str = "X-API-Key"
    require_api_key: bool = False
    valid_api_keys: List[str] = field(default_factory=list)
    
    log_level: str = "INFO"
    log_requests: bool = True
    
    enable_metrics: bool = True
    metrics_port: int = 9090


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    keep_last_n: int = 5
    
    save_best_only: bool = False
    monitor_metric: str = "val/loss"
    monitor_mode: str = "min"
    
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng_state: bool = True


@dataclass
class Config:
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    image: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    audio: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    video: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    music: MusicEncoderConfig = field(default_factory=MusicEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    model_name: str = "multimodal-ai-v1"
    version: str = "1.0.0"
    
    def validate(self) -> List[str]:
        warnings = []
        
        latent_dim = self.fusion.latent_dim
        encoder_dims = {
            "text": self.text.d_model,
            "image": self.image.d_model,
            "audio": self.audio.d_model,
            "video": self.video.d_model,
            "music": self.music.d_model,
        }
        
        for name, dim in encoder_dims.items():
            if dim != latent_dim and name in self.fusion.active_modalities:
                warnings.append(
                    f"[{name}] d_model={dim} != fusion.latent_dim={latent_dim}. "
                    f"A projection layer will be added."
                )
        
        total_params_estimate = sum(
            self._estimate_encoder_params(name)
            for name in self.fusion.active_modalities
        )
        if total_params_estimate > 1e9:
            warnings.append(
                f"Estimated model size >1B parameters ({total_params_estimate/1e9:.1f}B). "
                f"Consider reducing layer counts or dimensions."
            )
        
        return warnings
    
    def _estimate_encoder_params(self, modality: str) -> int:
        configs = {
            "text": (self.text, 6),
            "image": (self.image, 6),
            "audio": (self.audio, 4),
            "video": (self.video, 6),
            "music": (self.music, 6),
        }
        if modality not in configs:
            return 0
        cfg, _ = configs[modality]
        d = cfg.d_model
        num_layers = getattr(cfg, 'num_layers', getattr(cfg, 'spatial_layers', 4))
        ff = cfg.dim_feedforward
        return int(4 * d * d * num_layers + 2 * d * ff * num_layers)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(
            text=TextEncoderConfig(**d.get("text", {})),
            image=ImageEncoderConfig(**d.get("image", {})),
            audio=AudioEncoderConfig(**d.get("audio", {})),
            video=VideoEncoderConfig(**d.get("video", {})),
            music=MusicEncoderConfig(**d.get("music", {})),
            fusion=FusionConfig(**d.get("fusion", {})),
            data=DataConfig(**d.get("data", {})),
            training=TrainingConfig(**d.get("training", {})),
            inference=InferenceConfig(**d.get("inference", {})),
            server=ServerConfig(**d.get("server", {})),
            checkpoint=CheckpointConfig(**d.get("checkpoint", {})),
            model_name=d.get("model_name", "multimodal-ai-v1"),
            version=d.get("version", "1.0.0"),
        )
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config. pip install pyyaml")
            else:
                data = json.load(f)
        
        return cls.from_dict(data)


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    if config_path and config_path.exists():
        config = Config.load(config_path)
        logger.info(f"Loaded config from {config_path}")
    else:
        config = get_default_config()
        logger.info("Using default config")
    
    if overrides:
        config = _apply_overrides(config, overrides)
    
    return config


def _apply_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    config_dict = config.to_dict()
    
    def deep_update(d: dict, u: dict) -> dict:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    config_dict = deep_update(config_dict, overrides)
    return Config.from_dict(config_dict)


def get_default_config() -> Config:
    return Config(
        text=TextEncoderConfig(
            vocab_size=_get_env("TEXT_VOCAB_SIZE", 32000, int),
            max_seq_len=_get_env("TEXT_MAX_SEQ_LEN", 512, int),
            d_model=_get_env("TEXT_D_MODEL", 512, int),
        ),
        training=TrainingConfig(
            num_epochs=_get_env("TRAIN_EPOCHS", 50, int),
            batch_size=_get_env("TRAIN_BATCH_SIZE", 16, int),
            learning_rate=_get_env("TRAIN_LR", 3e-4, float),
            seed=_get_env("TRAIN_SEED", 42, int),
        ),
        inference=InferenceConfig(
            device=_get_env("INFERENCE_DEVICE", "auto", str),
        ),
        server=ServerConfig(
            port=_get_env("SERVER_PORT", 8000, int),
            workers=_get_env("SERVER_WORKERS", 1, int),
        ),
    )
