"""
Main training entry point.

Usage:
    python train.py                          # Train with default config
    python train.py --config config.yaml     # Train with custom config
    python train.py --resume best            # Resume from best checkpoint
    python train.py --epochs 100             # Override number of epochs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config.settings import Config, load_config
from core.multimodal_core import MultiModalCore
from modalities.text import TextEncoder
from modalities.image import ImageEncoder
from modalities.audio import AudioEncoder
from modalities.video import VideoEncoder
from modalities.music import MusicEncoder, MusicDecoder
from data.tokenizer import BPETokenizer
from data.music_processor import MusicProcessor
from data.dataset import MultiModalDataset, create_dataloader
from training.trainer import Trainer
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def build_model(config: Config) -> MultiModalCore:
    """Build and configure the multi-modal model."""
    core = MultiModalCore(config.fusion)
    
    if "text" in config.fusion.active_modalities:
        core.register_encoder(TextEncoder(config.text))
        logger.info("Registered text encoder")
    
    if "image" in config.fusion.active_modalities:
        core.register_encoder(ImageEncoder(config.image))
        logger.info("Registered image encoder")
    
    if "audio" in config.fusion.active_modalities:
        core.register_encoder(AudioEncoder(config.audio))
        logger.info("Registered audio encoder")
    
    if "video" in config.fusion.active_modalities:
        core.register_encoder(VideoEncoder(config.video))
        logger.info("Registered video encoder")
    
    if "music" in config.fusion.active_modalities:
        core.register_encoder(MusicEncoder(config.music))
        logger.info("Registered music encoder")
    
    for q, c in [
        ("text", "image"),
        ("text", "audio"),
        ("image", "audio"),
        ("text", "video"),
    ]:
        if q in core.registered_modalities and c in core.registered_modalities:
            core.add_cross_attention(q, c)
    
    logger.info(f"Total parameters: {core.count_parameters():,}")
    
    return core


def build_dataloaders(config: Config):
    """Build training and validation dataloaders."""
    tokenizer = None
    image_processor = None
    audio_processor = None
    video_processor = None
    music_processor = None
    
    modalities = config.fusion.active_modalities
    
    if "text" in modalities:
        tokenizer = BPETokenizer()
        tokenizer_path = Path(config.data.cache_dir) / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = BPETokenizer.load(tokenizer_path)
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
    
    if "image" in modalities:
        try:
            from data.image_processor import ImageProcessor
            image_processor = ImageProcessor()
        except ImportError:
            logger.warning("ImageProcessor not available, using synthetic data only")
    
    if "audio" in modalities:
        try:
            from data.audio_processor import AudioProcessor
            audio_processor = AudioProcessor()
        except ImportError:
            logger.warning("AudioProcessor not available, using synthetic data only")
    
    if "video" in modalities:
        try:
            from data.video_processor import VideoProcessor
            video_processor = VideoProcessor()
        except ImportError:
            logger.warning("VideoProcessor not available, using synthetic data only")
    
    if "music" in modalities:
        music_processor = MusicProcessor()
    
    data_path = Path(config.data.data_root)
    train_manifest = data_path / "train.jsonl"
    val_manifest = data_path / "val.jsonl"
    
    train_dataset = MultiModalDataset(
        data_path=train_manifest if train_manifest.exists() else [],
        tokenizer=tokenizer,
        image_processor=image_processor,
        audio_processor=audio_processor,
        video_processor=video_processor,
        music_processor=music_processor,
        modalities=modalities,
        is_training=True,
        max_text_length=config.data.max_text_length,
        max_audio_frames=config.data.max_audio_frames,
        max_video_frames=config.data.max_video_frames,
        max_music_tokens=config.data.max_music_tokens,
    )
    
    val_dataset = MultiModalDataset(
        data_path=val_manifest if val_manifest.exists() else [],
        tokenizer=tokenizer,
        image_processor=image_processor,
        audio_processor=audio_processor,
        video_processor=video_processor,
        music_processor=music_processor,
        modalities=modalities,
        is_training=False,
        max_text_length=config.data.max_text_length,
        max_audio_frames=config.data.max_audio_frames,
        max_video_frames=config.data.max_video_frames,
        max_music_tokens=config.data.max_music_tokens,
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )
    
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.data.eval_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=False,
        )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    if val_loader:
        logger.info(f"Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train MultiModal AI")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint or 'best'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    args = parser.parse_args()
    
    setup_logging(
        level=args.log_level,
        log_dir=Path("logs"),
    )
    
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    
    warnings = config.validate()
    for warning in warnings:
        logger.warning(warning)
    
    config.save(Path(config.checkpoint.checkpoint_dir) / "config.json")
    
    model = build_model(config)
    train_loader, val_loader = build_dataloaders(config)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
    )
    
    if args.resume:
        trainer.resume(args.resume)
    
    final_metrics = trainer.train()
    
    logger.info(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
