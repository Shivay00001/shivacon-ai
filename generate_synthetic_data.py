"""
Generate synthetic multi-modal training data for testing and development.

Creates:
- Text samples with descriptions
- Synthetic images (random tensors)
- Synthetic audio (random mel-spectrograms)
- Synthetic video (random frame sequences)
- Synthetic music (random token sequences)

Usage:
    python generate_synthetic_data.py --samples 1000
    python generate_synthetic_data.py --samples 500 --output data/synthetic/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEXT_DESCRIPTIONS = [
    "A cat sitting on a comfortable couch in a living room",
    "A dog playing happily in the park on a sunny day",
    "A bird flying gracefully in the clear blue sky",
    "A car driving smoothly on a winding mountain road",
    "A person walking peacefully on a quiet street",
    "A flower blooming beautifully in a spring garden",
    "A tall tree swaying gently in the warm breeze",
    "A sailboat gliding across calm ocean waters",
    "A snow-capped mountain under a starry night sky",
    "A stunning sunset over the Pacific Ocean horizon",
    "A child playing joyfully with a red ball",
    "A musician skillfully playing an acoustic guitar",
    "A chef preparing a gourmet meal in the kitchen",
    "An artist painting a colorful abstract canvas",
    "A dancer performing elegantly on stage",
    "A peaceful lake reflecting the autumn trees",
    "A bustling city street filled with people",
    "A quiet library with shelves of old books",
    "A modern office building with glass windows",
    "A cozy coffee shop with warm lighting",
    "A farmer working in a golden wheat field",
    "A scientist conducting an experiment in a lab",
    "A soccer player scoring a winning goal",
    "A singer performing at a sold-out concert",
    "A teacher explaining a lesson to students",
    "A doctor examining a patient in a clinic",
    "A pilot flying a plane above the clouds",
    "A baker decorating a wedding cake",
    "A mechanic repairing a classic car",
    "A photographer capturing a beautiful landscape",
    "A swimmer diving into crystal clear water",
    "A cyclist racing through the countryside",
    "A painter restoring an ancient artwork",
    "A writer typing a novel at a desk",
    "A gardener tending to rose bushes",
    "A fisherman casting a line at dawn",
    "A hiker reaching a mountain summit",
    "A surfer riding a massive wave",
    "A skier descending a snowy slope",
    "A camper sitting by a crackling fire",
]

MUSIC_PROMPTS = [
    "A happy upbeat melody in major key",
    "A sad melancholic tune in minor key",
    "Jazz improvisation with swing rhythm",
    "Classical piano piece with elegant phrasing",
    "Electronic dance music with heavy bass",
    "Rock guitar solo with distortion",
    "Ambient background music for meditation",
    "Folk acoustic song with storytelling",
    "Hip hop beat with strong groove",
    "Orchestral symphony with dramatic swells",
    "Reggae track with laid-back vibes",
    "Country song with storytelling lyrics",
    "Blues progression with emotional depth",
    "Pop ballad with catchy melody",
    "Metal riff with aggressive energy",
]


def generate_text_samples(num_samples: int) -> List[Dict[str, Any]]:
    """Generate text samples."""
    samples = []
    for i in range(num_samples):
        samples.append({
            "sample_id": f"text_{i:05d}",
            "text": random.choice(TEXT_DESCRIPTIONS),
            "modality": "text",
        })
    return samples


def generate_image_text_samples(num_samples: int) -> List[Dict[str, Any]]:
    """Generate image-text paired samples (synthetic)."""
    samples = []
    for i in range(num_samples):
        text = random.choice(TEXT_DESCRIPTIONS)
        samples.append({
            "sample_id": f"img_{i:05d}",
            "text": text,
            "image_type": "synthetic",
            "height": 64,
            "width": 64,
            "channels": 3,
            "modality": "image_text",
        })
    return samples


def generate_audio_text_samples(num_samples: int) -> List[Dict[str, Any]]:
    """Generate audio-text paired samples (synthetic)."""
    samples = []
    for i in range(num_samples):
        text = random.choice(TEXT_DESCRIPTIONS[:20])
        samples.append({
            "sample_id": f"audio_{i:05d}",
            "text": text,
            "audio_type": "synthetic",
            "n_mels": 80,
            "frames": random.randint(100, 300),
            "sample_rate": 16000,
            "modality": "audio_text",
        })
    return samples


def generate_video_text_samples(num_samples: int) -> List[Dict[str, Any]]:
    """Generate video-text paired samples (synthetic)."""
    samples = []
    for i in range(num_samples):
        text = random.choice(TEXT_DESCRIPTIONS[:15])
        samples.append({
            "sample_id": f"video_{i:05d}",
            "text": text,
            "video_type": "synthetic",
            "num_frames": random.choice([4, 8]),
            "height": 64,
            "width": 64,
            "channels": 3,
            "modality": "video_text",
        })
    return samples


def generate_music_text_samples(num_samples: int) -> List[Dict[str, Any]]:
    """Generate music-text paired samples (synthetic)."""
    samples = []
    for i in range(num_samples):
        prompt = random.choice(MUSIC_PROMPTS)
        num_tokens = random.randint(50, 200)
        tokens = [random.randint(3, 511) for _ in range(num_tokens)]
        samples.append({
            "sample_id": f"music_{i:05d}",
            "text": prompt,
            "music_tokens": tokens,
            "modality": "music_text",
        })
    return samples


def generate_multi_modal_samples(num_samples: int) -> List[Dict[str, Any]]:
    """Generate samples with multiple modalities combined."""
    samples = []
    for i in range(num_samples):
        text = random.choice(TEXT_DESCRIPTIONS)
        music_prompt = random.choice(MUSIC_PROMPTS)
        music_tokens = [random.randint(3, 511) for _ in range(random.randint(30, 60))]
        
        samples.append({
            "sample_id": f"multi_{i:05d}",
            "text": text,
            "image_type": "synthetic",
            "image_height": 64,
            "image_width": 64,
            "audio_type": "synthetic",
            "audio_frames": random.randint(50, 64),
            "video_type": "synthetic",
            "video_frames": 4,
            "height": 64,
            "width": 64,
            "music_tokens": music_tokens,
            "music_prompt": music_prompt,
            "modality": "multi",
        })
    return samples


def train_tokenizer(samples: List[Dict], output_path: Path, vocab_size: int = 2000) -> None:
    """Train BPE tokenizer on all text in samples."""
    logger.info("Training tokenizer...")
    
    from data.tokenizer import BPETokenizer, TokenizerConfig
    
    texts = []
    for sample in samples:
        if "text" in sample:
            texts.append(sample["text"])
        if "music_prompt" in sample:
            texts.append(sample["music_prompt"])
    
    if len(texts) < 100:
        texts = texts * (100 // len(texts) + 1)
    
    config = TokenizerConfig(vocab_size=vocab_size)
    tokenizer = BPETokenizer(config)
    tokenizer.train(texts, show_progress=True)
    tokenizer.save(output_path)
    
    logger.info(f"Tokenizer saved to {output_path} (vocab: {tokenizer.vocab_size})")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-modal data")
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples per modality",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="all",
        choices=["all", "text", "image", "audio", "video", "music"],
        help="Which modalities to generate",
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    if args.modalities in ("all", "text"):
        logger.info("Generating text samples...")
        text_samples = generate_text_samples(args.samples)
        all_samples.extend(text_samples)
        logger.info(f"  Generated {len(text_samples)} text samples")
    
    if args.modalities in ("all", "image"):
        logger.info("Generating image-text samples...")
        image_samples = generate_image_text_samples(args.samples)
        all_samples.extend(image_samples)
        logger.info(f"  Generated {len(image_samples)} image-text samples")
    
    if args.modalities in ("all", "audio"):
        logger.info("Generating audio-text samples...")
        audio_samples = generate_audio_text_samples(args.samples)
        all_samples.extend(audio_samples)
        logger.info(f"  Generated {len(audio_samples)} audio-text samples")
    
    if args.modalities in ("all", "video"):
        logger.info("Generating video-text samples...")
        video_samples = generate_video_text_samples(args.samples // 2)
        all_samples.extend(video_samples)
        logger.info(f"  Generated {len(video_samples)} video-text samples")
    
    if args.modalities in ("all", "music"):
        logger.info("Generating music-text samples...")
        music_samples = generate_music_text_samples(args.samples // 2)
        all_samples.extend(music_samples)
        logger.info(f"  Generated {len(music_samples)} music-text samples")
    
    if args.modalities == "all":
        logger.info("Generating multi-modal samples...")
        multi_samples = generate_multi_modal_samples(args.samples // 4)
        all_samples.extend(multi_samples)
        logger.info(f"  Generated {len(multi_samples)} multi-modal samples")
    
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    logger.info(f"Saved {len(train_samples)} training samples to {train_path}")
    
    val_path = output_dir / "val.jsonl"
    with open(val_path, "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")
    logger.info(f"Saved {len(val_samples)} validation samples to {val_path}")
    
    cache_dir = output_dir.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = cache_dir / "tokenizer.json"
    train_tokenizer(all_samples, tokenizer_path)
    
    print("\n" + "="*60)
    print("Synthetic data generation complete!")
    print("="*60)
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Training: {len(train_samples)}")
    print(f"  Validation: {len(val_samples)}")
    print(f"  Tokenizer: {tokenizer_path}")
    print("\nTo train:")
    print("  python train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
