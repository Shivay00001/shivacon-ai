"""
Download and prepare public datasets for multi-modal training.

Datasets:
- COCO 2017 (images + captions) for text-image alignment
- LibriSpeech (audio) for audio-text alignment  
- Synthetic video data

Usage:
    python download_data.py --datasets coco librispeech
    python download_data.py --datasets coco --samples 1000  # Limit samples
    python download_data.py --all  # Download all
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import io
import random
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
CACHE_DIR = Path(".cache")


def download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Download a file with progress."""
    logger.info(f"{desc}: {url}")
    
    class ProgressReporter:
        def __init__(self):
            self.downloaded = 0
            self.last_report = 0
        
        def report(self, count, block_size, total_size):
            self.downloaded = count * block_size
            if total_size > 0 and self.downloaded - self.last_report > 10 * 1024 * 1024:
                pct = (self.downloaded / total_size) * 100
                mb = self.downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                logger.info(f"  Progress: {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
                self.last_report = self.downloaded
    
    progress = ProgressReporter()
    urllib.request.urlretrieve(url, dest, progress.report)
    logger.info(f"  Saved to {dest}")


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract zip or tar archive."""
    logger.info(f"Extracting {archive_path}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffix in ('.tar', '.gz', '.tgz'):
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(dest_dir)
    
    logger.info(f"  Extracted to {dest_dir}")


def download_coco(max_samples: Optional[int] = None) -> Dict[str, List[str]]:
    """
    Download COCO 2017 dataset (images + captions).
    
    Returns dict with paths to images and captions.
    """
    coco_dir = DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "http://images.cocodataset.org"
    
    train_images_url = f"{base_url}/zips/train2017.zip"
    val_images_url = f"{base_url}/zips/val2017.zip"
    captions_url = f"{base_url}/annotations/annotations_trainval2017.zip"
    
    train_images_dir = coco_dir / "train2017"
    val_images_dir = coco_dir / "val2017"
    annotations_dir = coco_dir / "annotations"
    
    results = {"train_images": [], "val_images": [], "captions": []}
    
    if not train_images_dir.exists():
        train_zip = CACHE_DIR / "train2017.zip"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not train_zip.exists():
            logger.info("Downloading COCO train2017 images (~19GB)...")
            download_file(train_images_url, train_zip, "COCO train images")
        extract_archive(train_zip, coco_dir)
    
    if not val_images_dir.exists():
        val_zip = CACHE_DIR / "val2017.zip"
        if not val_zip.exists():
            logger.info("Downloading COCO val2017 images (~1GB)...")
            download_file(val_images_url, val_zip, "COCO val images")
        extract_archive(val_zip, coco_dir)
    
    if not annotations_dir.exists():
        captions_zip = CACHE_DIR / "annotations_trainval2017.zip"
        if not captions_zip.exists():
            logger.info("Downloading COCO annotations...")
            download_file(captions_url, captions_zip, "COCO annotations")
        extract_archive(captions_zip, coco_dir)
    
    train_images = list(train_images_dir.glob("*.jpg"))
    val_images = list(val_images_dir.glob("*.jpg"))
    
    if max_samples:
        train_images = train_images[:max_samples]
        val_images = val_images[:max_samples // 5]
    
    results["train_images"] = [str(p) for p in train_images]
    results["val_images"] = [str(p) for p in val_images]
    
    logger.info(f"Found {len(train_images)} train images, {len(val_images)} val images")
    
    return results


def create_coco_manifest(
    annotations_path: Path,
    images_dir: Path,
    output_path: Path,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> int:
    """
    Create JSONL manifest from COCO annotations.
    
    Returns number of samples written.
    """
    captions_file = annotations_path / f"captions_{split}2017.json"
    
    if not captions_file.exists():
        logger.warning(f"Captions file not found: {captions_file}")
        return 0
    
    with open(captions_file) as f:
        data = json.load(f)
    
    image_id_to_file = {}
    for img in data["images"]:
        image_id_to_file[img["id"]] = img["file_name"]
    
    image_to_captions = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        caption = ann["caption"]
        if img_id in image_id_to_file:
            fname = image_id_to_file[img_id]
            if fname not in image_to_captions:
                image_to_captions[fname] = []
            image_to_captions[fname].append(caption)
    
    samples = []
    for fname, captions in image_to_captions.items():
        img_path = images_dir / fname
        if img_path.exists():
            caption = random.choice(captions)
            samples.append({
                "sample_id": fname.replace(".jpg", ""),
                "image_path": str(img_path),
                "text": caption,
                "modality": "image_text",
            })
    
    if max_samples:
        samples = samples[:max_samples]
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Created {output_path} with {len(samples)} samples")
    return len(samples)


def download_librispeech(max_samples: Optional[int] = None) -> Dict[str, str]:
    """
    Download LibriSpeech dataset (audio).
    
    Returns paths to audio files.
    """
    libri_dir = DATA_DIR / "librispeech"
    libri_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    
    audio_dir = libri_dir / "LibriSpeech" / "dev-clean"
    
    if not audio_dir.exists():
        tar_path = CACHE_DIR / "dev-clean.tar.gz"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not tar_path.exists():
            logger.info("Downloading LibriSpeech dev-clean (~350MB)...")
            download_file(url, tar_path, "LibriSpeech")
        extract_archive(tar_path, libri_dir)
    
    audio_files = list(audio_dir.rglob("*.flac"))
    
    if max_samples:
        audio_files = audio_files[:max_samples]
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    return {"audio_files": [str(p) for p in audio_files]}


def create_librispeech_manifest(
    audio_dir: Path,
    output_path: Path,
    max_samples: Optional[int] = None,
) -> int:
    """Create JSONL manifest for LibriSpeech."""
    transcripts = list(audio_dir.rglob("*.txt"))
    
    samples = []
    for trans_file in transcripts:
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    audio_id, text = parts
                    audio_file = trans_file.parent / f"{audio_id}.flac"
                    if audio_file.exists():
                        samples.append({
                            "sample_id": audio_id,
                            "audio_path": str(audio_file),
                            "text": text,
                            "modality": "audio_text",
                        })
    
    if max_samples:
        samples = samples[:max_samples]
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Created {output_path} with {len(samples)} samples")
    return len(samples)


def create_synthetic_video_manifest(
    output_path: Path,
    num_samples: int = 100,
) -> int:
    """
    Create manifest for synthetic video data.
    
    Uses random noise as placeholder - in production you'd use real videos.
    """
    samples = []
    video_dir = DATA_DIR / "synthetic_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    descriptions = [
        "A person walking in the park",
        "Cars driving on a highway",
        "Birds flying in the sky",
        "Waves crashing on the beach",
        "Children playing in a playground",
        "A cat sleeping on a couch",
        "Dogs running in a field",
        "Sunset over mountains",
        "City traffic at night",
        "Rain falling on a window",
    ]
    
    for i in range(num_samples):
        sample_id = f"video_{i:05d}"
        samples.append({
            "sample_id": sample_id,
            "video_type": "synthetic",
            "num_frames": 16,
            "height": 224,
            "width": 224,
            "channels": 3,
            "text": random.choice(descriptions),
            "modality": "video_text",
        })
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Created {output_path} with {len(samples)} synthetic video samples")
    return len(samples)


def create_synthetic_music_manifest(
    output_path: Path,
    num_samples: int = 100,
) -> int:
    """
    Create manifest for synthetic music data.
    
    Generates random token sequences as placeholder.
    """
    samples = []
    
    prompts = [
        "A happy upbeat melody",
        "A sad melancholic tune",
        "Jazz improvisation",
        "Classical piano piece",
        "Electronic dance music",
        "Rock guitar solo",
        "Ambient background music",
        "Folk acoustic song",
        "Hip hop beat",
        "Orchestral symphony",
    ]
    
    for i in range(num_samples):
        sample_id = f"music_{i:05d}"
        
        tokens = [random.randint(3, 511) for _ in range(random.randint(50, 200))]
        
        samples.append({
            "sample_id": sample_id,
            "music_tokens": tokens,
            "text": random.choice(prompts),
            "modality": "music_text",
        })
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Created {output_path} with {len(samples)} synthetic music samples")
    return len(samples)


def create_combined_manifest(
    manifests: List[Path],
    output_path: Path,
) -> int:
    """Combine multiple manifests into one training manifest."""
    total_samples = 0
    
    with open(output_path, "w") as out_f:
        for manifest_path in manifests:
            if manifest_path.exists():
                with open(manifest_path) as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_samples += 1
    
    logger.info(f"Created combined manifest {output_path} with {total_samples} samples")
    return total_samples


def train_tokenizer(corpus_path: Path, output_path: Path, vocab_size: int = 8000) -> None:
    """Train BPE tokenizer on text corpus."""
    logger.info("Training tokenizer...")
    
    from data.tokenizer import BPETokenizer, TokenizerConfig
    
    texts = []
    with open(corpus_path) as f:
        for line in f:
            sample = json.loads(line)
            if "text" in sample:
                texts.append(sample["text"])
    
    if len(texts) < 100:
        logger.warning(f"Only {len(texts)} texts for tokenizer training, adding more...")
        extra_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "Actions speak louder than words.",
        ] * 100
        texts.extend(extra_texts)
    
    config = TokenizerConfig(vocab_size=vocab_size)
    tokenizer = BPETokenizer(config)
    tokenizer.train(texts, show_progress=True)
    tokenizer.save(output_path)
    
    logger.info(f"Tokenizer saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare public datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["coco", "librispeech", "video", "music", "all"],
        default=["coco"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Maximum samples per dataset",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, only create manifests",
    )
    args = parser.parse_args()
    
    if args.all:
        args.datasets = ["coco", "librispeech", "video", "music"]
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    manifests = []
    
    if "coco" in args.datasets:
        if not args.skip_download:
            try:
                download_coco(max_samples=args.samples)
            except Exception as e:
                logger.error(f"Failed to download COCO: {e}")
                logger.info("Creating synthetic image-text data instead...")
        
        coco_dir = DATA_DIR / "coco"
        train_images_dir = coco_dir / "train2017"
        val_images_dir = coco_dir / "val2017"
        annotations_dir = coco_dir / "annotations"
        
        if annotations_dir.exists():
            train_manifest = DATA_DIR / "coco_train.jsonl"
            val_manifest = DATA_DIR / "coco_val.jsonl"
            
            create_coco_manifest(
                annotations_dir, train_images_dir, train_manifest,
                split="train", max_samples=args.samples
            )
            create_coco_manifest(
                annotations_dir, val_images_dir, val_manifest,
                split="val", max_samples=args.samples
            )
            manifests.extend([train_manifest, val_manifest])
        else:
            logger.info("COCO annotations not found, creating synthetic image-text data...")
            synth_manifest = DATA_DIR / "synthetic_image_text.jsonl"
            create_synthetic_image_text_manifest(synth_manifest, args.samples or 1000)
            manifests.append(synth_manifest)
    
    if "librispeech" in args.datasets:
        if not args.skip_download:
            try:
                download_librispeech(max_samples=args.samples)
            except Exception as e:
                logger.error(f"Failed to download LibriSpeech: {e}")
        
        libri_audio_dir = DATA_DIR / "librispeech" / "LibriSpeech" / "dev-clean"
        
        if libri_audio_dir.exists():
            audio_manifest = DATA_DIR / "librispeech.jsonl"
            create_librispeech_manifest(libri_audio_dir, audio_manifest, args.samples)
            manifests.append(audio_manifest)
        else:
            logger.info("Creating synthetic audio-text data...")
            audio_manifest = DATA_DIR / "synthetic_audio_text.jsonl"
            create_synthetic_audio_manifest(audio_manifest, args.samples or 500)
            manifests.append(audio_manifest)
    
    if "video" in args.datasets:
        video_manifest = DATA_DIR / "synthetic_video.jsonl"
        create_synthetic_video_manifest(video_manifest, args.samples or 200)
        manifests.append(video_manifest)
    
    if "music" in args.datasets:
        music_manifest = DATA_DIR / "synthetic_music.jsonl"
        create_synthetic_music_manifest(music_manifest, args.samples or 200)
        manifests.append(music_manifest)
    
    combined_train = DATA_DIR / "train.jsonl"
    combined_val = DATA_DIR / "val.jsonl"
    
    if manifests:
        if len(manifests) > 1:
            create_combined_manifest(manifests[:-1], combined_train)
            if len(manifests) > 1:
                create_combined_manifest([manifests[-1]], combined_val)
        else:
            m = manifests[0]
            with open(m) as f:
                lines = f.readlines()
            split_idx = int(len(lines) * 0.9)
            with open(combined_train, "w") as f:
                f.writelines(lines[:split_idx])
            with open(combined_val, "w") as f:
                f.writelines(lines[split_idx:])
    
    corpus_path = DATA_DIR / "corpus.jsonl"
    if combined_train.exists():
        import shutil
        shutil.copy(combined_train, corpus_path)
        
        tokenizer_path = CACHE_DIR / "tokenizer.json"
        train_tokenizer(corpus_path, tokenizer_path)
    
    logger.info("\n" + "="*50)
    logger.info("Data preparation complete!")
    logger.info(f"  Training manifest: {combined_train}")
    logger.info(f"  Validation manifest: {combined_val}")
    logger.info(f"  Tokenizer: {CACHE_DIR / 'tokenizer.json'}")
    logger.info("\nTo start training:")
    logger.info("  python train.py --data-dir data/")


def create_synthetic_image_text_manifest(output_path: Path, num_samples: int) -> int:
    """Create synthetic image-text pairs."""
    descriptions = [
        "A cat sitting on a couch",
        "A dog playing in the park",
        "A bird flying in the sky",
        "A car driving on the road",
        "A person walking on the street",
        "A flower blooming in the garden",
        "A tree swaying in the wind",
        "A boat sailing on the water",
        "A mountain covered in snow",
        "A sunset over the ocean",
    ]
    
    samples = []
    for i in range(num_samples):
        samples.append({
            "sample_id": f"synth_img_{i:05d}",
            "image_type": "synthetic",
            "height": 224,
            "width": 224,
            "channels": 3,
            "text": random.choice(descriptions),
            "modality": "image_text",
        })
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Created {output_path} with {len(samples)} synthetic image-text samples")
    return len(samples)


def create_synthetic_audio_manifest(output_path: Path, num_samples: int) -> int:
    """Create synthetic audio-text pairs."""
    transcriptions = [
        "Hello world this is a test",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is transforming technology",
        "Artificial intelligence is the future",
        "Deep learning models require lots of data",
        "Neural networks can learn complex patterns",
        "Natural language processing enables understanding",
        "Computer vision allows machines to see",
        "Speech recognition converts audio to text",
        "Multi-modal AI combines different inputs",
    ]
    
    samples = []
    for i in range(num_samples):
        samples.append({
            "sample_id": f"synth_audio_{i:05d}",
            "audio_type": "synthetic",
            "n_mels": 80,
            "frames": random.randint(100, 500),
            "text": random.choice(transcriptions),
            "modality": "audio_text",
        })
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    logger.info(f"Created {output_path} with {len(samples)} synthetic audio samples")
    return len(samples)


if __name__ == "__main__":
    main()
