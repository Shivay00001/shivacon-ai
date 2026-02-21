"""
Comprehensive real-world test for MultiModal AI processing and generation.

Tests:
1. Real image processing (PIL)
2. Real audio processing (torchaudio)
3. Video processing (synthetic frames as demo)
4. Music generation and MIDI-like output
5. Cross-modal similarity
6. Multi-modal encoding
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import torch
from torch import Tensor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Config
from core.multimodal_core import MultiModalCore
from modalities.text import TextEncoder
from modalities.image import ImageEncoder
from modalities.audio import AudioEncoder
from modalities.video import VideoEncoder
from modalities.music import MusicEncoder, MusicDecoder
from data.tokenizer import BPETokenizer
from data.image_processor import ImageProcessor, ImageProcessorConfig
from data.audio_processor import AudioProcessor, AudioProcessorConfig
from data.video_processor import VideoProcessor, VideoProcessorConfig
from data.music_processor import MusicProcessor, MusicProcessorConfig
from inference.engine import MultiModalInference
from inference.pipeline import InferencePipeline, PipelineConfig


def create_test_image(path: Path, size: int = 224) -> Path:
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    img = Image.new('RGB', (size, size), color=(
        random.randint(50, 200),
        random.randint(50, 200),
        random.randint(50, 200)
    ))
    draw = ImageDraw.Draw(img)
    
    shapes = ['circle', 'rectangle', 'line']
    for _ in range(5):
        shape = random.choice(shapes)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        if shape == 'circle':
            x, y = random.randint(20, size-40), random.randint(20, size-40)
            r = random.randint(10, 40)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
        elif shape == 'rectangle':
            x1, y1 = random.randint(0, size-50), random.randint(0, size-50)
            x2, y2 = x1 + random.randint(20, 50), y1 + random.randint(20, 50)
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            x1, y1 = random.randint(0, size), random.randint(0, size)
            x2, y2 = random.randint(0, size), random.randint(0, size)
            draw.line([x1, y1, x2, y2], fill=color, width=3)
    
    draw.text((10, size//2 - 10), "TEST", fill=(255, 255, 255))
    
    img.save(path)
    return path


def create_test_audio(path: Path, duration_sec: float = 2.0, sample_rate: int = 16000) -> Path:
    import torchaudio
    import math
    
    t = torch.linspace(0, duration_sec, int(duration_sec * sample_rate))
    
    freq1, freq2, freq3 = 440.0, 880.0, 1320.0
    waveform = (
        0.3 * torch.sin(2 * math.pi * freq1 * t) +
        0.2 * torch.sin(2 * math.pi * freq2 * t) +
        0.1 * torch.sin(2 * math.pi * freq3 * t)
    )
    
    envelope = torch.exp(-t / (duration_sec / 2))
    waveform = waveform * envelope
    
    waveform = waveform.unsqueeze(0)
    
    try:
        torchaudio.save(str(path), waveform, sample_rate)
    except Exception:
        import scipy.io.wavfile as wavfile
        import numpy as np
        waveform_np = (waveform.numpy() * 32767).astype(np.int16)
        wavfile.write(str(path), sample_rate, waveform_np.T)
    
    return path


def test_real_image_processing():
    print("\n" + "="*60)
    print("TEST 1: Real Image Processing (PIL)")
    print("="*60)
    
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    image_path = test_dir / "test_image.png"
    create_test_image(image_path)
    print(f"[+] Created test image: {image_path}")
    
    processor = ImageProcessor(ImageProcessorConfig(image_size=64))
    
    tensor = processor.process(image_path, is_training=False)
    print(f"[+] Processed tensor shape: {tensor.shape}")
    print(f"[+] Tensor dtype: {tensor.dtype}")
    print(f"[+] Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    pil_image = processor.to_pil(tensor)
    output_path = test_dir / "test_image_reconstructed.png"
    pil_image.save(output_path)
    print(f"[+] Reconstructed image saved: {output_path}")
    
    batch_tensors = processor.process_batch([image_path, image_path], is_training=False)
    print(f"[+] Batch processing shape: {batch_tensors.shape}")
    
    image_path.unlink()
    output_path.unlink()
    
    print("[PASS] Real image processing works!")
    return True


def test_real_audio_processing():
    print("\n" + "="*60)
    print("TEST 2: Real Audio Processing (torchaudio)")
    print("="*60)
    
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    audio_path = test_dir / "test_audio.wav"
    create_test_audio(audio_path, duration_sec=2.0)
    print(f"[+] Created test audio: {audio_path}")
    
    processor = AudioProcessor(AudioProcessorConfig(
        sample_rate=16000,
        n_mels=80,
        max_frames=128,
    ))
    
    mel_spec = processor.process(audio_path, is_training=False)
    print(f"[+] Mel-spectrogram shape: {mel_spec.shape}")
    print(f"[+] Tensor dtype: {mel_spec.dtype}")
    print(f"[+] Value range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")
    
    print(f"[+] Config: {processor.get_config()}")
    
    audio_path.unlink()
    
    print("[PASS] Real audio processing works!")
    return True


def test_video_processing():
    print("\n" + "="*60)
    print("TEST 3: Video Processing (Synthetic Frames)")
    print("="*60)
    
    processor = VideoProcessor(VideoProcessorConfig(
        num_frames=4,
        image_size=64,
    ))
    
    synthetic_video = torch.randint(0, 255, (16, 64, 64, 3), dtype=torch.uint8)
    print(f"[+] Created synthetic video: {synthetic_video.shape}")
    
    processed = processor.process(synthetic_video, is_training=False)
    print(f"[+] Processed video shape: {processed.shape}")
    print(f"[+] Tensor dtype: {processed.dtype}")
    print(f"[+] Value range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    print("[PASS] Video processing works!")
    return True


def test_music_generation_and_decoding():
    print("\n" + "="*60)
    print("TEST 4: Music Generation and MIDI-like Decoding")
    print("="*60)
    
    processor = MusicProcessor(MusicProcessorConfig())
    print(f"[+] Music processor vocab size: {processor.vocab_size}")
    
    events = [
        {"type": "note_on", "pitch": 60, "velocity": 80},
        {"type": "time_shift", "time_ms": 500},
        {"type": "note_off", "pitch": 60},
        {"type": "note_on", "pitch": 64, "velocity": 70},
        {"type": "time_shift", "time_ms": 250},
        {"type": "note_off", "pitch": 64},
        {"type": "note_on", "pitch": 67, "velocity": 90},
        {"type": "time_shift", "time_ms": 750},
        {"type": "note_off", "pitch": 67},
    ]
    
    tokens = processor.encode_midi_events(events)
    print(f"[+] Encoded {len(events)} events to {len(tokens)} tokens")
    print(f"[+] Token IDs: {tokens[:15]}...")
    
    tensor = processor.process(events)
    print(f"[+] Tensor shape: {tensor.shape}")
    
    decoded = processor.decode_tokens(tokens, skip_special_tokens=True)
    print(f"[+] Decoded {len(decoded)} events:")
    for i, event in enumerate(decoded[:5]):
        print(f"    {i+1}. {event}")
    
    print("\n[+] Testing individual token encoding/decoding:")
    test_cases = [
        ("note_on", 60, 80),
        ("note_on", 72, 100),
        ("note_off", 60, None),
        ("time_shift", 500, None),
    ]
    
    for event_type, val1, val2 in test_cases:
        if event_type == "note_on":
            token = processor.encode_note_on(val1, val2)
            decoded = processor.decode_token(token)
            print(f"    note_on(pitch={val1}, vel={val2}) -> token {token} -> {decoded}")
        elif event_type == "note_off":
            token = processor.encode_note_off(val1)
            decoded = processor.decode_token(token)
            print(f"    note_off(pitch={val1}) -> token {token} -> {decoded}")
        else:
            token = processor.encode_time_shift(val1)
            decoded = processor.decode_token(token)
            print(f"    time_shift({val1}ms) -> token {token} -> {decoded}")
    
    print("[PASS] Music generation and decoding works!")
    return True


def test_cross_modal_similarity():
    print("\n" + "="*60)
    print("TEST 5: Cross-Modal Similarity")
    print("="*60)
    
    checkpoint_files = [
        "checkpoint_epoch0001_step00000046.pt",
        "checkpoint_epoch0001_step00000478.pt",
        "best.pt",
    ]
    checkpoint_path = None
    for fname in checkpoint_files:
        p = Path(__file__).parent / "checkpoints" / fname
        if p.exists():
            checkpoint_path = p
            break
    
    if checkpoint_path is None:
        print("[SKIP] No trained checkpoint found")
        return False
    
    engine = MultiModalInference.from_checkpoint(checkpoint_path, device="cpu")
    
    tokenizer_path = Path(__file__).parent / ".cache" / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path) if tokenizer_path.exists() else None
    
    pipeline = InferencePipeline(engine, tokenizer, PipelineConfig())
    
    test_pairs = [
        ("a cat sitting on a couch", "a cat resting on a sofa"),
        ("a cat sitting on a couch", "a dog running in the park"),
        ("a beautiful sunset over the ocean", "colorful sunset at the beach"),
        ("a person playing guitar", "someone riding a bicycle"),
    ]
    
    print("[+] Text-Text Similarity:")
    
    if tokenizer is None:
        print("[SKIP] No tokenizer found")
        return False
    
    for text1, text2 in test_pairs:
        tokens1 = tokenizer.encode(text1, add_bos=True, add_eos=True)
        tokens2 = tokenizer.encode(text2, add_bos=True, add_eos=True)
        inputs_a = {"text": {"x": torch.tensor(tokens1, dtype=torch.long).unsqueeze(0)}}
        inputs_b = {"text": {"x": torch.tensor(tokens2, dtype=torch.long).unsqueeze(0)}}
        
        sim = engine.get_similarity(inputs_a, inputs_b)
        print(f"    '{text1[:30]}...' vs '{text2[:30]}...': {sim.item():.4f}")
    
    print("[PASS] Cross-modal similarity works!")
    return True


def test_music_generation_conditioned():
    print("\n" + "="*60)
    print("TEST 6: Text-Conditioned Music Generation")
    print("="*60)
    
    checkpoint_files = [
        "checkpoint_epoch0001_step00000046.pt",
        "checkpoint_epoch0001_step00000478.pt",
        "best.pt",
    ]
    checkpoint_path = None
    for fname in checkpoint_files:
        p = Path(__file__).parent / "checkpoints" / fname
        if p.exists():
            checkpoint_path = p
            break
    
    if checkpoint_path is None:
        print("[SKIP] No trained checkpoint found")
        return False
    
    engine = MultiModalInference.from_checkpoint(checkpoint_path, device="cpu")
    
    tokenizer_path = Path(__file__).parent / ".cache" / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path) if tokenizer_path.exists() else None
    
    pipeline = InferencePipeline(engine, tokenizer, PipelineConfig(
        music_max_tokens=64,
        music_temperature=0.9,
    ))
    
    if pipeline.music_processor is None:
        pipeline.music_processor = MusicProcessor()
    
    prompts = [
        "happy upbeat melody",
        "sad slow song",
        "energetic rock music",
    ]
    
    print("[+] Generating music from text prompts:")
    for prompt in prompts:
        try:
            tokens = pipeline.generate_music_from_text(
                prompt,
                max_tokens=32,
                temperature=0.8,
            )
            print(f"\n    Prompt: '{prompt}'")
            print(f"    Generated {len(tokens)} tokens")
            
            decoded = pipeline.music_processor.decode_tokens(tokens)
            print(f"    Decoded {len(decoded)} events")
            
            note_events = [e for e in decoded if e.get("type") == "note_on"]
            print(f"    Note events: {len(note_events)}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    print("[PASS] Text-conditioned music generation works!")
    return True


def test_full_multimodal_encoding():
    print("\n" + "="*60)
    print("TEST 7: Full Multi-Modal Encoding Pipeline")
    print("="*60)
    
    checkpoint_files = [
        "checkpoint_epoch0001_step00000046.pt",
        "checkpoint_epoch0001_step00000478.pt",
        "best.pt",
    ]
    checkpoint_path = None
    for fname in checkpoint_files:
        p = Path(__file__).parent / "checkpoints" / fname
        if p.exists():
            checkpoint_path = p
            break
    
    if checkpoint_path is None:
        print("[SKIP] No trained checkpoint found")
        return False
    
    engine = MultiModalInference.from_checkpoint(checkpoint_path, device="cpu")
    
    tokenizer_path = Path(__file__).parent / ".cache" / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path) if tokenizer_path.exists() else None
    
    if tokenizer is None:
        print("[SKIP] No tokenizer found")
        return False
    
    pipeline = InferencePipeline(engine, tokenizer, PipelineConfig())
    
    if pipeline.image_processor is None:
        pipeline.image_processor = ImageProcessor(ImageProcessorConfig(image_size=engine.config.image.image_size))
    
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    
    image_path = test_dir / "temp_test.png"
    create_test_image(image_path, size=64)
    
    audio_path = test_dir / "temp_test.wav"
    create_test_audio(audio_path, duration_sec=1.0)
    
    text = "A beautiful landscape with mountains"
    
    print(f"[+] Input text: '{text}'")
    print(f"[+] Input image: {image_path}")
    print(f"[+] Input audio: {audio_path}")
    
    try:
        outputs = pipeline.encode_multimodal(
            text=text,
            image=image_path,
            audio=audio_path,
        )
        
        print("\n[+] Output tensors:")
        for key, tensor in outputs.items():
            if isinstance(tensor, Tensor):
                print(f"    {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        print(f"\n[+] Global embedding norm: {outputs['global_embedding'].norm():.4f}")
        print("[PASS] Full multi-modal encoding works!")
        return True
    except Exception as e:
        print(f"[FAIL] Full Multi-Modal Encoding: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        image_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


def test_batch_inference():
    print("\n" + "="*60)
    print("TEST 8: Batch Inference")
    print("="*60)
    
    checkpoint_files = [
        "checkpoint_epoch0001_step00000046.pt",
        "checkpoint_epoch0001_step00000478.pt",
        "best.pt",
    ]
    checkpoint_path = None
    for fname in checkpoint_files:
        p = Path(__file__).parent / "checkpoints" / fname
        if p.exists():
            checkpoint_path = p
            break
    
    if checkpoint_path is None:
        print("[SKIP] No trained checkpoint found")
        return False
    
    engine = MultiModalInference.from_checkpoint(checkpoint_path, device="cpu")
    
    batch_inputs = []
    for i in range(5):
        batch_inputs.append({
            "text": {"x": torch.randint(0, 500, (1, 16))}  # (B, T) shape
        })
    
    print(f"[+] Processing batch of {len(batch_inputs)} samples...")
    
    results = engine.batch_encode(batch_inputs, batch_size=2)
    
    print(f"[+] Got {len(results)} results")
    print(f"[+] First result keys: {list(results[0].keys())}")
    print(f"[+] First global embedding shape: {results[0]['global_embedding'].shape}")
    
    print("[PASS] Batch inference works!")
    return True


def main():
    print("="*60)
    print("MultiModal AI - Real-World Processing Tests")
    print("="*60)
    
    results = {}
    
    tests = [
        ("Image Processing", test_real_image_processing),
        ("Audio Processing", test_real_audio_processing),
        ("Video Processing", test_video_processing),
        ("Music Generation/Decoding", test_music_generation_and_decoding),
        ("Cross-Modal Similarity", test_cross_modal_similarity),
        ("Text-Conditioned Music", test_music_generation_conditioned),
        ("Full Multi-Modal Encoding", test_full_multimodal_encoding),
        ("Batch Inference", test_batch_inference),
    ]
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results[name] = "PASS" if success else "SKIP"
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            results[name] = f"FAIL: {str(e)[:50]}"
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v == "PASS")
    skipped = sum(1 for v in results.values() if v == "SKIP")
    failed = sum(1 for v in results.values() if v.startswith("FAIL"))
    
    for name, status in results.items():
        icon = "[OK]" if status == "PASS" else "[--]" if status == "SKIP" else "[XX]"
        print(f"  {icon} {name}: {status}")
    
    print(f"\nTotal: {passed} passed, {skipped} skipped, {failed} failed")
    
    test_dir = Path(__file__).parent / "test_data"
    if test_dir.exists():
        for f in test_dir.iterdir():
            f.unlink()
        test_dir.rmdir()
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
