import time
import torch
import torch.nn as nn
from core.multimodal_core import MultiModalCore
from config.settings import Config
from modalities.text import TextEncoder
from modalities.image import ImageEncoder
from training.finetune import FineTuner, FineTuneConfig

def benchmark_llm_finetuning():
    print("====================================================")
    print(" 🚀 OMNICORE LLM SCALE-UP & FINE-TUNING BENCHMARK 🚀 ")
    print("====================================================\n")
    
    # 1. Initialize Base Architecture
    print("[1] Initializing Base OmniCore Architecture...")
    config = Config()
    
    # Scale down sizes slightly just for local CPU/GPU synthetic benchmark throughput 
    config.text.d_model = 256
    config.text.num_layers = 4
    config.fusion.latent_dim = 256
    config.fusion.num_cross_attn_layers = 2
    
    model = MultiModalCore(config.fusion)
    model.register_encoder(TextEncoder(config.text))
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f" -> Base Model Parameters: {total_params:,}")
    
    # 2. Configure LoRA Fine-Tuning
    print("\n[2] Injecting LoRA (Low-Rank Adaptation) Matrices...")
    ft_config = FineTuneConfig(
        mode="lora",
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        learning_rate=2e-4,
        max_steps=5,  # Just benchmark 5 steps
        batch_size=2
    )
    
    # We use a dummy tokenizer for the benchmark wrapper
    class DummyTokenizer:
        def encode(self, text, **kwargs):
            return [1, 2, 3, 4, 5]
            
    tuner = FineTuner(model, DummyTokenizer(), ft_config)
    
    # 3. Memory & Trainable Parameter Analysis
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> LoRA Trainable Parameters: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    
    # 4. Throughput Benchmarking (Simulated Data)
    print("\n[3] Running Synthetic Training Forward/Backward Passes...")
    
    # Simulate a dataloader
    dummy_input = torch.randint(0, 1000, (ft_config.batch_size, 32)).to(tuner.device)
    dummy_labels = torch.randint(0, 1000, (ft_config.batch_size, 32)).to(tuner.device)
    dummy_loader = [{"input_ids": dummy_input, "labels": dummy_labels} for _ in range(5)]
    
    start_time = time.time()
    try:
        tuner.train(dummy_loader)
        end_time = time.time()
        
        step_time = (end_time - start_time) / ft_config.max_steps
        tokens_per_sec = (ft_config.batch_size * 32) / step_time
        
        print(f"\n✅ Benchmark Successful!")
        print(f" -> Average Step Latency: {step_time:.3f} seconds")
        print(f" -> Training Throughput: {tokens_per_sec:.2f} tokens/sec")
        
    except Exception as e:
        print(f"\n❌ Benchmark Failed: {e}")

    print("\n====================================================")
    print(" BENCHMARK COMPLETE ")
    print("====================================================")

if __name__ == "__main__":
    benchmark_llm_finetuning()
