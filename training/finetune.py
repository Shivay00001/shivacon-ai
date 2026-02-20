"""
SHIVACON AI - Fine-Tuning Module
================================
Fine-tune Shivacon on custom datasets:
- Full Parameter Fine-tuning
- LoRA (Low-Rank Adaptation)
- Freeze/Unfreeze layers
- Checkpoint management
- Training from pretrained
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning."""
    
    # Training mode
    mode: str = "full"  # "full", "lora", "freeze"
    
    # LoRA config
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Freeze config
    freeze_embeddings: bool = True
    freeze_encoder_layers: int = 0
    
    # Training params
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    gradient_clip: float = 1.0
    gradient_accumulation: int = 4
    
    # Data
    batch_size: int = 4
    max_seq_length: int = 128
    train_split: float = 0.9
    
    # Checkpointing
    save_every: int = 100
    eval_every: int = 100
    checkpoint_dir: str = "checkpoints/finetune"
    
    # Model
    pretrained_path: Optional[str] = None
    output_path: str = "checkpoints/finetuned"


class FineTuneDataset(Dataset):
    """Dataset for fine-tuning."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 128,
        modality: str = "text",
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.modality = modality
        self.samples = self._load_data()
    
    def _load_data(self) -> List[Dict]:
        samples = []
        
        if self.data_path.suffix == ".jsonl":
            with open(self.data_path) as f:
                for line in f:
                    data = json.loads(line)
                    samples.append(data)
        
        elif self.data_path.suffix == ".json":
            with open(self.data_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]
        
        elif self.data_path.suffix == ".txt":
            with open(self.data_path) as f:
                for line in f:
                    if line.strip():
                        samples.append({"text": line.strip()})
        
        logger.info(f"Loaded {len(samples)} samples from {self.data_path}")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        if self.modality == "text":
            text = sample.get("text", "")
            tokens = self.tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True,
                max_length=self.max_length,
            )
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "labels": torch.tensor(tokens, dtype=torch.long),
            }
        
        elif self.modality == "image":
            # For image, return dummy (actual image loading would be here)
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.zeros(self.max_length, dtype=torch.long),
            }
        
        else:
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.zeros(self.max_length, dtype=torch.long),
            }


class LoRALayer(nn.Module):
    """LoRA layer implementation."""
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        
        # Freeze original layer
        for param in original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        original_output = self.original_layer(x)
        
        # LoRA forward
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return original_output + lora_output * self.scaling
    
    def merge_weights(self):
        """Merge LoRA weights into original layer."""
        with torch.no_grad():
            # W = W + B * A * alpha/rank
            lora_weight = self.lora_B @ self.lora_A * self.scaling
            self.original_layer.weight.data += lora_weight


class FineTuner:
    """
    Fine-tuning engine for Shivacon.
    
    Supports:
    - Full parameter fine-tuning
    - LoRA (Low-Rank Adaptation)
    - Freeze embeddings
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: FineTuneConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Apply fine-tuning setup
        self._setup_model()
        
        # Optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.best_loss = float("inf")
        
        logger.info(f"Fine-tuner initialized in {config.mode} mode")
        logger.info(f"Device: {self.device}")
    
    def _setup_model(self):
        """Setup model for fine-tuning."""
        
        if self.config.mode == "lora":
            self._apply_lora()
        
        elif self.config.mode == "freeze":
            self._freeze_layers()
        
        elif self.config.mode == "full":
            # All parameters trainable
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    def _apply_lora(self):
        """Apply LoRA to model layers."""
        
        def apply_lora_to_linear(module: nn.Module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Check if it's a target module
                    for target in self.config.lora_target_modules:
                        if target in name:
                            lora_layer = LoRALayer(
                                child,
                                rank=self.config.lora_rank,
                                alpha=self.config.lora_alpha,
                                dropout=self.config.lora_dropout,
                            )
                            setattr(module, name, lora_layer)
                            logger.info(f"Applied LoRA to {name}")
                else:
                    apply_lora_to_linear(child)
        
        apply_lora_to_linear(self.model)
    
    def _freeze_layers(self):
        """Freeze model layers."""
        
        # Freeze embeddings
        if self.config.freeze_embeddings:
            if hasattr(self.model, 'embedding') or hasattr(self.model, 'embeddings'):
                for param in self.model.parameters():
                    if 'embed' in str(type(param)).lower():
                        param.requires_grad = False
        
        # Freeze encoder layers
        if self.config.freeze_encoder_layers > 0:
            frozen = 0
            for param in self.model.parameters():
                if frozen < self.config.freeze_encoder_layers:
                    param.requires_grad = False
                    frozen += 1
    
    def prepare_data(
        self,
        train_path: Union[str, Path],
        val_path: Optional[Union[str, Path]] = None,
        modality: str = "text",
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare data loaders for fine-tuning."""
        
        train_dataset = FineTuneDataset(
            train_path,
            self.tokenizer,
            max_length=self.config.max_seq_length,
            modality=modality,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        val_loader = None
        if val_path:
            val_dataset = FineTuneDataset(
                val_path,
                self.tokenizer,
                max_length=self.config.max_seq_length,
                modality=modality,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
            )
        
        return train_loader, val_loader
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Train the model."""
        
        # Setup optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        
        # Move model to device
        self.model.to(self.device)
        self.model.train()
        
        # Training loop
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }
        
        logger.info(f"Starting training for {self.config.max_steps} steps")
        
        step = 0
        epoch = 0
        
        while step < self.config.max_steps:
            for batch in train_loader:
                if step >= self.config.max_steps:
                    break
                
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    {"text": {"x": input_ids}},
                    active_modalities=["text"],
                )
                
                # Compute loss (simplified)
                loss = torch.nn.functional.cross_entropy(
                    outputs["global_embedding"],
                    labels[:, 0].to(self.device) if labels.numel() > 0 else input_ids[:, 0],
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                history["train_loss"].append(loss.item())
                history["learning_rate"].append(self.scheduler.get_last_lr()[0])
                
                # Validation
                if val_loader and step % self.config.eval_every == 0:
                    val_loss = self._evaluate(val_loader)
                    history["val_loss"].append(val_loss)
                    
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint("best")
                
                # Save checkpoint
                if step % self.config.save_every == 0:
                    self.save_checkpoint(f"step_{step}")
                
                step += 1
                self.global_step = step
                
                if step % 10 == 0:
                    logger.info(
                        f"Step {step}/{self.config.max_steps} | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                    )
            
            epoch += 1
        
        logger.info("Training complete!")
        return history
    
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate the model."""
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    {"text": {"x": input_ids}},
                    active_modalities=["text"],
                )
                
                loss = torch.nn.functional.cross_entropy(
                    outputs["global_embedding"],
                    labels[:, 0],
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def save_checkpoint(self, name: str):
        """Save fine-tuned checkpoint."""
        
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"finetune_{name}.pt"
        
        # Collect state
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "config": {
                "mode": self.config.mode,
                "lora_rank": self.config.lora_rank,
                "learning_rate": self.config.learning_rate,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load fine-tuned checkpoint."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint.get("global_step", 0)
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    
    def merge_lora(self):
        """Merge LoRA weights for inference."""
        
        def merge(module: nn.Module):
            for child in module.children():
                if isinstance(child, LoRALayer):
                    child.merge_weights()
                else:
                    merge(child)
        
        merge(self.model)
        logger.info("Merged LoRA weights")


def create_fine_tuner(
    model: nn.Module,
    tokenizer,
    mode: str = "lora",
    **kwargs,
) -> FineTuner:
    """Create a fine-tuner with specified configuration."""
    
    config = FineTuneConfig(mode=mode, **kwargs)
    return FineTuner(model, tokenizer, config)


def load_pretrained_model(
    model_path: str,
    tokenizer_path: str,
    device: str = "auto",
) -> Tuple[nn.Module, Any]:
    """Load pretrained model for fine-tuning."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # This would load the actual model architecture
    # For now, return dummy
    logger.info(f"Loaded pretrained from {model_path}")
    
    # Load tokenizer
    from data.tokenizer import BPETokenizer
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    return None, tokenizer


# Example usage
if __name__ == "__main__":
    print("""
    SHIVACON AI - Fine-Tuning Module
    =================================
    
    Usage:
    
    1. Create fine-tuner:
       from training.finetune import FineTuner, FineTuneConfig
      
       config = FineTuneConfig(
           mode="lora",           # "full", "lora", "freeze"
           learning_rate=1e-4,
           max_steps=1000,
           batch_size=4,
       )
      
       tuner = FineTuner(model, tokenizer, config)
    
    2. Prepare data:
       train_loader, val_loader = tuner.prepare_data(
           train_path="data/train.jsonl",
           val_path="data/val.jsonl",
       )
    
    3. Train:
       history = tuner.train(train_loader, val_loader)
    
    4. Save:
       tuner.save_checkpoint("final")
    
    5. Load and use:
       tuner.load_checkpoint("checkpoints/finetune/best.pt")
    """)
