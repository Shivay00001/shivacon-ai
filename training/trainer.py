"""
Production-grade training loop implementation.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config.settings import Config, TrainingConfig
from core.multimodal_core import MultiModalCore
from training.losses import MultiModalLoss
from training.scheduler import build_lr_scheduler
from utils.checkpoint_utils import CheckpointManager
from utils.metrics import MetricsTracker
from utils.device_utils import get_device, move_to_device, get_autocast_context

logger = logging.getLogger(__name__)


class Trainer:
    """
    Production training orchestrator.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Gradient clipping
    - Warmup + cosine scheduling
    - Checkpoint management
    - Metrics tracking
    - Early stopping
    """
    
    def __init__(
        self,
        model: MultiModalCore,
        train_loader: DataLoader,
        config: Config,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.train_cfg = config.training
        
        self.device = get_device(config.inference.device)
        self.model.to(self.device)
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model parameters: {model.count_parameters():,}")
        
        self.optimizer = self._build_optimizer()
        
        total_steps = len(train_loader) * self.train_cfg.num_epochs
        self.scheduler = build_lr_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.train_cfg.lr_scheduler,
            warmup_steps=self.train_cfg.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=self.train_cfg.min_lr_ratio,
        )
        
        self.use_amp = (
            self.train_cfg.mixed_precision and
            self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        
        self.loss_fn = loss_fn or MultiModalLoss(
            contrastive_weight=self.train_cfg.loss_weights.get("contrastive", 1.0),
            reconstruction_weight=self.train_cfg.loss_weights.get("reconstruction", 1.0),
            alignment_weight=self.train_cfg.loss_weights.get("alignment", 0.5),
        )
        
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(config.checkpoint.checkpoint_dir),
            keep_last_n=config.checkpoint.keep_last_n,
            monitor_metric=config.checkpoint.monitor_metric,
            monitor_mode=config.checkpoint.monitor_mode,
        )
        
        self.metrics = MetricsTracker(
            log_dir=Path(config.checkpoint.log_dir),
            tensorboard=self.train_cfg.use_tensorboard,
            wandb=self.train_cfg.use_wandb,
            wandb_project=self.train_cfg.wandb_project,
        )
        
        self._global_step = 0
        self._current_epoch = 0
        self._best_metric = float("inf")
        
        self._seed_everything(self.train_cfg.seed)
    
    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if self.train_cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _build_optimizer(self) -> AdamW:
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "bias" in name or "norm" in name or "pos_embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return AdamW(
            param_groups,
            lr=self.train_cfg.learning_rate,
            betas=self.train_cfg.optimizer_betas,
            eps=self.train_cfg.optimizer_eps,
        )
    
    def _batch_to_device(
        self,
        batch: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return move_to_device(batch, self.device)
    
    def train(self) -> Dict[str, float]:
        """
        Run the full training loop.
        
        Returns:
            Final metrics dictionary
        """
        logger.info(
            f"Starting training: {self.train_cfg.num_epochs} epochs, "
            f"batch_size={self.train_cfg.batch_size}, "
            f"AMP={'on' if self.use_amp else 'off'}"
        )
        
        for epoch in range(self._current_epoch, self.train_cfg.num_epochs):
            self._current_epoch = epoch
            self.metrics.set_epoch(epoch)
            
            train_loss = self._train_epoch(epoch)
            
            self.metrics.log(
                {"train/loss": train_loss, "train/epoch": epoch},
                step=self._global_step,
            )
            
            logger.info(f"Epoch {epoch:04d} | train_loss={train_loss:.4f}")
            
            if (epoch + 1) % self.train_cfg.eval_every_n_epochs == 0:
                if self.val_loader is not None:
                    val_loss = self._eval_epoch(epoch)
                    self.metrics.log(
                        {"val/loss": val_loss, "val/epoch": epoch},
                        step=self._global_step,
                    )
                    logger.info(f"Epoch {epoch:04d} | val_loss={val_loss:.4f}")
                    
                    if val_loss < self._best_metric:
                        self._best_metric = val_loss
                        self.checkpoint_manager.save(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=epoch,
                            global_step=self._global_step,
                            metrics={"val/loss": val_loss},
                            is_best=True,
                        )
            
            if (epoch + 1) % self.train_cfg.save_every_n_epochs == 0:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    epoch=epoch,
                    global_step=self._global_step,
                    metrics={"train/loss": train_loss},
                )
        
        self.metrics.close()
        logger.info("Training complete.")
        
        return self.metrics.summary()
    
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            step_start = time.perf_counter()
            
            batch = self._batch_to_device(batch)
            
            with get_autocast_context(
                self.device,
                enabled=self.use_amp,
                dtype=torch.bfloat16 if self.train_cfg.precision == "bf16" else torch.float16,
            ):
                outputs = self.model(batch)
                
                loss_dict = self.loss_fn(
                    fused_dict=outputs.get("fused", {}),
                    global_embedding=outputs["global_embedding"],
                )
                loss = loss_dict["total"]
                
                loss = loss / self.train_cfg.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.train_cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg.max_grad_norm,
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self._global_step += 1
            
            step_loss = loss.item() * self.train_cfg.gradient_accumulation_steps
            total_loss += step_loss
            num_batches += 1
            
            step_time = time.perf_counter() - step_start
            
            if self._global_step % self.train_cfg.log_every_n_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.metrics.log(
                    {
                        "train/step_loss": step_loss,
                        "train/lr": lr,
                        "train/step_time": step_time,
                    },
                    step=self._global_step,
                )
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = self._batch_to_device(batch)
            
            with get_autocast_context(
                self.device,
                enabled=self.use_amp,
            ):
                outputs = self.model(batch)
                
                loss_dict = self.loss_fn(
                    fused_dict=outputs.get("fused", {}),
                    global_embedding=outputs["global_embedding"],
                )
            
            total_loss += loss_dict["total"].item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        state = self.checkpoint_manager.load(
            model=self.model,
            checkpoint_path=checkpoint_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )
        
        self._current_epoch = state.get("epoch", 0) + 1
        self._global_step = state.get("global_step", 0)
        
        logger.info(
            f"Resumed from epoch {self._current_epoch - 1}, "
            f"step {self._global_step}"
        )
