"""
Production-grade checkpoint management.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    epoch: int
    global_step: int
    timestamp: str
    metrics: Dict[str, float]
    config_hash: Optional[str] = None
    model_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(**d)


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last_n: int = 5,
        monitor_metric: str = "val/loss",
        monitor_mode: str = "min",
        save_best_only: bool = False,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.save_best_only = save_best_only
        
        self._best_metric: Optional[float] = None
        self._saved_checkpoints: List[Path] = []
        self._metadata_file = self.checkpoint_dir / "checkpoint_history.json"
        
        self._load_history()
    
    def _load_history(self) -> None:
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    history = json.load(f)
                if history:
                    last_entry = history[-1]
                    self._best_metric = last_entry.get("best_metric")
            except (json.JSONDecodeError, KeyError):
                pass
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> Optional[Path]:
        metrics = metrics or {}
        
        current_metric = metrics.get(self.monitor_metric)
        
        is_best_checkpoint = False
        if current_metric is not None:
            if self._best_metric is None:
                is_best_checkpoint = True
            elif self.monitor_mode == "min" and current_metric < self._best_metric:
                is_best_checkpoint = True
            elif self.monitor_mode == "max" and current_metric > self._best_metric:
                is_best_checkpoint = True
            
            if is_best_checkpoint:
                self._best_metric = current_metric
        
        if self.save_best_only and not is_best_checkpoint:
            return None
        
        checkpoint_name = f"checkpoint_epoch{epoch:04d}_step{global_step:08d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        
        if extra_state:
            state["extra_state"] = extra_state
        
        state["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()
        
        torch.save(state, checkpoint_path)
        self._saved_checkpoints.append(checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        if is_best_checkpoint:
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copy2(checkpoint_path, best_path)
            logger.info(f"New best checkpoint! {self.monitor_metric}={current_metric:.4f}")
        
        self._prune_old_checkpoints()
        self._update_metadata(epoch, global_step, metrics, is_best_checkpoint)
        
        return checkpoint_path
    
    def load(
        self,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: Optional[torch.device] = None,
        strict: bool = True,
        load_rng: bool = True,
    ) -> Dict[str, Any]:
        if checkpoint_path is None or checkpoint_path == "best":
            checkpoint_path = str(self.checkpoint_dir / "best.pt")
        elif checkpoint_path == "latest":
            checkpoint_path = str(self._get_latest_checkpoint())
        
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        map_location = device if device else "cpu"
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state_dict"], strict=strict
        )
        
        if missing:
            logger.warning(f"Missing keys in checkpoint: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected[:5]}...")
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.debug("Loaded optimizer state")
        
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.debug("Loaded scheduler state")
        
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.debug("Loaded scaler state")
        
        if load_rng and "rng_state" in checkpoint:
            rng_state = checkpoint["rng_state"]
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if "cuda" in rng_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            logger.debug("Restored RNG state")
        
        epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        metrics = checkpoint.get("metrics", {})
        
        logger.info(
            f"Loaded checkpoint from {path} "
            f"(epoch={epoch}, step={global_step})"
        )
        
        return {
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
        }
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    def _prune_old_checkpoints(self) -> None:
        if self.keep_last_n <= 0:
            return
        
        while len(self._saved_checkpoints) > self.keep_last_n:
            old_checkpoint = self._saved_checkpoints.pop(0)
            if old_checkpoint.exists() and old_checkpoint.name != "best.pt":
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def _update_metadata(
        self,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        history = []
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = []
        
        entry = {
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "is_best": is_best,
            "best_metric": self._best_metric,
        }
        history.append(entry)
        
        with open(self._metadata_file, "w") as f:
            json.dump(history, f, indent=2)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        checkpoints = []
        for ckpt_path in self.checkpoint_dir.glob("checkpoint_epoch*.pt"):
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                checkpoints.append({
                    "path": str(ckpt_path),
                    "epoch": state.get("epoch", 0),
                    "global_step": state.get("global_step", 0),
                    "metrics": state.get("metrics", {}),
                })
            except Exception as e:
                logger.warning(f"Could not read checkpoint {ckpt_path}: {e}")
        
        return sorted(checkpoints, key=lambda x: x["global_step"])
