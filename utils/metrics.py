"""
Training metrics tracking and aggregation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import time

logger = logging.getLogger(__name__)


@dataclass
class MetricRecord:
    step: int
    value: float
    timestamp: float
    epoch: Optional[int] = None


class MetricsTracker:
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        tensorboard: bool = False,
        wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir) if log_dir else None
        self.tensorboard_enabled = tensorboard
        self.wandb_enabled = wandb
        
        self._metrics: Dict[str, List[MetricRecord]] = defaultdict(list)
        self._epoch_metrics: Dict[str, List[float]] = defaultdict(list)
        self._current_epoch: int = 0
        self._global_step: int = 0
        self._start_time: float = time.time()
        
        self._tensorboard_writer = None
        self._wandb_run = None
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if self.tensorboard_enabled:
            self._init_tensorboard()
        
        if self.wandb_enabled:
            self._init_wandb(wandb_project)
    
    def _init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.log_dir / "tensorboard" if self.log_dir else None
            self._tensorboard_writer = SummaryWriter(log_dir=str(tb_dir) if tb_dir else None)
            logger.info(f"TensorBoard initialized: {tb_dir}")
        except ImportError:
            logger.warning("tensorboard not installed. Install with: pip install tensorboard")
            self.tensorboard_enabled = False
    
    def _init_wandb(self, project: Optional[str]) -> None:
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=project or "multimodal-ai",
                config={},
                reinit=True,
            )
            logger.info(f"W&B initialized: {self._wandb_run.url}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.wandb_enabled = False
        except Exception as e:
            logger.warning(f"Could not initialize W&B: {e}")
            self.wandb_enabled = False
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        step = step if step is not None else self._global_step
        timestamp = time.time()
        
        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            
            record = MetricRecord(
                step=step,
                value=float(value),
                timestamp=timestamp,
                epoch=self._current_epoch,
            )
            self._metrics[name].append(record)
            self._epoch_metrics[name].append(float(value))
        
        if self._tensorboard_writer:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tensorboard_writer.add_scalar(name, value, step)
        
        if self._wandb_run:
            import wandb
            wandb.log(metrics, step=step)
    
    def log_histogram(self, name: str, values: Any, step: Optional[int] = None) -> None:
        step = step if step is not None else self._global_step
        
        if self._tensorboard_writer:
            import torch
            if isinstance(values, torch.Tensor):
                self._tensorboard_writer.add_histogram(name, values, step)
        
        if self._wandb_run:
            import wandb
            wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch
    
    def set_step(self, step: int) -> None:
        self._global_step = step
    
    def step(self) -> None:
        self._global_step += 1
    
    def get_epoch_average(self, name: str) -> Optional[float]:
        if name not in self._epoch_metrics or not self._epoch_metrics[name]:
            return None
        values = self._epoch_metrics[name]
        return sum(values) / len(values)
    
    def get_all_epoch_averages(self) -> Dict[str, float]:
        return {
            name: avg
            for name, avg in (
                (name, self.get_epoch_average(name))
                for name in self._epoch_metrics
            )
            if avg is not None
        }
    
    def reset_epoch_metrics(self) -> None:
        self._epoch_metrics.clear()
    
    def get_latest(self, name: str) -> Optional[float]:
        if name not in self._metrics or not self._metrics[name]:
            return None
        return self._metrics[name][-1].value
    
    def get_history(self, name: str) -> List[MetricRecord]:
        return self._metrics.get(name, [])
    
    def save(self, path: Optional[Path] = None) -> None:
        path = path or (self.log_dir / "metrics.json" if self.log_dir else None)
        if path is None:
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            name: [
                {"step": r.step, "value": r.value, "timestamp": r.timestamp, "epoch": r.epoch}
                for r in records
            ]
            for name, records in self._metrics.items()
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved metrics to {path}")
    
    def close(self) -> None:
        if self._tensorboard_writer:
            self._tensorboard_writer.close()
        
        if self._wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        
        if self.log_dir:
            self.save()
    
    def summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        return {
            "total_steps": self._global_step,
            "current_epoch": self._current_epoch,
            "elapsed_seconds": elapsed,
            "latest_metrics": {
                name: self.get_latest(name)
                for name in self._metrics
            },
            "epoch_averages": self.get_all_epoch_averages(),
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
