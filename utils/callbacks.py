"""Training callbacks: early stopping, checkpoint management."""

import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best: Optional[float] = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class CheckpointManager:
    """Saves periodic checkpoints and keeps the best model separately."""

    def __init__(self, output_dir: str, save_every: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.best_metric: Optional[float] = None

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer,
        scheduler,
        metrics: dict,
        cfg,
        monitor: str = "auroc",
    ) -> str:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": cfg.to_dict(),
        }

        path = ""

        # Periodic save
        if self.save_every > 0 and epoch % self.save_every == 0:
            path = str(self.output_dir / f"checkpoint_epoch{epoch:04d}.pt")
            torch.save(state, path)

        # Best model
        current = metrics.get(monitor, float("-inf"))
        if self.best_metric is None or current > self.best_metric:
            self.best_metric = current
            best_path = str(self.output_dir / "best_model.pt")
            torch.save(state, best_path)
            path = path or best_path

        return path

    def load(self, path: str, model: nn.Module, optimizer=None, scheduler=None):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer and ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt.get("epoch", 0), ckpt.get("metrics", {})
