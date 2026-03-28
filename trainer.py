"""Core training loop with AMP, Kornia GPU augmentation, TensorBoard, and rich logging."""

import os
import time
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data.transforms import KorniaAugmentationModule
from utils.metrics import MetricTracker
from utils.callbacks import EarlyStopping, CheckpointManager

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer,
        scheduler,
        cfg: Config,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)

        # Move criterion buffers (e.g. class weights) to device
        self.criterion = criterion.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.scaler = GradScaler(enabled=cfg.use_amp and self.device.type == "cuda")

        # GPU-side Kornia augmentations (train only)
        self.kornia_aug = KorniaAugmentationModule(cfg).to(self.device)
        self.kornia_aug.train()

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
            mode="max",
        )
        self.ckpt_manager = CheckpointManager(cfg.output_dir, cfg.save_every)

        # TensorBoard
        self.writer: Optional[SummaryWriter] = None
        if cfg.use_tensorboard:
            Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=cfg.log_dir)

        # Rich console
        self.console = Console() if (_RICH and cfg.use_rich) else None

        self.start_epoch = 0

        # Resume from checkpoint
        if cfg.resume:
            self.start_epoch, _ = self.ckpt_manager.load(
                cfg.resume, self.model, self.optimizer, self.scheduler
            )
            self.start_epoch += 1
            self._log(f"Resumed from {cfg.resume} at epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        best_metrics = {}

        for epoch in range(self.start_epoch, self.cfg.epochs):
            # Unfreeze backbone after freeze_epochs
            if epoch == self.cfg.freeze_epochs and self.cfg.freeze_epochs > 0:
                self.model.unfreeze_backbone()
                self._log(f"Epoch {epoch}: backbone unfrozen.")

            t0 = time.time()
            train_metrics = self._run_epoch(train_loader, epoch, training=True)
            val_metrics   = self._run_epoch(val_loader,   epoch, training=False)
            elapsed = time.time() - t0

            self._log_epoch(epoch, train_metrics, val_metrics, elapsed)

            if self.writer:
                self._write_tb(epoch, train_metrics, "train")
                self._write_tb(epoch, val_metrics,   "val")
                self._write_lr(epoch)

            # Scheduler step (OneCycleLR steps per batch, handled inside _run_epoch)
            if self.cfg.scheduler != "onecycle":
                self.scheduler.step()

            # Checkpoint
            self.ckpt_manager.save(
                epoch, self.model, self.optimizer, self.scheduler, val_metrics, self.cfg
            )

            # Early stopping on val AUROC
            monitor_val = val_metrics.get("auroc", val_metrics.get("acc", 0.0))
            if self.early_stopping(monitor_val):
                self._log(f"Early stopping at epoch {epoch} (patience={self.cfg.early_stopping_patience})")
                break

            best_metrics = val_metrics

        if self.writer:
            self.writer.close()

        return best_metrics

    def evaluate(self, loader: DataLoader) -> dict:
        """Run one eval pass and return metrics + full report."""
        self.model.eval()
        tracker = MetricTracker(self.cfg.num_classes, self.cfg.class_names)

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with autocast(enabled=self.cfg.use_amp and self.device.type == "cuda"):
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)
                tracker.update(logits, labels, loss.item())

        metrics = tracker.compute()
        report  = tracker.classification_report()
        cm      = tracker.confusion_matrix()

        self._log("\n" + report)
        return metrics

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, epoch: int, training: bool) -> dict:
        self.model.train(training)
        self.kornia_aug.train(training)

        tracker = MetricTracker(self.cfg.num_classes, self.cfg.class_names)
        use_amp = self.cfg.use_amp and self.device.type == "cuda"

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if training:
                images = self.kornia_aug(images)

            with autocast(enabled=use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # OneCycleLR steps per batch
                if self.cfg.scheduler == "onecycle":
                    self.scheduler.step()

            tracker.update(logits, labels, loss.item())

        return tracker.compute()

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.console:
            self.console.print(msg)
        else:
            print(msg)

    def _log_epoch(self, epoch: int, train: dict, val: dict, elapsed: float) -> None:
        if self.console:
            table = Table(title=f"Epoch {epoch}/{self.cfg.epochs - 1}  ({elapsed:.1f}s)", show_header=True)
            table.add_column("Split")
            for k in train:
                table.add_column(k)
            table.add_row("train", *[f"{v:.4f}" for v in train.values()])
            table.add_row("val",   *[f"{v:.4f}" for v in val.values()])
            self.console.print(table)
        else:
            t_str = "  ".join(f"train_{k}={v:.4f}" for k, v in train.items())
            v_str = "  ".join(f"val_{k}={v:.4f}"   for k, v in val.items())
            print(f"[Epoch {epoch:03d}]  {t_str}  {v_str}  ({elapsed:.1f}s)")

    def _write_tb(self, epoch: int, metrics: dict, prefix: str) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, epoch)

    def _write_lr(self, epoch: int) -> None:
        for i, pg in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f"lr/group_{i}", pg["lr"], epoch)
