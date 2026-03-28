"""Learning-rate scheduler factory."""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    StepLR,
    SequentialLR,
    LinearLR,
    LambdaLR,
)
from config import Config


def build_scheduler(cfg: Config, optimizer: Optimizer, steps_per_epoch: int):
    """Return a scheduler.  All schedulers are stepped every epoch."""

    total_epochs = cfg.epochs

    if cfg.scheduler == "cosine_warmup":
        warmup = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=cfg.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - cfg.warmup_epochs,
            eta_min=cfg.min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[cfg.warmup_epochs],
        )

    elif cfg.scheduler == "onecycle":
        # OneCycleLR steps per batch, so we wrap it to step once per epoch
        return OneCycleLR(
            optimizer,
            max_lr=cfg.learning_rate,
            epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=cfg.warmup_epochs / total_epochs,
            anneal_strategy="cos",
        )

    elif cfg.scheduler == "step":
        return StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma)

    elif cfg.scheduler == "none":
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    else:
        raise ValueError(f"Unknown scheduler {cfg.scheduler!r}")
