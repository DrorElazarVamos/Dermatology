"""Optimizer factory."""

import torch
from torch.optim import AdamW, Adam, SGD
from config import Config
from models.builder import DermModel


def build_optimizer(cfg: Config, model: DermModel) -> torch.optim.Optimizer:
    param_groups = model.parameter_groups(cfg.learning_rate, cfg.backbone_lr_multiplier)

    if cfg.optimizer == "adamw":
        return AdamW(param_groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        return Adam(param_groups, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return SGD(
            param_groups,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer!r}")
