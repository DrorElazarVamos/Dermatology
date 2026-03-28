"""Loss functions: cross-entropy, focal, weighted cross-entropy."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
        weight: Per-class weights (same semantics as ``nn.CrossEntropyLoss``).
        label_smoothing: Applied before focal weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C), targets: (N,)
        log_p = F.log_softmax(logits, dim=1)           # (N, C)
        p     = log_p.exp()                             # (N, C)

        # Gather probabilities for the true class (used for focal weight)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)

        # Standard CE with optional class weights (per-sample, no reduction)
        ce = F.nll_loss(
            log_p,
            targets,
            weight=self.weight,
            reduction="none",
        )  # (N,)

        # Apply label smoothing manually if requested
        if self.label_smoothing > 0.0:
            smooth_loss = -log_p.mean(dim=1)  # uniform distribution target
            ce = (1.0 - self.label_smoothing) * ce + self.label_smoothing * smooth_loss

        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = (focal_weight * ce).mean()
        return loss


def build_criterion(cfg: Config, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Return the appropriate loss module, moved to the correct device later by the trainer."""

    if cfg.loss == "cross_entropy":
        w = class_weights if cfg.use_class_weights else None
        return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)

    elif cfg.loss == "weighted_ce":
        if class_weights is None:
            raise ValueError("weighted_ce requires class_weights")
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

    elif cfg.loss == "focal":
        w = class_weights if cfg.use_class_weights else None
        return FocalLoss(
            gamma=cfg.focal_gamma,
            weight=w,
            label_smoothing=cfg.label_smoothing,
        )

    else:
        raise ValueError(f"Unknown loss {cfg.loss!r}. Choose from cross_entropy, weighted_ce, focal")
