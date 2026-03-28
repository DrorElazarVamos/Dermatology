"""Model factory supporting multiple torchvision / timm backbones."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as tvm

from config import Config


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

def _resnet50(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = tvm.resnet50(weights=weights)
    in_features = m.fc.in_features
    m.fc = nn.Identity()
    return m, in_features


def _efficientnet_b3(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.efficientnet_b3(weights=weights)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Identity()
    return m, in_features


def _convnext_small(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.convnext_small(weights=weights)
    in_features = m.classifier[2].in_features
    m.classifier = nn.Identity()
    return m, in_features


def _densenet121(pretrained: bool) -> Tuple[nn.Module, int]:
    weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.densenet121(weights=weights)
    in_features = m.classifier.in_features
    m.classifier = nn.Identity()
    return m, in_features


_REGISTRY = {
    "resnet50":        _resnet50,
    "efficientnet_b3": _efficientnet_b3,
    "convnext_small":  _convnext_small,
    "densenet121":     _densenet121,
}


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DermModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Some backbones return nested structures; flatten
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.dim() > 2:
            features = features.flatten(1)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def parameter_groups(self, base_lr: float, multiplier: float):
        """Return parameter groups with separate LRs for backbone vs head."""
        return [
            {"params": self.backbone.parameters(), "lr": base_lr * multiplier},
            {"params": self.head.parameters(),     "lr": base_lr},
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: Config) -> DermModel:
    if cfg.arch not in _REGISTRY:
        raise ValueError(f"Unknown arch {cfg.arch!r}. Choose from {list(_REGISTRY)}")

    backbone, in_features = _REGISTRY[cfg.arch](cfg.pretrained)
    head = ClassificationHead(in_features, cfg.num_classes, cfg.dropout)
    model = DermModel(backbone, head)

    if cfg.freeze_epochs > 0:
        model.freeze_backbone()
        print(f"Backbone frozen for the first {cfg.freeze_epochs} epoch(s).")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.arch} | params: {total:,} | trainable: {trainable:,}")

    return model
