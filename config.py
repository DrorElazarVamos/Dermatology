"""Default configuration dataclass for the dermatology CNN training framework."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class Config:
    # Data
    data_dir: str = "data/raw"
    output_dir: str = "outputs"
    num_classes: int = 9
    image_size: int = 224
    num_workers: int = 4

    # Model
    arch: str = "resnet50"  # resnet50, efficientnet_b3, convnext_small, densenet121
    pretrained: bool = True
    dropout: float = 0.4
    freeze_epochs: int = 0  # Epochs to keep backbone frozen (0 = never freeze)

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    backbone_lr_multiplier: float = 0.1  # LR multiplier for backbone vs head
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD

    # Optimizer
    optimizer: str = "adamw"  # adamw, sgd, adam

    # Scheduler
    scheduler: str = "cosine_warmup"  # cosine_warmup, onecycle, step, none
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    step_size: int = 10
    step_gamma: float = 0.1

    # Loss
    loss: str = "cross_entropy"  # cross_entropy, focal, weighted_ce
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    use_class_weights: bool = True

    # Augmentation
    use_kornia: bool = True
    cutout_size: int = 32

    # Mixed precision
    use_amp: bool = True

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Checkpointing
    save_every: int = 10  # Save checkpoint every N epochs
    resume: Optional[str] = None  # Path to checkpoint to resume from

    # Logging
    use_tensorboard: bool = True
    log_dir: str = "outputs/logs"
    use_rich: bool = True

    # Misc
    seed: int = 42
    device: str = "cuda"  # cuda, cpu
    val_split: float = 0.15
    test_split: float = 0.15

    # Class names (populated at runtime if not set)
    class_names: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert config to plain dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Instantiate Config from a dictionary (ignores unknown keys)."""
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def __post_init__(self) -> None:
        # Resolve output directories as Path objects for downstream use
        self.output_dir = str(Path(self.output_dir))
        self.log_dir = str(Path(self.log_dir))
