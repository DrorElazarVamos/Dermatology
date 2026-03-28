"""Entry point: parse args, build everything, run training + final test eval."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from config import Config
from data import build_dataloaders
from models import build_model
from utils import build_criterion, build_optimizer, build_scheduler
from trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Dermatology CNN Training Framework")

    # Override any Config field via CLI: --key value
    parser.add_argument("--data_dir",    type=str)
    parser.add_argument("--output_dir",  type=str)
    parser.add_argument("--arch",        type=str)
    parser.add_argument("--epochs",      type=int)
    parser.add_argument("--batch_size",  type=int)
    parser.add_argument("--lr",          dest="learning_rate", type=float)
    parser.add_argument("--loss",        type=str)
    parser.add_argument("--optimizer",   type=str)
    parser.add_argument("--scheduler",   type=str)
    parser.add_argument("--device",      type=str)
    parser.add_argument("--seed",        type=int)
    parser.add_argument("--resume",      type=str)
    parser.add_argument("--no_amp",      action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_kornia",   action="store_true")
    parser.add_argument("--config_json", type=str, help="Load config overrides from a JSON file")

    args = parser.parse_args()

    cfg = Config()

    # JSON overrides (lower priority than CLI)
    if args.config_json:
        with open(args.config_json) as f:
            cfg = Config.from_dict({**cfg.to_dict(), **json.load(f)})

    # CLI overrides
    for attr in ["data_dir", "output_dir", "arch", "epochs", "batch_size",
                 "learning_rate", "loss", "optimizer", "scheduler", "device",
                 "seed", "resume"]:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)

    if args.no_amp:
        cfg.use_amp = False
    if args.no_pretrained:
        cfg.pretrained = False
    if args.no_kornia:
        cfg.use_kornia = False

    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Dermatology CNN Training Framework")
    print(f"{'='*60}")
    print(f"  arch={cfg.arch}  loss={cfg.loss}  optimizer={cfg.optimizer}")
    print(f"  scheduler={cfg.scheduler}  amp={cfg.use_amp}  kornia={cfg.use_kornia}")
    print(f"  device={cfg.device}  seed={cfg.seed}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, test_loader, class_names, class_weights = build_dataloaders(cfg)

    # Model
    model = build_model(cfg)

    # Loss
    criterion = build_criterion(cfg, class_weights)

    # Optimizer + scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    # Save config snapshot
    config_path = Path(cfg.output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"Config saved → {config_path}\n")

    # Train
    trainer = Trainer(model, criterion, optimizer, scheduler, cfg, class_weights)
    best_val = trainer.fit(train_loader, val_loader)

    print(f"\n{'='*60}")
    print("  Final test-set evaluation (best model)")
    print(f"{'='*60}\n")

    # Load best checkpoint for test eval
    best_ckpt = str(Path(cfg.output_dir) / "best_model.pt")
    if Path(best_ckpt).exists():
        trainer.ckpt_manager.load(best_ckpt, model)

    test_metrics = trainer.evaluate(test_loader)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save test results
    results_path = Path(cfg.output_dir) / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nTest results saved → {results_path}")


if __name__ == "__main__":
    main()
