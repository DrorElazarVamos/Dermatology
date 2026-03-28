"""Metrics tracking: accuracy, balanced accuracy, AUROC, F1."""

from typing import List, Optional
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class MetricTracker:
    """Accumulates predictions and labels over an epoch, then computes metrics."""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.reset()

    def reset(self) -> None:
        self._logits: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._loss_sum = 0.0
        self._loss_n = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float) -> None:
        self._logits.append(logits.detach().cpu().float().numpy())
        self._labels.append(labels.detach().cpu().numpy())
        self._loss_sum += loss
        self._loss_n += 1

    def compute(self) -> dict:
        logits = np.concatenate(self._logits, axis=0)   # (N, C)
        labels = np.concatenate(self._labels, axis=0)   # (N,)
        probs  = _softmax(logits)                        # (N, C)
        preds  = logits.argmax(axis=1)                  # (N,)

        metrics = {
            "loss":      self._loss_sum / max(self._loss_n, 1),
            "acc":       float(accuracy_score(labels, preds)),
            "bal_acc":   float(balanced_accuracy_score(labels, preds)),
            "f1_macro":  float(f1_score(labels, preds, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

        try:
            if self.num_classes == 2:
                metrics["auroc"] = float(roc_auc_score(labels, probs[:, 1]))
            else:
                metrics["auroc"] = float(
                    roc_auc_score(labels, probs, multi_class="ovr", average="macro")
                )
        except ValueError:
            metrics["auroc"] = float("nan")

        return metrics

    def confusion_matrix(self) -> np.ndarray:
        labels = np.concatenate(self._labels, axis=0)
        logits = np.concatenate(self._logits, axis=0)
        preds  = logits.argmax(axis=1)
        return confusion_matrix(labels, preds)

    def classification_report(self) -> str:
        labels = np.concatenate(self._labels, axis=0)
        logits = np.concatenate(self._logits, axis=0)
        preds  = logits.argmax(axis=1)
        return classification_report(labels, preds, target_names=self.class_names, zero_division=0)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
