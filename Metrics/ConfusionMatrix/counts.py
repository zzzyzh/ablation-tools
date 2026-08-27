"""Construction, validation, and TP/FP/FN/TN extraction for confusion matrices."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor


def validate_confusion_matrix(confusion_matrix: Tensor) -> Tensor:
    """Validate a non-negative square matrix with rows=true and columns=predicted."""

    if not isinstance(confusion_matrix, Tensor):
        raise TypeError(
            f"confusion_matrix must be a torch.Tensor, got {type(confusion_matrix)!r}"
        )
    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError(
            "confusion_matrix must be a non-empty square [classes, classes] tensor, "
            f"got {tuple(confusion_matrix.shape)}"
        )
    if confusion_matrix.shape[0] == 0:
        raise ValueError("confusion_matrix must contain at least one class")
    if confusion_matrix.dtype == torch.bool or confusion_matrix.is_complex():
        raise TypeError("confusion_matrix must contain real numeric counts")
    if not torch.isfinite(confusion_matrix).all():
        raise ValueError("confusion_matrix must contain only finite values")
    if (confusion_matrix < 0).any():
        raise ValueError("confusion_matrix counts must be non-negative")
    if confusion_matrix.is_floating_point():
        if confusion_matrix.dtype in {torch.float16, torch.bfloat16}:
            return confusion_matrix.float()
        return confusion_matrix
    return confusion_matrix.to(torch.int64)


@dataclass(frozen=True)
class ConfusionMatrixCounts:
    """One-vs-rest counts extracted for every class."""

    true_positive: Tensor
    false_positive: Tensor
    false_negative: Tensor
    true_negative: Tensor
    support: Tensor
    predicted_count: Tensor
    total_count: Tensor

    @property
    def class_count(self) -> int:
        return self.true_positive.shape[0]


def extract_confusion_matrix_counts(confusion_matrix: Tensor) -> ConfusionMatrixCounts:
    """Extract per-class one-vs-rest counts.

    Matrix entry ``[target_class, predicted_class]`` stores one count or sample
    weight. Rows therefore sum to target support and columns to predicted count.
    """

    matrix = validate_confusion_matrix(confusion_matrix)
    true_positive = matrix.diagonal()
    support = matrix.sum(dim=1)
    predicted_count = matrix.sum(dim=0)
    false_negative = support - true_positive
    false_positive = predicted_count - true_positive
    total = matrix.sum()
    true_negative = total - true_positive - false_positive - false_negative
    return ConfusionMatrixCounts(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        support=support,
        predicted_count=predicted_count,
        total_count=total,
    )


def compute_confusion_matrix(
    predictions: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    ignore_index: int | None = None,
) -> Tensor:
    """Build an integer confusion matrix from hard class-index tensors.

    The returned matrix has rows=true/target and columns=predicted. Target
    ``ignore_index`` positions are removed before prediction range validation.
    """

    if not isinstance(num_classes, Integral) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an integer")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    num_classes = int(num_classes)
    for labels, name in ((predictions, "predictions"), (targets, "targets")):
        if not isinstance(labels, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if labels.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        if labels.is_floating_point() or labels.is_complex():
            raise TypeError(f"{name} must contain integer class indices")
        if labels.dtype == torch.bool and num_classes != 2:
            raise TypeError(f"boolean {name} is supported only when num_classes=2")
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must be on the same device")
    if ignore_index is not None and (
        not isinstance(ignore_index, Integral) or isinstance(ignore_index, bool)
    ):
        raise TypeError("ignore_index must be an integer or None")

    prediction_values = predictions.reshape(-1).to(torch.int64)
    target_values = targets.reshape(-1).to(torch.int64)
    valid = torch.ones_like(target_values, dtype=torch.bool)
    if ignore_index is not None:
        valid = target_values != int(ignore_index)
    prediction_values = prediction_values[valid]
    target_values = target_values[valid]
    for labels, name in ((prediction_values, "predictions"), (target_values, "targets")):
        if labels.numel() and ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError(f"valid {name} values must be in [0, {num_classes - 1}]")
    flattened = target_values * num_classes + prediction_values
    return torch.bincount(flattened, minlength=num_classes**2).reshape(
        num_classes,
        num_classes,
    )


__all__ = [
    "ConfusionMatrixCounts",
    "compute_confusion_matrix",
    "extract_confusion_matrix_counts",
    "validate_confusion_matrix",
]
