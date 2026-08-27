"""Precision from a standard classification confusion matrix."""

from __future__ import annotations

from torch import Tensor

from ._common import AverageMode, compute_averaged_ratio
from .counts import extract_confusion_matrix_counts


def compute_precision(
    confusion_matrix: Tensor,
    *,
    average: AverageMode = "none",
    zero_division: float = 0.0,
) -> Tensor:
    """Compute per-class, macro, or micro precision."""

    counts = extract_confusion_matrix_counts(confusion_matrix)
    return compute_averaged_ratio(
        counts.true_positive,
        counts.true_positive + counts.false_positive,
        average=average,
        zero_division=zero_division,
    )


__all__ = ["compute_precision"]
