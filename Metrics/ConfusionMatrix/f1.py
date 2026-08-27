"""F1 score from a standard classification confusion matrix."""

from __future__ import annotations

from torch import Tensor

from ._common import AverageMode, compute_averaged_ratio
from .counts import extract_confusion_matrix_counts


def compute_f1_score(
    confusion_matrix: Tensor,
    *,
    average: AverageMode = "none",
    zero_division: float = 0.0,
) -> Tensor:
    """Compute F1 directly from TP/FP/FN counts."""

    counts = extract_confusion_matrix_counts(confusion_matrix)
    numerator = 2 * counts.true_positive
    denominator = numerator + counts.false_positive + counts.false_negative
    return compute_averaged_ratio(
        numerator,
        denominator,
        average=average,
        zero_division=zero_division,
    )


__all__ = ["compute_f1_score"]
