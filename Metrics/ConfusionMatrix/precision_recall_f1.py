"""Combined Precision, Recall, and F1 reporting from one confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from ._common import AverageMode, compute_averaged_ratio
from .counts import ConfusionMatrixCounts, extract_confusion_matrix_counts


@dataclass(frozen=True)
class PrecisionRecallF1Result:
    """Per-class, macro, and micro classification metrics and raw counts."""

    per_class_precision: Tensor
    per_class_recall: Tensor
    per_class_f1: Tensor
    macro_precision: Tensor
    macro_recall: Tensor
    macro_f1: Tensor
    micro_precision: Tensor
    micro_recall: Tensor
    micro_f1: Tensor
    counts: ConfusionMatrixCounts


def compute_precision_recall_f1(
    confusion_matrix: Tensor,
    *,
    zero_division: float = 0.0,
) -> PrecisionRecallF1Result:
    """Compute all common Precision/Recall/F1 views from one validated matrix."""

    counts = extract_confusion_matrix_counts(confusion_matrix)
    precision_numerator = counts.true_positive
    precision_denominator = counts.true_positive + counts.false_positive
    recall_numerator = counts.true_positive
    recall_denominator = counts.true_positive + counts.false_negative
    f1_numerator = 2 * counts.true_positive
    f1_denominator = f1_numerator + counts.false_positive + counts.false_negative

    def compute(numerator: Tensor, denominator: Tensor, average: AverageMode) -> Tensor:
        return compute_averaged_ratio(
            numerator,
            denominator,
            average=average,
            zero_division=zero_division,
        )

    return PrecisionRecallF1Result(
        per_class_precision=compute(precision_numerator, precision_denominator, "none"),
        per_class_recall=compute(recall_numerator, recall_denominator, "none"),
        per_class_f1=compute(f1_numerator, f1_denominator, "none"),
        macro_precision=compute(precision_numerator, precision_denominator, "macro"),
        macro_recall=compute(recall_numerator, recall_denominator, "macro"),
        macro_f1=compute(f1_numerator, f1_denominator, "macro"),
        micro_precision=compute(precision_numerator, precision_denominator, "micro"),
        micro_recall=compute(recall_numerator, recall_denominator, "micro"),
        micro_f1=compute(f1_numerator, f1_denominator, "micro"),
        counts=counts,
    )


__all__ = ["PrecisionRecallF1Result", "compute_precision_recall_f1"]
