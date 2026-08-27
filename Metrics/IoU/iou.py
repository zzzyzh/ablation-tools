"""Classwise and micro Intersection over Union for hard class labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from .._overlap import (
    AbsentClassPolicy,
    aggregate_ratio,
    compute_classwise_ratio,
    compute_overlap_counts,
)


@dataclass(frozen=True)
class IoUResult:
    """Classwise IoU, micro IoU, and the counts used to compute them."""

    per_class_iou: Tensor
    micro_iou: Tensor
    intersection_count: Tensor
    union_count: Tensor
    prediction_count: Tensor
    target_count: Tensor
    selected_class_mask: Tensor
    valid_class_mask: Tensor


def compute_iou(
    predictions: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    ignore_index: int | None = None,
    class_indices: Sequence[int] | Tensor | None = None,
    absent_class: AbsentClassPolicy = "ignore",
) -> IoUResult:
    """Compute hard-label per-class and micro IoU.

    Inputs are class-index tensors of identical shape. This function does not
    apply argmax, thresholds, one-hot conversion, or bbox geometry.
    """

    counts = compute_overlap_counts(
        predictions,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_indices=class_indices,
    )
    per_class, defined = compute_classwise_ratio(
        counts.intersection_count,
        counts.union_count,
        absent_class=absent_class,
        selected_class_mask=counts.selected_class_mask,
    )
    micro = aggregate_ratio(
        counts.intersection_count,
        counts.union_count,
        counts.selected_class_mask,
        absent_class=absent_class,
    )
    return IoUResult(
        per_class_iou=per_class,
        micro_iou=micro,
        intersection_count=counts.intersection_count,
        union_count=counts.union_count,
        prediction_count=counts.prediction_count,
        target_count=counts.target_count,
        selected_class_mask=counts.selected_class_mask,
        valid_class_mask=defined,
    )


__all__ = ["IoUResult", "compute_iou"]
