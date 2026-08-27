"""Mean and frequency-weighted IoU aggregations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from .._overlap import AbsentClassPolicy, aggregate_frequency_weighted, aggregate_mean
from .iou import compute_iou


@dataclass(frozen=True)
class MeanIoUResult:
    """Classwise, macro, micro, and frequency-weighted IoU."""

    per_class_iou: Tensor
    mean_iou: Tensor
    micro_iou: Tensor
    frequency_weighted_iou: Tensor
    selected_class_mask: Tensor
    mean_class_mask: Tensor


def compute_mean_iou(
    predictions: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    ignore_index: int | None = None,
    class_indices: Sequence[int] | Tensor | None = None,
    absent_class: AbsentClassPolicy = "ignore",
) -> MeanIoUResult:
    """Compute mIoU together with micro and frequency-weighted variants."""

    result = compute_iou(
        predictions,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_indices=class_indices,
        absent_class=absent_class,
    )
    mean_iou, mean_mask = aggregate_mean(
        result.per_class_iou,
        result.valid_class_mask,
        result.selected_class_mask,
        absent_class=absent_class,
    )
    frequency_weighted = aggregate_frequency_weighted(
        result.per_class_iou,
        result.target_count,
        result.selected_class_mask,
        absent_class=absent_class,
    )
    return MeanIoUResult(
        per_class_iou=result.per_class_iou,
        mean_iou=mean_iou,
        micro_iou=result.micro_iou,
        frequency_weighted_iou=frequency_weighted,
        selected_class_mask=result.selected_class_mask,
        mean_class_mask=mean_mask,
    )


__all__ = ["MeanIoUResult", "compute_mean_iou"]
