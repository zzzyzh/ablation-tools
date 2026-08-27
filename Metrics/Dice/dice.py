"""Classwise and micro Sørensen-Dice for hard class labels."""

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
class DiceResult:
    """Classwise Dice, micro Dice, and the counts used to compute them."""

    per_class_dice: Tensor
    micro_dice: Tensor
    intersection_count: Tensor
    denominator_count: Tensor
    prediction_count: Tensor
    target_count: Tensor
    selected_class_mask: Tensor
    valid_class_mask: Tensor


def compute_dice(
    predictions: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    ignore_index: int | None = None,
    class_indices: Sequence[int] | Tensor | None = None,
    absent_class: AbsentClassPolicy = "ignore",
) -> DiceResult:
    """Compute hard-label per-class and micro Dice.

    The numerator is ``2 * intersection`` and the denominator is prediction
    count plus target count. Logits, probabilities, and bbox inputs are rejected.
    """

    counts = compute_overlap_counts(
        predictions,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_indices=class_indices,
    )
    numerator = 2 * counts.intersection_count
    denominator = counts.dice_denominator_count
    per_class, defined = compute_classwise_ratio(
        numerator,
        denominator,
        absent_class=absent_class,
        selected_class_mask=counts.selected_class_mask,
    )
    micro = aggregate_ratio(
        numerator,
        denominator,
        counts.selected_class_mask,
        absent_class=absent_class,
    )
    return DiceResult(
        per_class_dice=per_class,
        micro_dice=micro,
        intersection_count=counts.intersection_count,
        denominator_count=denominator,
        prediction_count=counts.prediction_count,
        target_count=counts.target_count,
        selected_class_mask=counts.selected_class_mask,
        valid_class_mask=defined,
    )


__all__ = ["DiceResult", "compute_dice"]
