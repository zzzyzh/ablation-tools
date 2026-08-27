"""Mean and frequency-weighted Dice aggregations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from .._overlap import AbsentClassPolicy, aggregate_frequency_weighted, aggregate_mean
from .dice import compute_dice


@dataclass(frozen=True)
class MeanDiceResult:
    """Classwise, macro, micro, and frequency-weighted Dice."""

    per_class_dice: Tensor
    mean_dice: Tensor
    micro_dice: Tensor
    frequency_weighted_dice: Tensor
    selected_class_mask: Tensor
    mean_class_mask: Tensor


def compute_mean_dice(
    predictions: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    ignore_index: int | None = None,
    class_indices: Sequence[int] | Tensor | None = None,
    absent_class: AbsentClassPolicy = "ignore",
) -> MeanDiceResult:
    """Compute mean Dice with micro and target-frequency-weighted variants."""

    result = compute_dice(
        predictions,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_indices=class_indices,
        absent_class=absent_class,
    )
    mean_dice, mean_mask = aggregate_mean(
        result.per_class_dice,
        result.valid_class_mask,
        result.selected_class_mask,
        absent_class=absent_class,
    )
    frequency_weighted = aggregate_frequency_weighted(
        result.per_class_dice,
        result.target_count,
        result.selected_class_mask,
        absent_class=absent_class,
    )
    return MeanDiceResult(
        per_class_dice=result.per_class_dice,
        mean_dice=mean_dice,
        micro_dice=result.micro_dice,
        frequency_weighted_dice=frequency_weighted,
        selected_class_mask=result.selected_class_mask,
        mean_class_mask=mean_mask,
    )


__all__ = ["MeanDiceResult", "compute_mean_dice"]
