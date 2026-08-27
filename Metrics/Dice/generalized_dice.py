"""Generalized Dice with explicit hard-label class weighting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .._overlap import (
    AbsentClassPolicy,
    compute_overlap_counts,
    validate_absent_class_policy,
)


GeneralizedDiceWeight = Literal["uniform", "inverse", "inverse_square"]


@dataclass(frozen=True)
class GeneralizedDiceResult:
    """Generalized Dice score and the per-class weights used."""

    generalized_dice: Tensor
    class_weights: Tensor
    intersection_count: Tensor
    prediction_count: Tensor
    target_count: Tensor
    selected_class_mask: Tensor


def compute_generalized_dice(
    predictions: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    ignore_index: int | None = None,
    class_indices: Sequence[int] | Tensor | None = None,
    class_weight: GeneralizedDiceWeight = "inverse_square",
    absent_class: AbsentClassPolicy = "ignore",
) -> GeneralizedDiceResult:
    """Compute hard-label generalized Dice across selected classes.

    ``inverse`` and ``inverse_square`` weight classes by target volume. Classes
    with zero target support receive zero weight rather than infinite weight.
    """

    if class_weight not in {"uniform", "inverse", "inverse_square"}:
        raise ValueError(
            "class_weight must be 'uniform', 'inverse', or 'inverse_square', "
            f"got {class_weight!r}"
        )
    policy = validate_absent_class_policy(absent_class)
    counts = compute_overlap_counts(
        predictions,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_indices=class_indices,
    )
    target = counts.target_count.to(torch.float64)
    weights = torch.zeros_like(target)
    present = target > 0
    if class_weight == "uniform":
        weights.fill_(1.0)
    elif class_weight == "inverse":
        weights[present] = target[present].reciprocal()
    else:
        weights[present] = target[present].square().reciprocal()
    weights *= counts.selected_class_mask.to(weights.dtype)

    numerator = 2 * (
        weights * counts.intersection_count.to(torch.float64)
    ).sum()
    denominator = (
        weights
        * (counts.prediction_count + counts.target_count).to(torch.float64)
    ).sum()
    if denominator > 0:
        score = numerator / denominator
    elif policy == "raise":
        raise ValueError("selected classes have no target support")
    else:
        fill = math.nan if policy == "ignore" else float(policy == "one")
        score = denominator.new_tensor(fill)
    return GeneralizedDiceResult(
        generalized_dice=score,
        class_weights=weights,
        intersection_count=counts.intersection_count,
        prediction_count=counts.prediction_count,
        target_count=counts.target_count,
        selected_class_mask=counts.selected_class_mask,
    )


__all__ = [
    "GeneralizedDiceResult",
    "GeneralizedDiceWeight",
    "compute_generalized_dice",
]
