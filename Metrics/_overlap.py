"""Shared hard-label counting and absent-class policies for overlap metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor


AbsentClassPolicy = Literal["ignore", "zero", "one", "raise"]


@dataclass(frozen=True)
class OverlapCounts:
    """Per-class counts pooled over all valid elements."""

    intersection_count: Tensor
    prediction_count: Tensor
    target_count: Tensor
    selected_class_mask: Tensor

    @property
    def union_count(self) -> Tensor:
        return self.prediction_count + self.target_count - self.intersection_count

    @property
    def dice_denominator_count(self) -> Tensor:
        return self.prediction_count + self.target_count


def validate_absent_class_policy(policy: str) -> AbsentClassPolicy:
    if policy not in {"ignore", "zero", "one", "raise"}:
        raise ValueError(
            "absent_class must be 'ignore', 'zero', 'one', or 'raise', "
            f"got {policy!r}"
        )
    return policy  # type: ignore[return-value]


def _validate_label_tensor(labels: Tensor, name: str, num_classes: int) -> None:
    if not isinstance(labels, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(labels)!r}")
    if labels.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if labels.is_floating_point() or labels.is_complex():
        raise TypeError(f"{name} must contain integer class indices")
    if labels.dtype == torch.bool and num_classes != 2:
        raise TypeError(f"boolean {name} is supported only when num_classes=2")


def _build_selected_class_mask(
    num_classes: int,
    class_indices: Sequence[int] | Tensor | None,
    device: torch.device,
) -> Tensor:
    if class_indices is None:
        return torch.ones(num_classes, dtype=torch.bool, device=device)
    if isinstance(class_indices, Tensor):
        if class_indices.ndim != 1:
            raise ValueError("class_indices tensor must be one-dimensional")
        values = class_indices.detach().cpu().tolist()
    elif isinstance(class_indices, Sequence) and not isinstance(class_indices, (str, bytes)):
        values = list(class_indices)
    else:
        raise TypeError("class_indices must be a sequence of unique integers or None")
    if not values:
        raise ValueError("class_indices must not be empty")
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
        raise TypeError("class_indices must contain only integers")
    normalized = [int(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError("class_indices must not contain duplicates")
    if any(value < 0 or value >= num_classes for value in normalized):
        raise ValueError(f"class_indices must be in [0, {num_classes - 1}]")
    mask = torch.zeros(num_classes, dtype=torch.bool, device=device)
    mask[torch.tensor(normalized, dtype=torch.int64, device=device)] = True
    return mask


def compute_overlap_counts(
    predictions: Tensor,
    targets: Tensor,
    *,
    num_classes: int,
    ignore_index: int | None = None,
    class_indices: Sequence[int] | Tensor | None = None,
) -> OverlapCounts:
    """Compute O(C) hard-label counts without allocating one-hot tensors."""

    if not isinstance(num_classes, Integral) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an integer")
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    num_classes = int(num_classes)
    _validate_label_tensor(predictions, "predictions", num_classes)
    _validate_label_tensor(targets, "targets", num_classes)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have identical shapes, got "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must be on the same device")
    if ignore_index is not None and (
        not isinstance(ignore_index, Integral) or isinstance(ignore_index, bool)
    ):
        raise TypeError("ignore_index must be an integer or None")

    predictions_flat = predictions.reshape(-1).to(torch.int64)
    targets_flat = targets.reshape(-1).to(torch.int64)
    valid = torch.ones_like(targets_flat, dtype=torch.bool)
    if ignore_index is not None:
        valid = targets_flat != int(ignore_index)
    predictions_valid = predictions_flat[valid]
    targets_valid = targets_flat[valid]

    for labels, name in (
        (predictions_valid, "predictions"),
        (targets_valid, "targets"),
    ):
        if labels.numel() and ((labels < 0) | (labels >= num_classes)).any():
            raise ValueError(f"valid {name} values must be in [0, {num_classes - 1}]")

    prediction_count = torch.bincount(predictions_valid, minlength=num_classes)
    target_count = torch.bincount(targets_valid, minlength=num_classes)
    matched = targets_valid[predictions_valid == targets_valid]
    intersection_count = torch.bincount(matched, minlength=num_classes)
    selected_class_mask = _build_selected_class_mask(
        num_classes,
        class_indices,
        predictions.device,
    )
    return OverlapCounts(
        intersection_count=intersection_count,
        prediction_count=prediction_count,
        target_count=target_count,
        selected_class_mask=selected_class_mask,
    )


def compute_classwise_ratio(
    numerator: Tensor,
    denominator: Tensor,
    *,
    absent_class: AbsentClassPolicy,
    selected_class_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    policy = validate_absent_class_policy(absent_class)
    numerator = numerator.to(torch.float64)
    denominator = denominator.to(torch.float64)
    defined = denominator > 0
    if policy == "raise" and ((~defined) & selected_class_mask).any():
        raise ValueError("at least one selected class has an undefined metric")
    fill_value = math.nan if policy in {"ignore", "raise"} else float(policy == "one")
    scores = torch.full_like(denominator, fill_value)
    scores[defined] = numerator[defined] / denominator[defined]
    return scores, defined


def aggregate_mean(
    values: Tensor,
    defined: Tensor,
    selected_class_mask: Tensor,
    *,
    absent_class: AbsentClassPolicy,
) -> tuple[Tensor, Tensor]:
    policy = validate_absent_class_policy(absent_class)
    mean_mask = selected_class_mask & defined if policy == "ignore" else selected_class_mask
    if mean_mask.any():
        return values[mean_mask].mean(), mean_mask
    if policy == "raise":
        raise ValueError("no selected class has a defined metric")
    fill = math.nan if policy == "ignore" else float(policy == "one")
    return values.new_tensor(fill), mean_mask


def aggregate_ratio(
    numerator: Tensor,
    denominator: Tensor,
    selected_class_mask: Tensor,
    *,
    absent_class: AbsentClassPolicy,
) -> Tensor:
    policy = validate_absent_class_policy(absent_class)
    numerator_sum = numerator[selected_class_mask].to(torch.float64).sum()
    denominator_sum = denominator[selected_class_mask].to(torch.float64).sum()
    if denominator_sum > 0:
        return numerator_sum / denominator_sum
    if policy == "raise":
        raise ValueError("selected classes have an undefined aggregate metric")
    fill = math.nan if policy == "ignore" else float(policy == "one")
    return numerator_sum.new_tensor(fill)


def aggregate_frequency_weighted(
    values: Tensor,
    target_count: Tensor,
    selected_class_mask: Tensor,
    *,
    absent_class: AbsentClassPolicy,
) -> Tensor:
    policy = validate_absent_class_policy(absent_class)
    weights = target_count.to(torch.float64)
    included = selected_class_mask & (weights > 0)
    denominator = weights[included].sum()
    if denominator > 0:
        return (values[included] * weights[included]).sum() / denominator
    if policy == "raise":
        raise ValueError("selected classes have no target support")
    fill = math.nan if policy == "ignore" else float(policy == "one")
    return weights.new_tensor(fill)


__all__ = ["AbsentClassPolicy", "OverlapCounts", "compute_overlap_counts"]
