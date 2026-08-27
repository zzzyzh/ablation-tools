"""Shared averaging and zero-division behavior for confusion-matrix metrics."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import torch
from torch import Tensor


AverageMode = Literal["none", "macro", "micro"]


def validate_average(average: str) -> AverageMode:
    if average not in {"none", "macro", "micro"}:
        raise ValueError("average must be 'none', 'macro', or 'micro'")
    return average  # type: ignore[return-value]


def validate_zero_division(zero_division: float) -> float:
    if not isinstance(zero_division, Real) or isinstance(zero_division, bool):
        raise TypeError("zero_division must be 0.0 or 1.0")
    value = float(zero_division)
    if not math.isfinite(value) or value not in {0.0, 1.0}:
        raise ValueError("zero_division must be 0.0 or 1.0")
    return value


def safe_ratio(numerator: Tensor, denominator: Tensor, zero_division: float) -> Tensor:
    numerator = numerator.to(torch.float64)
    denominator = denominator.to(torch.float64)
    fill = validate_zero_division(zero_division)
    result = torch.full_like(denominator, fill)
    defined = denominator > 0
    result[defined] = numerator[defined] / denominator[defined]
    return result


def compute_averaged_ratio(
    numerator: Tensor,
    denominator: Tensor,
    *,
    average: AverageMode,
    zero_division: float,
) -> Tensor:
    mode = validate_average(average)
    per_class = safe_ratio(numerator, denominator, zero_division)
    if mode == "none":
        return per_class
    if mode == "macro":
        return per_class.mean()
    return safe_ratio(
        numerator.to(torch.float64).sum().reshape(1),
        denominator.to(torch.float64).sum().reshape(1),
        zero_division,
    ).squeeze(0)


__all__ = ["AverageMode"]
