"""Mean Absolute Error with explicit reduction semantics."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from .._regression import ErrorReduction, prepare_regression_inputs, reduce_regression_error


def compute_mae(
    predictions: Tensor,
    targets: Tensor,
    *,
    reduction: ErrorReduction = "mean",
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Compute absolute error, returning its mean by default."""

    predictions, targets = prepare_regression_inputs(predictions, targets)
    absolute_error = (predictions - targets).abs()
    return reduce_regression_error(
        absolute_error,
        reduction=reduction,
        dim=dim,
        keepdim=keepdim,
    )


compute_mean_absolute_error = compute_mae


__all__ = ["compute_mae", "compute_mean_absolute_error"]
