"""Mean Squared Error with explicit reduction semantics."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from .._regression import ErrorReduction, prepare_regression_inputs, reduce_regression_error


def compute_mse(
    predictions: Tensor,
    targets: Tensor,
    *,
    reduction: ErrorReduction = "mean",
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> Tensor:
    """Compute squared error, returning its mean by default."""

    predictions, targets = prepare_regression_inputs(predictions, targets)
    squared_error = (predictions - targets).square()
    return reduce_regression_error(
        squared_error,
        reduction=reduction,
        dim=dim,
        keepdim=keepdim,
    )


compute_mean_squared_error = compute_mse


__all__ = ["compute_mean_squared_error", "compute_mse"]
