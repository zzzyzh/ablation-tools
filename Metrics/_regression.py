"""Shared validation and reduction behavior for pointwise regression errors."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor


ErrorReduction = Literal["none", "mean", "sum"]

def _validate_integer_precision(values: Tensor, name: str) -> None:
    if values.is_floating_point():
        return
    exact_limit = 2**52
    dtype_range = torch.iinfo(values.dtype)
    if dtype_range.min < 0:
        outside = (values < -exact_limit) | (values > exact_limit)
    else:
        outside = values > exact_limit
    if outside.any():
        raise ValueError(
            f"integer {name} must stay within [-2**52, 2**52] for exact float64 subtraction"
        )


def prepare_regression_inputs(predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Validate aligned real tensors and promote them before subtraction."""

    for values, name in ((predictions, "predictions"), (targets, "targets")):
        if not isinstance(values, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(values)!r}")
        if values.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        if values.dtype == torch.bool or values.is_complex():
            raise TypeError(f"{name} must contain real numeric values, not {values.dtype}")
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        _validate_integer_precision(values, name)
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must have identical shapes, got "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must be on the same device")

    if not predictions.is_floating_point() or not targets.is_floating_point():
        dtype = torch.float64
    else:
        dtype = torch.promote_types(predictions.dtype, targets.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    return predictions.to(dtype), targets.to(dtype)


def _normalize_reduction_dims(
    dim: int | Sequence[int] | None,
    ndim: int,
) -> int | tuple[int, ...] | None:
    if dim is None:
        return None
    if isinstance(dim, Integral) and not isinstance(dim, bool):
        values = [int(dim)]
        return_single = True
    elif isinstance(dim, Sequence) and not isinstance(dim, (str, bytes)):
        values = list(dim)
        return_single = False
    else:
        raise TypeError("dim must be an integer, a sequence of integers, or None")
    if not values:
        raise ValueError("dim sequence must not be empty")
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
        raise TypeError("dim must contain only integers")
    normalized = []
    for value in values:
        axis = int(value)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"dim {value} is out of range for a {ndim}-D error tensor")
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        raise ValueError("dim must not contain duplicate axes")
    return normalized[0] if return_single else tuple(normalized)


def reduce_regression_error(
    error: Tensor,
    *,
    reduction: ErrorReduction,
    dim: int | Sequence[int] | None,
    keepdim: bool,
) -> Tensor:
    """Apply the shared ``none``/``mean``/``sum`` reduction contract."""

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not isinstance(keepdim, bool):
        raise TypeError("keepdim must be a bool")
    if not torch.isfinite(error).all():
        raise ValueError("error is not representable in the computation dtype")
    if reduction == "none":
        if dim is not None or keepdim:
            raise ValueError("dim and keepdim require reduction='mean' or 'sum'")
        return error
    normalized_dim = _normalize_reduction_dims(dim, error.ndim)
    if reduction == "mean":
        result = error.mean(dim=normalized_dim, keepdim=keepdim)
    else:
        result = error.sum(dim=normalized_dim, keepdim=keepdim)
    if not torch.isfinite(result).all():
        raise ValueError("reduced error is not representable in the computation dtype")
    return result


__all__ = ["ErrorReduction"]
