"""Shared validation, normalization, and reduction for cosine similarity."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Literal

import torch
from torch import Tensor


CosineReduction = Literal["none", "mean", "sum"]
ZeroVectorPolicy = Literal["zero", "raise"]


def validate_cosine_tensor(values: Tensor, name: str) -> None:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(values)!r}")
    if values.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if values.dtype == torch.bool or values.is_complex():
        raise TypeError(f"{name} must contain real numeric values")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if not values.is_floating_point():
        exact_limit = 2**52
        dtype_range = torch.iinfo(values.dtype)
        if dtype_range.min < 0:
            outside = (values < -exact_limit) | (values > exact_limit)
        else:
            outside = values > exact_limit
        if outside.any():
            raise ValueError(
                f"integer {name} must stay within [-2**52, 2**52] for exact float64 conversion"
            )


def prepare_cosine_inputs(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
    validate_cosine_tensor(first, "first")
    validate_cosine_tensor(second, "second")
    if first.device != second.device:
        raise ValueError("first and second must be on the same device")
    if not first.is_floating_point() or not second.is_floating_point():
        dtype = torch.float64
    else:
        dtype = torch.promote_types(first.dtype, second.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    return first.to(dtype), second.to(dtype)


def normalize_feature_dim(dim: int, ndim: int) -> int:
    if not isinstance(dim, Integral) or isinstance(dim, bool):
        raise TypeError("dim must be an integer")
    normalized = int(dim)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"dim {dim} is out of range for a {ndim}-D tensor")
    return normalized


def validate_eps(eps: float, dtype: torch.dtype, device: torch.device) -> Tensor:
    if not isinstance(eps, Real) or isinstance(eps, bool):
        raise TypeError("eps must be a positive finite number")
    value = float(eps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("eps must be positive and finite")
    represented = torch.tensor(value, dtype=dtype, device=device)
    if not torch.isfinite(represented) or represented <= 0:
        raise ValueError(f"eps={eps} is not representable in computation dtype {dtype}")
    return represented


def validate_zero_vector_policy(policy: str) -> ZeroVectorPolicy:
    if policy not in {"zero", "raise"}:
        raise ValueError("zero_vector must be 'zero' or 'raise'")
    return policy  # type: ignore[return-value]


def normalize_vectors(
    values: Tensor,
    *,
    dim: int,
    eps: Tensor,
    zero_vector: ZeroVectorPolicy,
) -> tuple[Tensor, Tensor]:
    norms = torch.linalg.vector_norm(values, dim=dim, keepdim=True)
    if not torch.isfinite(norms).all():
        raise ValueError("vector norms are not representable in the computation dtype")
    zero_mask = norms == 0
    if zero_vector == "raise" and zero_mask.any():
        raise ValueError("cosine similarity is undefined for exact zero vectors")
    normalized = values / norms.clamp_min(eps)
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized vectors are not finite")
    return normalized, zero_mask.squeeze(dim)


def reduce_cosine_scores(scores: Tensor, reduction: CosineReduction) -> Tensor:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not torch.isfinite(scores).all():
        raise ValueError("cosine similarity scores must be finite")
    scores = scores.clamp(-1.0, 1.0)
    if reduction == "none":
        return scores
    result = scores.mean() if reduction == "mean" else scores.sum()
    if not torch.isfinite(result):
        raise ValueError("reduced cosine similarity is not representable")
    return result


__all__ = ["CosineReduction", "ZeroVectorPolicy"]
