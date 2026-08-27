"""Aligned cosine similarity with no implicit broadcasting."""

from __future__ import annotations

import torch
from torch import Tensor

from ._common import (
    CosineReduction,
    ZeroVectorPolicy,
    normalize_feature_dim,
    normalize_vectors,
    prepare_cosine_inputs,
    reduce_cosine_scores,
    validate_eps,
    validate_zero_vector_policy,
)


def compute_cosine_similarity(
    first: Tensor,
    second: Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
    zero_vector: ZeroVectorPolicy = "zero",
    reduction: CosineReduction = "none",
) -> Tensor:
    """Compute aligned cosine similarity along one feature dimension.

    Inputs must have identical, non-empty shapes. ``reduction='none'`` removes
    only the feature dimension; mean/sum aggregate all resulting paired scores.
    """

    if not isinstance(first, Tensor) or not isinstance(second, Tensor):
        raise TypeError("first and second must be torch.Tensor values")
    if first.shape != second.shape:
        raise ValueError(
            f"first and second must have identical shapes, got {first.shape} and {second.shape}"
        )
    if first.ndim == 0:
        raise ValueError("cosine similarity requires at least one feature dimension")
    feature_dim = normalize_feature_dim(dim, first.ndim)
    first, second = prepare_cosine_inputs(first, second)
    policy = validate_zero_vector_policy(zero_vector)
    epsilon = validate_eps(eps, first.dtype, first.device)
    first_normalized, first_zero = normalize_vectors(
        first,
        dim=feature_dim,
        eps=epsilon,
        zero_vector=policy,
    )
    second_normalized, second_zero = normalize_vectors(
        second,
        dim=feature_dim,
        eps=epsilon,
        zero_vector=policy,
    )
    scores = (first_normalized * second_normalized).sum(dim=feature_dim)
    if policy == "zero":
        scores = torch.where(first_zero | second_zero, torch.zeros_like(scores), scores)
    return reduce_cosine_scores(scores, reduction)


__all__ = ["compute_cosine_similarity"]
