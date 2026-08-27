"""All-pairs cosine similarity for two explicit 2-D vector collections."""

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


def compute_pairwise_cosine_similarity(
    first: Tensor,
    second: Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
    zero_vector: ZeroVectorPolicy = "zero",
    reduction: CosineReduction = "none",
) -> Tensor:
    """Compute an ``[first_vectors, second_vectors]`` cosine matrix.

    Both inputs must be two-dimensional. ``dim`` identifies their feature axis;
    the other axis identifies vectors. No batch flattening or broadcasting occurs.
    """

    if not isinstance(first, Tensor) or not isinstance(second, Tensor):
        raise TypeError("first and second must be torch.Tensor values")
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("pairwise cosine similarity requires two 2-D tensors")
    first_dim = normalize_feature_dim(dim, first.ndim)
    second_dim = normalize_feature_dim(dim, second.ndim)
    if first.shape[first_dim] != second.shape[second_dim]:
        raise ValueError("first and second must have equal feature lengths")
    first, second = prepare_cosine_inputs(first, second)
    if first_dim == 0:
        first = first.transpose(0, 1)
        second = second.transpose(0, 1)
    policy = validate_zero_vector_policy(zero_vector)
    epsilon = validate_eps(eps, first.dtype, first.device)
    first_normalized, first_zero = normalize_vectors(
        first,
        dim=1,
        eps=epsilon,
        zero_vector=policy,
    )
    second_normalized, second_zero = normalize_vectors(
        second,
        dim=1,
        eps=epsilon,
        zero_vector=policy,
    )
    scores = first_normalized @ second_normalized.transpose(0, 1)
    if policy == "zero":
        undefined = first_zero.unsqueeze(1) | second_zero.unsqueeze(0)
        scores = torch.where(undefined, torch.zeros_like(scores), scores)
    return reduce_cosine_scores(scores, reduction)


__all__ = ["compute_pairwise_cosine_similarity"]
