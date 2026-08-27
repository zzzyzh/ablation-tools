"""Aligned and pairwise Cosine Similarity Score metrics."""

from ._common import CosineReduction, ZeroVectorPolicy
from .cosine_similarity import compute_cosine_similarity
from .pairwise_cosine_similarity import compute_pairwise_cosine_similarity

__all__ = [
    "CosineReduction",
    "ZeroVectorPolicy",
    "compute_cosine_similarity",
    "compute_pairwise_cosine_similarity",
]
