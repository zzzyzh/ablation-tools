"""Reusable dimensionality-reduction methods."""

from .pca import FittedPCA, PCAProjection, compute_pca, fit_pca
from .tsne import (
    JointTSNEResult,
    TSNEConfig,
    TSNEResult,
    fit_joint_tsne,
    fit_tsne,
)

__all__ = [
    "FittedPCA",
    "JointTSNEResult",
    "PCAProjection",
    "TSNEConfig",
    "TSNEResult",
    "compute_pca",
    "fit_joint_tsne",
    "fit_pca",
    "fit_tsne",
]
