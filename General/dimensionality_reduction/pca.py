"""Pure-PyTorch principal component analysis for reusable feature studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor


def _validate_feature_matrix(features: Tensor, *, minimum_samples: int = 1) -> None:
    if not isinstance(features, Tensor):
        raise TypeError(f"features must be a torch.Tensor, got {type(features)!r}")
    if features.ndim != 2:
        raise ValueError(f"features must have shape [samples, features], got {features.shape}")
    if features.shape[0] < minimum_samples or features.shape[1] == 0:
        raise ValueError(
            f"features must contain at least {minimum_samples} sample(s) and one feature"
        )
    if not features.is_floating_point():
        raise TypeError(f"features must be floating point, got {features.dtype}")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")


def _validate_rank_rtol(value: float | None, features: Tensor) -> float:
    if value is None:
        return max(features.shape) * torch.finfo(features.dtype).eps
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError("rank_rtol must be a non-negative finite number or None")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"rank_rtol must be non-negative and finite, got {value}")
    return result


def _resolve_component_count(
    n_components: int | float | None,
    explained_variance_ratio: Tensor,
) -> int:
    effective_rank = explained_variance_ratio.shape[0]
    if n_components is None:
        return effective_rank
    if isinstance(n_components, Integral) and not isinstance(n_components, bool):
        if not 1 <= n_components <= effective_rank:
            raise ValueError(
                f"integer n_components must be in [1, effective_rank={effective_rank}], "
                f"got {n_components}"
            )
        return int(n_components)
    if isinstance(n_components, Real) and not isinstance(n_components, bool):
        threshold = float(n_components)
        if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("float n_components must be a variance ratio in (0, 1]")
        if threshold == 1.0:
            return effective_rank
        cumulative = explained_variance_ratio.cumsum(dim=0)
        index = torch.searchsorted(
            cumulative,
            cumulative.new_tensor(threshold),
            right=False,
        )
        return min(int(index.item()) + 1, effective_rank)
    raise TypeError("n_components must be an integer, a float variance ratio, or None")


def _stabilize_component_signs(components: Tensor) -> Tensor:
    pivot_indices = components.abs().argmax(dim=1)
    pivot_values = components[
        torch.arange(components.shape[0], device=components.device),
        pivot_indices,
    ]
    signs = torch.where(
        pivot_values < 0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    return components * signs.unsqueeze(1)


@dataclass(frozen=True)
class FittedPCA:
    """Detached fitted PCA state and reusable transforms.

    ``components`` contains only the requested subset, while ``effective_rank``
    records the numerical rank available at fit time.
    """

    mean: Tensor
    scale: Tensor
    components: Tensor
    singular_values: Tensor
    explained_variance: Tensor
    explained_variance_ratio: Tensor
    cumulative_explained_variance_ratio: Tensor
    total_variance: Tensor
    sample_count: int
    feature_count: int
    effective_rank: int
    correction: int
    rank_rtol: float
    standardize: bool
    whiten: bool

    @property
    def component_count(self) -> int:
        return self.components.shape[0]

    def transform(self, features: Tensor) -> Tensor:
        """Project ``[..., input_features]`` using the fitted training state."""

        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim < 1 or features.shape[-1] != self.feature_count:
            raise ValueError(
                f"features must end with {self.feature_count} values, got {features.shape}"
            )
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if not torch.isfinite(features).all():
            raise ValueError("features must contain only finite values")
        if features.device != self.mean.device:
            raise ValueError("features and fitted PCA state must be on the same device")
        working = features.to(self.mean.dtype)
        scores = ((working - self.mean) / self.scale) @ self.components.transpose(0, 1)
        if self.whiten:
            scores = scores / self.explained_variance.sqrt()
        return scores

    def inverse_transform(self, component_scores: Tensor) -> Tensor:
        """Map component scores back to the original feature space."""

        if not isinstance(component_scores, Tensor):
            raise TypeError("component_scores must be a torch.Tensor")
        if component_scores.ndim < 1 or component_scores.shape[-1] != self.component_count:
            raise ValueError(
                f"component_scores must end with {self.component_count} values, "
                f"got {component_scores.shape}"
            )
        if not component_scores.is_floating_point():
            raise TypeError("component_scores must be floating point")
        if not torch.isfinite(component_scores).all():
            raise ValueError("component_scores must contain only finite values")
        if component_scores.device != self.mean.device:
            raise ValueError("component_scores and fitted PCA state must share a device")
        working = component_scores.to(self.mean.dtype)
        if self.whiten:
            working = working * self.explained_variance.sqrt()
        standardized = working @ self.components
        return standardized * self.scale + self.mean


@dataclass(frozen=True)
class PCAProjection:
    """PCA embedding paired with the fitted state needed to reproduce it."""

    embedding: Tensor
    fitted_pca: FittedPCA


def fit_pca(
    features: Tensor,
    *,
    n_components: int | float | None = None,
    standardize: bool = False,
    whiten: bool = False,
    correction: int = 1,
    rank_rtol: float | None = None,
) -> FittedPCA:
    """Fit PCA with SVD on a ``[samples, features]`` tensor.

    A float ``n_components`` in ``(0, 1]`` selects the smallest number of
    effective-rank components reaching that explained-variance ratio.
    """

    _validate_feature_matrix(features, minimum_samples=2)
    if not isinstance(standardize, bool) or not isinstance(whiten, bool):
        raise TypeError("standardize and whiten must be bool values")
    if not isinstance(correction, Integral) or isinstance(correction, bool):
        raise TypeError("correction must be an integer")
    if not 0 <= correction < features.shape[0]:
        raise ValueError("correction must be in [0, sample_count)")

    working = (
        features.float()
        if features.dtype in {torch.float16, torch.bfloat16}
        else features
    )
    resolved_rank_rtol = _validate_rank_rtol(rank_rtol, working)
    mean = working.mean(dim=0)
    centered = working - mean
    if standardize:
        scale = torch.linalg.vector_norm(centered, dim=0) / math.sqrt(
            features.shape[0] - correction
        )
        if not torch.isfinite(scale).all():
            raise ValueError("standardization scale is not finite; fit PCA in float64")
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    else:
        scale = torch.ones_like(mean)
    normalized = centered / scale
    if not torch.isfinite(normalized).all():
        raise ValueError("preprocessed features are not finite; fit PCA in float64")

    _, singular_values_all, right_vectors = torch.linalg.svd(
        normalized,
        full_matrices=False,
    )
    structural_max = min(features.shape[1], features.shape[0] - 1)
    singular_values_all = singular_values_all[:structural_max]
    right_vectors = right_vectors[:structural_max]
    if singular_values_all.numel() == 0:
        raise ValueError("features have no PCA components after centering")
    rank_threshold = singular_values_all[0] * resolved_rank_rtol
    effective_rank = int((singular_values_all > rank_threshold).sum().item())
    if effective_rank == 0:
        raise ValueError("features have numerical rank zero after preprocessing")

    components_effective = _stabilize_component_signs(right_vectors[:effective_rank])
    singular_values_effective = singular_values_all[:effective_rank]
    variance_effective = singular_values_effective.square() / (
        features.shape[0] - correction
    )
    if not torch.isfinite(variance_effective).all() or (variance_effective <= 0).any():
        raise ValueError("PCA variance is not representable; fit PCA in float64")
    total_variance = variance_effective.sum()
    if not torch.isfinite(total_variance):
        raise ValueError("PCA total variance is not finite; fit PCA in float64")
    relative_singular_values = singular_values_effective / singular_values_effective[0]
    relative_variance = relative_singular_values.square()
    relative_total = relative_variance.sum()
    if not torch.isfinite(relative_total) or relative_total <= 0:
        raise ValueError("PCA explained-variance ratios are not representable")
    ratio_effective = relative_variance / relative_total
    component_count = _resolve_component_count(n_components, ratio_effective)

    components = components_effective[:component_count]
    singular_values = singular_values_effective[:component_count]
    explained_variance = variance_effective[:component_count]
    explained_variance_ratio = ratio_effective[:component_count]
    if whiten and (explained_variance <= 0).any():
        raise ValueError("whitening requires positive variance for every component")

    return FittedPCA(
        mean=mean.detach(),
        scale=scale.detach(),
        components=components.detach(),
        singular_values=singular_values.detach(),
        explained_variance=explained_variance.detach(),
        explained_variance_ratio=explained_variance_ratio.detach(),
        cumulative_explained_variance_ratio=explained_variance_ratio.cumsum(dim=0).detach(),
        total_variance=total_variance.detach(),
        sample_count=features.shape[0],
        feature_count=features.shape[1],
        effective_rank=effective_rank,
        correction=int(correction),
        rank_rtol=resolved_rank_rtol,
        standardize=standardize,
        whiten=whiten,
    )


def compute_pca(
    features: Tensor,
    *,
    n_components: int | float | None = None,
    standardize: bool = False,
    whiten: bool = False,
    correction: int = 1,
    rank_rtol: float | None = None,
) -> PCAProjection:
    """Fit PCA and return the training projection together with its state."""

    fitted_pca = fit_pca(
        features,
        n_components=n_components,
        standardize=standardize,
        whiten=whiten,
        correction=correction,
        rank_rtol=rank_rtol,
    )
    return PCAProjection(
        embedding=fitted_pca.transform(features),
        fitted_pca=fitted_pca,
    )


__all__ = ["FittedPCA", "PCAProjection", "compute_pca", "fit_pca"]
