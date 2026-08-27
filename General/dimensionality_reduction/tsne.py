"""Explicit scikit-learn t-SNE workflows for high-dimensional feature studies."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Literal

import torch
from torch import Tensor

from .pca import fit_pca


TSNEInit = Literal["pca", "random"]
TSNEMethod = Literal["barnes_hut", "exact"]


def _validate_positive_finite(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return result

def _validate_nonnegative_finite(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be non-negative and finite, got {value}")
    return result




@dataclass(frozen=True)
class TSNEConfig:
    """Common t-SNE parameters with explicit optional PCA pre-reduction."""

    n_components: int = 2
    perplexity: float = 30.0
    early_exaggeration: float = 12.0
    learning_rate: float | Literal["auto"] = "auto"
    max_iter: int = 1000
    n_iter_without_progress: int = 300
    min_grad_norm: float = 1e-7
    metric: str = "euclidean"
    init: TSNEInit = "pca"
    method: TSNEMethod = "barnes_hut"
    angle: float = 0.5
    n_jobs: int | None = None
    random_state: int | None = 0
    verbose: int = 0
    pca_components: int | None = None
    pca_standardize: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.n_components, "n_components"),
            (self.max_iter, "max_iter"),
            (self.n_iter_without_progress, "n_iter_without_progress"),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")
        if self.max_iter < 250:
            raise ValueError("max_iter must be at least 250")
        if self.n_iter_without_progress < -1:
            raise ValueError("n_iter_without_progress must be at least -1")
        _validate_positive_finite(self.perplexity, "perplexity")
        early_exaggeration = _validate_positive_finite(
            self.early_exaggeration, "early_exaggeration"
        )
        if early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1.0")
        _validate_nonnegative_finite(self.min_grad_norm, "min_grad_norm")
        if self.learning_rate != "auto":
            _validate_positive_finite(self.learning_rate, "learning_rate")
        if not isinstance(self.metric, str) or not self.metric:
            raise TypeError("metric must be a non-empty string")
        if self.metric == "precomputed":
            raise ValueError("precomputed distances are outside the feature-matrix API")
        if self.init not in {"pca", "random"}:
            raise ValueError("init must be 'pca' or 'random'")
        if self.method not in {"barnes_hut", "exact"}:
            raise ValueError("method must be 'barnes_hut' or 'exact'")
        if self.method == "barnes_hut" and self.n_components > 3:
            raise ValueError("barnes_hut supports at most 3 output components")
        if not isinstance(self.angle, Real) or isinstance(self.angle, bool):
            raise TypeError("angle must be a finite number in [0, 1]")
        if not math.isfinite(float(self.angle)) or not 0.0 <= self.angle <= 1.0:
            raise ValueError("angle must be finite and in [0, 1]")
        if self.n_jobs is not None and (
            not isinstance(self.n_jobs, Integral)
            or isinstance(self.n_jobs, bool)
            or self.n_jobs == 0
        ):
            raise ValueError("n_jobs must be a non-zero integer or None")
        if self.random_state is not None and (
            not isinstance(self.random_state, Integral)
            or isinstance(self.random_state, bool)
        ):
            raise TypeError("random_state must be an integer or None")
        if not isinstance(self.verbose, Integral) or isinstance(self.verbose, bool):
            raise TypeError("verbose must be a non-negative integer")
        if self.verbose < 0:
            raise ValueError("verbose must be non-negative")
        if self.pca_components is not None and (
            not isinstance(self.pca_components, Integral)
            or isinstance(self.pca_components, bool)
            or self.pca_components <= 0
        ):
            raise ValueError("pca_components must be a positive integer or None")
        if not isinstance(self.pca_standardize, bool):
            raise TypeError("pca_standardize must be a bool")
        if self.pca_standardize and self.pca_components is None:
            raise ValueError("pca_standardize requires pca_components")


@dataclass(frozen=True)
class TSNEResult:
    """CPU t-SNE coordinates and diagnostics needed to reproduce them."""

    embedding: Tensor
    kl_divergence: float
    n_iter: int
    effective_learning_rate: float
    config: TSNEConfig
    input_shape: tuple[int, int]
    tsne_input_feature_count: int
    pca_components_used: int | None
    pca_explained_variance_ratio: Tensor | None


@dataclass(frozen=True)
class JointTSNEResult:
    """One joint t-SNE fit split back into named feature groups."""

    embeddings: Mapping[str, Tensor]
    group_slices: Mapping[str, tuple[int, int]]
    combined_result: TSNEResult


def _validate_features(features: Tensor, *, minimum_samples: int = 2) -> Tensor:
    if not isinstance(features, Tensor):
        raise TypeError(f"features must be a torch.Tensor, got {type(features)!r}")
    if features.ndim != 2:
        raise ValueError(f"features must have shape [samples, features], got {features.shape}")
    if features.shape[0] < minimum_samples or features.shape[1] == 0:
        raise ValueError(f"features need at least {minimum_samples} sample(s) and one feature")
    if not features.is_floating_point():
        raise TypeError("features must be floating point")
    if not torch.isfinite(features).all():
        raise ValueError("features must contain only finite values")
    detached = features.detach().cpu()
    if detached.dtype in {torch.float16, torch.bfloat16}:
        detached = detached.float()
    return detached


def _load_tsne_class() -> type[Any]:
    try:
        from sklearn.manifold import TSNE
    except ImportError as error:  # pragma: no cover - environment-specific
        raise ImportError(
            "t-SNE requires scikit-learn; install "
            "ablation-tools[dimensionality-reduction]"
        ) from error
    return TSNE


def _build_estimator(tsne_class: type[Any], config: TSNEConfig) -> Any:
    parameters = inspect.signature(tsne_class).parameters
    iteration_name = "max_iter" if "max_iter" in parameters else "n_iter"
    options: dict[str, Any] = {
        "n_components": config.n_components,
        "perplexity": float(config.perplexity),
        "early_exaggeration": float(config.early_exaggeration),
        "learning_rate": config.learning_rate,
        iteration_name: int(config.max_iter),
        "n_iter_without_progress": int(config.n_iter_without_progress),
        "min_grad_norm": float(config.min_grad_norm),
        "metric": config.metric,
        "init": config.init,
        "verbose": int(config.verbose),
        "random_state": config.random_state,
        "method": config.method,
        "angle": float(config.angle),
        "n_jobs": config.n_jobs,
    }
    return tsne_class(**options)


def _preprocess_with_optional_pca(
    features: Tensor,
    config: TSNEConfig,
) -> tuple[Tensor, int | None, Tensor | None]:
    if config.pca_components is None or features.shape[1] <= config.pca_components:
        return features, None, None
    fitted = fit_pca(
        features,
        n_components=None,
        standardize=config.pca_standardize,
        whiten=False,
    )
    actual_components = min(int(config.pca_components), fitted.effective_rank)
    transformed = fitted.transform(features)[:, :actual_components]
    ratio = fitted.explained_variance_ratio[:actual_components].clone()
    return transformed, actual_components, ratio


def _effective_learning_rate(config: TSNEConfig, sample_count: int, estimator: Any) -> float:
    learned = getattr(estimator, "learning_rate_", None)
    if learned is not None:
        return float(learned)
    if config.learning_rate != "auto":
        return float(config.learning_rate)
    return max(sample_count / float(config.early_exaggeration) / 4.0, 50.0)


def fit_tsne(features: Tensor, *, config: TSNEConfig | None = None) -> TSNEResult:
    """Fit t-SNE once; no out-of-sample transform is implied or provided."""

    resolved_config = TSNEConfig() if config is None else config
    if not isinstance(resolved_config, TSNEConfig):
        raise TypeError("config must be a TSNEConfig or None")
    prepared = _validate_features(features)
    if not float(resolved_config.perplexity) < prepared.shape[0]:
        raise ValueError(
            f"perplexity must be smaller than sample count {prepared.shape[0]}, "
            f"got {resolved_config.perplexity}"
        )
    tsne_input, pca_components_used, pca_ratio = _preprocess_with_optional_pca(
        prepared,
        resolved_config,
    )
    tsne_class = _load_tsne_class()
    estimator = _build_estimator(tsne_class, resolved_config)
    embedding_array = estimator.fit_transform(tsne_input.numpy())
    embedding = torch.as_tensor(embedding_array).detach().cpu().clone()
    expected_shape = (prepared.shape[0], resolved_config.n_components)
    if embedding.shape != expected_shape:
        raise ValueError(
            f"t-SNE returned shape {tuple(embedding.shape)}, expected {expected_shape}"
        )
    if not embedding.is_floating_point() or not torch.isfinite(embedding).all():
        raise ValueError("t-SNE embedding must contain finite floating-point values")
    kl_divergence = float(getattr(estimator, "kl_divergence_"))
    if not math.isfinite(kl_divergence):
        raise ValueError("t-SNE kl_divergence_ must be finite")
    n_iter = int(getattr(estimator, "n_iter_"))
    return TSNEResult(
        embedding=embedding,
        kl_divergence=kl_divergence,
        n_iter=n_iter,
        effective_learning_rate=_effective_learning_rate(
            resolved_config,
            prepared.shape[0],
            estimator,
        ),
        config=resolved_config,
        input_shape=(prepared.shape[0], prepared.shape[1]),
        tsne_input_feature_count=tsne_input.shape[1],
        pca_components_used=pca_components_used,
        pca_explained_variance_ratio=pca_ratio,
    )


def fit_joint_tsne(
    feature_groups: Mapping[str, Tensor],
    *,
    config: TSNEConfig | None = None,
) -> JointTSNEResult:
    """Concatenate named groups, fit one shared t-SNE, then split coordinates."""

    if not isinstance(feature_groups, Mapping) or not feature_groups:
        raise ValueError("feature_groups must be a non-empty mapping")
    names = list(feature_groups)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("feature group names must be non-empty strings")
    prepared_groups = [
        _validate_features(feature_groups[name], minimum_samples=1) for name in names
    ]
    feature_count = prepared_groups[0].shape[1]
    if any(group.shape[1] != feature_count for group in prepared_groups):
        raise ValueError("all feature groups must have the same feature count")

    slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, group in zip(names, prepared_groups):
        slices[name] = (offset, offset + group.shape[0])
        offset += group.shape[0]
    combined = torch.cat(prepared_groups, dim=0)
    combined_result = fit_tsne(combined, config=config)
    embeddings = {
        name: combined_result.embedding[start:end]
        for name, (start, end) in slices.items()
    }
    return JointTSNEResult(
        embeddings=embeddings,
        group_slices=slices,
        combined_result=combined_result,
    )


__all__ = [
    "JointTSNEResult",
    "TSNEConfig",
    "TSNEResult",
    "fit_joint_tsne",
    "fit_tsne",
]
