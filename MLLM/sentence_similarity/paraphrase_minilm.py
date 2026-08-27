"""Sentence-Transformers MiniLM semantic similarity scored with shared cosine metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from Metrics.CosineSimilarity import ZeroVectorPolicy, compute_cosine_similarity


def _validate_text_batch(texts: Sequence[str], name: str) -> list[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError(f"{name} must be a non-empty sequence of strings")
    values = list(texts)
    if not values:
        raise ValueError(f"{name} must not be empty")
    if not all(isinstance(text, str) for text in values):
        raise TypeError(f"{name} must contain only strings")
    return values


def _load_sentence_transformer_class() -> type[Any]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - dependency-specific
        raise ImportError(
            "ParaphraseMiniLMCosineScorer requires sentence-transformers; "
            "install ablation-tools[sentence-similarity]"
        ) from error
    return SentenceTransformer


@dataclass(frozen=True)
class MiniLMCosineSimilarityResult:
    """Aligned semantic similarity scores and reproducibility metadata."""

    scores: Tensor
    mean_score: Tensor
    prediction_count: int
    embedding_dimension: int
    normalize_embeddings: bool
    eps: float
    zero_vector: ZeroVectorPolicy
    model_name_or_path: str | None
    prediction_embeddings: Tensor | None
    reference_embeddings: Tensor | None


class ParaphraseMiniLMCosineScorer:
    """Encode aligned text pairs and compute cosine semantic similarity.

    The encoder is injected for testing/custom loading. Use
    :meth:`from_pretrained` for a standard Sentence-Transformers checkpoint.
    """

    def __init__(self, encoder: Any, *, model_name_or_path: str | None = None) -> None:
        if not hasattr(encoder, "encode") or not callable(encoder.encode):
            raise TypeError("encoder must provide a callable encode method")
        if model_name_or_path is not None and not isinstance(model_name_or_path, str):
            raise TypeError("model_name_or_path must be a string or None")
        if hasattr(encoder, "eval"):
            evaluated = encoder.eval()
            if evaluated is not None:
                encoder = evaluated
        self.encoder = encoder
        self.model_name_or_path = model_name_or_path

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        local_files_only: bool = True,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> ParaphraseMiniLMCosineScorer:
        """Load a Sentence-Transformers model from a local path or model ID."""

        identifier = str(model_name_or_path)
        if not identifier:
            raise ValueError("model_name_or_path must not be empty")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a bool")
        options = dict(model_kwargs or {})
        options.setdefault("device", str(torch.device(device)))
        options.setdefault("local_files_only", local_files_only)
        sentence_transformer = _load_sentence_transformer_class()
        encoder = sentence_transformer(identifier, **options)
        return cls(encoder, model_name_or_path=identifier)

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> Tensor:
        """Encode one non-empty text batch into finite ``[N, D]`` embeddings."""

        values = _validate_text_batch(texts, "texts")
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(normalize_embeddings, bool) or not isinstance(
            show_progress_bar, bool
        ):
            raise TypeError("normalize_embeddings and show_progress_bar must be bool values")
        options = dict(encode_kwargs or {})
        reserved = {
            "batch_size",
            "show_progress_bar",
            "convert_to_tensor",
            "normalize_embeddings",
        }
        overlap = reserved.intersection(options)
        if overlap:
            raise ValueError(f"encode_kwargs must not override reserved options: {sorted(overlap)}")
        with torch.inference_mode():
            embeddings = self.encoder.encode(
                values,
                batch_size=int(batch_size),
                show_progress_bar=show_progress_bar,
                convert_to_tensor=True,
                normalize_embeddings=normalize_embeddings,
                **options,
            )
        if not isinstance(embeddings, Tensor):
            embeddings = torch.as_tensor(embeddings)
        embeddings = embeddings.detach()
        if embeddings.ndim != 2 or embeddings.shape[0] != len(values):
            raise ValueError(
                "encoder must return [text_count, embedding_dimension], got "
                f"{tuple(embeddings.shape)} for {len(values)} texts"
            )
        if embeddings.shape[1] == 0 or not embeddings.is_floating_point():
            raise TypeError("encoder embeddings must be non-empty floating-point vectors")
        if not torch.isfinite(embeddings).all():
            raise ValueError("encoder embeddings must contain only finite values")
        return embeddings

    def score(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        eps: float = 1e-8,
        zero_vector: ZeroVectorPolicy = "raise",
        show_progress_bar: bool = False,
        return_embeddings: bool = False,
        encode_kwargs: Mapping[str, Any] | None = None,
    ) -> MiniLMCosineSimilarityResult:
        """Score aligned prediction/reference strings with one shared encode call."""

        prediction_values = _validate_text_batch(predictions, "predictions")
        reference_values = _validate_text_batch(references, "references")
        if len(prediction_values) != len(reference_values):
            raise ValueError("predictions and references must have the same length")
        if not isinstance(return_embeddings, bool):
            raise TypeError("return_embeddings must be a bool")
        combined = prediction_values + reference_values
        embeddings = self.encode_texts(
            combined,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
            encode_kwargs=encode_kwargs,
        )
        count = len(prediction_values)
        prediction_embeddings = embeddings[:count]
        reference_embeddings = embeddings[count:]
        scores = compute_cosine_similarity(
            prediction_embeddings,
            reference_embeddings,
            dim=-1,
            eps=eps,
            zero_vector=zero_vector,
            reduction="none",
        )
        return MiniLMCosineSimilarityResult(
            scores=scores,
            mean_score=scores.mean(),
            prediction_count=count,
            embedding_dimension=embeddings.shape[1],
            normalize_embeddings=normalize_embeddings,
            eps=float(eps),
            zero_vector=zero_vector,
            model_name_or_path=self.model_name_or_path,
            prediction_embeddings=prediction_embeddings if return_embeddings else None,
            reference_embeddings=reference_embeddings if return_embeddings else None,
        )


__all__ = ["MiniLMCosineSimilarityResult", "ParaphraseMiniLMCosineScorer"]
