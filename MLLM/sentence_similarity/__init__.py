"""Sentence-embedding similarity for MLLM-generated text."""

from .paraphrase_minilm import (
    MiniLMCosineSimilarityResult,
    ParaphraseMiniLMCosineScorer,
)

__all__ = ["MiniLMCosineSimilarityResult", "ParaphraseMiniLMCosineScorer"]
