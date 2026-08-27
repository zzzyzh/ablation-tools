"""Dependency-free ROUGE-L for tokenized or raw multilingual text."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real


RougeToken = str | int
RougeTokenizer = Callable[[str], Sequence[RougeToken]]
RougeInput = str | Sequence[RougeToken]


def _validate_binary_fill(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be 0.0 or 1.0")
    result = float(value)
    if not math.isfinite(result) or result not in {0.0, 1.0}:
        raise ValueError(f"{name} must be 0.0 or 1.0")
    return result


def _validate_tokens(tokens: Sequence[RougeToken], name: str) -> tuple[RougeToken, ...]:
    if isinstance(tokens, (str, bytes)):
        raise TypeError(f"{name} tokens must be a sequence, not one string")
    result = tuple(tokens)
    if not all(
        isinstance(token, str)
        or (isinstance(token, Integral) and not isinstance(token, bool))
        for token in result
    ):
        raise TypeError(f"{name} tokens must contain only strings or integer token IDs")
    return tuple(int(token) if isinstance(token, Integral) else token for token in result)


def tokenize_rouge_text(text: str, *, lowercase: bool = False) -> tuple[str, ...]:
    """Tokenize text by whitespace without punctuation removal or stemming."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text)!r}")
    if not isinstance(lowercase, bool):
        raise TypeError("lowercase must be a bool")
    normalized = text.lower() if lowercase else text
    return tuple(normalized.split())



def tokenize_rouge_characters(
    text: str,
    *,
    lowercase: bool = False,
    ignore_whitespace: bool = True,
) -> tuple[str, ...]:
    """Tokenize Unicode code points for an explicit character-level baseline."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text)!r}")
    if not isinstance(lowercase, bool) or not isinstance(ignore_whitespace, bool):
        raise TypeError("lowercase and ignore_whitespace must be bool values")
    normalized = text.lower() if lowercase else text
    return tuple(
        character
        for character in normalized
        if not (ignore_whitespace and character.isspace())
    )
def _prepare_rouge_tokens(
    value: RougeInput,
    *,
    name: str,
    tokenizer: RougeTokenizer | None,
    lowercase: bool,
) -> tuple[RougeToken, ...]:
    if isinstance(value, str):
        normalized = value.lower() if lowercase else value
        if tokenizer is None:
            return tokenize_rouge_text(normalized)
        if not callable(tokenizer):
            raise TypeError("tokenizer must be callable or None")
        tokens = tokenizer(normalized)
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            raise TypeError("tokenizer must return a token sequence, not one string")
        return _validate_tokens(tokens, name)
    if tokenizer is not None and not callable(tokenizer):
        raise TypeError("tokenizer must be callable or None")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be text or a token sequence")
    tokens = _validate_tokens(value, name)
    if lowercase:
        return tuple(token.lower() if isinstance(token, str) else token for token in tokens)
    return tokens


def compute_lcs_length(
    prediction_tokens: Sequence[RougeToken],
    reference_tokens: Sequence[RougeToken],
) -> int:
    """Compute longest-common-subsequence length in O(min(m, n)) memory."""

    prediction = _validate_tokens(prediction_tokens, "prediction")
    reference = _validate_tokens(reference_tokens, "reference")
    if len(prediction) < len(reference):
        shorter, longer = prediction, reference
    else:
        shorter, longer = reference, prediction
    previous = [0] * (len(shorter) + 1)
    for longer_token in longer:
        current = [0]
        for index, shorter_token in enumerate(shorter, start=1):
            if longer_token == shorter_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class RougeLScore:
    """ROUGE-L Precision, Recall, F-beta, and underlying sequence lengths."""

    precision: float
    recall: float
    fmeasure: float
    lcs_length: int
    prediction_length: int
    reference_length: int
    beta: float


@dataclass(frozen=True)
class MultiReferenceRougeLScore:
    """Best ROUGE-L score and its stable reference index."""

    score: RougeLScore
    reference_index: int
    reference_count: int


def compute_rouge_l(
    prediction: RougeInput,
    reference: RougeInput,
    *,
    tokenizer: RougeTokenizer | None = None,
    lowercase: bool = False,
    beta: float = 1.0,
    zero_division: float = 0.0,
) -> RougeLScore:
    """Compute sentence-level ROUGE-L from one prediction/reference pair."""

    if not isinstance(lowercase, bool):
        raise TypeError("lowercase must be a bool")
    if not isinstance(beta, Real) or isinstance(beta, bool):
        raise TypeError("beta must be a positive finite number")
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    fill = _validate_binary_fill(zero_division, "zero_division")
    prediction_tokens = _prepare_rouge_tokens(
        prediction,
        name="prediction",
        tokenizer=tokenizer,
        lowercase=lowercase,
    )
    reference_tokens = _prepare_rouge_tokens(
        reference,
        name="reference",
        tokenizer=tokenizer,
        lowercase=lowercase,
    )
    lcs_length = compute_lcs_length(prediction_tokens, reference_tokens)
    precision = (
        lcs_length / len(prediction_tokens) if prediction_tokens else fill
    )
    recall = lcs_length / len(reference_tokens) if reference_tokens else fill
    if beta >= 1.0:
        inverse_beta_squared = (1.0 / beta) ** 2
        denominator = precision + inverse_beta_squared * recall
        numerator = (1.0 + inverse_beta_squared) * precision * recall
    else:
        beta_squared = beta * beta
        denominator = recall + beta_squared * precision
        numerator = (1.0 + beta_squared) * precision * recall
    fmeasure = numerator / denominator if denominator > 0 else fill
    return RougeLScore(
        precision=float(precision),
        recall=float(recall),
        fmeasure=float(fmeasure),
        lcs_length=lcs_length,
        prediction_length=len(prediction_tokens),
        reference_length=len(reference_tokens),
        beta=beta,
    )


def compute_rouge_l_multi_reference(
    prediction: RougeInput,
    references: Sequence[RougeInput],
    *,
    tokenizer: RougeTokenizer | None = None,
    lowercase: bool = False,
    beta: float = 1.0,
    zero_division: float = 0.0,
) -> MultiReferenceRougeLScore:
    """Select the highest-F score across references, keeping the first tie."""

    if isinstance(references, (str, bytes)) or not isinstance(references, Sequence):
        raise TypeError("references must be a non-empty sequence")
    if not references:
        raise ValueError("references must not be empty")
    scores = tuple(
        compute_rouge_l(
            prediction,
            reference,
            tokenizer=tokenizer,
            lowercase=lowercase,
            beta=beta,
            zero_division=zero_division,
        )
        for reference in references
    )
    best_index = max(
        range(len(scores)),
        key=lambda index: (
            scores[index].fmeasure,
            scores[index].recall,
            scores[index].precision,
            -index,
        ),
    )
    return MultiReferenceRougeLScore(
        score=scores[best_index],
        reference_index=best_index,
        reference_count=len(scores),
    )


def compute_rouge_l_batch(
    predictions: Sequence[RougeInput],
    references: Sequence[RougeInput],
    *,
    tokenizer: RougeTokenizer | None = None,
    lowercase: bool = False,
    beta: float = 1.0,
    zero_division: float = 0.0,
) -> tuple[RougeLScore, ...]:
    """Compute aligned one-reference ROUGE-L pairs without hidden aggregation."""

    for values, name in ((predictions, "predictions"), (references, "references")):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} must be a non-empty sequence")
        if not values:
            raise ValueError(f"{name} must not be empty")
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same batch length")
    return tuple(
        compute_rouge_l(
            prediction,
            reference,
            tokenizer=tokenizer,
            lowercase=lowercase,
            beta=beta,
            zero_division=zero_division,
        )
        for prediction, reference in zip(predictions, references)
    )


__all__ = [
    "MultiReferenceRougeLScore",
    "RougeInput",
    "RougeLScore",
    "RougeToken",
    "RougeTokenizer",
    "compute_lcs_length",
    "compute_rouge_l",
    "compute_rouge_l_batch",
    "compute_rouge_l_multi_reference",
    "tokenize_rouge_text",
    "tokenize_rouge_characters",
]
