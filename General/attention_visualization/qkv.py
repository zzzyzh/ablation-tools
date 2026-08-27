"""Model-agnostic utilities for analysing query, key, and value tensors.

The functions in this module operate on already-projected tensors.  They do
not register hooks, inspect a model, or assume a particular transformer
implementation.  Query and key tensors use the canonical shapes
``[..., query_tokens, channels]`` and ``[..., key_tokens, channels]``.  Value
tensors use ``[..., key_tokens, value_channels]``.

Boolean attention masks use ``True`` for visible positions.  Floating-point
masks are interpreted as additive logit biases, which makes both causal masks
and soft attention biases possible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor


__all__ = [
    "QKVAttentionResult",
    "QKVFeatureMaps",
    "compute_qk_attention_logits",
    "compute_qkv_attention",
    "compute_qkv_feature_maps",
    "compute_token_feature_map",
]


@dataclass(frozen=True)
class QKVAttentionResult:
    """Outputs reconstructed from query, key, and value tensors.

    Attributes:
        logits: Scaled (or unscaled) QK similarities with shape
            ``[..., query_tokens, key_tokens]``.  Any mask is already applied.
        probabilities: Row-wise attention probabilities with the same shape as
            ``logits``.  A fully masked row is represented by zeros.
        context: Probability-weighted values with shape
            ``[..., query_tokens, value_channels]``.
    """

    logits: Tensor
    probabilities: Tensor
    context: Tensor


@dataclass(frozen=True)
class QKVFeatureMaps:
    """Scalar feature maps derived independently from Q, K, and V tensors."""

    query: Tensor
    key: Tensor
    value: Tensor


def _validate_feature_tensor(tensor: Tensor, name: str) -> None:
    """Validate the tensor properties shared by all public operations."""

    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} must have at least two dimensions [..., tokens, channels], "
            f"got shape {tuple(tensor.shape)}."
        )
    if tensor.shape[-2] <= 0:
        raise ValueError(f"{name} must contain at least one token.")
    if tensor.shape[-1] <= 0:
        raise ValueError(f"{name} must contain at least one feature channel.")
    if any(size <= 0 for size in tensor.shape[:-2]):
        raise ValueError(
            f"{name} must not contain empty leading dimensions, got shape "
            f"{tuple(tensor.shape)}."
        )
    if not tensor.is_floating_point():
        raise TypeError(
            f"{name} must have a floating-point dtype, got {tensor.dtype}."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")


def _broadcast_shapes(
    left: Sequence[int],
    right: Sequence[int],
    *,
    description: str,
) -> tuple[int, ...]:
    """Return a broadcast shape while replacing PyTorch errors with context."""

    try:
        return tuple(torch.broadcast_shapes(tuple(left), tuple(right)))
    except RuntimeError as error:
        raise ValueError(
            f"{description} are not broadcast-compatible: "
            f"{tuple(left)} and {tuple(right)}."
        ) from error


def _apply_attention_mask(logits: Tensor, attention_mask: Tensor | None) -> Tensor:
    """Apply a validated boolean visibility mask or additive floating mask."""

    if attention_mask is None:
        return logits
    if not isinstance(attention_mask, Tensor):
        raise TypeError(
            "attention_mask must be a torch.Tensor or None, got "
            f"{type(attention_mask).__name__}."
        )
    if attention_mask.device != logits.device:
        raise ValueError(
            "attention_mask must be on the same device as query and key; "
            f"got {attention_mask.device} and {logits.device}."
        )

    broadcast_shape = _broadcast_shapes(
        attention_mask.shape,
        logits.shape,
        description="attention_mask and attention logits shapes",
    )
    if broadcast_shape != tuple(logits.shape):
        raise ValueError(
            "attention_mask may broadcast over attention logits but must not "
            "introduce additional dimensions; got mask shape "
            f"{tuple(attention_mask.shape)} for logits shape {tuple(logits.shape)}."
        )

    if attention_mask.dtype == torch.bool:
        return logits.masked_fill(~attention_mask, -torch.inf)
    if attention_mask.is_floating_point():
        if torch.isnan(attention_mask).any() or torch.isposinf(attention_mask).any():
            raise ValueError(
                "floating attention_mask may contain finite biases or negative "
                "infinity, but not NaN or positive infinity."
            )
        # Attention masks commonly use float32 even when activations use a
        # lower-precision dtype.  Applying the bias in the logits dtype keeps
        # the subsequent probability/value matmul well-defined.
        return logits + attention_mask.to(dtype=logits.dtype)
    raise TypeError(
        "attention_mask must have bool or floating-point dtype; "
        f"got {attention_mask.dtype}."
    )


def compute_qk_attention_logits(
    query: Tensor,
    key: Tensor,
    *,
    scale: bool = True,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Compute model-independent dot-product attention logits.

    Args:
        query: Query features shaped ``[..., query_tokens, channels]``.
        key: Key features shaped ``[..., key_tokens, channels]``.  Its leading
            dimensions must broadcast with those of ``query``.
        scale: Divide similarities by ``sqrt(channels)`` when ``True``.
        attention_mask: Optional mask broadcastable *to* the resulting
            ``[..., query_tokens, key_tokens]`` shape.  A boolean mask uses
            ``True`` for visible positions; a floating mask is added to the
            logits (use ``-inf`` to hide positions).

    Returns:
        The masked attention logits.  Masked boolean positions contain
        negative infinity.

    Raises:
        TypeError: If tensor dtypes or argument types are unsupported.
        ValueError: If feature sizes, devices, or broadcast dimensions do not
            agree.
    """

    _validate_feature_tensor(query, "query")
    _validate_feature_tensor(key, "key")
    if not isinstance(scale, bool):
        raise TypeError(f"scale must be a bool, got {type(scale).__name__}.")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "query and key must have the same feature size; got "
            f"{query.shape[-1]} and {key.shape[-1]}."
        )
    if query.device != key.device:
        raise ValueError(
            "query and key must be on the same device; "
            f"got {query.device} and {key.device}."
        )
    if query.dtype != key.dtype:
        raise TypeError(
            "query and key must have the same dtype; "
            f"got {query.dtype} and {key.dtype}."
        )

    _broadcast_shapes(
        query.shape[:-2],
        key.shape[:-2],
        description="query and key leading dimensions",
    )

    logits = torch.matmul(query, key.transpose(-2, -1))
    if scale:
        logits = logits / math.sqrt(query.shape[-1])
    return _apply_attention_mask(logits, attention_mask)


def _safe_attention_softmax(logits: Tensor) -> Tensor:
    """Apply softmax while mapping fully masked (all ``-inf``) rows to zero."""

    fully_masked = torch.isneginf(logits).all(dim=-1, keepdim=True)
    safe_logits = torch.where(fully_masked, torch.zeros_like(logits), logits)
    probabilities = torch.softmax(safe_logits, dim=-1)
    return torch.where(fully_masked, torch.zeros_like(probabilities), probabilities)


def compute_qkv_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: bool = True,
    attention_mask: Tensor | None = None,
) -> QKVAttentionResult:
    """Reconstruct attention logits, probabilities, and context features.

    ``query`` and ``key`` follow the shapes documented by
    :func:`compute_qk_attention_logits`.  ``value`` has shape
    ``[..., key_tokens, value_channels]``.  Q/K/V must use the same dtype and
    device; value leading dimensions may broadcast to, but cannot expand, the
    QK attention batch/head shape.

    Fully masked query rows produce all-zero probabilities and context values
    rather than ``NaN``.

    Args:
        query: Query features shaped ``[..., query_tokens, channels]``.
        key: Key features shaped ``[..., key_tokens, channels]``.
        value: Value features shaped
            ``[..., key_tokens, value_channels]``.
        scale: Divide QK similarities by ``sqrt(channels)`` when ``True``.
        attention_mask: Boolean visibility mask or floating additive mask.

    Returns:
        A :class:`QKVAttentionResult` containing logits, probabilities, and
        the probability-weighted value context.
    """

    _validate_feature_tensor(query, "query")
    _validate_feature_tensor(key, "key")
    _validate_feature_tensor(value, "value")
    if value.shape[-2] != key.shape[-2]:
        raise ValueError(
            "key and value must contain the same number of tokens; got "
            f"{key.shape[-2]} and {value.shape[-2]}."
        )
    if value.device != query.device:
        raise ValueError(
            "query, key, and value must be on the same device; "
            f"got {query.device}, {key.device}, and {value.device}."
        )
    if value.dtype != query.dtype:
        raise TypeError(
            "query, key, and value must have the same dtype; "
            f"got {query.dtype}, {key.dtype}, and {value.dtype}."
        )

    logits = compute_qk_attention_logits(
        query,
        key,
        scale=scale,
        attention_mask=attention_mask,
    )

    attention_leading_shape = tuple(logits.shape[:-2])
    context_leading_shape = _broadcast_shapes(
        attention_leading_shape,
        value.shape[:-2],
        description="attention and value leading dimensions",
    )
    if context_leading_shape != attention_leading_shape:
        raise ValueError(
            "value leading dimensions may broadcast to the attention shape "
            "but must not expand it; got value shape "
            f"{tuple(value.shape[:-2])} and attention shape "
            f"{attention_leading_shape}."
        )

    probabilities = _safe_attention_softmax(logits)
    context = torch.matmul(probabilities, value)
    return QKVAttentionResult(
        logits=logits,
        probabilities=probabilities,
        context=context,
    )


def _validate_map_options(
    head_dim: int | None,
    head_fusion: str,
    feature_reduction: str,
    spatial_shape: Sequence[int] | None,
    prefix_tokens: int,
    normalize: bool,
    eps: float,
) -> tuple[int, int] | None:
    """Validate feature-map options and return a canonical spatial shape."""

    if head_dim is not None and (
        not isinstance(head_dim, Integral) or isinstance(head_dim, bool)
    ):
        raise TypeError("head_dim must be an integer axis index or None.")
    if not isinstance(head_fusion, str):
        raise TypeError("head_fusion must be a string.")
    if head_fusion not in {"mean", "max", "sum"}:
        raise ValueError(
            "head_fusion must be one of {'mean', 'max', 'sum'}, got "
            f"{head_fusion!r}."
        )
    if not isinstance(feature_reduction, str):
        raise TypeError("feature_reduction must be a string.")
    if feature_reduction not in {"l1", "l2", "mean", "max", "mean_abs"}:
        raise ValueError(
            "feature_reduction must be one of "
            "{'l1', 'l2', 'mean', 'max', 'mean_abs'}, got "
            f"{feature_reduction!r}."
        )
    if not isinstance(prefix_tokens, Integral) or isinstance(prefix_tokens, bool):
        raise TypeError("prefix_tokens must be a non-negative integer.")
    if prefix_tokens < 0:
        raise ValueError(f"prefix_tokens must be non-negative, got {prefix_tokens}.")
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a bool, got {type(normalize).__name__}.")
    if not isinstance(eps, Real) or isinstance(eps, bool):
        raise TypeError("eps must be a positive finite real number.")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError(f"eps must be a positive finite number, got {eps!r}.")

    if spatial_shape is None:
        return None
    if isinstance(spatial_shape, (str, bytes)) or not isinstance(
        spatial_shape, Sequence
    ):
        raise TypeError("spatial_shape must be a two-item sequence or None.")
    if len(spatial_shape) != 2:
        raise ValueError(
            "spatial_shape must contain exactly (height, width), got "
            f"{tuple(spatial_shape)}."
        )
    height, width = spatial_shape
    if any(
        not isinstance(size, Integral) or isinstance(size, bool)
        for size in (height, width)
    ):
        raise TypeError("spatial_shape height and width must be integers.")
    if height <= 0 or width <= 0:
        raise ValueError(
            "spatial_shape height and width must be positive, got "
            f"{(height, width)}."
        )
    return int(height), int(width)


def _canonical_head_axis(head_dim: int, ndim: int, name: str) -> int:
    """Normalize and validate an existing head-axis index."""

    axis = int(head_dim)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"head_dim={head_dim} is out of range for {name} with {ndim} dimensions."
        )
    if axis >= ndim - 2:
        raise ValueError(
            f"head_dim must identify a leading head axis in {name}, not its "
            f"token or feature axis; got axis {head_dim} for a shape with "
            f"{ndim} dimensions."
        )
    return axis


def _reduce_feature_channels(tensor: Tensor, reduction: str) -> Tensor:
    """Reduce the final feature-channel axis to a scalar token score."""

    if reduction == "l1":
        return torch.linalg.vector_norm(tensor, ord=1, dim=-1)
    if reduction == "l2":
        return torch.linalg.vector_norm(tensor, ord=2, dim=-1)
    if reduction == "mean":
        return tensor.mean(dim=-1)
    if reduction == "max":
        return tensor.amax(dim=-1)
    # Validated by _validate_map_options.
    return tensor.abs().mean(dim=-1)


def _build_feature_map(
    tensor: Tensor,
    name: str,
    *,
    head_dim: int | None,
    head_fusion: str,
    feature_reduction: str,
    spatial_shape: tuple[int, int] | None,
    prefix_tokens: int,
    normalize: bool,
    eps: float,
) -> Tensor:
    """Build one Q/K/V map after all arguments have been validated."""

    feature_map = _reduce_feature_channels(tensor, feature_reduction)

    if head_dim is not None:
        head_axis = _canonical_head_axis(head_dim, tensor.ndim, name)
        if head_fusion == "mean":
            feature_map = feature_map.mean(dim=head_axis)
        elif head_fusion == "max":
            feature_map = feature_map.amax(dim=head_axis)
        else:
            feature_map = feature_map.sum(dim=head_axis)

    token_count = feature_map.shape[-1]
    if prefix_tokens >= token_count:
        raise ValueError(
            f"prefix_tokens={prefix_tokens} must be smaller than the "
            f"{token_count} tokens in {name}."
        )
    if prefix_tokens:
        feature_map = feature_map[..., prefix_tokens:]

    spatial_dims = 1
    if spatial_shape is not None:
        height, width = spatial_shape
        expected_tokens = height * width
        actual_tokens = feature_map.shape[-1]
        if actual_tokens != expected_tokens:
            raise ValueError(
                f"{name} has {actual_tokens} tokens after removing "
                f"{prefix_tokens} prefix token(s), but spatial_shape "
                f"{spatial_shape} requires {expected_tokens}."
            )
        feature_map = feature_map.reshape(*feature_map.shape[:-1], height, width)
        spatial_dims = 2

    if normalize:
        reduction_dims = tuple(range(feature_map.ndim - spatial_dims, feature_map.ndim))
        original_dtype = feature_map.dtype
        working_map = (
            feature_map.float()
            if original_dtype in {torch.float16, torch.bfloat16}
            else feature_map
        )
        minimum = working_map.amin(dim=reduction_dims, keepdim=True)
        maximum = working_map.amax(dim=reduction_dims, keepdim=True)
        effective_eps = max(float(eps), torch.finfo(working_map.dtype).tiny)
        denominator = (maximum - minimum).clamp_min(effective_eps)
        feature_map = ((working_map - minimum) / denominator).to(original_dtype)

    return feature_map


def compute_token_feature_map(
    features: Tensor,
    *,
    head_dim: int | None = None,
    head_fusion: str = "mean",
    feature_reduction: str = "l2",
    spatial_shape: tuple[int, int] | None = None,
    prefix_tokens: int = 0,
    normalize: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Reduce token features to a scalar token map or spatial feature map.

    This single-tensor API is the preferred building block for cross-attention,
    where query and key/value streams often have different prefix tokens and
    spatial grids. It never guesses a head split or token layout.

    Args:
        features: Floating-point tensor shaped ``[..., tokens, channels]``.
        head_dim: Optional axis of an existing head dimension. It must be a
            leading axis, not the token or channel axis.
        head_fusion: Head reduction: ``"mean"``, ``"max"``, or ``"sum"``.
        feature_reduction: Channel reduction: ``"l1"``, ``"l2"``, ``"mean"``,
            ``"max"``, or ``"mean_abs"``.
        spatial_shape: Optional ``(height, width)`` for the remaining tokens.
        prefix_tokens: Number of leading non-spatial tokens to remove.
        normalize: Independently min-max normalize each remaining map.
        eps: Positive denominator floor for min-max normalization.

    Returns:
        A floating-point tensor on the input device. Its last axis is tokens
        when ``spatial_shape`` is omitted; otherwise its final axes are
        ``[height, width]``. The head axis is absent when ``head_dim`` is set.

    Raises:
        TypeError: If the tensor or option types are unsupported.
        ValueError: If values, axes, token counts, or spatial shape are invalid.
    """

    _validate_feature_tensor(features, "features")
    canonical_spatial_shape = _validate_map_options(
        head_dim,
        head_fusion,
        feature_reduction,
        spatial_shape,
        prefix_tokens,
        normalize,
        eps,
    )
    return _build_feature_map(
        features,
        "features",
        head_dim=int(head_dim) if head_dim is not None else None,
        head_fusion=head_fusion,
        feature_reduction=feature_reduction,
        spatial_shape=canonical_spatial_shape,
        prefix_tokens=int(prefix_tokens),
        normalize=normalize,
        eps=float(eps),
    )


def compute_qkv_feature_maps(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    head_dim: int | None = None,
    head_fusion: str = "mean",
    feature_reduction: str = "l2",
    spatial_shape: tuple[int, int] | None = None,
    prefix_tokens: int = 0,
    normalize: bool = True,
    eps: float = 1e-6,
) -> QKVFeatureMaps:
    """Convert Q/K/V feature tensors into comparable scalar maps.

    The last axis is first reduced to one score per token.  If ``head_dim`` is
    provided, it is interpreted as the axis index of an *existing* head
    dimension (for example, ``1`` for ``[batch, heads, tokens, channels]``),
    and the per-head scalar maps are then fused.  This function deliberately
    does not guess a number of heads or split a packed feature dimension.

    The same ``prefix_tokens`` and ``spatial_shape`` are applied to all three
    streams. For cross-attention layouts with different token grids, call
    :func:`compute_token_feature_map` separately for query, key, and value.

    Args:
        query: Query features shaped ``[..., query_tokens, channels]``.
        key: Key features shaped ``[..., key_tokens, channels]``.
        value: Value features shaped ``[..., value_tokens, channels]``.
        head_dim: Optional head-axis index.  It must refer to a leading axis,
            never the final token or feature axis.
        head_fusion: Head reduction: ``"mean"``, ``"max"``, or ``"sum"``.
        feature_reduction: Channel reduction: ``"l1"``, ``"l2"``,
            ``"mean"``, ``"max"``, or ``"mean_abs"``.
        spatial_shape: Optional ``(height, width)`` used to reshape the final
            token axis after prefix tokens are removed.
        prefix_tokens: Number of leading tokens (for example CLS/register
            tokens) to omit from every map.
        normalize: Independently min-max normalize every token/spatial map to
            ``[0, 1]``.  Batch dimensions and any unfused head dimension are
            normalized independently.
        eps: Positive denominator floor used by min-max normalization.

    Returns:
        A :class:`QKVFeatureMaps` containing processed query, key, and value
        maps.  Without ``spatial_shape`` each final axis is tokens; with it,
        the final two axes are height and width.

    Raises:
        TypeError: If inputs or option types are unsupported.
        ValueError: If axes, token counts, or spatial dimensions are invalid.
    """

    _validate_feature_tensor(query, "query")
    _validate_feature_tensor(key, "key")
    _validate_feature_tensor(value, "value")
    canonical_spatial_shape = _validate_map_options(
        head_dim,
        head_fusion,
        feature_reduction,
        spatial_shape,
        prefix_tokens,
        normalize,
        eps,
    )

    common_options = {
        "head_dim": int(head_dim) if head_dim is not None else None,
        "head_fusion": head_fusion,
        "feature_reduction": feature_reduction,
        "spatial_shape": canonical_spatial_shape,
        "prefix_tokens": int(prefix_tokens),
        "normalize": normalize,
        "eps": float(eps),
    }
    return QKVFeatureMaps(
        query=_build_feature_map(query, "query", **common_options),
        key=_build_feature_map(key, "key", **common_options),
        value=_build_feature_map(value, "value", **common_options),
    )
