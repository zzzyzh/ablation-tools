"""Model-agnostic Grad-CAM utilities for activations and attention maps.

The functions in this module operate on tensors that have already been produced
by a model.  They intentionally make no assumptions about a model architecture,
layer naming scheme, input modality, or visualization backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


DEFAULT_NORMALIZATION_EPS = 1e-8

OutputSelector = int | str | Callable[[Any], Tensor]


def _validate_eps(eps: float) -> float:
    if isinstance(eps, bool):
        raise TypeError("eps must be a positive finite number, not a boolean.")
    try:
        value = float(eps)
    except (TypeError, ValueError) as error:
        raise TypeError("eps must be a positive finite number.") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"eps must be positive and finite, but got {eps!r}.")
    return value


def _canonicalize_dim(dim: int, ndim: int, *, name: str) -> int:
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError(f"{name} must be an integer, but got {type(dim).__name__}.")
    if dim < -ndim or dim >= ndim:
        raise ValueError(
            f"{name}={dim} is out of range for a tensor with {ndim} dimensions."
        )
    return dim % ndim


def _canonicalize_spatial_dims(
    spatial_dims: int | Sequence[int] | None,
    *,
    ndim: int,
    channel_dim: int,
) -> tuple[int, ...]:
    if spatial_dims is None:
        dimensions = tuple(
            dim for dim in range(ndim) if dim not in (0, channel_dim)
        )
    else:
        raw_dimensions: Sequence[int]
        if isinstance(spatial_dims, int) and not isinstance(spatial_dims, bool):
            raw_dimensions = (spatial_dims,)
        elif isinstance(spatial_dims, Sequence) and not isinstance(
            spatial_dims, (str, bytes)
        ):
            raw_dimensions = spatial_dims
        else:
            raise TypeError("spatial_dims must be an integer, a sequence, or None.")

        dimensions = tuple(
            _canonicalize_dim(dim, ndim, name="spatial dimension")
            for dim in raw_dimensions
        )

    if not dimensions:
        raise ValueError("At least one spatial dimension is required for Grad-CAM.")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError(f"spatial_dims contains duplicate dimensions: {dimensions}.")
    if 0 in dimensions:
        raise ValueError("spatial_dims must not contain the batch dimension (dimension 0).")
    if channel_dim in dimensions:
        raise ValueError(
            "spatial_dims must not contain the channel dimension "
            f"(dimension {channel_dim})."
        )
    return dimensions


def _validate_feature_tensors(activations: Tensor, gradients: Tensor) -> None:
    if not isinstance(activations, Tensor):
        raise TypeError(
            "activations must be a torch.Tensor, "
            f"but got {type(activations).__name__}."
        )
    if not isinstance(gradients, Tensor):
        raise TypeError(
            "gradients must be a torch.Tensor, "
            f"but got {type(gradients).__name__}."
        )
    if activations.shape != gradients.shape:
        raise ValueError(
            "activations and gradients must have identical shapes, but got "
            f"{tuple(activations.shape)} and {tuple(gradients.shape)}."
        )
    if activations.ndim < 3:
        raise ValueError(
            "Grad-CAM expects at least batch, spatial, and channel dimensions; "
            f"got a {activations.ndim}-dimensional tensor."
        )
    if any(size == 0 for size in activations.shape):
        raise ValueError(
            "Grad-CAM does not support empty tensor dimensions; got shape "
            f"{tuple(activations.shape)}."
        )
    if not activations.is_floating_point():
        raise TypeError(
            f"activations must have a floating-point dtype, got {activations.dtype}."
        )
    if not gradients.is_floating_point():
        raise TypeError(
            f"gradients must have a floating-point dtype, got {gradients.dtype}."
        )
    if activations.device != gradients.device:
        raise ValueError(
            "activations and gradients must be on the same device, but got "
            f"{activations.device} and {gradients.device}."
        )
    if not torch.isfinite(activations).all():
        raise ValueError("activations must contain only finite values.")
    if not torch.isfinite(gradients).all():
        raise ValueError("gradients must contain only finite values.")


def normalize_heatmap(
    heatmap: Tensor,
    eps: float = DEFAULT_NORMALIZATION_EPS,
) -> Tensor:
    """Min-max normalize every heatmap in a batch independently.

    Dimension zero is interpreted as the batch dimension.  All remaining
    dimensions are flattened only for computing each sample's minimum and
    maximum; the returned tensor keeps the original shape.  Constant heatmaps
    become zero heatmaps.

    Args:
        heatmap: Floating-point tensor shaped ``[batch, ...]``.
        eps: Positive lower bound for the per-sample value range.

    Returns:
        A tensor with the same shape, device, and dtype as ``heatmap``.  Values
        are normally in ``[0, 1]``; maps whose range is smaller than ``eps``
        occupy a correspondingly smaller interval.
    """

    if not isinstance(heatmap, Tensor):
        raise TypeError(
            f"heatmap must be a torch.Tensor, got {type(heatmap).__name__}."
        )
    if heatmap.ndim < 2:
        raise ValueError(
            "heatmap must include a batch dimension and at least one map "
            f"dimension, but got shape {tuple(heatmap.shape)}."
        )
    if any(size == 0 for size in heatmap.shape):
        raise ValueError(
            f"heatmap must not contain empty dimensions, got {tuple(heatmap.shape)}."
        )
    if not heatmap.is_floating_point():
        raise TypeError(
            f"heatmap must have a floating-point dtype, got {heatmap.dtype}."
        )
    if not torch.isfinite(heatmap).all():
        raise ValueError("heatmap must contain only finite values.")

    eps_value = _validate_eps(eps)

    # Float32 reductions avoid an underflowing epsilon for float16/bfloat16
    # tensors, while the final cast preserves the public dtype contract.
    working_heatmap = (
        heatmap.float()
        if heatmap.dtype in (torch.float16, torch.bfloat16)
        else heatmap
    )
    flattened = working_heatmap.reshape(working_heatmap.shape[0], -1)
    sample_minimum = flattened.amin(dim=1)
    sample_range = flattened.amax(dim=1) - sample_minimum

    broadcast_shape = (working_heatmap.shape[0],) + (1,) * (
        working_heatmap.ndim - 1
    )
    sample_minimum = sample_minimum.reshape(broadcast_shape)
    sample_range = sample_range.reshape(broadcast_shape).clamp_min(eps_value)
    normalized = (working_heatmap - sample_minimum) / sample_range
    return normalized.to(dtype=heatmap.dtype)


def compute_gradcam(
    activations: Tensor,
    gradients: Tensor,
    *,
    channel_dim: int = 1,
    spatial_dims: int | Sequence[int] | None = None,
    relu: bool = True,
    normalize: bool = True,
    eps: float = DEFAULT_NORMALIZATION_EPS,
) -> Tensor:
    """Compute classical Grad-CAM from an activation tensor and its gradients.

    Gradients are globally averaged over ``spatial_dims`` to produce one weight
    per channel.  The weighted activations are then summed over ``channel_dim``.
    Dimension zero is always treated as the batch dimension.

    The defaults support image features shaped ``[B, C, H, W]``.  Token
    features shaped ``[B, T, C]`` are supported by setting ``channel_dim=-1``.

    Args:
        activations: Floating-point feature tensor.
        gradients: Gradient of the target score with respect to ``activations``;
            its shape must exactly match ``activations``.
        channel_dim: Channel/embedding dimension.  It cannot be the batch axis.
        spatial_dims: Dimensions over which to average gradients.  By default,
            every dimension except batch and channel is used.  Supplying a
            subset can preserve group-specific weights, for example per-frame
            Grad-CAM from ``[B, C, T, H, W]`` by using ``(H, W)`` axes only.
        relu: Clamp negative attribution values to zero when true.
        normalize: Independently min-max normalize each output sample.
        eps: Positive normalization stability value.

    Returns:
        A heatmap with ``channel_dim`` removed.  For example, image features
        produce ``[B, H, W]`` and token features produce ``[B, T]``.
    """

    _validate_feature_tensors(activations, gradients)
    eps_value = _validate_eps(eps)
    canonical_channel_dim = _canonicalize_dim(
        channel_dim, activations.ndim, name="channel_dim"
    )
    if canonical_channel_dim == 0:
        raise ValueError("channel_dim must not be the batch dimension (dimension 0).")
    canonical_spatial_dims = _canonicalize_spatial_dims(
        spatial_dims,
        ndim=activations.ndim,
        channel_dim=canonical_channel_dim,
    )
    if not isinstance(relu, bool):
        raise TypeError(f"relu must be a bool, got {type(relu).__name__}.")
    if not isinstance(normalize, bool):
        raise TypeError(
            f"normalize must be a bool, got {type(normalize).__name__}."
        )

    channel_weights = gradients.mean(dim=canonical_spatial_dims, keepdim=True)
    heatmap = (channel_weights * activations).sum(dim=canonical_channel_dim)

    if relu:
        heatmap = heatmap.relu()
    if normalize:
        heatmap = normalize_heatmap(heatmap, eps=eps_value)
    return heatmap


def compute_attention_gradcam(
    attention_weights: Tensor,
    gradients: Tensor,
    *,
    head_dim: int = 1,
    relu: bool = True,
    normalize: bool = True,
    eps: float = DEFAULT_NORMALIZATION_EPS,
) -> Tensor:
    """Compute classical Grad-CAM across an attention tensor's heads.

    For attention shaped ``[B, H, Q, K]``, the gradient is averaged over query
    and key positions to obtain one coefficient per head.  The weighted heads
    are summed to produce ``[B, Q, K]``.  Other head layouts are supported via
    ``head_dim`` as long as the tensor still has exactly four dimensions.

    Args:
        attention_weights: Floating-point attention tensor with four dimensions.
        gradients: Gradient of a target score with respect to the attention
            tensor; its shape must match ``attention_weights``.
        head_dim: Dimension containing attention heads (never the batch axis).
        relu: Clamp negative attribution values to zero when true.
        normalize: Independently min-max normalize each output sample.
        eps: Positive normalization stability value.

    Returns:
        Attention attribution shaped ``[B, Q, K]`` (with the original non-head
        dimension order preserved).
    """

    _validate_feature_tensors(attention_weights, gradients)
    if attention_weights.ndim != 4:
        raise ValueError(
            "attention_weights must have four dimensions (batch, heads, query, "
            f"key in any head-axis layout), got {tuple(attention_weights.shape)}."
        )

    canonical_head_dim = _canonicalize_dim(
        head_dim, attention_weights.ndim, name="head_dim"
    )
    if canonical_head_dim == 0:
        raise ValueError("head_dim must not be the batch dimension (dimension 0).")
    attention_spatial_dims = tuple(
        dim
        for dim in range(attention_weights.ndim)
        if dim not in (0, canonical_head_dim)
    )
    return compute_gradcam(
        attention_weights,
        gradients,
        channel_dim=canonical_head_dim,
        spatial_dims=attention_spatial_dims,
        relu=relu,
        normalize=normalize,
        eps=eps,
    )


class GradCAMCapture:
    """Capture a target module's activations and activation gradients.

    The forward hook is installed at construction time.  After a forward pass,
    backpropagate the scalar target of interest and call :meth:`compute`.
    Using the object as a context manager guarantees hook removal::

        with GradCAMCapture(model.target_layer) as capture:
            scores = model(inputs)
            scores[:, class_index].sum().backward()
            heatmap = capture.compute()

    Tensor outputs need no selector.  For tuple/list outputs, pass an integer;
    for mapping outputs, pass a string key; arbitrary structures can be handled
    by a callable such as ``lambda output: output.attention``.  When a module is
    invoked repeatedly, only its most recent activation is retained.

    Args:
        target_module: Module whose output activation should be captured.
        output_selector: Optional integer index, mapping key, or callable that
            extracts a tensor from the module output.
    """

    def __init__(
        self,
        target_module: nn.Module,
        output_selector: OutputSelector | None = None,
    ) -> None:
        if not isinstance(target_module, nn.Module):
            raise TypeError(
                "target_module must be a torch.nn.Module, "
                f"but got {type(target_module).__name__}."
            )
        if output_selector is not None and not (
            callable(output_selector)
            or isinstance(output_selector, str)
            or (
                isinstance(output_selector, int)
                and not isinstance(output_selector, bool)
            )
        ):
            raise TypeError(
                "output_selector must be an integer, string, callable, or None."
            )

        self.target_module = target_module
        self.output_selector = output_selector
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._activation_hook_handle: RemovableHandle | None = None
        self._forward_hook_handle: RemovableHandle | None = (
            target_module.register_forward_hook(self._capture_activation)
        )
        self._closed = False

    @property
    def activations(self) -> Tensor | None:
        """Most recently selected module output, or ``None`` before forward."""

        return self._activations

    @property
    def gradients(self) -> Tensor | None:
        """Most recently captured activation gradient, or ``None`` before backward."""

        return self._gradients

    @property
    def closed(self) -> bool:
        """Whether the module hook has been removed."""

        return self._closed

    def _select_output(self, output: Any) -> Tensor:
        selector = self.output_selector
        if selector is None:
            selected_output = output
            if not isinstance(selected_output, Tensor):
                raise TypeError(
                    "The target module returned a non-tensor output. Provide "
                    "output_selector to choose its activation tensor."
                )
        elif callable(selector):
            selected_output = selector(output)
        elif isinstance(selector, int):
            if not isinstance(output, (tuple, list)):
                raise TypeError(
                    "An integer output_selector requires a tuple or list module "
                    f"output, got {type(output).__name__}."
                )
            try:
                selected_output = output[selector]
            except IndexError as error:
                raise IndexError(
                    f"output_selector index {selector} is out of range for a "
                    f"module output of length {len(output)}."
                ) from error
        else:
            if not isinstance(output, Mapping):
                raise TypeError(
                    "A string output_selector requires a mapping module output, "
                    f"got {type(output).__name__}."
                )
            try:
                selected_output = output[selector]
            except KeyError as error:
                raise KeyError(
                    f"output_selector key {selector!r} is absent from the module "
                    "output."
                ) from error

        if not isinstance(selected_output, Tensor):
            raise TypeError(
                "output_selector must select a torch.Tensor, but selected "
                f"{type(selected_output).__name__}."
            )
        return selected_output

    def _capture_activation(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if self._closed:
            return

        self.clear()
        activation = self._select_output(output)
        self._activations = activation
        if activation.requires_grad:
            activation.retain_grad()
            self._activation_hook_handle = activation.register_hook(
                self._capture_gradient
            )

    def _capture_gradient(self, gradient: Tensor) -> None:
        # Detaching the stored gradient avoids retaining a higher-order graph.
        # Returning None leaves the model's backward computation unchanged.
        self._gradients = gradient.detach()

    def compute(
        self,
        *,
        channel_dim: int = 1,
        spatial_dims: int | Sequence[int] | None = None,
        relu: bool = True,
        normalize: bool = True,
        eps: float = DEFAULT_NORMALIZATION_EPS,
    ) -> Tensor:
        """Compute Grad-CAM from the most recently captured forward/backward pair.

        Raises:
            RuntimeError: If no forward activation or no corresponding gradient
                has been captured.  The latter usually means that backward has
                not run, gradient tracking was disabled, or the selected tensor
                did not contribute to the target score.
        """

        if self._activations is None:
            raise RuntimeError(
                "No activation has been captured. Run the target module after "
                "constructing GradCAMCapture."
            )
        if self._gradients is None:
            raise RuntimeError(
                "No activation gradient has been captured. Backpropagate a target "
                "that depends on the selected activation before calling compute()."
            )
        return compute_gradcam(
            self._activations,
            self._gradients,
            channel_dim=channel_dim,
            spatial_dims=spatial_dims,
            relu=relu,
            normalize=normalize,
            eps=eps,
        )

    def clear(self) -> None:
        """Release captured tensors while leaving the module hook installed."""

        if self._activation_hook_handle is not None:
            self._activation_hook_handle.remove()
            self._activation_hook_handle = None
        self._activations = None
        self._gradients = None

    def close(self) -> None:
        """Remove all hooks and release captured tensors; safe to call repeatedly."""

        if self._forward_hook_handle is not None:
            self._forward_hook_handle.remove()
            self._forward_hook_handle = None
        self.clear()
        self._closed = True

    def __enter__(self) -> GradCAMCapture:
        if self._closed:
            raise RuntimeError("A closed GradCAMCapture cannot be reused.")
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "DEFAULT_NORMALIZATION_EPS",
    "GradCAMCapture",
    "OutputSelector",
    "compute_attention_gradcam",
    "compute_gradcam",
    "normalize_heatmap",
]
