"""Rendering helpers for attention and attribution heatmaps.

The functions in this module deliberately return PyTorch tensors instead of
figures.  This keeps them usable in notebooks, evaluation workers, and logging
pipelines without requiring a plotting backend.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


ColormapName = Literal["turbo", "jet", "grayscale"]


def _validate_heatmap(heatmap: Tensor) -> None:
    if not isinstance(heatmap, Tensor):
        raise TypeError(f"heatmap must be a torch.Tensor, got {type(heatmap)!r}")
    if heatmap.ndim not in (2, 3):
        raise ValueError(
            "heatmap must have shape [height, width] or "
            f"[batch, height, width], got {tuple(heatmap.shape)}"
        )
    if heatmap.numel() == 0:
        raise ValueError("heatmap must not be empty")
    if heatmap.is_complex():
        raise TypeError("heatmap must contain real-valued scalars")
    if not torch.isfinite(heatmap).all():
        raise ValueError("heatmap must contain only finite values")


def _validate_eps(eps: float) -> float:
    if not isinstance(eps, Real) or isinstance(eps, bool):
        raise TypeError("eps must be a positive finite real number")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be positive and finite, got {eps!r}")
    return eps_value


def _normalize_heatmap(heatmap: Tensor, eps: float) -> Tensor:
    eps = _validate_eps(eps)
    heatmap = heatmap.float()
    flat = heatmap.reshape(-1, heatmap.shape[-2] * heatmap.shape[-1])
    minimum = flat.amin(dim=-1, keepdim=True)
    maximum = flat.amax(dim=-1, keepdim=True)
    scale = maximum - minimum
    normalized = torch.where(
        scale > eps,
        (flat - minimum) / scale.clamp_min(eps),
        torch.zeros_like(flat),
    )
    return normalized.reshape_as(heatmap)


def resize_heatmap(
    heatmap: Tensor,
    output_size: Sequence[int],
    *,
    mode: Literal["nearest", "bilinear", "bicubic"] = "bilinear",
) -> Tensor:
    """Resize one heatmap or a batch of heatmaps.

    Args:
        heatmap: Tensor shaped ``[H, W]`` or ``[B, H, W]``.
        output_size: Target ``(height, width)``.
        mode: Interpolation algorithm used by :func:`torch.nn.functional.interpolate`.

    Returns:
        A floating-point tensor with the same batch convention as ``heatmap``.
    """

    _validate_heatmap(heatmap)
    if isinstance(output_size, (str, bytes)) or not isinstance(output_size, Sequence):
        raise TypeError("output_size must be a two-item sequence of positive integers")
    if len(output_size) != 2:
        raise ValueError(f"output_size must contain two integers, got {output_size}")
    if any(not isinstance(size, Integral) or isinstance(size, bool) for size in output_size):
        raise TypeError(f"output_size values must be integers, got {output_size}")
    if any(size <= 0 for size in output_size):
        raise ValueError(f"output_size values must be positive, got {output_size}")
    if mode not in ("nearest", "bilinear", "bicubic"):
        raise ValueError(
            f"unsupported interpolation mode {mode!r}; expected nearest, bilinear, or bicubic"
        )

    unbatched = heatmap.ndim == 2
    batched = heatmap.float().unsqueeze(0) if unbatched else heatmap.float()
    interpolation_input = batched.unsqueeze(1)
    align_corners = False if mode in ("bilinear", "bicubic") else None
    resized = F.interpolate(
        interpolation_input,
        size=(int(output_size[0]), int(output_size[1])),
        mode=mode,
        align_corners=align_corners,
    ).squeeze(1)
    return resized.squeeze(0) if unbatched else resized


def _apply_colormap(values: Tensor, colormap: ColormapName) -> Tensor:
    """Map ``[B, H, W]`` values in ``[0, 1]`` to ``[B, 3, H, W]``."""

    if colormap == "grayscale":
        return values.unsqueeze(1).expand(-1, 3, -1, -1)

    if colormap == "jet":
        red = (1.5 - (4.0 * values - 3.0).abs()).clamp(0.0, 1.0)
        green = (1.5 - (4.0 * values - 2.0).abs()).clamp(0.0, 1.0)
        blue = (1.5 - (4.0 * values - 1.0).abs()).clamp(0.0, 1.0)
        return torch.stack((red, green, blue), dim=1)

    if colormap == "turbo":
        # Polynomial approximation published with Google's Turbo colormap.
        coefficients = values.new_tensor(
            [
                [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
                [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
                [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
            ]
        )
        powers = torch.stack([values.pow(power) for power in range(6)], dim=1)
        return torch.einsum("bcij,rc->brij", powers, coefficients).clamp(0.0, 1.0)

    raise ValueError(
        f"unsupported colormap {colormap!r}; expected 'turbo', 'jet', or 'grayscale'"
    )


def render_heatmap(
    heatmap: Tensor,
    *,
    colormap: ColormapName = "turbo",
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Convert scalar heatmaps into RGB tensors.

    Args:
        heatmap: Tensor shaped ``[H, W]`` or ``[B, H, W]``.
        colormap: Color mapping used for rendering.
        normalize: Independently min-max normalize each heatmap before rendering.
        eps: Numerical guard for constant heatmaps.

    Returns:
        ``[3, H, W]`` for an unbatched input or ``[B, 3, H, W]`` for a
        batched input. Values are floating point and clipped to ``[0, 1]``.
    """

    _validate_heatmap(heatmap)
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a bool, got {type(normalize).__name__}")
    unbatched = heatmap.ndim == 2
    batched = heatmap.unsqueeze(0) if unbatched else heatmap
    values = _normalize_heatmap(batched, eps) if normalize else batched.float().clamp(0.0, 1.0)
    rendered = _apply_colormap(values, colormap)
    return rendered.squeeze(0) if unbatched else rendered


def _prepare_image(image: Tensor) -> tuple[Tensor, bool]:
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image)!r}")
    if image.ndim not in (2, 3, 4):
        raise ValueError(
            "image must have shape [H, W], [C, H, W], or [B, C, H, W], "
            f"got {tuple(image.shape)}"
        )
    if image.numel() == 0:
        raise ValueError("image must not be empty")
    if image.is_complex():
        raise TypeError("image must contain real-valued pixels")
    if not torch.isfinite(image).all():
        raise ValueError("image must contain only finite values")

    unbatched = image.ndim < 4
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)

    if image.shape[1] not in (1, 3):
        raise ValueError(f"image channel dimension must be 1 or 3, got {image.shape[1]}")

    if image.dtype == torch.bool:
        prepared = image.float()
    elif image.is_floating_point():
        prepared = image.float()
    else:
        maximum = float(torch.iinfo(image.dtype).max)
        prepared = image.float() / maximum
    prepared = prepared.clamp(0.0, 1.0)
    if prepared.shape[1] == 1:
        prepared = prepared.expand(-1, 3, -1, -1)
    return prepared, unbatched


def overlay_heatmap(
    image: Tensor,
    heatmap: Tensor,
    *,
    alpha: float = 0.5,
    colormap: ColormapName = "turbo",
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Blend heatmaps over channel-first images.

    Heatmap magnitude controls local opacity, while ``alpha`` sets its maximum.
    A zero-valued region leaves the source image unchanged.

    Args:
        image: Real tensor shaped ``[H, W]``, ``[C, H, W]``, or
            ``[B, C, H, W]``, with one or three channels. Integer inputs are
            scaled by the dtype maximum; floating inputs are clipped to
            ``[0, 1]``.
        heatmap: Finite tensor shaped ``[H, W]`` or ``[B, H, W]``. It is
            resized to the image resolution.
        alpha: Maximum heatmap opacity in ``[0, 1]``.
        colormap: ``"turbo"``, ``"jet"``, or ``"grayscale"``.
        normalize: Independently min-max normalize each heatmap before blending.
        eps: Positive finite denominator guard for normalization.

    Returns:
        Channel-first RGB float values in ``[0, 1]``. The shape is
        ``[3, H, W]`` only when both inputs are unbatched; otherwise it is
        ``[B, 3, H, W]``. Singleton batches broadcast to the other input.

    Raises:
        TypeError: If inputs or option types are unsupported.
        ValueError: If shapes, devices, ranges, or finite-value checks fail.
    """

    if not isinstance(alpha, Real) or isinstance(alpha, bool):
        raise TypeError("alpha must be a real number in [0, 1]")
    if not math.isfinite(float(alpha)) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be finite and in [0, 1], got {alpha}")
    if not isinstance(normalize, bool):
        raise TypeError(f"normalize must be a bool, got {type(normalize).__name__}")
    _validate_heatmap(heatmap)
    prepared_image, image_was_unbatched = _prepare_image(image)
    if prepared_image.device != heatmap.device:
        raise ValueError(
            "image and heatmap must be on the same device, got "
            f"{prepared_image.device} and {heatmap.device}"
        )
    heatmap_was_unbatched = heatmap.ndim == 2
    batched_heatmap = heatmap.unsqueeze(0) if heatmap_was_unbatched else heatmap
    batched_heatmap = resize_heatmap(batched_heatmap, prepared_image.shape[-2:])
    values = (
        _normalize_heatmap(batched_heatmap, eps)
        if normalize
        else batched_heatmap.float().clamp(0.0, 1.0)
    )

    image_batch = prepared_image.shape[0]
    heatmap_batch = values.shape[0]
    if image_batch != heatmap_batch:
        if image_batch == 1:
            prepared_image = prepared_image.expand(heatmap_batch, -1, -1, -1)
        elif heatmap_batch == 1:
            values = values.expand(image_batch, -1, -1)
        else:
            raise ValueError(
                "image and heatmap batch sizes must match or one must be 1, "
                f"got {image_batch} and {heatmap_batch}"
            )

    colored = _apply_colormap(values, colormap)
    opacity = values.unsqueeze(1) * alpha
    overlaid = prepared_image * (1.0 - opacity) + colored * opacity
    if image_was_unbatched and heatmap_was_unbatched:
        return overlaid.squeeze(0)
    return overlaid


def save_rgb_image(image: Tensor, output_path: str | Path) -> Path:
    """Save a single channel-first RGB tensor using Pillow only.

    Args:
        image: Finite tensor shaped ``[3, H, W]``. Values are converted to
            float, clipped to ``[0, 1]``, and quantized to eight-bit RGB.
        output_path: Destination filename. Missing parent directories are made.

    Returns:
        The destination as a :class:`pathlib.Path`.

    Raises:
        ImportError: If the optional Pillow dependency is unavailable.
        TypeError: If pixels are complex-valued.
        ValueError: If shape, emptiness, or finite-value validation fails.
    """

    if not isinstance(image, Tensor) or image.ndim != 3 or image.shape[0] != 3:
        shape = tuple(image.shape) if isinstance(image, Tensor) else None
        raise ValueError(f"image must have shape [3, H, W], got {shape}")
    if image.numel() == 0:
        raise ValueError("image must not be empty")
    if image.is_complex():
        raise TypeError("image must contain real-valued pixels")
    if not torch.isfinite(image).all():
        raise ValueError("image must contain only finite values")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "save_rgb_image requires Pillow; install ablation-tools[visualization]"
        ) from error

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = (
        image.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
    )
    height, width = pixels.shape[:2]
    pixel_bytes = bytes(pixels.flatten().tolist())
    Image.frombytes("RGB", (width, height), pixel_bytes).save(destination)
    return destination


__all__ = [
    "ColormapName",
    "overlay_heatmap",
    "render_heatmap",
    "resize_heatmap",
    "save_rgb_image",
]
