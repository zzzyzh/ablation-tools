"""Tensor contracts and geometry helpers for region-of-interest boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor


ImageSize = tuple[int, int]


def validate_image_size(image_size: ImageSize) -> ImageSize:
    """Validate and return an ``(height, width)`` image size."""

    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise TypeError("image_size must be a (height, width) tuple")
    height, width = image_size
    if any(not isinstance(size, Integral) or isinstance(size, bool) for size in image_size):
        raise TypeError("image_size height and width must be integers")
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size values must be positive, got {image_size}")
    return int(height), int(width)


def _validate_boxes_xyxy(boxes_xyxy: Tensor) -> None:
    if not isinstance(boxes_xyxy, Tensor):
        raise TypeError(f"boxes_xyxy must be a torch.Tensor, got {type(boxes_xyxy)!r}")
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[-1] != 4:
        raise ValueError(
            f"boxes_xyxy must have shape [num_boxes, 4], got {tuple(boxes_xyxy.shape)}"
        )
    if not boxes_xyxy.is_floating_point():
        raise TypeError(f"boxes_xyxy must be floating point, got {boxes_xyxy.dtype}")
    if not torch.isfinite(boxes_xyxy).all():
        raise ValueError("boxes_xyxy must contain only finite values")


def clip_bounding_boxes_xyxy(boxes_xyxy: Tensor, image_size: ImageSize) -> Tensor:
    """Clip absolute-pixel XYXY boxes to image boundaries.

    ``boxes_xyxy`` uses continuous ``[left, top, right, bottom]`` coordinates.
    Right and bottom are half-open edges and may equal image width and height.
    Shape and candidate count are preserved; use
    :func:`compute_valid_bounding_box_mask` to remove degenerate boxes.
    """

    _validate_boxes_xyxy(boxes_xyxy)
    height, width = validate_image_size(image_size)
    lower = boxes_xyxy.new_zeros(4)
    upper = boxes_xyxy.new_tensor((width, height, width, height))
    return torch.minimum(torch.maximum(boxes_xyxy, lower), upper)


def compute_valid_bounding_box_mask(boxes_xyxy: Tensor) -> Tensor:
    """Return a boolean mask for finite, positive-area XYXY boxes."""

    _validate_boxes_xyxy(boxes_xyxy)
    return (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (
        boxes_xyxy[:, 3] > boxes_xyxy[:, 1]
    )


def compute_bounding_box_areas(boxes_xyxy: Tensor) -> Tensor:
    """Compute continuous pixel areas for valid absolute-pixel XYXY boxes."""

    _validate_boxes_xyxy(boxes_xyxy)
    widths = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp_min(0)
    heights = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp_min(0)
    return widths * heights


def round_bounding_boxes_xyxy_outward(
    boxes_xyxy: Tensor,
    image_size: ImageSize,
) -> Tensor:
    """Convert continuous boxes to outward-rounded half-open integer boxes.

    Left/top use ``floor`` and right/bottom use ``ceil`` after clipping. The
    returned ``int64`` tensor can be used directly as ``image[y0:y1, x0:x1]``.
    """

    clipped = clip_bounding_boxes_xyxy(boxes_xyxy, image_size)
    minimum = torch.floor(clipped[:, :2])
    maximum = torch.ceil(clipped[:, 2:])
    return torch.cat((minimum, maximum), dim=-1).to(torch.int64)


def compute_border_touching_mask(
    boxes_xyxy: Tensor,
    image_size: ImageSize,
    *,
    border_margin: float = 0.0,
) -> Tensor:
    """Return boxes that touch an image border within ``border_margin`` pixels."""

    _validate_boxes_xyxy(boxes_xyxy)
    height, width = validate_image_size(image_size)
    if not isinstance(border_margin, Real) or isinstance(border_margin, bool):
        raise TypeError("border_margin must be a non-negative finite number")
    margin = float(border_margin)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"border_margin must be non-negative and finite, got {border_margin}")
    return (
        (boxes_xyxy[:, 0] <= margin)
        | (boxes_xyxy[:, 1] <= margin)
        | (boxes_xyxy[:, 2] >= width - margin)
        | (boxes_xyxy[:, 3] >= height - margin)
    )


@dataclass(frozen=True)
class ROIDetections:
    """Detections for one image in canonical absolute-pixel XYXY format.

    Attributes:
        boxes_xyxy: Finite floating tensor shaped ``[N, 4]``. Coordinates are
            clipped to the image and every box has positive area.
        scores: Floating confidence tensor shaped ``[N]`` with values in
            ``[0, 1]`` on the same device as ``boxes_xyxy``.
        labels: Tuple of ``N`` text labels aligned with boxes and scores.
        image_size: Original ``(height, width)``.
    """

    boxes_xyxy: Tensor
    scores: Tensor
    labels: tuple[str, ...]
    image_size: ImageSize

    def __post_init__(self) -> None:
        _validate_boxes_xyxy(self.boxes_xyxy)
        image_size = validate_image_size(self.image_size)
        object.__setattr__(self, "image_size", image_size)
        if not isinstance(self.scores, Tensor):
            raise TypeError(f"scores must be a torch.Tensor, got {type(self.scores)!r}")
        if self.scores.ndim != 1 or self.scores.shape[0] != self.boxes_xyxy.shape[0]:
            raise ValueError(
                "scores must have shape [num_boxes] aligned with boxes, got "
                f"{tuple(self.scores.shape)} and {tuple(self.boxes_xyxy.shape)}"
            )
        if not self.scores.is_floating_point():
            raise TypeError(f"scores must be floating point, got {self.scores.dtype}")
        if self.scores.device != self.boxes_xyxy.device:
            raise ValueError("scores and boxes_xyxy must be on the same device")
        if not torch.isfinite(self.scores).all():
            raise ValueError("scores must contain only finite values")
        if ((self.scores < 0) | (self.scores > 1)).any():
            raise ValueError("scores must be in [0, 1]")
        labels = tuple(self.labels)
        if isinstance(self.labels, (str, bytes)):
            raise TypeError("labels must be a sequence of strings, not one string")
        if len(labels) != self.boxes_xyxy.shape[0] or not all(
            isinstance(label, str) for label in labels
        ):
            raise ValueError("labels must contain one string per detection")
        object.__setattr__(self, "labels", labels)

        height, width = image_size
        boxes = self.boxes_xyxy
        if boxes.numel() and (
            (boxes[:, 0] < 0).any()
            or (boxes[:, 1] < 0).any()
            or (boxes[:, 2] > width).any()
            or (boxes[:, 3] > height).any()
        ):
            raise ValueError("boxes_xyxy must be clipped to image_size")
        if not compute_valid_bounding_box_mask(boxes).all():
            raise ValueError("boxes_xyxy must contain only positive-area boxes")

    @property
    def count(self) -> int:
        """Number of detections."""

        return self.boxes_xyxy.shape[0]

    @property
    def is_empty(self) -> bool:
        """Whether the image has no valid detections."""

        return self.count == 0

    @property
    def areas(self) -> Tensor:
        """Continuous pixel area for every detection."""

        return compute_bounding_box_areas(self.boxes_xyxy)

    @property
    def area_fractions(self) -> Tensor:
        """Detection areas divided by full image area."""

        height, width = self.image_size
        return self.areas / (height * width)


__all__ = [
    "ImageSize",
    "ROIDetections",
    "clip_bounding_boxes_xyxy",
    "compute_border_touching_mask",
    "compute_bounding_box_areas",
    "compute_valid_bounding_box_mask",
    "round_bounding_boxes_xyxy_outward",
    "validate_image_size",
]
