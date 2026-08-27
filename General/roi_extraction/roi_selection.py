"""Explicit, task-independent selection policies for ROI detections."""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor

from .bounding_boxes import ROIDetections, compute_border_touching_mask


def _validate_unit_interval(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    return result


@dataclass(frozen=True)
class ROISelection:
    """Selected detections and their indices in the original candidate list."""

    detections: ROIDetections
    source_indices: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.source_indices, Tensor):
            raise TypeError("source_indices must be a torch.Tensor")
        if self.source_indices.dtype != torch.int64 or self.source_indices.ndim != 1:
            raise TypeError("source_indices must be a one-dimensional int64 tensor")
        if self.source_indices.device != self.detections.boxes_xyxy.device:
            raise ValueError("source_indices and detections must be on the same device")
        if self.source_indices.shape[0] != self.detections.count:
            raise ValueError("source_indices must align with selected detections")


def select_roi_detections(
    detections: ROIDetections,
    *,
    score_threshold: float = 0.0,
    exclude_border: bool = False,
    border_margin: float = 0.0,
    minimum_area_fraction: float = 0.0,
    maximum_area_fraction: float = 1.0,
    allowed_labels: Collection[str] | None = None,
    max_detections: int | None = None,
) -> ROISelection:
    """Filter and stably rank ROI detections by descending score.

    Selection uses ``score >= score_threshold``. Equal scores preserve their
    source order. No NMS, fallback, or camera-specific policy is applied.
    """

    if not isinstance(detections, ROIDetections):
        raise TypeError("detections must be an ROIDetections instance")
    threshold = _validate_unit_interval(score_threshold, "score_threshold")
    minimum_area = _validate_unit_interval(minimum_area_fraction, "minimum_area_fraction")
    maximum_area = _validate_unit_interval(maximum_area_fraction, "maximum_area_fraction")
    if minimum_area > maximum_area:
        raise ValueError("minimum_area_fraction must not exceed maximum_area_fraction")
    if not isinstance(exclude_border, bool):
        raise TypeError("exclude_border must be a bool")
    if max_detections is not None and (
        not isinstance(max_detections, Integral)
        or isinstance(max_detections, bool)
        or max_detections <= 0
    ):
        raise ValueError("max_detections must be a positive integer or None")

    label_set: set[str] | None = None
    if allowed_labels is not None:
        if isinstance(allowed_labels, (str, bytes)):
            raise TypeError("allowed_labels must be a collection of strings, not one string")
        label_set = set(allowed_labels)
        if not all(isinstance(label, str) for label in label_set):
            raise TypeError("allowed_labels must contain only strings")

    keep = detections.scores >= threshold
    areas = detections.area_fractions
    keep &= (areas >= minimum_area) & (areas <= maximum_area)
    border_touching = compute_border_touching_mask(
        detections.boxes_xyxy,
        detections.image_size,
        border_margin=border_margin,
    )
    if exclude_border:
        keep &= ~border_touching
    if label_set is not None:
        label_keep = torch.tensor(
            [label in label_set for label in detections.labels],
            dtype=torch.bool,
            device=detections.boxes_xyxy.device,
        )
        keep &= label_keep

    source_indices = torch.where(keep)[0]
    if source_indices.numel():
        scores = detections.scores[source_indices]
        order = torch.argsort(scores, descending=True, stable=True)
        source_indices = source_indices[order]
    if max_detections is not None:
        source_indices = source_indices[: int(max_detections)]

    labels = tuple(detections.labels[index] for index in source_indices.cpu().tolist())
    selected = ROIDetections(
        boxes_xyxy=detections.boxes_xyxy[source_indices],
        scores=detections.scores[source_indices],
        labels=labels,
        image_size=detections.image_size,
    )
    return ROISelection(detections=selected, source_indices=source_indices)


__all__ = ["ROISelection", "select_roi_detections"]
