"""Region-of-interest bounding-box extraction and explicit selection."""

from .bounding_boxes import (
    ImageSize,
    ROIDetections,
    clip_bounding_boxes_xyxy,
    compute_border_touching_mask,
    compute_bounding_box_areas,
    compute_valid_bounding_box_mask,
    round_bounding_boxes_xyxy_outward,
    validate_image_size,
)
from .grounding_dino import (
    GroundedBoxExtractor,
    GroundingDINOBoxExtractor,
    normalize_grounding_dino_prompt,
)
from .roi_selection import ROISelection, select_roi_detections

__all__ = [
    "GroundedBoxExtractor",
    "GroundingDINOBoxExtractor",
    "ImageSize",
    "ROIDetections",
    "ROISelection",
    "clip_bounding_boxes_xyxy",
    "compute_border_touching_mask",
    "compute_bounding_box_areas",
    "compute_valid_bounding_box_mask",
    "normalize_grounding_dino_prompt",
    "round_bounding_boxes_xyxy_outward",
    "select_roi_detections",
    "validate_image_size",
]
