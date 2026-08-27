"""Hugging Face Grounding DINO adapter for extracting ROI bounding boxes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from .bounding_boxes import (
    ImageSize,
    ROIDetections,
    clip_bounding_boxes_xyxy,
    compute_valid_bounding_box_mask,
    validate_image_size,
)


def normalize_grounding_dino_prompt(prompt: str) -> str:
    """Normalize a Grounding DINO prompt to lowercase dot-terminated text.

    Grounding DINO treats dots as category separators. This helper is explicit
    so experiments can record whether prompt normalization was applied.
    """

    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt)!r}")
    normalized = " ".join(prompt.strip().lower().split())
    if not normalized:
        raise ValueError("prompt must not be empty")
    if not normalized.endswith("."):
        normalized += "."
    return normalized


def _validate_threshold(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    return result


def _prepare_image(image: Any) -> tuple[Any, ImageSize]:
    if isinstance(image, Tensor):
        if image.device.type != "cpu":
            raise ValueError("tensor images must be on CPU before processor input")
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("tensor images must have uint8 RGB shape [3, height, width]")
        return image, validate_image_size((int(image.shape[1]), int(image.shape[2])))

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - dependency-specific
        raise ImportError(
            "PIL image input requires Pillow; install ablation-tools[grounding-dino]"
        ) from error
    if not isinstance(image, Image.Image):
        raise TypeError("images must be PIL images or CPU uint8 RGB tensors")
    converted = image.convert("RGB")
    return converted, validate_image_size((converted.height, converted.width))


def _normalize_prompt_batch(
    prompts: str | Sequence[str],
    batch_size: int,
    *,
    normalize_prompts: bool,
) -> list[str]:
    if isinstance(prompts, str):
        prompt_list = [prompts] * batch_size
    elif isinstance(prompts, Sequence) and not isinstance(prompts, (str, bytes)):
        prompt_list = list(prompts)
        if len(prompt_list) != batch_size:
            raise ValueError(
                f"prompts must contain one string per image, got {len(prompt_list)} "
                f"for {batch_size} images"
            )
    else:
        raise TypeError("prompts must be a string or a sequence of strings")
    if not all(isinstance(prompt, str) and prompt.strip() for prompt in prompt_list):
        raise ValueError("every prompt must be a non-empty string")
    if normalize_prompts:
        return [normalize_grounding_dino_prompt(prompt) for prompt in prompt_list]
    if not all("." in prompt for prompt in prompt_list):
        raise ValueError(
            "unmodified batched Grounding DINO prompts must contain dot separators; "
            "otherwise Transformers interprets the list as labels for one image"
        )
    return prompt_list


def _move_processor_inputs(inputs: Any, device: torch.device) -> Mapping[str, Any]:
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    elif isinstance(inputs, Mapping):
        inputs = {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in inputs.items()
        }
    if not isinstance(inputs, Mapping):
        raise TypeError("processor output must be a mapping of model inputs")
    if "input_ids" not in inputs:
        raise KeyError("processor output must contain input_ids")
    return inputs


def _as_float_tensor(value: Any, *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    if tensor.is_complex():
        raise TypeError(f"{name} must be real-valued")
    if not tensor.is_floating_point():
        tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _parse_post_processed_result(
    result: Mapping[str, Any],
    image_size: ImageSize,
) -> ROIDetections:
    if not isinstance(result, Mapping):
        raise TypeError("each post-processed result must be a mapping")
    if "boxes" not in result or "scores" not in result:
        raise KeyError("post-processed results must contain boxes and scores")

    boxes = _as_float_tensor(result["boxes"], name="boxes")
    scores = _as_float_tensor(result["scores"], name="scores")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"post-processed boxes must have shape [N, 4], got {tuple(boxes.shape)}")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError("post-processed scores must have shape [N] aligned with boxes")

    labels_value = result.get("text_labels")
    if labels_value is None:
        labels_value = result.get("labels")
    if labels_value is None:
        raise KeyError("post-processed results must contain text_labels or labels")
    if isinstance(labels_value, (str, bytes)):
        raise TypeError("post-processed labels must be a sequence, not one string")
    if isinstance(labels_value, Tensor):
        labels_value = labels_value.detach().cpu().tolist()
    labels = tuple(str(label) for label in labels_value)
    if len(labels) != boxes.shape[0]:
        raise ValueError("post-processed labels must align with boxes and scores")

    boxes = clip_bounding_boxes_xyxy(boxes, image_size)
    valid = compute_valid_bounding_box_mask(boxes)
    boxes = boxes[valid]
    scores = scores[valid]
    valid_indices = torch.where(valid)[0].cpu().tolist()
    labels = tuple(labels[index] for index in valid_indices)
    if scores.numel():
        order = torch.argsort(scores, descending=True, stable=True)
        boxes = boxes[order]
        scores = scores[order]
        labels = tuple(labels[index] for index in order.cpu().tolist())
    return ROIDetections(
        boxes_xyxy=boxes,
        scores=scores,
        labels=labels,
        image_size=image_size,
    )


@runtime_checkable
class GroundedBoxExtractor(Protocol):
    """Protocol for text-grounded ROI box extractors."""

    def extract_roi_boxes(
        self,
        images: Sequence[Any],
        prompts: str | Sequence[str],
        **kwargs: Any,
    ) -> tuple[ROIDetections, ...]:
        """Extract one variable-length ROI detection set per image."""


class GroundingDINOBoxExtractor:
    """Extract Grounding DINO ROI candidates through an injected HF backend.

    Construct directly with compatible ``processor`` and ``model`` objects for
    testing or custom loading. :meth:`from_pretrained` supplies the standard
    Transformers 4.57 adapter without importing Transformers at module import.
    """

    def __init__(self, processor: Any, model: Any, *, device: str | torch.device) -> None:
        if not callable(processor):
            raise TypeError("processor must be callable")
        if not callable(model):
            raise TypeError("model must be callable")
        self.processor = processor
        self.device = torch.device(device)
        if hasattr(model, "to"):
            moved_model = model.to(self.device)
            if moved_model is not None:
                model = moved_model
        if hasattr(model, "eval"):
            evaluated_model = model.eval()
            if evaluated_model is not None:
                model = evaluated_model
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        device: str | torch.device,
        local_files_only: bool = False,
        torch_dtype: torch.dtype | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> GroundingDINOBoxExtractor:
        """Load a Hugging Face Grounding DINO processor and model."""

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:  # pragma: no cover - dependency-specific
            raise ImportError(
                "GroundingDINOBoxExtractor.from_pretrained requires Transformers; "
                "install ablation-tools[grounding-dino]"
            ) from error

        identifier = str(model_name_or_path)
        if not identifier:
            raise ValueError("model_name_or_path must not be empty")
        processor_options = dict(processor_kwargs or {})
        model_options = dict(model_kwargs or {})
        processor_options.setdefault("local_files_only", local_files_only)
        model_options.setdefault("local_files_only", local_files_only)
        if torch_dtype is not None:
            model_options.setdefault("torch_dtype", torch_dtype)
        processor = AutoProcessor.from_pretrained(identifier, **processor_options)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            identifier,
            **model_options,
        )
        return cls(processor, model, device=device)

    def extract_roi_boxes(
        self,
        images: Sequence[Any],
        prompts: str | Sequence[str],
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        normalize_prompts: bool = True,
    ) -> tuple[ROIDetections, ...]:
        """Extract all valid grounded boxes for a batch of images.

        Args:
            images: Non-empty sequence of PIL images or CPU ``uint8`` RGB
                tensors shaped ``[3, H, W]``.
            prompts: One prompt broadcast to every image or one prompt per
                image. Normalized prompts are lowercase and dot-terminated.
            box_threshold: Candidate score threshold passed as ``threshold`` to
                Transformers 4.57 Grounding DINO post-processing.
            text_threshold: Token-to-phrase decoding threshold.
            normalize_prompts: Apply :func:`normalize_grounding_dino_prompt`.

        Returns:
            Tuple aligned with ``images``. Each element contains score-sorted,
            clipped, positive-area absolute-pixel XYXY boxes.
        """

        if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
            raise TypeError("images must be a non-empty sequence")
        image_list = list(images)
        if not image_list:
            raise ValueError("images must not be empty")
        if not isinstance(normalize_prompts, bool):
            raise TypeError("normalize_prompts must be a bool")
        candidate_threshold = _validate_threshold(box_threshold, "box_threshold")
        phrase_threshold = _validate_threshold(text_threshold, "text_threshold")

        prepared_images: list[Any] = []
        image_sizes: list[ImageSize] = []
        for image in image_list:
            prepared, image_size = _prepare_image(image)
            prepared_images.append(prepared)
            image_sizes.append(image_size)
        prompt_list = _normalize_prompt_batch(
            prompts,
            len(prepared_images),
            normalize_prompts=normalize_prompts,
        )

        processor_inputs = self.processor(
            images=prepared_images,
            text=prompt_list,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = _move_processor_inputs(processor_inputs, self.device)
        with torch.inference_mode():
            model_outputs = self.model(**model_inputs)
        results = self.processor.post_process_grounded_object_detection(
            model_outputs,
            model_inputs["input_ids"],
            threshold=candidate_threshold,
            text_threshold=phrase_threshold,
            target_sizes=image_sizes,
        )
        if not isinstance(results, Sequence) or len(results) != len(prepared_images):
            result_count = len(results) if isinstance(results, Sequence) else None
            raise ValueError(
                "post-processing must return one result per image, got "
                f"{result_count} for {len(prepared_images)} images"
            )
        return tuple(
            _parse_post_processed_result(result, image_size)
            for result, image_size in zip(results, image_sizes)
        )


__all__ = [
    "GroundedBoxExtractor",
    "GroundingDINOBoxExtractor",
    "normalize_grounding_dino_prompt",
]
