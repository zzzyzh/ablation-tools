"""Task-independent metric implementations."""

from .Dice import (
    DiceResult,
    GeneralizedDiceResult,
    GeneralizedDiceWeight,
    MeanDiceResult,
    compute_dice,
    compute_generalized_dice,
    compute_mean_dice,
)
from .IoU import IoUResult, MeanIoUResult, compute_iou, compute_mean_iou

__all__ = [
    "DiceResult",
    "GeneralizedDiceResult",
    "GeneralizedDiceWeight",
    "IoUResult",
    "MeanDiceResult",
    "MeanIoUResult",
    "compute_dice",
    "compute_generalized_dice",
    "compute_iou",
    "compute_mean_dice",
    "compute_mean_iou",
]
