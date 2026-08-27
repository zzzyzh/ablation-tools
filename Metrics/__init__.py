"""Task-independent metric implementations."""

from .ConfusionMatrix import (
    AverageMode,
    ConfusionMatrixCounts,
    PrecisionRecallF1Result,
    compute_confusion_matrix,
    compute_f1_score,
    compute_precision,
    compute_precision_recall_f1,
    compute_recall,
    extract_confusion_matrix_counts,
    validate_confusion_matrix,
)
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
    "AverageMode",
    "ConfusionMatrixCounts",
    "DiceResult",
    "GeneralizedDiceResult",
    "GeneralizedDiceWeight",
    "IoUResult",
    "MeanDiceResult",
    "MeanIoUResult",
    "PrecisionRecallF1Result",
    "compute_confusion_matrix",
    "compute_dice",
    "compute_f1_score",
    "compute_generalized_dice",
    "compute_iou",
    "compute_mean_dice",
    "compute_mean_iou",
    "compute_precision",
    "compute_precision_recall_f1",
    "compute_recall",
    "extract_confusion_matrix_counts",
    "validate_confusion_matrix",
]
