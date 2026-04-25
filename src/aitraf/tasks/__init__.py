"""Task entrypoints grouped by modality."""

from . import (
    score_prediction,
    score_prediction_binary,
    score_prediction_pairwise,
    trick_classifier,
)

__all__ = [
    "score_prediction",
    "score_prediction_binary",
    "score_prediction_pairwise",
    "trick_classifier",
]
