"""Classification prediction utilities."""

from collections import Counter
from collections.abc import Sequence
from typing import List

import numpy as np
import torch

from aitraf_core.inference.classification import (
    classification_confidences_from_logits,
    classification_ids_from_logits,
)


def compute_dummy_classification_pred_ids(labels: List[int]) -> List[int]:
    most_common = Counter(labels).most_common(1)[0][0]
    return np.full(len(labels), most_common)


def compute_pred_ids(logits: Sequence[Sequence[float]] | torch.Tensor) -> np.ndarray:
    return classification_ids_from_logits(logits)


def compute_pred_confidences(
    logits: Sequence[Sequence[float]] | torch.Tensor,
) -> np.ndarray:
    return classification_confidences_from_logits(logits)


__all__ = [
    "compute_dummy_classification_pred_ids",
    "compute_pred_confidences",
    "compute_pred_ids",
]
