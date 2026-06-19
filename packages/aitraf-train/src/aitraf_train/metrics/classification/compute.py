"""Classification prediction utilities."""

from collections import Counter
from typing import List

import numpy as np
import torch


def compute_pred_ids(logits: List[float]) -> List[int]:
    return np.argmax(logits, axis=-1)


def compute_pred_confidences(logits: List[float]) -> np.ndarray:
    """Return the max softmax probability for each row of logits."""
    tensor_logits = torch.as_tensor(logits)
    probs = torch.softmax(tensor_logits, dim=-1)
    return probs.max(dim=-1).values.numpy()


def compute_dummy_classification_pred_ids(labels: List[int]) -> List[int]:
    most_common = Counter(labels).most_common(1)[0][0]
    return np.full(len(labels), most_common)


__all__ = [
    "compute_dummy_classification_pred_ids",
    "compute_pred_confidences",
    "compute_pred_ids",
]
