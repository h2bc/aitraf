"""Classification prediction utilities."""

from collections import Counter
from typing import List

import numpy as np

from aitraf_core.inference import (
    compute_pred_confidences,
    compute_pred_ids,
)


def compute_dummy_classification_pred_ids(labels: List[int]) -> List[int]:
    most_common = Counter(labels).most_common(1)[0][0]
    return np.full(len(labels), most_common)


__all__ = [
    "compute_dummy_classification_pred_ids",
    "compute_pred_confidences",
    "compute_pred_ids",
]
