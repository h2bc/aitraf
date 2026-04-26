"""Regression prediction utilities."""

from typing import Sequence

import numpy as np


def compute_dummy_regression_preds(actual_values: Sequence[float]) -> np.ndarray:
    """Return a constant prediction equal to the mean of the labels."""

    values = np.asarray(actual_values, dtype=np.float32)
    return np.full(values.shape, values.mean())


__all__ = [
    "compute_dummy_regression_preds",
]
