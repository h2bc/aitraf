"""Task-local helpers for pairwise score prediction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_pairwise_pred_labels(logits: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert pairwise logits into binary labels using zero threshold."""

    return (np.asarray(logits).reshape(-1) >= 0).astype(int)


__all__ = [
    "compute_pairwise_pred_labels",
]
