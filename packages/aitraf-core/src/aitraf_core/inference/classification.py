"""Classification inference utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def compute_pred_ids(logits: Sequence[Sequence[float]] | torch.Tensor) -> np.ndarray:
    return np.argmax(_as_numpy(logits), axis=-1)


def compute_pred_confidences(
    logits: Sequence[Sequence[float]] | torch.Tensor,
) -> np.ndarray:
    probs = torch.softmax(torch.as_tensor(logits), dim=-1)
    return probs.max(dim=-1).values.numpy()


def decode_classification_logits(
    logits: torch.Tensor,
    id2label: dict[str, str],
) -> tuple[str, float]:
    probs = torch.softmax(logits, dim=-1)
    class_id = int(torch.argmax(probs).item())
    return str(id2label[str(class_id)]), float(probs[class_id].item())


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = [
    "compute_pred_confidences",
    "compute_pred_ids",
    "decode_classification_logits",
]
