"""Classification prediction helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def classification_ids_from_logits(
    logits: Sequence[Sequence[float]] | torch.Tensor,
) -> np.ndarray:
    return torch.argmax(torch.as_tensor(logits), dim=-1).cpu().numpy()


def classification_confidences_from_logits(
    logits: Sequence[Sequence[float]] | torch.Tensor,
) -> np.ndarray:
    probs = torch.softmax(torch.as_tensor(logits), dim=-1)
    return probs.max(dim=-1).values.cpu().numpy()


def classification_label_from_logits(
    logits: torch.Tensor,
    id2label: Mapping[int | str, str],
) -> tuple[str, float]:
    probs = torch.softmax(logits, dim=-1)
    class_id = int(torch.argmax(probs).item())
    if class_id in id2label:
        label = id2label[class_id]
    else:
        label = id2label[str(class_id)]
    return str(label), float(probs[class_id].item())


__all__ = [
    "classification_confidences_from_logits",
    "classification_ids_from_logits",
    "classification_label_from_logits",
]
