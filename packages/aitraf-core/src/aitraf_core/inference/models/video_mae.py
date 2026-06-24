"""VideoMAE inference helpers."""

from __future__ import annotations

import torch
from torch import nn

from aitraf_core.inference.classification import (
    classification_label_from_logits,
)


def predict_feature_vectors(
    *,
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    backbone = getattr(model, "videomae", model)

    with torch.no_grad():
        output = backbone(pixel_values=inputs)

    sequence_output = output.last_hidden_state
    feature_norm = getattr(model, "fc_norm", None)

    if feature_norm is not None:
        return feature_norm(sequence_output.mean(1)).float()

    return sequence_output[:, 0].float()


def predict(
    *,
    classifier_head: nn.Module,
    feature_vectors: torch.Tensor,
    id2label: dict[int, str],
) -> tuple[str, float]:
    classifier_head.eval()

    with torch.no_grad():
        logits = classifier_head(feature_vectors)
    return classification_label_from_logits(logits[0].float(), id2label)


__all__ = [
    "predict",
    "predict_feature_vectors",
]
