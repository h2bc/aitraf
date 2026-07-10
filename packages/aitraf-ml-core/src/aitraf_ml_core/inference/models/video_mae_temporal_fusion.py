"""VideoMAE temporal-fusion inference helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from aitraf_ml_core.inference.classification import classification_label_from_logits


def predict(
    *,
    model: nn.Module,
    features: torch.Tensor,
    id2label: Mapping[int | str, str],
) -> tuple[str, float]:
    model.eval()
    model_device = next(model.parameters()).device
    with torch.inference_mode():
        output = model(features=features.to(model_device).unsqueeze(0))

    logits = output["logits"][0].float()
    return classification_label_from_logits(logits, id2label)


__all__ = ["predict"]
