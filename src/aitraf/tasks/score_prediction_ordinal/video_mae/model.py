"""Task-local ordinal classification model for VideoMAE score prediction."""

from __future__ import annotations

import torch
from dlordinal.losses import EMDLoss
from torch import nn


class ScorePredictionOrdinalModel(nn.Module):
    """Train a VideoMAE classifier with an ordinal-aware EMD loss."""

    def __init__(self, classifier: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.classifier = classifier
        self.loss_fn = EMDLoss(num_classes=num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.classifier(pixel_values=pixel_values)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        result: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            result["loss"] = self.loss_fn(logits, labels.long())

        return result


__all__ = ["ScorePredictionOrdinalModel"]
