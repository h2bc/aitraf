"""Task-local pairwise comparison model for VideoMAE scorers."""

from __future__ import annotations

import torch
from torch import nn


class ScorePredictionPairwiseModel(nn.Module):
    """Score two clips with a shared scorer and train on their ordering."""

    def __init__(self, scorer: nn.Module, margin: float = 0.0) -> None:
        super().__init__()
        self.scorer = scorer
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def _score(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.scorer(pixel_values=pixel_values)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return logits.squeeze(-1)

    def forward(
        self,
        left_pixel_values: torch.Tensor,
        right_pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        left_scores = self._score(left_pixel_values)
        right_scores = self._score(right_pixel_values)
        pair_logits = right_scores - left_scores

        outputs: dict[str, torch.Tensor] = {"logits": pair_logits.unsqueeze(-1)}
        if labels is not None:
            targets = labels.float().mul(2).sub(1)
            outputs["loss"] = self.loss_fn(right_scores, left_scores, targets)
        return outputs


__all__ = ["ScorePredictionPairwiseModel"]
