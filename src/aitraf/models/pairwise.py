"""Generic pairwise ranking wrappers."""

from __future__ import annotations

import torch
from torch import nn


class PairwiseRanker(nn.Module):
    """Apply a shared scorer to two clips and compare their scalar scores."""

    def __init__(self, scorer: nn.Module, loss_fn: nn.Module | None = None) -> None:
        super().__init__()
        self.scorer = scorer
        self.loss_fn = loss_fn or nn.BCEWithLogitsLoss()

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
            outputs["loss"] = self.loss_fn(pair_logits, labels.float())
        return outputs


__all__ = ["PairwiseRanker"]
