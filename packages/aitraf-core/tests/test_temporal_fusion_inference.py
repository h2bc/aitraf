from __future__ import annotations

import pytest
import torch
from torch import nn

from aitraf_core.inference.models.video_mae_temporal_fusion import (
    predict_temporal_fusion_label,
    predict_temporal_fusion_logits,
)


class LogitModel(nn.Module):
    def forward(self, *, features: torch.Tensor) -> dict[str, torch.Tensor]:
        _ = features
        return {"logits": torch.tensor([[0.1, 2.0, 0.3]])}


def test_predict_temporal_fusion_logits_returns_single_row() -> None:
    logits = predict_temporal_fusion_logits(
        model=LogitModel(),
        features=torch.ones(6, 4),
    )

    assert logits.shape == (3,)
    assert torch.argmax(logits).item() == 1


def test_predict_temporal_fusion_label_decodes_confidence() -> None:
    label, confidence = predict_temporal_fusion_label(
        model=LogitModel(),
        features=torch.ones(6, 4),
        id2label={"0": "1", "1": "3", "2": "5"},
    )

    assert label == "3"
    assert 0.0 <= confidence <= 1.0


def test_predict_temporal_fusion_logits_rejects_output_without_logits() -> None:
    class BadModel(nn.Module):
        def forward(self, *, features: torch.Tensor) -> dict[str, torch.Tensor]:
            _ = features
            return {"not_logits": torch.tensor([1.0])}

    with pytest.raises(ValueError, match="missing logits"):
        predict_temporal_fusion_logits(model=BadModel(), features=torch.ones(6, 4))
